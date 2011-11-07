##-*- Mode: python -*-
## objstream.py - Sequentially write/read objects to file
## ---------------------------------------------------------------------
## This file is part of the hlab package and released as:
## Copyright (C) 2009, Matthew Hagy (hagy@gatech.edu)
## All rights reserved.
## ---------------------------------------------------------------------
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY# without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.
## ---------------------------------------------------------------------

from __future__ import with_statement
from __future__ import absolute_import

import sys
import cPickle as pickle
import zlib
import bz2
from __builtin__ import open as builtin_open

from .pathutils import FilePath
from .tempfile import temp_file_proxy


def open(filename, mode='r', **kwds):
    '''open a file for writing/reading objects
       return the proper file-like object
    '''
    if mode=='r':
        return Reader(builtin_open(filename, 'rb'), **kwds)
    elif mode=='w':
        return (BatchingWriter if 'batch_size' in kwds else
                Writer)(builtin_open(filename, 'wb'), **kwds)
    elif mode=='a':
        return (BatchingWriter if 'batch_size' in kwds else
                Writer)(builtin_open(filename, 'ab'), **kwds)
    else:
        raise ValueError('bad mode %r' % (mode,))

# File Format
#-----------------------------------------------------------
# Object are serialized to file with pickle.  Additionally,
# larger objects can also be compressed. Each object in the file
# is prefixed by a mode character to tell us how to deserialize
# the object later.  Additionally, we also store the number of
# bytes of which the object is composed of to faciliate seeking
# over the object, without actually reading it
#
# The format looks like:
#  __________________________________________________________________
#  | Mode (1 byte) | Size (* bytes) | Serialized Bytes (Size bytes) |
#  ------------------------------------------------------------------
#
# There are three compression modes
#    raw - No compression, direct pickling
#    bz2 compression - Compression using bz2 algorithm
#    zlib compression - Compression using DEFLATE (zlib) algorithm
#
# By default compression is accomplished with bz2
#

def serialize_unsigned_integer(op):
    '''serialize an integer using a variable number of bytes.
       first byte encodes number of following bytes that represent
       the number.
       these following bytes encode a big endian unsigned integer.
    '''
    assert isinstance(op, (int,long))
    assert op >= 0
    parts = []
    while op:
        parts.append(op & 0xff)
        op >>= 8
    parts.reverse()
    parts.insert(0, len(parts))
    return ''.join(map(chr, parts))

def write_size_t(fileobj, sz):
    fileobj.write(serialize_unsigned_integer(sz))

def read_size_t(fileobj):
    '''read variable byte size_t from current position in fileobj.
       leaves fileobj at end of size_t entry.
    '''
    n_bytes = fileobj.read(1)
    if not n_bytes:
        raise EOFError
    n_bytes = ord(n_bytes)
    bytes = fileobj.read(n_bytes)
    if len(bytes) < n_bytes:
        raise EOFError
    acc = 0
    for part in map(ord, bytes):
        acc <<= 8
        acc += part
    return acc


class Compressor(object):

    def __init__(self, mode_char, name, compress_func, decompress_func):
        self.mode_char = mode_char
        self.name = name
        self.compress_func = compress_func
        self.decompress_func = decompress_func

    def __repr__(self):
        return '<%s %s>' % (self.__class__.__name__, self.name)
    __str__ = __repr__

def identity(op): return op

compressors = dict((name, Compressor(mode, name, compress_func, decompress_func))
                   for name, mode, compress_func, decompress_func in [
    ['raw',  'R', identity,      identity],
    ['bz2',  'B', bz2.compress,  bz2.decompress],
    ['zlib', 'C', zlib.compress, zlib.decompress]])

compressor_modes = dict((c.mode_char,c) for c in compressors.itervalues())


class FileWrapper(object):

    def __init__(self, fileobj):
        self.fileobj = fileobj
        self.closed = False

    def close(self):
        if not self.closed:
            self.fileobj.close()
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
        return False

    def __del__(self):
        self.close()

    def ensure_not_closed(self):
        if self.closed:
            raise IOError("performing operation on a closed object stream")


class Writer(FileWrapper):
    '''Object to for serial writing of objects to an objstream
    '''

    def flush(self):
        self.ensure_not_closed()
        self.fileobj.flush()
        return self

    def write(self, op):
        self.ensure_not_closed()
        compressor, bytes = self.serialize(op)
        #rollback partial writes
        current = self.fileobj.tell()
        try:
            self.fileobj.write(compressor.mode_char)
            write_size_t(self.fileobj, len(bytes))
            self.fileobj.write(bytes)
        except:
            try:
                self.fileobj.truncate(current)
            except:
                pass
            raise
        return self

    pickle_protocol = 0

    def serialize(self, op):
        raw_bytes = pickle.dumps(op, self.pickle_protocol)
        compresor = self.get_compressor(raw_bytes)
        compress_bytes = compresor.compress_func(raw_bytes)
        if len(compress_bytes) < len(raw_bytes):
            return compresor, compress_bytes
        return compressors['raw'], raw_bytes

    min_compress_length = 256
    compressor = 'bz2'

    def get_compressor(self, raw_bytes):
        compressor = self.compressor
        if len(raw_bytes) < self.min_compress_length:
            compressor = 'raw'
        if isinstance(compressor, str):
            compressor = compressors[compressor]
        assert isinstance(compressor, Compressor)
        return compressor


class BatchingWriter(Writer):

    def __init__(self, fileobj, batch_size=10):
        super(BatchingWriter, self).__init__(fileobj)
        assert batch_size >= 1
        self.batch_size = batch_size
        self.acc_batch = []

    def write(self, obj):
        self.ensure_not_closed()
        self.acc_batch.append(obj)
        if len(self.acc_batch) == self.batch_size:
            self.flush()

    def flush(self):
        self.ensure_not_closed()
        batch = self.acc_batch
        self.acc_batch = []
        # could lose elements if error in write
        # likely not worth guarding against as files is likely
        # already corrupt at this point
        for op in batch:
            super(BatchingWriter, self).write(op)
        return super(BatchingWriter, self).flush()

    def close(self):
        if self.closed:
            assert not len(self.acc_batch)
        else:
            self.flush()
        super(BatchingWriter, self).close()


class CorruptFile(Exception):
    pass

class ObjectLocator(object):
    '''Describes how to read a specific object from file
    '''

    __slots__ = ['compressor','byte_offset','size']

    def __init__(self, compressor, byte_offset, size):
        '''compressor: Compressor for this object
           byte_offset: where the objects bytes begin
           size: number of bytes which make up object
        '''
        self.compressor = compressor
        self.byte_offset = byte_offset
        self.size = size

    def __repr__(self):
        return '<%s compressor=%s byte_offset=%d size=%d>' % (
            self.__class__.__name__, self.compressor.name,
            self.byte_offset, self.size)
    __str__ = __repr__

    def seek_to_start(self, fileobj):
        fileobj.seek(self.byte_offset)

    def seek_to_end(self, fileobj):
        fileobj.seek(self.byte_offset + self.size)

    def read_object(self, fileobj):
        self.seek_to_start(fileobj)
        if self.compressor is compressors['raw']:
            start = fileobj.tell()
            try:
                obj = pickle.load(fileobj)
            except EOFError:
                self.eof_error()
            except ValueError,e:
                self.value_error(e)
            assert fileobj.tell() - start == self.size
            return obj
        else:
            bytes = fileobj.read(self.size)
            if len(bytes) != self.size:
                assert len(bytes) < self.size
                self.eof_error()
            try:
                return pickle.loads(self.compressor.decompress_func(bytes))
            except ValueError, e:
                self.value_error(e)

    @staticmethod
    def eof_error():
        #shouldn't ever be called due to sanity checks when reading locators
        raise CorruptFile("EOF when reading object")

    @staticmethod
    def value_error(e):
        raise CorruptFile(str(e))


class Reader(FileWrapper):
    '''Provides random access to objects in an objstream
    '''

    def __init__(self, fileobj):
        super(Reader, self).__init__(fileobj)
        self.obj_locators = []
        self.next_index = 0
        self.seen_end = False

    def close(self):
        if not self.closed:
            del self.obj_locators
        super(Reader, self).close()

    # fileobj API
    def read(self):
        '''sequentially read object from file
        '''
        self.ensure_not_closed()
        obj =  self.read_index(self.next_index)
        self.next_index += 1
        return obj

    def seek(self, offset, whence=None):
        '''seek for sequential read using same API as fileobj seek
        '''
        self.ensure_not_closed()
        if not isinstance(offset, (int,long)):
            raise TypeError("bad offset %r" % (offset,))
        if whence==None or whence==0:
            if offset < 0:
                raise ValueError("can't seek beyond begining of file")
            self.set_index(offset)
        elif whence==1:
            new_index = self.next_index + offset
            if new_index < 0:
                raise ValueError("offset %d from %d is beyond begining of file" % (offset, self.next_index))
            self.set_index(new_index)
        elif whence==2:
            if offset >= 0:
                raise ValueError("can't seek beyond end of file")
            self.set_index(offset)
        else:
            raise ValueError("bad whence %s" % (whence,))

    # sequence API
    def __len__(self):
        self.ensure_not_closed()
        self.read_all_locators()
        return len(self.obj_locators)

    def __iter__(self):
        return self.iterslice()

    def __getitem__(self, item):
        self.ensure_not_closed()
        if isinstance(item, slice):
            return list(self.iterslice(item))
        try:
            itr = iter(item)
        except:
            pass
        else:
            return list(self[i] for i in itr)
        try:
            return self.read_index(item)
        except (EOFError, ValueError):
            raise IndexError("can't read item %d in stream of length %d" % (item, len(self)))

    def iterslice(self, start=None, stop=None, step=None):
        self.ensure_not_closed()
        if stop is None and step is None and isinstance(start, slice):
            slc = start
            start = slc.start
            stop = slc.stop
            step = slc.step
            del slc
        if step is None:
            step = 1
        if start is None:
            start = 0
        else:
            start = self.normalize_index(start)
        if stop is not None:
            stop = self.normalize_index(stop)
            if step * (stop - start) < 0:
                self.step_error(start, stop, step)
        elif step < 0:
            self.step_error(start, 'oo', step)
        if step==0 and start < stop:
            self.step_error(start, stop, step)
        inx = start
        while stop is None or (inx < stop if step > 0 else inx > stop):
            try:
                op = self.read_index(inx)
            except EOFError:
                break
            yield op
            inx += step
            self.set_index(inx)

    @staticmethod
    def step_error(start, stop, step):
        raise ValueError("can't step from %d to %s by %d" % (start, stop, step))

    # internals
    def set_index(self, inx):
        '''chose the index of the object to read next for
           sequential read.  can chose index beyond end of
           file, and this will raise EOFError on the next
           read.  negative indices imply from end of file,
           and this forces the reading of all locations to
           determine file size
        '''
        self.next_index = self.normalize_index(inx)
        # If we have the index, we can possibly decrease disk
        # latency for the next read by seeking to this position
        # now as it hints to the filesystem we may need this file
        # region soon.
        locator = self.get_locator(self.next_index)
        if locator is not None:
            locator.seek_to_start(self.fileobj)

    def read_index(self, index):
        '''read the object at a specific index
        '''
        locator = self.get_locator(self.normalize_index(index))
        if not locator:
            raise EOFError
        try:
            return locator.read_object(self.fileobj)
        except IOError, e:
            self.handle_io_error(e)
            raise EOFError

    def normalize_index(self, index):
        '''index object through same API as sequences (ie. negative
           numbers are from end)
        '''
        if index < 0:
            # reading backwards from end of file, need to know length
            self.read_all_locators()
            real_index = len(self.obj_locators) + index
            if real_index < 0:
                raise ValueError("index is %d before start of file with %d objects" %
                                 (index, len(self.obj_locators)))
            return real_index
        #don't yet check if index is beyond file, as more elements could
        #be added or index may not be used for reading
        return index

    def get_locator(self, real_index):
        assert real_index >= 0
        while real_index >= len(self.obj_locators) and not self.seen_end:
            self.read_next_locator()
        try:
            return self.obj_locators[real_index]
        except IndexError:
            return None

    def read_all_locators(self):
        while not self.seen_end:
            self.read_next_locator()

    def read_next_locator(self):
        '''read the next locator from the end of the stream
        '''
        if self.seen_end:
            return None
        if not self.obj_locators:
            self.fileobj.seek(0)
        else:
            self.obj_locators[-1].seek_to_end(self.fileobj)
        locator = self.read_locator()
        if locator is not None:
            self.obj_locators.append(locator)
        else:
            self.seen_end = True
        return locator

    def read_locator(self):
        mode_char = self.fileobj.read(1)
        if not mode_char:
            return self.handle_eof_mode()
        try:
            compressor = compressor_modes[mode_char]
        except KeyError:
            return self.handle_corrupt_mode(mode_char)
        try:
            size = read_size_t(self.fileobj)
        except EOFError:
            return self.handle_eof_size()

        #check there are sufficient bytes to read this object
        start_offset = self.fileobj.tell()
        try:
            self.fileobj.seek(start_offset + size - 1)
        except OverflowError:
            return self.handle_overflow_seek()

        last_byte = self.fileobj.read(1)
        if not last_byte:
            return self.handle_insufficient_obj_bytes()
        return ObjectLocator(compressor, start_offset, size)

    # there are various errors we can encounter while reading
    # from file.  these are handled by specifc methods to
    # allow overiding default actions

    ignore_corrupt_entries = False

    def handle_eof_mode(self):
        #end of stream
        return None

    def handle_corrupt_mode(self, mode_char):
        if self.ignore_corrupt_entries:
            return None
        raise CorruptFile("bad mode charcter \\x%02x" % (ord(mode_char),))

    def handle_eof_size(self):
        if self.ignore_corrupt_entries:
            return None
        raise CorruptFile("EOF within size header")

    def handle_overflow_seek(self):
        if self.ignore_corrupt_entries:
            return None
        raise CorruptFile("overflow in seek; likely corrupt size")

    def handle_insufficient_obj_bytes(self):
        if self.ignore_corrupt_entries:
            return None
        raise CorruptFile("there are insufficient bytes to read next object from file")

    def handle_io_error(self, e):
        if self.ignore_corrupt_entries:
            return None
        raise CorruptFile(str(e))


def open_cached_locators(filepath, cache_path=None, ignore_corrupt_entries=False):
    filepath = FilePath(filepath)
    fp = open(filepath)
    fp.ignore_corrupt_entries = ignore_corrupt_entries

    if cache_path is None:
        cache_path = FilePath(filepath + '.locs')
    cache_path = FilePath(cache_path)

    cache_was_read = False

    if cache_path.exists() and cache_path.mtime() > filepath.mtime():
        try:
            with open(cache_path, 'r') as cache_fp:
                fp.obj_locators = cache_fp.read()
        except (CorruptFile, EOFError):
            pass
        else:
            cache_was_read = True

    if not cache_was_read:
        fp.read_all_locators()
        with temp_file_proxy(cache_path, 'w', open=open) as cache_fp:
            cache_fp.pickle_protocol = 2
            cache_fp.write(fp.obj_locators)

    return fp


def fix_objstream(path, pickle_protocol=2, verbose=False):
    '''truncates corrupt entries at the end of an objstream
    '''
    splice_objstream(path, indices=None, out_path=path,
                     ignore_corrupt_entries=True,
                     pickle_protocol=pickle_protocol, verbose=verbose)

def splice_objstream(in_path, indices=None, out_path=None, pickle_protocol=2, verbose=False, ignore_corrupt_entries=False):
    if isinstance(indices, (int,long)):
        indices = xrange(indices)

    if out_path is None:
        out_path = in_path

    with temp_file_proxy(out_path, 'w', open=open) as out_fp:
        out_fp.pickle_protocol = pickle_protocol

        with open(in_path) as in_fp:
            in_fp.ignore_corrupt_entries = ignore_corrupt_entries

            try:
                for op in (iter(in_fp) if indices is None else (in_fp[i] for i in indices)):
                    if verbose:
                        print '.',
                        sys.stdout.flush()
                    out_fp.write(op)
            except CorruptFile:
                pass

            if verbose:
                print
