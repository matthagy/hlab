'''Stream of fixed-size arrays
'''

from __future__ import with_statement
from __future__ import absolute_import

from __builtin__ import open as builtin_open
from os import SEEK_END

import numpy as np

from .pathutils import FilePath
from .tempfile import temp_file_proxy
from .objstream import FileWrapper


def open(filename, mode, dtype, truncate_corrupted=False, **kwds):
    '''open a file for writing/reading objects
       return the proper file-like object
    '''
    if mode=='r':
        return Reader(builtin_open(filename, 'rb'), dtype, **kwds)
    elif mode=='w':
        return Writer(builtin_open(filename, 'wb'), dtype, **kwds)
    elif mode=='a':
        check_for_appending(filename, dtype, truncate_corrupted)
        return Writer(builtin_open(filename, 'ab'), dtype, **kwds)
    else:
        raise ValueError('bad mode %r' % (mode,))


class FixedFileWrapper(FileWrapper):

    def __init__(self, fileobj, dtype):
        super(FixedFileWrapper, self).__init__(fileobj)
        self.dtype = np.dtype(dtype)

class CorruptFile(IOError):
    pass

def check_for_appending(filename, dtype, truncate_corrupted):
    dtype = np.dtype(dtype)
    with builtin_open(filename, 'rb') as fp:
        fp.seek(0, SEEK_END)
        n_bytes = fp.tell()
    n,extra = divmod(n_bytes, dtype.itemsize)

    if not extra:
        return

    if not truncate_corrupted:
        raise CorruptFile('extra bytes at end of file')

    with builtin_open(filename, 'ab') as fp:
        fp.seek(n_bytes - extra)
        fp.truncate()


class Writer(FixedFileWrapper):
    '''Object to for serial writing of objects to an objstream
    '''

    def __init__(self, fileobj, dtype, batch_size=1):
        super(Writer, self).__init__(fileobj, dtype)
        assert batch_size is None or isinstance(batch_size, (int,long))
        assert batch_size is None or batch_size >= 1
        self.batch_size = batch_size
        self.acc_batch = []

    def write(self, op):
        self.ensure_not_closed()
        if isinstance(op, list):
            op = tuple(op)
        arr = np.asarray(op, dtype=self.dtype)

        assert self.batch_size is None or len(self.acc_batch) < self.batch_size
        self.acc_batch.append(arr)

        if self.batch_size is not None and len(self.acc_batch) == self.batch_size:
            self.flush()
        assert self.batch_size is None or len(self.acc_batch) < self.batch_size

    def flush(self):
        if not self.acc_batch:
            return

        batch = np.array(self.acc_batch, dtype=self.dtype)
        self.acc_batch = []

        #attempt to rollback partial writes
        current = self.fileobj.tell()
        try:
            batch.tofile(self.fileobj)
            self.fileobj.flush()
        except:
            try:
                self.fileobj.truncate(current)
            except:
                pass
            raise

    def close(self):
        if self.closed:
            assert not len(self.acc_batch)
        else:
            self.flush()
        super(Writer, self).close()



class Reader(FixedFileWrapper):

    def __len__(self):
        try:
            return self.cached_length
        except AttributeError:
            self.fileobj.seek(0, SEEK_END)
            bytes = self.fileobj.tell()
            self.cached_length = bytes // self.dtype.itemsize
            return self.cached_length

    def normalize_index(self, org_inx, default=0):
        if org_inx is None:
            org_inx = default

        N = len(self)
        inx = org_inx
        if inx < 0:
            inx = N + inx

        if not (0 <= inx < N):
            raise IndexError("bad index %d not in [0:%d)" % (org_inx, N))
        return inx

    def read_batch(self, start_index=0, count=1):
        self.ensure_not_closed()
        start_index = self.normalize_index(start_index)
        assert start_index >= 0
        assert count >= 1
        self.fileobj.seek(self.dtype.itemsize * start_index)
        arr = np.fromfile(self.fileobj, dtype=self.dtype, count=count)
        if arr.shape[0] < count:
            raise EOFError
        assert arr.shape[0] == count
        return arr

    def read_one(self, index=0):
        return self.read_batch(start_index=index, count=1)[0]

    def chunking_iter(self, start=0, stop=None, chunk_size=None):
        if chunk_size is None:
            chunk_size = self.calculate_default_chunk_size()

        start = self.normalize_index(start, 0)
        stop = self.normalize_index(stop, -1)
        count = stop - start
        assert count > 0, 'bad count %d' % (count,)

        n_chunks, extra = divmod(count, chunk_size)
        for chunk_i in xrange(n_chunks):
            for el in self.read_batch(start_index=start + chunk_i, count=chunk_size):
                yield el

        if extra:
            for el in self.read_batch(start_index=start + n_chunks, count=extra):
                yield el

    default_chunk_bytes = 0x1000

    def calculate_default_chunk_size(self):
        return self.default_chunk_bytes // self.dtype.itemsize

    def __iter__(self):
        return self.chunking_iter()

    def stepping_iter(self, start=0, stop=None, step=1):
        start = self.normalize_index(start, 0)
        stop = self.normalize_index(stop, -1)

        for index in xrange(start, stop, step):
            yield self.read_one(index)

    def __getitem__(self, item):
        if np.isscalar(item):
            return self.read_one(item)

        if isinstance(item, slice):
            if item.step == 1:
                return np.array(list(self.chunking_iter(item.start, item.step)))
            elif item.step == -1:
                return np.array(list(self.chunking_iter(item.stop, item.start)))[::-1]
            else:
                return np.array(list(self.stepping_iter(item.start, item.stop, item.step)))

        base_indices = np.asarray(item)
        return np.array([self.read_batch(self.normalize_index(index), count=1)
                         for index in base_indices.flat]).reshape(base_indices.shape)
