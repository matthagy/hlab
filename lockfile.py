'''Utilties for locking exclusive access to a file path.
'''

import sys
import os
import socket
import random
import time
import atexit
import errno
from threading import Thread, Condition
from contextlib import contextmanager

from .pathutils import FilePath

DEFAULT_DELAY = 0.0

class LockError(Exception):
    pass

class LockAquireFailed(LockError):
    pass

@contextmanager
def ignore_exist_error():
    try:
        yield
    except OSError,e:
        if e.errno != errno.EEXIST:
            raise

class LockFile(object):

    # List of all lock files for this process
    # Cleared in atexit
    _lock_files = []

    def __init__(self, filename, delay=None, aquire=True, lid=None):
        '''Create a lock file (filename [,delay=None] [,aquire=True], [lid=None])
               filename - Path of file to lock
               delay - Seconds to wait for lock if already aquired (defaults to DEFAULT_DELAY)
               aquire - Whether to aquire lock at creation of LockFile
               lid -  Lock ID (Resoure identifier string) Generated randomly by default
        '''

        # should add a thread lock for this operation (see _prune_locks)
        self._lock_files.append(self)

        self.locked = False
        self.filename = FilePath(filename)
        self.lockpath = self.make_lock_path(self.filename)
        if lid is None:
           self.lid = self.make_random_lock_id()
        else:
           self.lid = str(lid)
        if aquire:
            self.lock(delay)

    @staticmethod
    def make_lock_path(path):
        return FilePath('%s.lock' % path)

    @classmethod
    def is_locked(cls, file):
        return cls.make_lock_path(file).exists()

    @classmethod
    def make_contents(cls, host, pid, lid):
        return '%s.%i.%s' % (host,pid,lid)

    @classmethod
    def read_lock(cls, filename):
        lockpath = cls.make_lock_path(filename)
        line = file(lockpath).readline().strip()
        host,pid,lid = line.split('.',2)
        pid = int(pid)
        return host,pid,lid

    @staticmethod
    def make_random_lock_id():
        return '%x' % random.randrange(1L<<100)

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, str(self.filename))

    def __enter__(self):
        self.lock()
        return self

    def __exit__(self, *exc_info):
        self.unlock()
        return False

    aquire_interval = 0.05

    def lock(self, delay=None):
        '''Aquires lock, if not already locked.
           lock([,delay=None])
               delay - Seconds to delay for a lock
                       Defaults to DEFAULT_DELAY
        '''

        if delay is None:
            delay = DEFAULT_DELAY
        remaining_delay = float(delay)
        if remaining_delay < 0:
            raise ValueError('delay must be positive, not %r' % remaining_delay)

        if self.locked:
            return

        host = socket.gethostname()
        pid = os.getpid()
        content = self.make_contents(host, pid, self.lid)
        tmp_path = FilePath('%s_%s_%i_%s' % (self.lockpath, host, pid, self.lid[:40]))
        assert not tmp_path.exists()

        try:
            first_pass = True
            while 1:
                self.attempt_aquire(content, tmp_path, first_pass)
                if (self.locked or remaining_delay <= 0.0):
                    break
                first_pass = False
                time.sleep(self.aquire_interval)
                remaining_delay -= self.aquire_interval
        finally:
            tmp_path.unlink_carefully()

        if not self.locked:
            raise LockAquireFailed(self.filename, 'Could not lock %s; lock exits' % (self.filename,))

    def attempt_aquire(self, content, tmp_path, first_pass):
        assert not self.locked

        # check if someone else already aquired lock
        if self.lockpath.exists():
            return

        if first_pass:
            with open(tmp_path, 'w') as fp:
                fp.write(content)

        try:
            os.link(tmp_path, self.lockpath)
        except OSError,e:
            if e.errno == errno.EEXIST:
                return
            raise

        #tmp_path.unlink()

        self.locked = True

    def unlock(self):
        '''Removes lockfile if we are locked'''
        if self.locked:
            self.lockpath.unlink_carefully()
            self.locked = False

    @classmethod
    def _prune_locks(cls):
        # should add a thread lock for this portion of code
        # can have issues if locks are added while this code is running
        while cls._lock_files:
            locks = cls._lock_files[::]
            del cls._lock_files[::]
            for lock in locks:
                try:
                    lock.unlock()
                except LockError,e:
                    print >>sys.stderr, str(e)

atexit.register(LockFile._prune_locks)

is_locked = LockFile.is_locked


class LockToucher(object):
    """Touches a LockFile's lock at a specific interval.
       Used to know which locks are still active and which locks
       can be associated with a process that died without proper cleanup.
    """

    def __init__(self, lockfile, touch_frequency):
        if isinstance(lockfile, LockFile):
            lockfile = lockfile.lockpath
        assert isinstance(lockfile, str)
        self.lockfile = lockfile
        self.touch_frequency = touch_frequency
        self.thread = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc_info):
        self.stop()
        return False

    def start(self):
        if self.thread is not None:
            return
        self.thread_stop = False
        self.thread = Thread(target=self.thread_target)
        self.condition = Condition()
        self.thread.start()

    def stop(self):
        thread = self.thread
        self.thread = None
        if thread is None:
            return
        self.condition.acquire()
        self.thread_stop = True
        self.condition.notify()
        self.condition.release()
        thread.join()

    def thread_target(self):
        if not self.touch():
            return
        while True:
            self.condition.acquire()
            try:
                self.condition.wait(self.touch_frequency)
                if self.thread_stop:
                    break
            finally:
                self.condition.release()
            if not self.touch():
                return

    def touch(self):
        try:
            os.utime(self.lockfile, None)
        except OSError,e:
            #likely means file was deleted
            return False
        return True




