'''
'''

import sys
import os
import socket
import random
import time
import atexit
import errno
from threading import Thread, Condition

from .pathutils import FilePath

DEFAULT_DELAY = 1.0

class LockError(Exception):
    pass


class LockFile(object):

    #List of all lock files for process
    #Cleared in atexit
    _lockFiles = []

    def __init__(self, filename, delay=None, aquire=True, lid=None):
        '''Create a lock file (filename [,delay=None] [,aquire=True], [lid=None])
               filename - Path of file to lock
               delay - Seconds to delay for lock if auqire (defaults to DEFAULT_DELAY)
               aquire - Should I attempt to aquire lock immediatley?
               lid -  Lock ID (Resoure identifier string)
                      Generated randomly by default
        '''

        self._lockFiles.append(self)
        self.locked = False
        self.filename = FilePath(filename)
        self.lockpath = self.mklockpath(self.filename)
        if lid is None:
           self.lid = self.randomLid()
        else:
           self.lid = str(lid)
        if aquire:
            self.lock(delay)

    @staticmethod
    def mklockpath(path):
        return FilePath('%s.lock' % path)

    @classmethod
    def islocked(cls, file):
        return cls.mklockpath(file).exists()

    @classmethod
    def makecontents(cls, host, pid, lid):
        return '%s.%i.%s' % (host,pid,lid)

    @classmethod
    def readlock(cls, filename):
        lockpath = cls.mklockpath(filename)
        line = file(lockpath).readline().strip()
        host,pid,lid = line.split('.',2)
        pid = int(pid)
        return host,pid,lid

    @staticmethod
    def randomLid():
        return '%x' % random.randrange(1L<<100)

    def __str__(self):
        return self.lockpath

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

        if self.locked:
            return
        if delay is None:
            delay = DEFAULT_DELAY
        delay = float(delay)
        if delay < 0:
            raise ValueError('Wait must be positive, not %r' % delay)

        host = socket.gethostname()
        pid = os.getpid()
        content = self.makecontents(host, pid, self.lid)
        tmp_path = FilePath('%s_%s_%i_%s' % (self.lockpath,host,pid,self.lid[:40]))

        try:
            while delay >= 0.0:
                self.attempt_aquire(content, tmp_path)
                if self.locked:
                    break
                if delay > 0:
                    time.sleep(self.aquire_interval)
                delay -= self.aquire_interval
        finally:
            tmp_path.unlink_carefully()

        if not self.locked:
            raise LockError(self.filename, 'Could not lock %s; lock exits' % (self.filename,))

    def attempt_aquire(self, content, tmp_path):
        assert not self.locked

        if self.lockpath.exists():
            return

        with open(tmp_path, 'w') as fp:
            fp.write(content)

        try:
            os.link(tmp_path, self.lockpath)
        except OSError,e:
            if e.errno == errno.EEXIST:
                return
            raise

        tmp_path.unlink()

        self.locked = True

    def unlock(self):
        '''Removes lockfile if we are locked'''
        if self.locked:
            self.lockpath.unlink_carefully()
            self.locked = False

    @classmethod
    def _pruneLocks(cls):
        while cls._lockFiles:
            locks = cls._lockFiles[:]
            del cls._lockFiles[:]
            for lock in locks:
                try:
                    lock.unlock()
                except LockError,e:
                    print >>sys.stderr,str(e)

atexit.register(LockFile._pruneLocks)

islocked = LockFile.islocked


class LockToucher(object):

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




