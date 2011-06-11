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
        self.filename = filename
        self.lockpath = self.mklockpath(self.filename)
        if lid is None:
           self.lid = self.randomLid()
        else:
           self.lid = str(lid)
        if aquire:
            self.lock(delay)

    @staticmethod
    def mklockpath(file):
        return '%s.lock' % file

    @classmethod
    def islocked(cls, file):
        return os.path.exists(cls.mklockpath(file))

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
        lockpath = self.lockpath
        host = socket.gethostname()
        pid = os.getpid()

        exists = os.path.exists
        rename = os.rename
        interval = 0.02

        #Write to tempfile in same dir
        content = self.makecontents(host, pid, self.lid)
        tmp = '%s_%s.%i_%s' % (lockpath,host,pid,self.lid[:40])

        def write_tmp():
            with file(tmp,'w') as fh:
                fh.write(content)

        write_tmp()
        try:
            while not self.locked:

                if not exists(lockpath):
                    try:
                        rename(tmp, lockpath)
                    except OSError:
                        # this can happen in various odd cases that I don't understand
                        # just try again
                        write_tmp()
                        continue
                    # Ensure that my rename succeeded
                    try:
                        with file(lockpath) as fh:
                            fhcontent = fh.read()
                    except (IOError, OSError):
                        pass
                    else:
                        if fhcontent == content:
                            self.locked = True
                            break

                    write_tmp()

                if delay <= 0.0:
                    raise LockError(self.filename, 'Could not lock %s; lock exits' % (self.filename,))

                time.sleep(interval)
                delay -= interval
        finally:
            try: os.unlink(tmp)
            except OSError:
                pass

    def unlock(self):
        '''Removes lockfile if we are locked'''
        if self.locked:
            if os.path.exists(self.lockpath):
                try: os.unlink(self.lockpath)
                except OSError,e:
                    if e.errno != errno.ENOENT:
                        raise LockError(self.filename, 'Failed to clear lock; %s' % (e,))
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




