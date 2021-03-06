
from __future__ import division
from __future__ import with_statement

import sys
import threading
import time
import cPickle as pickle
from functools import wraps


class NoWorkToDo(Exception):
    pass

@wraps
def synchronised(func):
    def wrapper(self, *args, **kwds):
        self.lock.acquire()
        try:
            return func(self, *args, **kwds)
        finally:
            self.lock.release()
    return wrapper


class IOThreadQueue(threading.Thread):

    open_flags = 'r'

    def __init__(self, fp):
        if isinstance(fp, str):
            fp = open(fp, self.open_flags)
        self.fp = fp
        self.que = []
        self.quit = False
        self.lock = threading.Lock()
        self.context_depth = 0
        super(IOThreadQueue, self).__init__()

    @synchronised
    def que_len(self):
        return len(self.que)

    @synchronised
    def que_pop(self, i=-1):
        return self.que.pop(i)

    @synchronised
    def que_push(self, op):
        self.que.append(op)

    @synchronised
    def que_try_pop(self, i=-1):
        try:
            return self.que.pop(i)
        except IndexError:
            return None

    SLEEP_DURATION = 0.05
    def run(self):
        while not self.quit:
            try:
                self.run_cycle()
            except NoWorkToDo:
                time.sleep(self.SLEEP_DURATION)

    def finish(self):
        self.quit = True
        if self.isAlive():
            self.join()
        self.fp.close()

    def __enter__(self):
        self.context_depth += 1
        if not self.isAlive():
            self.start()
        return self

    def __exit__(self, *exc_info):
        self.context_depth -= 1
        if not self.context_depth:
            self.finish()


class OutputThread(IOThreadQueue):

    open_flags = 'w'

    def run_cycle(self):
        if not self.que_len():
            raise NoWorkToDo()
        self.dump_output()
        self.fp.flush()


class OverwrittingPicklingThread(OutputThread):

    def dump_output(self):
        while self.que_len():
            state = self.que_pop(0)
        self.fp.seek(0)
        self.fp.truncate()
        pickle.dump(state, self.fp)


class AppendingPicklingThread(OutputThread):

    open_flags = 'w'

    def dump_output(self):
        if not self.que_len():
            raise NoWorkToDo
        while self.que_len():
            state = self.que_pop(0)
            pickle.dump(state, self.fp)
        self.fp.flush()


class ReadingThread(IOThreadQueue):

    max_que = 5
    open_flags = 'r'

    def process(self, op):
        return op

    def run_cycle(self):
        if self.que_len() > self.max_que:
            raise NoWorkToDo
        try:
            state = pickle.load(self.fp)
        except (ValueError,IOError,EOFError):
            self.quit = True
        else:
            entry = self.process(state)
            if entry is not None:
                self.que_push(entry)

    def __iter__(self):
        with self:
            while not self.quit:
                if not self.isAlive():
                    raise EOFError
                state = self.que_try_pop(0)
                if state is None:
                    time.sleep(0.05)
                else:
                    yield state



class ReaderMultiplexer(object):

    STATES = ST_INIT, ST_RUNNING, ST_EXHAUSTED = range(3)

    def __init__(self, readers):
        self.readers = list(readers)
        self.reset_pending_state()
        self.state = self.ST_INIT
        self.context_depth = 0

    @classmethod
    def fromfiles(cls, files, reader_cls=ReadingThread):
        return cls(reader_cls(f) for f in files)

    def reset_pending_state(self):
        self.pending_state = [None] * len(self.readers)

    def pending_state_complete(self):
        return None not in self.pending_state

    def try_pull(self):
        if not self.state==self.ST_RUNNING:
            raise RuntimeError('not running')
        something_new = False
        for i,r in enumerate(self.readers):
            sys.stdout.flush()
            if self.pending_state[i] is not None:
                continue
            if r.quit or not r.isAlive():
                raise EOFError
            self.pending_state[i] = r.que_try_pop(0)
            something_new = something_new or self.pending_state[i] is not None
        return something_new

    def pull(self):
        while not self.pending_state_complete():
            if not self.try_pull():
                time.sleep(0.05)
        state = self.pending_state
        self.reset_pending_state()
        return state

    def start(self):
        if self.state==self.ST_INIT:
            for r in self.readers:
                r.start()
            self.state = self.ST_RUNNING

    def finish(self):
        if self.state==self.ST_RUNNING:
            self.state = self.ST_EXHAUSTED
            for r in self.readers:
                r.finish()

    def __enter__(self):
        self.context_depth += 1
        self.start()

    def __exit__(self, *exc_info):
        self.context_depth -= 1
        if not self.context_depth:
            self.finish()

    def __iter__(self):
        with self:
            while True:
                try:
                    state = self.pull()
                except EOFError:
                    break
                yield state


