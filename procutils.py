'''Process utilities
'''

from __future__ import division
from __future__ import absolute_import

import sys
import os
import time
import errno
import subprocess
from socket import gethostname

from .pathutils import FilePath


def which(proc, which_path='/usr/bin/which'):
    '''uses unix program `which` to locate a program
    '''
    p = do_while_interrupted(lambda : subprocess.Popen([which_path, proc],
                                                  stdout=subprocess.PIPE, stdin=subprocess.PIPE))
    do_while_interrupted(lambda : p.stdin.close())
    output = do_while_interrupted(lambda : p.stdout.read())
    r = do_while_interrupted(lambda : p.wait())
    if r:
        raise subprocess.CalledProcessError(r, which_path)
    paths = map(FilePath, filter(None, output.split()))
    if not paths:
        raise ValueError('no such program %r' % (proc,))
    return paths[0]

def run_program(prog, *args, **kwds):
    prog = which(prog)
    p = do_while_interrupted(lambda : subprocess.Popen((prog,) + args, **kwds))
    run_program_notify(p, prog, args, kwds)
    r = do_while_interrupted(lambda : p.wait())
    if r:
        raise subprocess.CalledProcessError(r, prog)

def do_while_interrupted(func):
    assert callable(func)
    while True:
        try:
            return func()
        except (OSError,IOError), e:
            if e.errno != errno.EINTR:
                raise

run_program_notify_hook = []

def run_program_notify(proc, prog, args, kwds):
    for func in run_program_notify_hook:
        func(proc, prog, args, kwds)

def add_run_program_notify(func):
    assert callable(func)
    run_program_notify_hook.append(func)


class PidRecorder(object):

    def __init__(self, pid_path):
        self.pid_path = pid_path
        self.fp = None if self.pid_path is None else open(self.pid_path, 'w')
        self.msg('host %s', gethostname())
        add_run_program_notify(self.run_program_notify)

    def record(self, pid, name):
        self.msg('pid=%d name=%s stat=%s', pid, name, self.get_sys_proc_stat(pid))

    def record_this(self):
        self.record(os.getpid(), sys.argv[0])

    def run_program_notify(self, proc, prog, args, kwds):
        self.record(proc.pid, prog)

    def msg(self, msg, *args):
        if self.fp is not None:
            print >>self.fp, self.get_time_stamp(), msg % args

    @staticmethod
    def get_time_stamp():
        return time.strftime('%Y-%m-%dT%H:%M:%S')

    @staticmethod
    def get_sys_proc_stat(pid):
        sys_proc_path = FilePath('/proc/%d/stat' % (pid,))
        if not sys_proc_path.exists():
            return 'None'
        with open(sys_proc_path) as fp:
            return '[%s]' % fp.readline().strip()

