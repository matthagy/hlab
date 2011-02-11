'''Process utilities
'''

from __future__ import division
from __future__ import absolute_import

import errno
import subprocess

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
    p = do_while_interrupted(lambda : subprocess.Popen((which(prog),) + args, **kwds))
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
