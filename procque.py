from __future__ import with_statement


import os
import sys
from time import sleep
import subprocess
from Queue import Empty
from multiprocessing import cpu_count, Queue, Process
import socket

from hlab.pathutils import DirPath, FilePath
from hlab import objstream
from pizza.dump import dump

from util import State

mydir = FilePath(__file__).parent()

queue = None
hostname = socket.gethostname()

def run_multiplex(worker_func, work_seq):
    global queue
    queue = Queue(-1)
    for op in work_seq:
        queue.put(op)
    procs = [Process(target=worker_target, args=(prid,worker_func)) for prid in xrange(cpu_count())]
    print 'starting',len(procs),'processes'
    for proc in procs:
        proc.start()
    print 'running'
    for proc in procs:
        proc.join()
    print 'finished'

def worker_target(prid, worker_func):
    global queue
    while 1:
        try:
            bytes,output_path = queue.get(block=False, timeout=2)
        except Empty:
            sleep(1)
            if queue.empty():
                print prid, 'exiting'
                return
            continue
        else:
            run(bytes, output_path, prid)
