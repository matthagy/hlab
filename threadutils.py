'''
'''

import ctypes
import threading
import inspect


def raise_in_main_thread(exctype):
    '''Allows an exception to be raised in a thread
    '''
    main_thread_id, main_thread = get_main_thread()
    async_raise(main_thread_id, exctype)

def async_raise(tid, exctype):
    '''Raises an exception in the threads with id tid
       Copied from http://stackoverflow.com/questions/323972/is-there-any-way-to-kill-a-thread-in-python
    '''
    if not inspect.isclass(exctype):
        raise TypeError("Only types can be raised (not instances)")
    ctypes.pythonapi.PyThreadState_SetAsyncExc.argtypes = [ctypes.c_long, ctypes.py_object]
    ctypes.pythonapi.restype = ctypes.c_int
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        # """if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"""
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, 0)
        raise SystemError("PyThreadState_SetAsyncExc failed")

def get_main_thread():
    for tid,thread in threading._active.items():
        if thread.getName() == 'MainThread':
            return tid, thread
    assert 0
