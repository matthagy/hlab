

from __future__ import division

import os
import new
import sys
from ctypes import (c_void_p, c_char, c_int, c_size_t, c_ulong, c_double,
                    cast, sizeof, pointer, POINTER, Structure, byref)

from hlab.pathutils import FilePath
from hlab.prof import Distribution
from hlab.xclib import deflib, as_ctype
import numpy

#from constants import *
NULL = cast(0, c_void_p)
c_int_p = POINTER(c_int)
c_int_pp = POINTER(c_int_p)
c_double_p = POINTER(c_double)
c_double_pp = POINTER(c_double_p)
c_char_p = POINTER(c_char)
c_char_pp = POINTER(c_char_p)
c_ulong_p = POINTER(c_ulong)
c_void = None
c_size_t_p = POINTER(c_size_t)

class BaseStructure(Structure):
    '''Extends ctypes.Structure
    '''

    _defaults_ = {}

    def __init__(self, *args, **kwds):
        if len(args) > len(self._fields_):
            raise ValueError("bad #args %d to %s constructor; takes at most %d" %
                             (len(args), self.__class__.__name__, len(self._fields_)))
        for n,v in self._defaults_.iteritems():
            setattr(self, n, v)
        for (n,_),v in zip(self._fields_, args):
            assert n not in kwds
            setattr(self, n, v)
        for n,v in kwds.iteritems():
            setattr(self, n, v)

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__,
                           ', '.join('%s=%r' % (n,getattr(self,n))
                                     for n,_ in self._fields_))
    def __str__(self):
        return repr(self)



def defstruct(name, *dfs, **kwds):
    '''DSL to declare structures
    '''
    fields = []
    for df in dfs:
        ct = df[0] #as_ctype(df[0])
        for n in df[1:]:
            fields.append((n, ct))
    dct = dict(_fields_=fields)
    dct.update(kwds)
    cls = new.classobj(name, (BaseStructure,), dct)
    gbls = sys._getframe(1).f_globals
    gbls[name] = cls
    gbls[name + '_p'] = POINTER(cls)


#vector.h
def vec_t_init(self, *args):
    if len(args)==1:
        args = args[0]
    self.x, self.y, self.z = args

defstruct('vec_t',
          (c_double, 'x', 'y', 'z'),
          __init__=vec_t_init,
          __iter__= lambda self: iter(list([self.x, self.y, self.z])),
          __array__=lambda self: numpy.array(list(self)))

defstruct('complex_t',
          (c_double, 'real', 'imag'))

def assign_vec(vec, seq):
    [vec.x, vec.y, vec.z] = seq
