'''eXperimental declarative interface to c-libraries through ctypes
'''

from __future__ import absolute_import

import sys
import new
import ctypes as C

from HH2.pathutils import FilePath

from jamenson.runtime import atypes
from jamenson.runtime.multimethod import MultiMethod, defmethod
from jamenson.runtime.atypes import intersection, Seq, typep, complement, anytype
from jamenson.runtime.atypes.accessors import Item

from .func import identity, partial, return_it

__all__ = '''
deflib defstruct
'''.split()

as_ctype = MultiMethod('as_ctype')

_CData = C._SimpleCData.mro()[1]
defmethod(as_ctype, 'lambda x: isinstance(x, type) and issubclass(x,_CData)')(identity)

defmethod(as_ctype, 'type(None)')(identity)

class CLibBase(object):
    pass

def deflib(filepath,
           symbol_translator=identity,
           defs=[]):
    filepath = FilePath(filepath)
    assert filepath.exists()
    dct = {}
    clib = C.CDLL(filepath)
    for df in defs:
        dolibdef(symbol_translator, clib, dct, df)
    return new.classobj(filepath.basename(),
                        (CLibBase,), dct)()

dolibdef = MultiMethod(name='dolibdef',
                    signature='symbol_translator, clib, class_dict, _def',
                    doc='''
                    Update dictionary to reflect information of this definition
                    ''',
                    cache=False)

seq_type = list,tuple
funcdef_type = intersection(seq_type,
                            lambda x : len(x) == 3,
                            Item(1, str),
                            Item(2, seq_type))

datadef_type = intersection(seq_type,
                            lambda x : len(x) >= 2,
                            Item(1, str),
 Item(slice(1, None), Seq(str)))

def assign_dct(dct, name, op):
    if name in dct:
        raise RuntimeError("redfining library attribute %s" % (name,))
    dct[name] = op


@defmethod(dolibdef, [anytype, anytype, dict, funcdef_type])
def meth(symbol_translator, clib, dct, fundef):
    restype, name, argtypes = fundef
    c_func = getattr(clib, symbol_translator(name))
    c_func.argtypes = map(as_ctype, argtypes)
    c_func.restype = as_ctype(restype)
    assign_dct(dct, name, c_func)

def base_library_func(c_func, name, restype, argtypes, *args):
    if not len(args) == len(argtypes):
        raise ValueError("bad #arguments to %s; expected %s got %s"
                         % (name, len(args), len(argtypes)))

def make_accessor(clib, name, var):
    return var
#     def g():
#         return var.value
#     def s(value):
#         var.value = value
#     return property(g, s)

@defmethod(dolibdef, [anytype, anytype, dict, datadef_type])
def meth(symbol_translator, clib, dct, datadef):
    ctp = as_ctype(datadef[0])
    for name in datadef[1:]:
        accessor = make_accessor(clib, name, ctp.in_dll(clib, symbol_translator(name)))
        assign_dct(dct, name, accessor)


class BaseStructure(C.Structure):
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
    gbls[name + '_p'] = C.POINTER(cls)

