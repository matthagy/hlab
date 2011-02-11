'''eXperimental algebric relationship framework using multimethods
'''

from __future__ import absolute_import

#from numpy import number as numpy_number_type

from jamenson.runtime import atypes
from jamenson.runtime.multimethod import (MultiMethod, defmethod, defboth_wrapper,
                                          current_method)
from jamenson.runtime.atypes import as_optimized_type, intersection, Seq, typep, complement, anytype
from jamenson.runtime.atypes.accessors import Item

from .func import identity, partial, return_it

__add__ = '''
   unop_names binop_names AlgebraBase scalar_number_type
'''.split()

scalar_number_type = as_optimized_type((int,long,float))


unop_names = '''
neg pos
'''.split()

__add__ += unop_names

binop_names = '''
add mul sub div truediv eq neq pow
'''.split()

__add__ += binop_names

def make_mms():
    gbls = globals()
    global mm_unop_base
    mm_unop_base = MultiMethod(name='mm_unop_base')
    for name in unop_names:
        mm_name = 'mm_' + name
        gbls[mm_name] = MultiMethod(name=mm_name,
                                    #signature='op',
                                    doc='''multimethod for unary operation %s
                                    ''' % (name,),
                                    inherit_from=[mm_unop_base])
    global mm_binop_base
    mm_binop_base = MultiMethod(name='mm_binop_base')
    for name in binop_names:
        mm_name = 'mm_' + name
        gbls[mm_name] = MultiMethod(name=mm_name,
                                    #signature='object,object',
                                    doc='''multimethod for binary operation %s
                                    ''' % (name,),
                                    inherit_from=[mm_binop_base])
        gbls['defboth_mm_' + name] = partial(defboth_wrapper, gbls[mm_name])

make_mms()
del make_mms

class AlgebraBase(object):

    def make_methods(locs=locals()):
        def make_wrapper(name, func):
            meth_name = '__%s__' % name
            func.func_name = meth_name
            locs[meth_name] = func
        def make_unop_wrapper(name):
            mm = gbls['mm_%s' % name]
            make_wrapper(name, lambda op: mm(op))
        def make_binop_wrapper(name):
            mm = gbls['mm_%s' % name]
            make_wrapper(name, lambda lop, rop: mm(lop, rop))
        def make_binrop_wrapper(name):
            mm = gbls['mm_%s' % name]
            make_wrapper('r'+name, lambda rop, lop: mm(lop, rop))
        gbls = globals()
        for name in unop_names:
            make_unop_wrapper(name)
        for name in binop_names:
            make_binop_wrapper(name)
            make_binrop_wrapper(name)

    make_methods()
    del make_methods


@defmethod(mm_unop_base, [anytype])
def meth(op):
    return NotImplemented
    raise TypeError("no method defined for %s wtih type %s" %
                    (current_method().name, type(op).__name__))


@defmethod(mm_binop_base, [anytype, anytype])
def meth(lop, rop):
    return NotImplemented
    raise TypeError("no method defined for %s with types (%s,%s)" %
                    (current_method().name,
                     type(lop).__name__,
                     type(rop).__name__))

@defmethod(mm_eq, [AlgebraBase, AlgebraBase])
def meth(a,b):
    return a is b

@defmethod(mm_neq, [AlgebraBase, AlgebraBase])
def meth(a,b):
    return not mm_eq(a,b)

@defmethod(mm_sub, [AlgebraBase, scalar_number_type])
def meth(a,s):
    return mm_add(a, -s)

@defmethod(mm_div, [AlgebraBase, scalar_number_type])
def meth(a,s):
    return mm_mul(a, 1.0 / float(s))


class DivAlgebraBase(AlgebraBase):

    pass

@defmethod(mm_truediv, [DivAlgebraBase, anytype])
def meth(dab, x):
    return mm_div(dab, x)

@defmethod(mm_truediv, [anytype, DivAlgebraBase])
def meth(x, dab):
    return mm_div(x, dab)
