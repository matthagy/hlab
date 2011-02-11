
import warnings
warnings.warn(
    'hlab.linalg is depreciated. Use corresponding routings in linutil'
    DeprecationWarning, stacklevel=1)


from math import *
from numpy import *


vec = lambda *args: array(args)
vlen = lambda v: dot(v,v) ** 0.5
vunit = lambda v: v / vlen(v)
vcos = lambda v1,v2: dot(v1,v2)/sqrt(dot(v1,v1)*dot(v2,v2))
vang = lambda v1,v2: acos(vcos(v1,v2))
vcross = cross
def vproj(v, dir):
    dir = unit(dir)
    return dot(v,dir) * dir

def vset(a):
    return set(map(tuple, a))

def outer_op_flat(op, a, b):
    '''calculates vectorwise (vetor order N) outer operation

       inputs:
         a.shape = (la, N)
         b.shape = (lb, N)
       output:
         c.shape = (la*lb, N)

    '''

    '''
        operation:

            |       vector i of a      |  vector i+1 of a
            | a_i0 | a_i1 | ... | a_iN |         ...
    v  -----|------------------------------------------------
       b_j0 | c_k0 |      |     |      |
    j  _____|______|______|_____|______|
       b_j1 |      | c_k1 |     |      |
    o  _____|______|______|_____|______|
    f  ...  |      |      | ... |      |
       ____ |______|______|_____|______|
    b  b_jN |      |      |     |      |
       _____|______|______|_____|______|
         .  |
         .  |
         .  |
    '''

    la,Na = a.shape
    lb,Nb = b.shape
    assert Na==Nb
    m = op.outer(a.ravel(), b.ravel())
    return concatenate([diag(m, Na*i) for i in xrange(-la, lb+1)]).reshape(la*lb, Na)

def vec_grid(spacings, order=None, index=None):
    if order is None:
        order = len(spacings)
    if index is None:
        index = 0
    offsets = multiply.outer([1 if i==index else 0 for i in xrange(order)],
                             asarray(spacings[0])
                             ).swapaxes(1,0).reshape(len(spacings[0]), order)
    if len(spacings)==1:
        return offsets
    else:
        return outer_op_flat(add, offsets, vec_grid(spacings[1:], order, index+1))
