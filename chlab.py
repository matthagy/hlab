'''Interface to c extension
'''

from __future__ import division
from __future__ import with_statement

import numpy as np

from . import cutil as C
from .pathutils import FilePath

libpath = FilePath(__file__).dsibling('chlab').child('libchlab.so')

if not libpath.exists():
    raise RuntimeError("missing extension %s" % (libchlab,))

libchlab = C.deflib(libpath,
                 symbol_translator = lambda s : 'CHLAB_' + s,
                 defs=[
    [C.c_void, 'acc_periodic_rs', [C.c_double_p, C.c_int,
                                   C.c_double, C.c_double,
                                   C.c_double_p, C.c_int,
                                   C.c_double_p]],
    [C.c_void, 'acc_periodic_orient', [C.c_double_p, C.c_int_p,
                                       C.c_int, C.c_double, C.c_double,
                                       C.c_double_p, C.c_double_p, C.c_int,
                                       C.c_double_p]]

    ])

acc_rs_dtype = np.dtype(C.c_int)

def acc_periodic_rs(acc_rs_array, r_min, r_prec,
                    positions, box_size):

    [acc_rs_array, r_min, r_prec, positions, box_size
     ] = acc_periodic_fixup(acc_rs_array, r_min, r_prec, positions, box_size)

    libchlab.acc_periodic_rs(
        acc_rs_array.ctypes.data_as(C.c_double_p),
        len(acc_rs_array),
        r_min, r_prec,
        positions.ctypes.data_as(C.c_double_p),
        len(positions),
        box_size.ctypes.data_as(C.c_double_p))

acc_orients_dtype = np.dtype(C.c_double)

def acc_periodic_orientation_rs(acc_rs_orient, acc_rs_Ns,
                                r_min, r_prec,
                                positions, orients, box_size):

    [acc_rs_Ns, r_min, r_prec, positions, box_size
     ] = acc_periodic_fixup(acc_rs_Ns, r_min, r_prec, positions, box_size)

    if not isinstance(acc_rs_orient, np.ndarray):
        raise TypeError("must accumulate into ndarray")

    orients = np.asarray(orients, dtype=np.dtype(C.c_double))

    libchlab.acc_periodic_orient(
        acc_rs_orient.ctypes.data_as(C.c_double_p),
        acc_rs_Ns.ctypes.data_as(C.c_int_p),
        len(acc_rs_Ns), r_min, r_prec,
        positions.ctypes.data_as(C.c_double_p),
        orients.ctypes.data_as(C.c_double_p),
        len(positions),
        box_size.ctypes.data_as(C.c_double_p))


def acc_periodic_fixup(acc_rs_array, r_min, r_prec, positions, box_size):

    if not isinstance(acc_rs_array, np.ndarray):
        raise TypeError("must accumulate into ndarray")
    if acc_rs_array.dtype != acc_rs_dtype:
        raise ValueError("bad array type; must be equivalent to c_int")
    if len(acc_rs_array.shape) != 1:
        raise ValueError("can only accumulate into flat arrays")

    r_min = float(r_min)
    if r_min < 0:
        raise ValueError("bad negative r_min=%r" % (r_min,))

    r_prec = float(r_prec)
    if r_prec < 0:
        raise ValueError("bad negative r_prec=%r" % (r_prec,))

    positions = np.asarray(positions, dtype=np.dtype(C.c_double))
    if len(positions.shape) != 2 or positions.shape[1] != 3:
        raise ValueError("bad positions shape %s" % (positions.shape,))

    box_size = np.asarray(box_size, dtype=np.dtype(C.c_double))
    if box_size.shape != (3,):
        raise ValueError("bad boxsize shape %s" % (box_size.shape,))

    return [acc_rs_array, r_min, r_prec, positions, box_size]


