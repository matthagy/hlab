'''Interface to c extension
'''

from __future__ import division
from __future__ import with_statement

import numpy as np

from . import cutil as C
from .pathutils import FilePath

libpath = FilePath(__file__).dsibling('chlab').child('libchlab.so')

if not libpath.exists():
    raise RuntimeError("missing extension %s" % (libpath,))

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
                                       C.c_double_p]],
    [C.c_void, 'acc_periodic_orient_position', [C.c_int_p,
                                                C.c_int, C.c_double,
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
    acc_rs_array = validate_acc_array(acc_rs_array)
    if len(acc_rs_array.shape) != 1:
        raise ValueError("can only accumulate into flat arrays")

    r_min = float(r_min)
    if r_min < 0:
        raise ValueError("bad negative r_min=%r" % (r_min,))

    r_prec = validate_prec(r_prec)
    positions = validate_positions(positions)
    box_size = validate_box_size(box_size)

    return [acc_rs_array, r_min, r_prec, positions, box_size]

def validate_acc_array(acc_array):
    if not isinstance(acc_array, np.ndarray):
        raise TypeError("must accumulate into ndarray")
    if acc_array.dtype != acc_rs_dtype:
        raise ValueError("bad array type; must be equivalent to c_int")
    return acc_array

def validate_prec(prec):
    prec = float(prec)
    if prec < 0:
        raise ValueError("bad negative prec=%r" % (prec,))
    return prec

def validate_positions(positions):
    positions = np.asarray(positions, dtype=np.dtype(C.c_double))
    if len(positions.shape) != 2 or positions.shape[1] != 3:
        raise ValueError("bad positions shape %s" % (positions.shape,))
    return positions

def validate_orientations(orientations, positions):
    orients = np.asarray(orients, dtype=np.dtype(C.c_double))
    if orients.shape != positions.shape:
        raise ValueError("inconsistent positions/orientations array shapes")
    return orientations

def validate_box_size(box_size):
    box_size = np.asarray(box_size, dtype=np.dtype(C.c_double))
    if box_size.shape != (3,):
        raise ValueError("bad boxsize shape %s" % (box_size.shape,))
    return box_size

def acc_periodic_orient_position(acc_count, prec, positions, orientations, box_size):
    acc_count = validate_acc_array(acc_count)
    if acc_count.ndim != 2:
        raise ValueError("acc_count must be a 2D array")
    if acc_count.shape[0] != acc_count.shape[1]:
        raise ValueError("acc_count must be a square array")

    prec = validate_prec(prec)
    positions = validate_positions(positions)
    orientations = validate_orientations(orientations, positions)

    libchlab.acc_periodic_orient_position(acc_count.ctypes.data_as(c_double_p),
                                          acc_count.shape[0],
                                          prec,
                                          positions.ctypes.data_as(c_double_p),
                                          orientations.ctypes.data_as(c_double_p),
                                          positions.shape[0],
                                          box_size.ctypes.data_as(c_double_p))
    return acc_count


