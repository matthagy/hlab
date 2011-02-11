'''
'''

from __future__ import division
from __future__ import absolute_import

import os
import sys
import time
from decimal import Decimal
from types import GeneratorType
import cPickle as pickle

import numpy as np


def msg(msg='', *args):
    print >>sys.stderr, msg%args if args else msg
    sys.stderr.flush()

def error(msg='error', *args):
    raise RuntimeError(msg%args)

def coere_listlike(op):
    return list(op) if not isinstance(op, (list,tuple,np.ndarray)) else op

def epsilon_eq(a, b, epsilon=None):
    if np.isnan(a) or np.isnan(b):
        return False
    if epsilon is None:
        epsilon = 1e-3 * (abs(a) + abs(b))
    assert epsilon >= 0, 'bad epsilon %r' % (epsilon,)
    return abs(a-b) <= epsilon

def all_epsilon_eq(seq, val, epsilon=None):
    seq = np.asarray(seq)
    if np.isnan(seq).any() or np.isnan(val):
        return False
    if epsilon is None:
        epsilon = 1e-3 * (abs(seq.mean() + abs(val)))
    assert epsilon >= 0, 'bad epsilon %r' % (epsilon,)
    return (abs(seq-val) <= epsilon).all()

def pow2(n):
    '''is n a power of 2?
       little bitwise trick
    '''
    return n & -n == n

def flatten(op, seq_type=(list, tuple, GeneratorType)):
    if not isinstance(op, seq_type):
        yield op
    else:
        for el in op:
            for sel in flatten(el, seq_type):
                yield sel

def decimal_coere(op):
    if isinstance(op, (int,long,str,Decimal)):
        return Decimal(op)
    if isinstance(op, numpy.integer):
        return decimal_coere(int(op))
    if isinstance(op, float):
        dop = Decimal(str(op))
        if not epsilon_eq(float(dop), op):
            raise ValueError("can't coere %r to decimal within reasonable epsilon" % (op,))
        return dop
    if isinstance(op, numpy.inexact):
        return decimal_coere(float(op))
    raise TypeError("can't coere %r of type %s to a decimal" % (op, type(op)))

def collect_list(func):
    def wrap(*args, **kwds):
        return list(func(*args, **kwds))
    return wrap

def collect_tuple(func):
    def wrap(*args, **kwds):
        return tuple(func(*args, **kwds))
    return wrap

def collect_sum(func):
    def wrap(*args, **kwds):
        return sum(func(*args, **kwds))
    return wrap


def make_emitter(fp):
    def emitter(msg='', *args):
        print >>fp, msg%args if args else msg
    return emitter

def calculate_periodic_deltas(pa, pb, box_sizes, delta=None):
    delta = np.subtract(pb, pa, delta)
    box_sizes = coere_listlike(box_sizes)
    if all_epsilon_eq(box_sizes, box_sizes[0]):
        #optimization for uniform box_sizes
        sz = box_sizes[0]
        delta[delta > +0.5 * sz] -= sz
        delta[delta < -0.5 * sz] += sz
    else:
        #periodize each dimension separately
        for index in range(len(box_sizes)):
            sz = box_sizes[index]
            l = delta[..., index]
            l[l > +0.5 * sz] -= sz
            l[l < -0.5 * sz] += sz
    return delta


class FauxRepr(object):
    def __init__(self, op):
        self.op = op
    def __repr__(self):
        return str(self.op)

def float_fix(op, places=2):
    factor = 10 ** places
    return FauxRepr(Decimal(int(round(op * factor))) / factor)
