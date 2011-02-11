
from __future__ import division
from __future__ import with_statement

import sys
import os
from bisect import bisect_right

import Gnuplot
from numpy import *

class BaseTable(object):

    def __radd__(self, other):
        return self+other

    def __sub__(self, other):
        return self+(-other)

    def __rmul__(self, op):
        return self*op

    def __div__(self, op):
        return self*(1.0/op)

    def plot(self, persist=True, **kwds):
        gp = Gnuplot.Gnuplot(persist=persist)
        gp.plot(self.gnuplot_item(**kwds))
        return gp


class HomogenousTable(BaseTable):

    def __init__(self, y, x_prec, x_min):
        self.y = asarray(y)
        self.x_prec = x_prec
        self.x_min = x_min
        assert 0, 'not done'

    @classmethod
    def from_func(cls, func, x_min, x_max, n_points):
        pass

    def interpolate(self, x):
        pass

    def __neg__(self):
        return self.__class__(-self.y, self.x_prec, self.x_min)

    def __add__(self, other):
        assert len(self.y) == len(other.y)
        assert self.x_prec == other.x_prec
        assert self.x_min == other.x_min
        return self.__class__(self.y + other.y, self.x_prec, self.x_min)




class InHomogenousTable(BaseTable):

    def __init__(self, x, y, check=True):
        self.x = asarray(x)
        self.y = asarray(y)
        if check:
            assert len(x.shape)==1
            assert x.shape == y.shape
            xc = self.x.copy()
            xc.sort()
            assert (xc==self.x).all()

    @classmethod
    def from_func(cls, func, x):
        x = asarray(x)
        return cls(x, map(func, x), True)

    def __neg__(self):
        return self.__class__(self.x, -self.y, False)

    def __add__(self, other):
        assert all(self.x == other.x)
        return self.__class__(self.x, self.y + other.y, False)

    def __mul__(self, other):
        assert all(self.x == other.x)
        return self.__class__(self.x, other*self.y, False)

    def diff(self):
        return self.__class__((self.x[1:] + self.x[:-1]) / 2,
                              (self.y[1:] - self.y[:-1]) /
                              (self.x[1:] - self.x[:-1]))

    def interpolate(self, x):
        if isinstance(x, ndarray):
            return array(map(self.interpolate, x.ravel())).reshape(x.shape)
        b = bisect_right(self.x, x)
        a = b-1
        if a < 0:
            raise ValueError('%r not in table' % (x,))
        wa = 1.0 / (x - self.x[a])
        if isinf(wa):
            return self.y[a]
        try:
            wb = 1.0 / (self.x[b] - x)
        except IndexError:
            if b == len(self.x) and epsilon_eq(x, self.x[-1]):
                return self.y[-1]
            raise ValueError('%r not in table' % (x,))
        assert wa >= 0, 'w=%g a=%d x[a]=%g x=%g' % (wa, a, self.x[a], x)
        assert wb >= 0, 'w=%g b=%d x[b]=%g x=%g' % (wb, b, self.x[b], x)
        return (wa*self.y[a] + wb*self.y[b]) / (wa+wb)

    def __call__(self, x):
        return self.interpolate(x)

    def gnuplot_item(self, title=None, with_='l'):
        return Gnuplot.Data(self.x, self.y, title=title, with_=with_)


def epsilon_eq(a, b, epsilon=None):
    if isnan(a) or isnan(b):
        return False
    if epsilon is None:
        epsilon = 1e-3 * (abs(a) + abs(b))
    assert epsilon >= 0, 'bad epsilon %r' % (epsilon,)
    return abs(a-b) <= epsilon
