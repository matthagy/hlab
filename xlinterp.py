
from __future__ import division
from __future__ import with_statement

import sys
import os
from bisect import bisect_right

import Gnuplot
from numpy import *

from jamenson.runtime.multimethod import defmethod

from hlab.xalgebra import (AlgebraBase, mm_neg, defboth_mm_eq,
                           defboth_mm_add, defboth_mm_mul, defboth_mm_div,
                           scalar_number_type)


class HomogenousTable(AlgebraBase):

    def __init__(self, y, x_min, x_prec):
        self.y = asarray(y)
        self.x_min = x_min
        self.x_prec = x_prec

    @classmethod
    def fromfunc(cls, func, x_min, x_max, n_points):
        x = linspace(x_min, x_max, n_points)
        x_prec = (x_max - x_min) / (n_points - 1)
        return cls(map(func, x), x_min, x_prec)

    def interpolate(self, x):
        if isinstance(x, ndarray):
            return array(map(self.interpolate, x))
        k = (x-self.x_min) / self.x_prec
        i = floor(k)
        ik = k-i
        return self.y[i] * (1 - ik) + self.y[i+1] * ik

    def __call__(self, x):
        return self.interpolate(x)

    def diff(self):
        return self.__class__(
            (self.y[1::] - self.y[:-1:]) / self.x_prec,
            self.x_min + self.x_prec * 0.5,
            self.x_prec)

@defmethod(mm_neg, 'HomogenousTable')
def meth(ht):
    return HomogenousTable(-ht.y, ht.x_min, ht.x_prec)


@defboth_mm_add('HomogenousTable, scalar_number_type')
def meth(ht, c):
    return HomogenousTable(ht.y + c, ht.x_min, ht.x_prec)

@defboth_mm_add('HomogenousTable, HomogenousTable')
def meth(a,b):
    if a.x_prec != b.x_prec or a.x_min != b.x_min:
        raise ValueError("tables not aligned")
    return HomogenousTable(a.y + b.y, a.x_min, a.x_prec)

@defboth_mm_mul('HomogenousTable, scalar_number_type')
def meth(ht, f):
    return HomogenousTable(ht.y * f, ht.x_min, ht.x_prec)

@defboth_mm_mul('HomogenousTable, HomogenousTable')
def meth(a,b):
    if a.x_prec != b.x_prec or a.x_min != b.x_min:
        raise ValueError("tables not aligned")
    return HomogenousTable(a.y * b.y, a.x_min, a.x_prec)



class InHomogenousTable(AlgebraBase):

    def __init__(self, x, y, check=True):
        self.x = asarray(x)
        self.y = asarray(y)
        if check:
            assert len(self.x.shape)==1
            assert self.x.shape == self.y.shape
            xc = self.x.copy()
            xc.sort()
            assert (xc==self.x).all()

    @classmethod
    def fromfunc(cls, func, x):
        x = asarray(x)
        return cls(x, map(func, x), True)

    def diff(self):
        return self.__class__((self.x[1:] + self.x[:-1]) / 2,
                              (self.y[1:] - self.y[:-1]) /
                              (self.x[1:] - self.x[:-1]))

    def interpolate(self, x):
        if isinstance(x, ndarray):
            return array(map(self.interpolate, x.ravel())).reshape(x.shape)
        b = bisect_right(self.x, x)
        a = b-1
        assert a>=0
        da = x - self.x[a]
        db = self.x[b] - x
        return (db*self.y[a] + da*self.y[b]) / (da+db)

    def __call__(self, x):
        return self.interpolate(x)


@defmethod(mm_neg, 'InHomogenousTable')
def meth(it):
    return InHomogenousTable(it.x, -it.y, check=False)

@defboth_mm_add('InHomogenousTable, (int,long,float)')
def meth(it, c):
    return InHomogenousTable(it.x, it.y + c, check=False)

@defboth_mm_add('InHomogenousTable, InHomogenousTable')
def meth(a,b):
    if not all(a.x==b.x):
        raise ValueError("tables not aligned")
    return InHomogenousTable(a.x, a.y + b.y, check=False)

@defboth_mm_mul('InHomogenousTable, (int,long,float)')
def meth(it, f):
    return InHomogenousTable(it.x, it.y * f, check=False)

@defboth_mm_mul('InHomogenousTable, InHomogenousTable')
def meth(a,b):
    if not all(a.x==b.x):
        raise ValueError("tables not aligned")
    return InHomogenousTable(a.x, a.y * b.y, check=False)

def main():
    from Gnuplot import Gnuplot, Data
    from time import sleep

    func = lambda x: 1.2*(x-2)**2 + 2*x + 10
    ht = HomogenousTable.fromfunc(func, -10, 10, 10)
    it = InHomogenousTable.fromfunc(func, [-11,-7,-5,-3,-2,-1,0,
                                           1,2,3,5,7,11])
    it2 = InHomogenousTable.fromfunc(func, [-8,-5,-2,0,
                                            2,5,8])
    gp = Gnuplot(persist=1)
    x = linspace(-5,5,20)
    gp.plot(Data(x, func(x), with_='l'),
            Data(x, ht(x), title='ht'),
            Data(x, (1.2*ht + 2)(x), title='1.2*ht +2'),
            Data(x, it(x), title='it'),
            Data(x, it2(x), title='it2'))
    sleep(1)
    exit()

__name__ == '__main__' and main()
