'''Propigation of uncertainity in calculations
'''

from __future__ import division

import math
import operator

from hlab.ihlab import IUncertainNumber, implements, registerAdapter

__all__ = '''
UncertainNumber
'''.split()


class UncertainNumber(object):
    '''
    Represents a value and an uncertainity in its measurments/calculation
    Uncertainity is propigated throughout calculations
    '''

    implements(IUncertainNumber)

    __slots__ = 'value dv'.split()

    def __init__(self, value, dv=0.0):
        self.value = value
        assert dv >= 0
        self.dv = dv

    def __getstate__(self):
        return self.value, self.dv

    def __setstate__(self, tp):
        self.__init__(*tp)

    def __repr__(self):
        return '%s(%r, %r)' % (self.__class__.__name__, self.value, self.dv)

    def __str__(self):
        if not self.dv:
            return str(self.value)
        p = math.log10(abs(self.value))
        if abs(p)<2:
            return '%s+\-%s' % (self.value, self.dv)
        p = int(math.floor(p))
        b = 10.0**p
        v = self.value / b
        dv = self.dv / b
        return '(%s+\-%s)e%s' % (v,dv,p)

    def __nonzero__(self):
        return bool(self.value or self.dv)

    def __float__(self):
        return float(self.value)

    def __neg__(self):
        return self.__class__(-self.value, self.dv)

    def makeBinFunction(loc=locals()):
        quat = lambda a,b: (a**2 + b**2)**0.5
        addsub = lambda A,da,B,db,val: quat(da, db)
        muldiv = lambda A,da,B,db,val: (abs(val)*quat(da/A, db/B)
                                        if A and B else 0)
        fmuldiv = lambda A,da,B,db,val: (abs(val)*quat(da//A,db//B)
                                         if A and B else 0)
        from math import log
        d = [['add', addsub],
             ['sub', addsub],
             ['mul', muldiv],
             ['div', muldiv],
             ['truediv', muldiv],
             ['floordiv', fmuldiv],
             ['pow', lambda A,da,B,dB,val:
                     quat((val*B/A*da),
                          (val*log(abs(A))*dB))]]
        def doit(name, ufunc):
            op = getattr(operator, name)
            def wrapper(self, other):
                other = IUncertainNumber(other)
                val = op(self.value, other.value)
                dv = ufunc(self.value,
                            self.dv,
                            other.value,
                            other.dv,
                            val)
                return self.__class__(val, dv)
            return wrapper
        def doright(name):
            name = '__%s__' % name
            def rwrapper(self, other):
                return getattr(IUncertainNumber(other), name)(self)
            return rwrapper
        for name,ufunc in d:
             loc['__%s__' % name] = doit(name,ufunc)
             loc['__r%s__'%name] = doright(name)

    makeBinFunction()
    del makeBinFunction

    def exp(self):
        return math.e ** self

registerAdapter(UncertainNumber, float, IUncertainNumber)
registerAdapter(UncertainNumber, int, IUncertainNumber)
registerAdapter(UncertainNumber, long, IUncertainNumber)

