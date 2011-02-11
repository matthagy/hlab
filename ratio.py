

from hlab.bases import AutoRepr
from hlab import xalgebra as A

from jamenson.runtime.multimethod import defmethod, MultiMethod


def gcf(a, b=0):
    while b!=0:
        a,b = b, a - b*(a//b)
    return a

class Ratio(AutoRepr, A.AlgebraBase):

    def __init__(self, num, den=None):
        if den is None:
            den = 1
        assert isinstance(num, (int,long))
        assert isinstance(den, (int,long))
        self.num = num
        self.den = den

    def repr_args(self):
        return (self.num, self.den) if self.den != 1 else (self.num,)

    def __str__(self):
        return '%d/%d' % (self.num, self.den)

    def normalized(self):
        f = gcf(self.num, self.den)
        if f==1:
            return self
        num,den = self.num // f, self.den // f
        if den<0:
            num,den = -num,-den
        return self.__class__(num, den)

as_ratio = MultiMethod('as_ratio')

@defmethod(as_ratio, [Ratio])
def meth(r):
    return r

@defmethod(as_ratio, [(int,long)])
def meth(i):
    return Ratio(i)

@defmethod(A.mm_eq, [Ratio, Ratio])
def meth(a, b):
    a = a.normalized()
    b = b.normalized()
    return a.num==b.num and a.den==b.den

@A.defboth_mm_eq([Ratio, (int,long)])
def meth(r, i):
    r = r.normalized()
    return r.den==1 and r.num==i

@defmethod(A.mm_neg, [Ratio])
def meth(r):
    return Ratio(-r.num, r.den)

@A.defboth_mm_add([Ratio, Ratio])
def meth(a, b):
    return Ratio(a.num * b.den + b.num * a.den,
                 a.den * b.den).normalized()

@A.defboth_mm_add([Ratio, (int,long)])
def meth(r, i):
    return Ratio(r.num + i * r.den, r.den).normalized()

@A.defboth_mm_mul([Ratio, Ratio])
def meth(a, b):
    return Ratio(a.num*b.num, a.den*b.den)

@A.defboth_mm_mul([Ratio, (int,long)])
def meth(r, i):
    return Ratio(r.num * i, r.den).normalized()

@defmethod(A.mm_pow, [Ratio, (int,long)])
def meth(r, i):
    return Ratio(r.num**i, r.den**i) if i>1 else Ratio(r.den**-i, r.num**-i)

@defmethod(A.mm_sub, [Ratio, (int,long,Ratio)])
@defmethod(A.mm_sub, [(int,long,Ratio), Ratio])
def meth(a, b):
    return a + -b

@defmethod(A.mm_div, [Ratio, Ratio])
@defmethod(A.mm_truediv, [Ratio, Ratio])
def meth(a, b):
    return a * b ** -1

@defmethod(A.mm_div, [Ratio, (int,long)])
@defmethod(A.mm_truediv, [Ratio, (int, long)])
def meth(r, i):
    return Ratio(r.num, r.div * i).normalized()

@defmethod(A.mm_div, [(int,long), Ratio])
@defmethod(A.mm_truediv, [(int,long), Ratio])
def meth(i, r):
    return Ratio(i) / r
