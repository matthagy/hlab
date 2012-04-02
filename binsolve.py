from __future__ import division

def binsolve(f, target, a, b, ftol=1e-12, xtol=1e-12):
    _abs = abs
    mlast = None
    av,bv = map(f, [a,b])
    f_scale = 1.0 / target if target != 0 else 1.0
    while True:
        m = (a+b) / 2
        mv = f(m)
        if m==mlast or _abs((mv-target) * f_scale) < ftol or _abs((a-b) / (0.5*(_abs(a)+_abs(b)))) < xtol:
            return m
        if (av>target and mv>target) or (av<target and mv<target):
            a = m
            av = mv
        elif (bv>target and mv>target) or (bv<target and mv<target):
            b = m
            bv = mv
        else:
            raise ValueError('invalid range or non-monotonic function')
        mlast = m
