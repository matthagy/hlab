
from __future__ import division


import types
import opcode
import random
from functools import partial

import numpy as N
from scipy.optimize import leastsq
import Gnuplot

from jamenson.runtime.multimethod import MultiMethod, defmethod, defboth_wrapper

from hlab.ihlab import IDomain
from hlab.domain import DomainError, open_domain
from hlab.xbrooke import bimport
from hlab.xalgebra import (AlgebraBase, scalar_number_type,
                           mm_neg, mm_pos, mm_eq,
                           defboth_mm_add, mm_sub,
                           defboth_mm_mul, mm_div)

EPSILON = 1e-6


class Polynomial(AlgebraBase):

    def __init__(self, terms=None, domain=None):
        self.terms = {}
        if terms is not None:
            self.set_terms(terms)
        self.domain = IDomain(domain) if domain else open_domain

    def set_term(self, n, A=1):
        assert isinstance(n, (int,long))
        assert isinstance(A, (int,long,float))
        assert n>=0
        self.terms[n] = A

    def get_term(self, n):
        return self.terms.get(n,0)

    def remove_zero_constants(self):
        for n in list(terms):
            if not self.terms[n]:
                del self.terms[n]

    def set_terms(self, terms):
        if isinstance(terms, dict):
            terms = terms.iteritems()
        else:
            if not isinstance(terms, (list,tuple)):
                terms = list(terms)
            if terms and isinstance(terms[0], (int,long,float)):
                terms = enumerate(terms)
        for n,A in terms:
            self.set_term(n, A)

    def iterterms(self):
        return self.terms.iteritems()

    def eval(self, x, domain=True):
        if isinstance(x, N.ndarray):
            return N.array(map(self.eval, x.ravel())).reshape(x.shape)
        if domain and x not in self.domain:
            raise DomainError(x, self.domain)
        return sum([A*x**n for n,A in self.iterterms()])

    def __call__(self, x):
        return self.eval(x)

    def make_equation_string(self, include_zeros=True,
                                independent='x',
                                constant_format='%r',
                                power_format='%r',
                                addition_sign='+',
                                multiply_sign='*',
                                power_sign='**'):
        term_format = ''.join([
                constant_format, multiply_sign,
                independent,power_sign,power_format])
        def str_terms():
            for n,A in sorted(self.iterterms()):
                if A == 0 and not include_zeros:
                    continue
                if n==0:
                    yield constant_format % (A,)
                elif n==1:
                    yield ''.join([constant_format, multiply_sign,
                                   independent]) % (A,)
                else:
                    yield term_format % (A,n)
        return addition_sign.join(str_terms())

    def __str__(self):
        return self.make_equation_string()

    def copy(self):
        return self.__class__(self.terms, self.domain)

    def differentiate(self):
        d = self.__class__(domain=self.domain)
        for n,A in self.iterterms():
            A *= n #constant (n=0) removed here
            if A:
                d.set_term(n-1,A)
        return d

    def integrate(self, constant=0):
        i = self.__class__(domain=self.domain)
        if constant:
            i.set_term(0,constant)
        for n,A in self.iterterms():
            n += 1
            assert n >= 0
            A /= n
            i.set_term(n,A)
        return i

    def iter_roots(self, only_real=False, redundant=True, domain=False, **kwds):
        if not redundant:
            seen = set()
        for r in RootSolver(self, **kwds).iter_roots():
            if ((only_real and isinstance(r, complex)) or
                (domain is not False and r not in self.domain) or
                (not redundant and r in seen)):
                pass
            else:
                if not redundant:
                    seen.add(r)
                yield r

    def roots(self, **kwds):
        return sorted(self.iter_roots(**kwds))

    def solve(self, const=0, **kwds):
        if const != 0:
           self = self - const
        return self.roots(**kwds)

    def iter_extremes(self, domain=False):
        d = self.differentiate()
        roots = d.roots(only_real=True, redundant=False, domain=domain)
        if not roots:
            return
        dd = d.differentiate()
        for r in roots:
            x = dd(r)
            if x<0:
                tp = 'max'
            elif x>0:
                tp = 'min'
            else:
                tp = 'inflect'
            yield tp, r

    def find_extreme_types(self, etypes, domain=False):
        if isinstance(etypes, str):
            etypes = [etypes]
        return [x for etype,x in self.iter_extremes(domain=domain)
                if etype in etypes]

    def find_minimums(self, domain=False):
        return self.find_extreme_types('min', domain)

    def find_maximums(self, domain=False):
        return self.find_extreme_types('max', domain)

    def find_inflections(self, domain=False):
        return self.find_extreme_types('inflex', domain)

    def _find_global_extreme(self, domain, extremes, index):
        extremes = list(extremes)
        if domain is not False:
            if domain is True or domain is None:
                domain = self.domain
            for e in domain.get_extremes():
                if -oo < e < oo:
                    extremes.append(e)
        extremes.sort(key=lambda x: self.eval(x))
        return (extremes or [None])[index]

    def find_global_minimum(self, domain=None):
        return self._find_global_extreme(domain, self.find_minimums(), 0)

    def find_global_maximum(self, domain=None):
        return self._find_global_extreme(domain, self.find_maximums(), -1)

    def find_global_minimum_value(self, domain=None):
        x = self.find_global_minimum(domain)
        if x is None:
            return None
        return self.eval(x)

    def find_global_maximum_value(self, domain=None):
        x = self.find_global_maximum(domain)
        if x is None:
            return None
        return self.eval(x)


    def gnuplot_item(self, title=None):
        if title is None:
            title = str(self)
        elif title is False:
            title = ''
        return Gnuplot.Func(str(self), title=title, with_='l')

    def plot(self, persist=True, **kwds):
        gp = Gnuplot.Gnuplot(persist=persist)
        gp.plot(self.gnuplot_item(**kwds))
        return gp

    def compile_to_function(self, check_domain=True):
        '''
        Create a statically compiled function that evalutes the current polynomial.
        On average this leads to a 10x speed up
        '''
        return bimport('polynomial').compile_to_function(self, check_domain)


@defmethod(mm_eq, [Polynomial, Polynomial])
def meth(a,b):
    return (a.terms==b.terms and
            a.domain==b.domain)

@defmethod(mm_pos, [Polynomial])
def meth(p):
    return p.copy()

@defmethod(mm_neg, [Polynomial])
def meth(p):
    cp = p.__class__(domain=p.domain)
    for n,A in p.iterterms():
        cp.set_term(n, -A)
    return cp

@defboth_mm_add([Polynomial, scalar_number_type])
def meth(p, s):
    cp = p.copy()
    cp.set_term(0, cp.get_term(0) + s)
    return cp

@defboth_mm_add([Polynomial, Polynomial])
def meth(a,b):
    p = p = a.copy()
    p.domain = self.domain & other.domain
    for n,A in b.iterterms():
        p.set_term(n, p.get_term(n)+A)
    return p

@defboth_mm_mul([Polynomial, scalar_number_type])
def meth(p, s):
    cp = p.copy()
    for n,A in p.iterterms():
        cp.set_term(n, A*s)
    return cp

@defboth_mm_mul([Polynomial, Polynomial])
def meth(a,b):
    p = a.__class__(domain=a.domain & b.domain)
    for na,Aa in a.iterterms():
        for nb,Ab in b.iterterms():
            p.set_term(na+nb, Aa*Ab + p.get_term(na+nb))
    return p


class RootSolver(object):
    """find all roots (real and complex) of a polynomial using Bairstow's algorithm
    """

    def __init__(self, poly, epsilon=None, rnd_range=(-100,100),
                 maxsteps=128, rndseed=0xC0EDA55):
        self.poly = poly
        self.epsilon = epsilon or EPSILON
        self.rnd_range = rnd_range
        self.maxsteps = maxsteps
        self.rnd = random.Random(rndseed)
        self.n_terms = max(self.poly.terms)

    def almost_zerop(self, op):
        return abs(op) < self.epsilon

    def iter_roots(self):
        #trival solutions
        if self.n_terms == 0:
            pass
        elif self.n_terms == 1:
            B = self.poly.get_term(1)
            C = self.poly.get_term(0)
            yield -C / B
        elif self.n_terms == 2:
            A = self.poly.get_term(2)
            B = self.poly.get_term(1)
            C = self.poly.get_term(0)
            a = 0j + -B / 2 / A
            b = N.sqrt(B*B - 4*A*C + 0j) / 2 / A
            if self.almost_zerop(a.imag):
                a = a.real
            if self.almost_zerop(b.imag):
                b = b.real
            yield a+b
            yield a-b
        else:
            for op in self.iter_roots_bairstows():
                yield op

    def iter_roots_bairstows(self):
        """
        reduce to quadratic equation and n-2 polymonimal through bairstows algorithm
        uses multivariable newton's method to determine constants u and v s.t.
            x**2 + u*x + v
        has the same roots as this polynomial and can be divided out of this polynomial
        see http://en.wikipedia.org/wiki/Bairstow's_method for details
        """
        converged = False
        while not converged:
            u = self.rnd_init()
            v = self.rnd_init()
            for step in xrange(self.maxsteps):
                c = self.c_func(u,v)
                d = self.d_func(u,v)
                if self.almost_zerop(c) and self.almost_zerop(d):
                    converged = True
                    break
                duv = partial(self.differentiate, u, v)
                j = N.array([ #Jacobian
                    [duv(self.c_func, 0), duv(self.c_func, 1)],
                    [duv(self.d_func, 0), duv(self.d_func, 1)]])
                j = N.linalg.inv(j)
                u,v = N.array([u,v]) - N.dot(j, N.array([c,d]))
            for r in self.iter_sub_roots(Polynomial([v,u,1])):
                yield r
        b = Polynomial()
        for i in xrange(self.n_terms-1):
            b.set_term(i, self.b_func(i,u,v))
        for r in self.iter_sub_roots(b):
            yield r

    def b_func(self, i,u,v):
        if i == self.n_terms or i==self.n_terms-1:
            return 0
        return self.poly.get_term(i+2) - u*self.b_func(i+1,u,v) - v*self.b_func(i+2,u,v)

    def c_func(self, u,v):
        return self.poly.get_term(1) - u*self.b_func(0,u,v) - v*self.b_func(1,u,v)

    def d_func(self, u,v):
        return self.poly.get_term(0) - v*self.b_func(0,u,v)

    def differentiate(self, u, v, func, n):
        args = [u,v]
        x = args[n]
        args[n] = x+self.epsilon
        y2 = func(*args)
        args[n] = x-self.epsilon
        y1 = func(*args)
        dy = y2-y1
        dx = 2*self.epsilon
        return dy/dx

    def rnd_init(self):
        r = 0
        while not r:
            r = self.rnd.randrange(*self.rnd_range)
        return r

    def iter_sub_roots(self, poly):
        return RootSolver(poly, self.epsilon, self.rnd_range,
                          self.maxsteps, self.rnd.randrange(100000)).iter_roots()




def fit_polynomial(x, y, Nterms=None, init_terms=None,
                  maxfev=10000, ftol=None, xtol=None):
    x = N.array(x)
    y = N.array(y)
    assert len(x)
    assert len(x) == len(y)
    if Nterms is None and init_terms:
        Nterms = len(init_terms)
    elif init_terms is None:
        init_terms = [0.1] * Nterms
    assert len(init_terms) == Nterms
    eq = Polynomial(init_terms, (min(x), max(x)))
    def residuals(ks):
        eq.set_terms(ks)
        ye = eq.eval(x, domain=False)
        r = y-ye
        return r
    ks0 = init_terms
    ks,_ = leastsq(residuals, ks0,
            maxfev=maxfev,
            ftol=ftol or EPSILON,
            xtol=xtol or EPSILON)
    if Nterms==1:
        ks = [ks]
    eq.set_terms(ks)
    return eq


def test():
    x = N.linspace(-0.5*N.pi, 0.5*N.pi, 64)
    y = N.cos(x)
    p = fit_polynomial(x,y,2)
    #p.plot()
    print 'poly',p
    from HH2 import polyutil
    pp = polyutil.polynomialFit(x,y,2)
    print 'pp', pp
    print N.array(p.roots(only_real=False)) / N.pi
    print N.array(pp.roots(onlyReal=False)) / N.pi


    from time import time
    x = 0
    start = time()
    yp = p(x)
    tp = time() - start

    f = p.compile_to_function(True)
    start = time()
    yf = f(x)
    tf = time() - start

    print 'answers', yp, yf
    print 'speed up', tp / tf

    #from dis import dis
    #dis(f)

__name__ == '__main__' and test()
