
from __future__ import division

import numpy as N
from scipy.optimize import leastsq, fminbound
import types
import opcode
import random


from hlab.ihlab import IDomain
from hlab.domain import DomainError, open_domain
from hlab.xbrooke import bimport

EPSILON = 1e-6


class Polynomial(object):

    def __init__(self, terms=None, domain=None):
        self.terms = {}
        if terms is not None:
            self.set_terms(terms)
        self.domain = IDomain(domain) if domain else open_domain

    def __eq__(self, other):
        if not isinstance(other, Polynomial):
            return NotImplemented
        return (self.terms==other.terms and
                self.domain==other.domain)

    def set_term(self, n, A=1):
        assert isinstance(n, (int,long))
        assert isinstance(A, (int,long,float))
        assert n>=0
        self.terms[n] = A

    def get_term(self, n):
        return self.terms.get(n,0)

    def iter_terms(self):
        return self.terms.iteritems()

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

    def eval(self, x):
        if x not in self.domain:
            raise DomainError(x, self.domain)
        return N.sum([A*x**n for n,A in self.iter_terms()], axis=0)

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
            for n,A in self.sorted_terms():
                if A == 0 and not include_zeros:
                    continue
                yield term_format % (A,n)
        return addition_sign.join(str_terms())

    def __str__(self):
        return self.make_equation_string()

    def copy(self):
        return self.__class__(self.terms, self.domain)

    def __pos__(self):
        return self.copy()

    def __neg__(self):
        p = self.__class__(domain=self.domain)
        for n,A in self.iter_terms():
            p.set_term(n,-A)
        return p

    def __add__(self, other):
        p = self.copy()
        if isinstance(other, (int,float,long)):
            p.set_term(0, p.get_term(0)+other)
        else:
            p.domain = self.domain & other.domain
            for n,A in other.iter_terms():
                p.set_term(n, p.get_term(n)+A)
        return p

    def __radd__(self, other):
        return self+other

    def __sub__(self, other):
        return self+(-other)

    def __rsub__(self, other):
        return (-self)+other

    def __mul__(self, other):
        if isinstance(other, (int,float,long)):
            p = self.copy()
            for n,A in p.iter_terms():
                p.set_term(n, A*other)
        else:
            p = self.__class__(domain=self.domain & other.domain)
            for n1,A1 in self.iter_terms():
                for n2,A2 in other.iter_terms():
                    n=n1+n2
                    p.set_term(n, A1*A2 + p.get_term(n))
        return p

    def __rmul__(self, other):
        return self*other

    def __div__(self, other):
        return self*(other**-1)

    def differentiate(self):
        d = self.__class__(domain=self.domain)
        for n,A in self.iter_terms():
            A *= n #constant (n=0) removed here
            if A:
                d.set_term(n-1,A)
        return d

    def integrate(self, constant=0):
        i = self.__class__(domain=self.domain)
        if constant:
            i.set_term(0,constant)
        for n,A in self.iter_terms():
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

    def compileToFunction(self):
        '''
        Create a statically compiled function that evalutes the current polynomial.
        On average this decrease execution time to about 1/20th origanl evaluation
        '''

        bimport('polynomial')

        constants = []
        varnames = (independentName,)
        names = [independentName]


        def constantIndex(op):
            tp = type(op)
            for i,iop in enumerate(constants):
                if type(iop) is tp and iop==op:
                    break
            else:
                i = len(constants)
                constants.append(op)
            return i
        def nameIndex(name):
            if name not in names:
                names.append(name)
            return names.index(name)


        def enforceDomain():
            dmin,dmax = self.domain
            dstr = "bad value %s not in domain " + domainString(dmin,dmax)
            dstri = constantIndex(dstr)

            def domainCmp(const, cmp):
                if const is None:
                    return
                yield ops.DUP_TOP
                ci = constantIndex(const)
                yield ops.LOAD_CONST, ci
                yield ops.COMPARE_OP, cmp
                yield ops.JUMP_IF_TRUE, 13
                # # # # # # # # # # # # # # # # # # # #
                yield ops.POP_TOP             # 1
                ni = nameIndex('ValueError')  #
                yield ops.LOAD_GLOBAL, ni     # 3  4
                yield ops.ROT_TWO             # 1  5
                yield ops.LOAD_CONST, dstri   # 3  8
                yield ops.ROT_TWO             # 1  9
                yield ops.BINARY_MODULO       # 1  10
                yield ops.RAISE_VARARGS, 2    # 3  13
                # # # # # # # # # # # # # # # # # # # #
                yield ops.POP_TOP
            for i in domainCmp(dmin, ocmp.GE):
                yield i
            for i in domainCmp(dmax, ocmp.LE):
                yield i

        def makeOperations():
            yield ops.LOAD_FAST, 0
            if self.enforceDomain:
                for op in enforceDomain():
                    yield op
            if not self.terms:
                ci = constantIndex(None)
                yield ops.LOAD_CONST, ci
                yield ops.RETURN_VALUE
            first_term = True
            last_term = False
            highest_term = self.highest_term()
            for n in xrange(highest_term+1):
                C = self.get_term(n)
                if C==0:
                    continue
                if n==highest_term:
                    last_term = True
                if n!=0:
                    if not first_term:
                        yield ops.ROT_TWO
                    if not last_term:
                        yield ops.DUP_TOP
                        if not first_term:
                            yield ops.ROT_THREE
                    if n!=1:
                        ci = constantIndex(n)
                        yield ops.LOAD_CONST, ci
                        yield ops.BINARY_POWER
                ci = constantIndex(C)
                yield ops.LOAD_CONST, ci
                if n!=0:
                    yield ops.BINARY_MULTIPLY
                if not first_term:
                    yield ops.BINARY_ADD
                if first_term:
                    first_term = False
            yield ops.RETURN_VALUE

        def writeCode(operations):
            for op in operations:
                if isinstance(op, tuple):
                    op, arg = op
                else:
                    arg = None
                assert isinstance(op, int)
                yield op
                if arg is not None:
                    hi,lo = divmod(arg, 256)
                    yield lo
                    yield hi

        codestring = ''.join(map(chr, writeCode(makeOperations())))
        constants = tuple(constants)
        names = tuple(names)
        argcount = 1
        nlocals = 1
        stacksize = 4
        firstlineno = 0
        lnotab = '\x00\x01'
        from compiler import consts
        consts.CO_NOFREE = 0x0040
        flags = consts.CO_NOFREE | consts.CO_NEWLOCALS | consts.CO_OPTIMIZED
        code = types.CodeType(argcount, nlocals, stacksize,
                 flags, codestring,
                 constants, names,
                 varnames, filename,
                 name, firstlineno, lnotab)
        func = types.FunctionType(code, gbls, name)
        return func



class RootSolver(object):
    """find all roots (real and complex) of a polynomial using Bairstow's algorithm
    """

    def __init__(self, poly, epsilon=EPSILON, rnd_range=(-100,100),
                 maxsteps=128, rndseed=0xC0EDA55):
        self.poly = poly
        self.epsilon = epsilon
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
            for op in self.iter_roots_bairstows(self):
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
                d = self.differentiate
                j = N.array([ #Jacobian
                    [d(self.c_func, 0), d(self.c_func, 1)],
                    [d(self.d_func, 0), d(self.d_func, 1)]])
                j = N.linalg.inv(j)
                u,v = N.array([u,v]) - N.dot(j, N.array([c,d]))
            for r in self.iter_sub_roots(Polynomial([v,u,1])):
                yield r
            b = Polynomial()
            for i in xrange(n_terms-1):
                b.set_term(i,self.b_func(i,u,v))
            for r in self.iter_sub_roots(b):
                yield r

    def b_func(self, i,u,v):
        if i == self.n_terms or i==self.n_terms-1:
            return 0
        return self.poly.get_term(i+2) - u*self.b_func(i+1,u,v) - v*self.b_func(i+2,u,v)

    def c_func(self, u,v):
        return self.poly.get_term(1) - u*self.b_func(0,u,v) - v*self.b_func(1,u,v)

    def d_func(u,v):
        return self.poly.get_term(0) - v*self.b_func(0,u,v)

    def differentiate(func, n):
        args = [u,v]
        x = args[n]
        args[n] = x+self.epsilon
        y2 = func(*args)
        args[n] = x-self.epsilon
        y1 = func(*args)
        dy = y2-y1
        dx = 2*self.epsilon
        return dy/dx

    def rnd_init():
        r = 0
        while not r:
            r = self.rnd.randrange(*self.rnd_range)
        return r

    def iter_sub_roots(self, poly):
        return RootSolver(poly, self.epsilon, self.rnd_range,
                          self.maxsteps, self.rndseed).iter_roots()



def polynomialFit(x, y, Nterms=None, init_terms=None,
                  maxfev=10000, ftol=1e-6, xtol=1e-6):

    x = N.array(x)
    y = N.array(y)
    assert len(x)
    assert len(x) == len(y)
    if Nterms is None and init_terms:
        Nterms = len(init_terms)
    elif init_terms is None:
        init_terms = [0.1] * Nterms
    assert len(init_terms) == Nterms
    domain = min(x), max(x)
    eq = Polynomial(init_terms, domain)
    def residuals(ks):
        eq.set_terms(ks)
        ye = N.array(map(eq.eval, x))
        r = y-ye
        return r
    ks0 = init_terms
    ks,_ = leastsq(residuals, ks0,
            maxfev=maxfev,
            ftol=ftol,
            xtol=xtol)
    if Nterms==1:
        ks = [ks]
    eq.set_terms(ks)
    return eq


