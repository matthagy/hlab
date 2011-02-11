'''A Domain is represented by a collection of non-overlapping DomainPairs
   where each DomainPair defines a range from a start-Boundary to and end-Boundary.
   Each Boundary maybe inclusive or exclusive.
'''


from __future__ import division

import operator

from hlab.ihlab import IDomainBoundary, IDomainPair, IDomain, implements, registerAdapter
from hlab.infinity import oo, InfinityType
from hlab.xbrooke import bimport


__all__ = '''
Boundary DomainPair Domain
null_domain open_domain
'''.split()


class DomainError(ValueError):

    def __init__(self, value, domain):
        self.value = value
        self.domain = domain

    def __str__(self):
        return '%r not in domain %s' % (self.value, self.domain)


class Boundary(object):

    implements(IDomainBoundary)

    def __init__(self, value, inclusive=True):
        self.value = value
        self.inclusive = not not inclusive

    def __repr__(self):
        return '%s(%r, inclusive=%s)' % (
            self.__class__.__name__,
            self.value, self.inclusive)

    def __str__(self):
        if self.inclusive:
            return str(self.value)
        return repr(self)

    #defines how to perform a comparison for a pair of values based
    #on whether each operand in inclusive.  Leads to 4 possible ways
    #to perform comparision for each operator
    def build_comparer(name, comparers, locs=locals()):
        #no mapping specified, comparision specified
        if not isinstance(comparers, list):
            comparers = [(i,j,comparers) for i in [0,1] for j in [0,1]]
        #all comparisions not specified return False
        cmpmap = dict(((i,j), lambda a,b: False)
                      for i in [0,1] for j in [0,1])
        cmpmap.update(((a_inclusive, b_inclusive), op)
                        for a_inclusive,b_inclusive,op in comparers)
        assert len(cmpmap)==4
        def comparer(a,b):
            b = IDomainBoundary(b, None)
            if b is None:
                return NotImplemented
            return cmpmap[a.inclusive,b.inclusive](a.value,b.value)
        name = '__%s__' % name
        comparer.func_name = name
        locs[name] = comparer

    o = operator
    [build_comparer(name,comparers)
         for name,comparers in [
            ['eq',[[0,0,o.eq],
                   [1,1,o.eq]]],
            ['gt',o.gt],
            ['lt',o.lt],
            ['ge',[[0,0,o.gt],
                   [0,1,o.gt],
                   [1,0,o.ge],
                   [1,1,o.ge]]],
            ['le',[[0,0,o.lt],
                   [0,1,o.lt],
                   [1,0,o.le],
                   [1,1,o.le]]]]]

    del build_comparer,o,name,comparers

    def __ne__(self, other):
        return not (self==other)

    def bound_lower(self, op):
        return (op >= self.value
                   if self.inclusive else
                op > self.value)

    def bound_upper(self, op):
        return (op <= self.value
                   if self.inclusive else
                op < self.value)

    def flipped(self):
        return self.__class__(self.value, not self.inclusive)



registerAdapter(Boundary, int, IDomainBoundary)
registerAdapter(Boundary, long, IDomainBoundary)
registerAdapter(Boundary, float, IDomainBoundary)
registerAdapter(Boundary, InfinityType, IDomainBoundary)


class DomainPair(object):

    implements(IDomainPair)

    def __init__(self, start=-oo, end=oo):
        start = IDomainBoundary(start)
        end = IDomainBoundary(end)
        assert end.value>=start.value
        self.start = start
        self.end = end

    def __iter__(self):
        yield self.start
        yield self.end

    def __eq__(self, other):
        other = IDomainPair(other, None)
        if other is None:
            return NotImplemented
        return (self.start==other.start and
                self.end==other.end)

    def __ne__(self, other):
        return not (self==other)

    def __gt__(self, other):
        other = IDomainPair(other, None)
        if other is None:
            return NotImplemented
        return ((self.start<other.start and
                 self.end>=other.end) or
                (self.start<=other.start and
                 self.end>other.end))

    def __ge__(self, other):
        other = IDomainPair(other, None)
        if other is None:
            return NotImplemented
        return (self.start<=other.start and
                self.end>=other.end)

    def __lt__(self, other):
        other = IDomainPair(other, None)
        if other is None:
            return NotImplemented
        return ((self.start>other.start and
                 self.end<=other.start) or
                (self.start>=other.start and
                 self.end<other.end))

    def __le__(self, other):
        other = IDomainPair(other, None)
        if other is None:
            return NotImplemented
        return (self.start>=other.start and
                self.end<=other.end)

    def __str__(self):
        return '%s(%s, %s)' % (self.__class__.__name__,
                               self.start,self.end)

    def __repr__(self):
        return '%s(%r, %r)' % (self.__class__.__name__,
                               self.start,self.end)

    def intersection(self, other):
        a = self
        b = other
        if a<=b:
            return a
        if b<=a:
            return b
        if b.start < a.start:
            a,b = b,a
        if a.start < b.start and a.end>b.start:
            return DomainPair(b.start, a.end)
        return None

    def union(self, other):
        if self.intersection(other):
            return [DomainPair(min(self.start, other.start),
                              max(self.end, other.end))]
        return [self, other]

    def __contains__(self, op):
        return (self.start.bound_lower(op) and
                self.end.bound_upper(op))

    def iter_points(self, step):
        value = self.start.value
        while value <= self.end:
            if value >= self.start:
                yield value
            value += step

registerAdapter(lambda (start,end): DomainPair(start,end), tuple, IDomainPair)

def unionize_pairs(pairs):
    if len(pairs)<2:
        return pairs
    acc_union_first,rest = pairs[0], pairs[1:]
    acc_rest = []
    for pair in rest:
        union = acc_union_first.union(pair)
        if len(union)==1:
            xpairs = unionize_pairs(union + acc_rest)
            acc_union_first, acc_rest = xpairs[0], xpairs[1:]
        else:
            acc_rest.append(pair)
    return [acc_union_first] + unionize_pairs(acc_rest)


class Domain(object):

    implements(IDomain)

    def __init__(self, pairs=[]):
        self.pairs = unionize_pairs(pairs)
        self.pairs.sort(key=lambda op: op.start)

    def is_empty(self):
        return not self.pairs

    def __iter__(self):
        return iter(self.pairs)

    def __repr__(self):
        return '%s([%s])' % (self.__class__.__name__,
                             ','.join(map(repr, self.pairs)))
    def __str__(self):
        return '[%s]' % ','.join('%s:%s' % (pair.start, pair.end)
                                 for pair in self.pairs)

    def union(self, other):
        return Domain(self.pairs + IDomain(other).pairs)

    def intersection(self, other):
        return Domain(filter(None, [ap.intersection(bp)
                                    for bp in IDomain(other).pairs
                                    for ap in self.pairs]))

    def __eq__(self, other):
        other = IDomain(other, None)
        if other is None:
            return NotImplemented
        return self.pairs == other.pairs

    def __or__(self, other):
        return self.union(other)

    def __and__(self, other):
        return self.intersection(other)

    def __contains__(self, op):
        return any(op in pair for pair in self.pairs)

    def __neg__(self):
        if not self.pairs:
            return Domain([[-oo,oo]])
        l = []
        if self.pairs[0].start.value != -oo:
            l.append(DomainPair(-oo, self.pairs[0].start.flipped()))
        last = self.pairs[0]
        for op in self.pairs[1:]:
            l.append(DomainPair(last.end.flipped(), op.start.flipped()))
            last = op
        if self.pairs[-1].end.value != oo:
            l.append(DomainPair(self.pairs[-1].end.flipped(), oo))
        return Domain(l)

    def iter_points(self, step=1):
        for pair in self.pairs:
            for point in pair.iter_points(step):
                yield point

    def iter_npoints(self, npoints):
        if npoints == 0:
            return []
        if npoints == 1:
            return [self.pairs[0].start.value]
        trange = 0
        nholes = 0
        for pair in self.pairs:
            trange += pair.end.value - pair.start.value
            nholes += sum(1 for bound in [pair.end, pair.start]
                              if not bound.inclusive)
        step = trange / (npoints + nholes - 1)
        return self.iter_points(step)

    def compile_to_function(self):
        '''write a function to perform contains operation for this domain
        '''
        return bimport('xdomain').compile_to_function(self)


registerAdapter(lambda dp: Domain([dp]), DomainPair, IDomain)
registerAdapter(lambda (a,b): Domain([DomainPair(a,b)]), tuple, IDomain)

class DomainSyntax(object):

    def __getitem__(self, item):
        if not isinstance(item, tuple):
            item = item,
        acc = None
        for slc in item:
            assert isinstance(slc, slice)
            assert slc.step is None
            assert slc.start is not None
            assert slc.stop is not None
            d = IDomain((slc.start, slc.stop))
            if acc:
                d = d | acc
            acc = d
        return acc

ds = DomainSyntax()

null_domain = Domain()
open_domain = IDomain((-oo,oo))

def test():
    def d(a,b):
        return IDomain((a,b))
    dd = ds[-oo:-10, -8:5, -2:-1, 4:8]
    check = dd.compile_to_function()
    print check
    for i in xrange(-15,7,3):
        print i,check(i), i in dd
    #from dis import dis
    #dis(check)

#__name__ == '__main__' and test()

