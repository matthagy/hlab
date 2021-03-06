
from __future__ import division

import operator
from itertools import imap

from hlab.uncertain import UncertainNumber

__all__ = '''
OrderedSet
'''.split()

class OrderedSet(list):
    '''Order set of items'''

    def __new__(cls, *args, **kwds):
        return list.__new__(cls)

    def __init__(self, entries=None):
        if entries:
             self.extend(entries)

    def create_subset(self, seq=None):
        cp = self.__new__(self.__class__)
        if seq:
            cp.extend(seq)
        return cp

    @staticmethod
    def as_criteria(criteria):
        return (operator.attrgetter(criteria)
                if isinstance(criteria,str) else criteria)

    @staticmethod
    def as_criterias(criterias):
        return [operator.attrgetter(criteria)
                if isinstance(criteria, str) else criteria
                for criteria in criterias]

    def sort(self, *criterias):
        '''Sorts the set for the specified attributes'''
        if not criterias:
            raise ValueError('Must specify sort criterias')
        criterias = self.as_criterias(criterias)
        if len(criterias) == 1:
            keyfunc, = criterias
        else:
            keyfunc = lambda entry: [criteria(entry) for criteria in criterias]
        list.sort(self, key=keyfunc)

    def sorted(self, *criterias):
        cp = self.create_subset(self)
        cp.sort(*criterias)
        return cp

    def require(self, criteria, value=True):
        criteria = self.as_criteria(criteria)
        if value is True:
            self[::] = [entry for entry in self if criteria(entry)]
        elif value is False:
            self[::] = [entry for entry in self if not criteria(entry)]
        elif value is None:
            self[::] = [entry for entry in self if criteria(entry) is None]
        else:
            self[::] = [entry for entry in self if criteria(entry) == value]

    def required(self, criteria, value=True):
        cp = self.create_subset(self)
        cp.require(criteria)
        return cp

    def align(self, criteria):
        '''
        Iterate through subsets where the value of criteria is the same
        in all entries in eahc subset.
        '''
        criteria = self.as_criteria(criteria)
        current = set_value = None
        s = [(criteria(entry), entry) for entry in self]
        s.sort()
        for sv,entry in s:
             if current is None or set_value != sv:
                 if current:
                     yield set_value, self.create_subset(current)
                 current = []
                 set_value = sv
             current.append(entry)
        if current:
            yield set_value, self.create_subset(current)

    def malign(self, criterias):
        '''
        Multiple alignements
        '''
        if isinstance(criterias, str):
            criterias = criterias.split()
        criterias = [operator.attrgetter(criteria)
                     if isinstance(criteria, str) else criteria
                     for criteria in criterias]
        return self.align(lambda entry: [critera(entry) for critera in criterias])

    def iter_criteria(self, criteria):
        return imap(self.as_criteria(criteria), self)

    def iter_criterias(self, *criterias):
        criterias = self.as_criterias(criterias)
        for entry in self:
            yield [criteria(entry) for criteria in criterias]

    def common(self, criteria, nocommon=None):
        '''
        If the value of criteria is the same on all entries return
        this common value.  Else return nocommon argument.
        '''
        s = set(self.iter_criteria(criteria))
        try:
            op, = s
        except ValueError:
            op = nocommon
        return op

    def have_common(self, criteria):
        nocommon = object()
        return self.common(criteria, nocommon) is not nocommon

    def __getitem__(self, op):
        if isinstance(op, slice):
            return self.create_subset(list.__getitem__(self, op))
        else:
            return list.__getitem__(self, op)

    def map(self, predicate):
        return [predicate(entry) for entry in self]

    def amap(self, predicate):
        return array(self.map(predicate))

    def _stat(self, predicate, corr_power=0.5):
        corr_power = 0 if corr_power is None else corr_power
        a = self.amap(predicate)
        return a.mean(axis=0), a.std(axis=0)/len(a)**corr_power

    def stat(self, predicate, corr_power=0.5):
        mean,err = self._stat(predicate, corr_power)
        return UncertainNumber(mean, err)


