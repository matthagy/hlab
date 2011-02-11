'''
'''

from __future__ import division
from __future__ import absolute_import

import numpy as np

from .calculated import calculated
from .util import coere_listlike


class PairCorrelation(Profile):
    '''Static pair correlation function g(r)
    '''

    @classmethod
    def from_density_distribution(cls, dist):
        '''From density distribution of pair separation distances p(r)
        '''
        indexes = np.arange(len(dist.data), dtype=np.float)
        V = 4/3*pi*dist.prec**3 * ((indexes+1)**3 - indexes**3)
        g = (dist.data/V) / (dist.data.sum()/V.sum())
        return cls(g, dist.prec)

    @calculated
    def r(self):
        return self.indices

    @calculated
    def g(self):
        return self.data

    @calculated
    def h(self):
        return self.data - 1

    @calculated
    def r_max(self):
        return self.indices[where(self.g == self.g.max())][0]



class PairCorrelationExtractor(Extractor):

    @classmethod
    def from_one_configuration(cls, config, **kwds):
        '''From one configuration where config is an Nxn array with
           either n=2 or n=3
        '''
        return cls.from_many_configurations([config])

    @classmethod
    def from_many_configurations(cls, configs):
        configs = coere_listlike(configs)

