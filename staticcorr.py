'''Static correlations in data
'''

from __future__ import division
from __future__ import absolute_import

import warnings

import numpy as np

from .calculated import calculated
from .util import coere_listlike, calculate_periodic_deltas
from .prof import Profile, Distribution
from .extractor import BaseExtractor
try:
    from .chlab import acc_periodic_rs, acc_periodic_orientation_rs, acc_rs_dtype, acc_orients_dtype
except ImportError:
    pass


class PairProfile(Profile):

    @calculated
    def r(self):
        return self.indices

class PairCorrelation(PairProfile):
    '''Static pair correlation function g(r)
    '''

    @classmethod
    def from_density_distribution(cls, dist):
        '''From density distribution of pair separation distances p(r)
        '''
        indexes = np.arange(len(dist.data), dtype=np.float)
        V = 4/3*np.pi*dist.prec**3 * ((indexes+1)**3 - indexes**3)
        g = (dist.data/V) / (dist.data.sum()/V.sum())
        return cls(g, dist.prec)

    @calculated
    def g(self):
        return self.data

    @calculated
    def h(self):
        return self.data - 1

    @calculated
    def r_max(self):
        return self.indices[where(self.g == self.g.max())][0]


class PairCorrelationExtractor(BaseExtractor):

    output_name = 'paircorr'

    @classmethod
    def from_one_configuration(cls, config, verbose=False, **kwds):
        '''From one configuration where config is an Nxn array with
           either n=2 or n=3
        '''
        return cls.from_many_configurations([config], box_size, verbose=verbose, **kwds)

    @classmethod
    def from_configurations(cls, configs, **kwds):
        return cls(configs, **kwds)

    def __init__(self, config_iterator, box_size=None, prec=None, dims=None, truncate_nonperiodic=True, **kwds):
        super(PairCorrelationExtractor, self).__init__(**kwds)
        self.config_iterator = config_iterator
        self.box_size = box_size if box_size is None else np.asarray(box_size)
        self.prec = prec
        self.dims = dims
        self.truncate_nonperiodic = truncate_nonperiodic

        self.pair_indices = {}

    def extract(self):
        acc_config_separations = []
        try:
            length = len(self.config_iterator)
        except (ValueError, AttributeError):
            length = None
        for i,config in enumerate(self.config_iterator):
            acc_config_separations.append(self.calculate_config_pair_separations(config))
            self.provide_info(i, length, None)

        if not len(acc_config_separations):
            raise ValueError("no configs")
        pair_separations = np.concatenate(acc_config_separations)
        dist = self.calculate_pair_density_distribution(pair_separations)
        g = PairCorrelation.from_density_distribution(dist)
        return g

    def calculate_config_pair_separations(self, config):

        config = np.asarray(config)

        if len(config.shape) != 2 or config.shape[1] not in (2,3):
            raise ValueError("bad shape for config %s; must be (N,2) or (N,3)" % (config.shape,))
        if self.dims is None:
            self.dims = config.shape[1]
        elif config.shape[1] != self.dims:
            raise ValueError("inconsistent dimensions %s; initally %s" % (config.shape[1], self.dims))

        pair_indices = self.get_pair_indices(config.shape[0])
        pairs = config[pair_indices.ravel()].reshape(pair_indices.shape + (self.dims,))
        r_ij = (calculate_periodic_deltas(pairs[:,0,:], pairs[:,1,:], self.box_size)
                if self.box_size is not None else
                pairs[:,0,:] - pairs[:,1,:])

        l_ij = (r_ij ** 2).sum(axis=1) ** 0.5
        assert (l_ij >= 0).all()

        if self.box_size is not None:
            assert (l_ij <= ((0.5*self.box_size)**2).sum() ** 0.5).all()
            if self.truncate_nonperiodic:
                l_ij = l_ij[l_ij <= 0.5 * self.box_size.min()]

        return l_ij

    def calculate_pair_density_distribution(self, pair_separations):
        prec = (self.prec if self.prec is not None
                else self.calculate_prec(pair_separations))
        return Distribution.fromseq(pair_separations, prec=prec)

    def calculate_prec(self, pair_separations):
        return pair_separations.min() * 0.01 #roughly one hundreths of a particle size

    def get_pair_indices(self, N):
        if N not in self.pair_indices:
            if N > 1000:
                warnings.warn(
                    'special code needs developed for N>1000',
                    DeprecationWarning, stacklevel=2)
            self.pair_indices[N] = np.array( np.array(np.triu_indices(N, 1)).T )
        return self.pair_indices[N]


class XBaseStreamingPairCorrelationExtractor(BaseExtractor):

    def __init__(self, config_iterator, box_size, prec,
                 max_r=None, truncate_nonperiodic=True,
                 length=None,
                 **kwds):
        super(XBaseStreamingPairCorrelationExtractor, self).__init__(**kwds)
        self.config_iterator = config_iterator

        box_size = np.asarray(box_size, dtype=float)
        if box_size.shape != (3,):
            raise ValueError("bad box_size shape")
        self.box_size = box_size

        if prec < 0:
            raise ValueError("bad prec %r" % (prec,))
        self.prec = prec

        if max_r is None:
            if truncate_nonperiodic:
                max_r = 0.5 * self.box_size.min()
            else:
                max_r = ((0.5*self.box_size)**2).sum() ** 0.5
        elif max_r < 0:
            raise ValueError("bad max_r %r" % (max_r,))
        self.max_r  = max_r

        self.length = length

    def extract(self):
        self.initialize_extract()
        if self.length is not None:
            length = self.length
        else:
            try:
                length = len(self.config_iterator)
            except (ValueError, AttributeError, TypeError):
                length = None
        for i,config in enumerate(self.config_iterator):
            self.acc_a_config(config)
            self.provide_info(i, length, None)
        return self.reduce_extraction()


class XStreamingPerodicPairCorrelationExtractor(XBaseStreamingPairCorrelationExtractor):

    output_name = 'xspaircorr'

    def initialize_extract(self):
        self.acc_one_config_rs = np.empty(1 + self.max_r / self.prec,
                                          dtype=acc_rs_dtype)
        self.acc_all_rs = np.zeros(self.acc_one_config_rs.shape, dtype=float)

    def acc_a_config(self, config):
        self.acc_one_config_rs.fill(0)
        acc_periodic_rs(self.acc_one_config_rs, 0, self.prec,
                        config, self.box_size)
        self.acc_all_rs += self.acc_one_config_rs

    def reduce_extraction(self):
        dist = Distribution(self.acc_all_rs, prec=self.prec)
        g = PairCorrelation.from_density_distribution(dist)
        return g


class PairOrientationCorrelation(object):

    def __init__(self, rs, orientations):
        self.rs = rs
        self.orientations = orientations


class XStreamingPeriodicPairOrientationExtractor(XBaseStreamingPairCorrelationExtractor):

    output_name = 'xspairorient'

    def initialize_extract(self):
        self.acc_one_config_n_rs = np.empty(1 + self.max_r // self.prec,
                                            dtype=acc_rs_dtype)
        self.acc_all_n_rs = np.zeros(self.acc_one_config_n_rs.shape, dtype=float)

        self.acc_one_config_orients = np.empty(1 + self.max_r / self.prec,
                                               dtype=acc_orients_dtype)
        self.acc_all_orients = np.zeros(self.acc_one_config_orients.shape, dtype=float)

    def acc_a_config(self, (positions, orientations)):
        self.acc_one_config_n_rs.fill(0)
        self.acc_one_config_orients.fill(0)

        acc_periodic_orientation_rs(self.acc_one_config_orients,
                                    self.acc_one_config_n_rs,
                                    0, self.prec, positions, orientations,
                                    self.box_size)
        self.acc_all_n_rs += self.acc_one_config_n_rs
        self.acc_all_orients += self.acc_one_config_orients

    def reduce_extraction(self):
        w = np.where(self.acc_all_n_rs != 0)
        rs = np.arange(0, 1 + self.max_r // self.prec, 1, dtype=float) * self.prec
        orients = self.acc_all_orients[w] / self.acc_all_n_rs[w]
        return PairOrientationCorrelation(rs, orients)