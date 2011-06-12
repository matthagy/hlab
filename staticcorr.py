'''Static correlations in data
'''

from __future__ import division
from __future__ import absolute_import

import warnings
import __builtin__
from multiprocessing import Pool
from contextlib import closing

import numpy as np

from .calculated import calculated
from .util import coere_listlike, calculate_periodic_deltas
from .prof import Profile, Distribution
from .extractor import BaseExtractor
from .smooth import smooth2D
try:
    from .chlab import (acc_periodic_rs, acc_periodic_orientation_rs, acc_rs_dtype,
                        acc_orients_dtype, acc_periodic_orient_position,
                        acc_periodic_pair_orient)
except ImportError:
    pass

def combine(*ops):
    return reduce(lambda a,b: a.combine(b), ops)

class PairProfile(Profile):

    @calculated
    def r(self):
        return self.indices

    def combine(self, other):
        if not isinstance(other, PairProfile):
            raise TypeError("can only combine PairProfiles")
        assert self.x_min == other.x_min
        assert self.prec == other.prec
        assert self._uncertainty is None
        assert other._uncertainty is None
        return self.__class__(combine_arrays(self.data, other.data), prec=self.prec, x_min=self.x_min)

def combine_arrays(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.ndims != b.ndims:
        raise ValueError("can't combine arrays with different dimensions")
    if a.shape == b.shape:
        return 0.5 * (a+b)
    if a.ndims != 1:
        raise ValueError("differently sized arrays can only be combined if one dimensional")
    if a.size > b.size:
        a,b = b,a
    assert a.size < b.size
    return np.concatenate([0.5 * (a + b[:a.size:]), b[a.size::]])


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

def identity(op):
    return op

def do_extract((func, func_args, func_kwds, op)):
    return func(op, *func_args, **func_kwds)

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

    def get_length(self):
        if self.length is not None:
            return self.length
        try:
            return len(self.config_iterator)
        except (ValueError, AttributeError, TypeError):
            return None

    @staticmethod
    def initialize_extract(asynchronous):
        pass

    @staticmethod
    def reduce_extractions(op, acc):
        return op + acc

    @staticmethod
    def wrap_extraction(op):
        return op

    global_extraction_function = staticmethod(identity)
    extract_args = ()
    extract_kwds = None

    def extract(self):
        return self.extraction_loop(False, (self.global_extraction_function(config, *self.extract_args, **(self.extract_kwds or {}))
                                            for config in self.config_iterator))

    def extract_pool(self, pool):
        return self.extraction_loop(True, pool.imap_unordered(do_extract, ((self.global_extraction_function,
                                                                            self.extract_args, (self.extract_kwds or {}), config)
                                                                           for config in self.config_iterator)))

    def extract_asynchronous(self, processes=None):
        with closing(Pool(processes)) as pool:
            return self.extract_pool(pool)

    def extraction_loop(self, asynchronous, extract_iter):
        self.initialize_extract(asynchronous)
        length = self.get_length()
        acc = None
        for i,extract in enumerate(extract_iter):
            self.provide_info(i, length, None)
            acc = extract if acc is None else self.reduce_extractions(acc, extract)
        if acc is None:
            return None
        return self.wrap_extraction(acc)


def calculate_periodic_rs(positions, prec, max_r, box_size):
    acc_config_rs = np.zeros(1 + max_r / prec, dtype=acc_rs_dtype)
    acc_periodic_rs(acc_config_rs, 0, prec, positions, box_size)
    return acc_config_rs

class XStreamingPerodicPairCorrelationExtractor(XBaseStreamingPairCorrelationExtractor):

    output_name = 'xspaircorr'

    global_extraction_function = staticmethod(calculate_periodic_rs)

    @staticmethod
    def reduce_extractions(op, acc):
        acc = acc.astype(np.float64)
        acc += op
        return acc

    def initialize_extract(self, asynchronous):
        self.extract_args = self.prec, self.max_r, self.box_size

    def wrap_extraction(self, config_rs):
        dist = Distribution(config_rs, prec=self.prec)
        return PairCorrelation.from_density_distribution(dist)


class OrientationCorrelation(object):

    def __init__(self, rs, orientations):
        self.rs = rs
        self.orientations = orientations

    def combine(self, other):
        if not isinstance(other, OrientationCorrelation):
            raise TypeError("can only combine OrientationCorrelations")
        if self.rs.size == other.rs.size and np.allclose(self.rs, other.rs):
            return self.__class__(self.rs, combine_arrays(self.orientations, other.orientations))
        map_a = dict(self.rs, self.orientations)
        map_b = dict(other.rs, other.orientations)
        acc = []
        for r in sorted(set(map_a) | set(map_b)):
            if r not in map_a:
                o = map_b[r]
            elif r not in map_b:
                o = map_a[r]
            else:
                o = 0.5 * (map_a[r] + map_b[r])
            acc.append([r, o])
        rs,orientations = np.array(acc).T
        return self.__class__(rs, orientations)


def calculate_pair_orientation_corr((positions, orientations), prec, max_r, box_size):
    n = 1 + int(max_r // prec)
    config_n_rs = np.zeros(n, dtype=acc_rs_dtype)
    config_orients = np.zeros(n, dtype=acc_orients_dtype)
    acc_periodic_orientation_rs(config_orients, config_n_rs,
                                0.0, prec, positions, orientations, box_size)
    return (config_n_rs, config_orients)

class XStreamingPeriodicPairOrientationExtractor(XBaseStreamingPairCorrelationExtractor):

    output_name = 'xspairorient'

    global_extraction_function = staticmethod(calculate_pair_orientation_corr)

    @staticmethod
    def reduce_extractions((n_rs, orients), (acc_n_rs, acc_orients)):
        acc_n_rs = acc_n_rs.astype(np.float64)
        acc_orients = acc_orients.astype(np.float64)
        acc_n_rs += n_rs
        acc_orients += orients
        return acc_n_rs, acc_orients

    def initialize_extract(self, asynchronous):
        self.extract_args = self.prec, self.max_r, self.box_size

    def wrap_extraction(self, (n_rs, orients)):
        w = np.where(n_rs != 0)
        rs = np.arange(0, 1 + n_rs.shape[0], 1, dtype=float)[w] * self.prec
        orients = orients[w] / n_rs[w]
        return OrientationCorrelation(rs, orients)


class BaseCorrelation2D(object):

    def __init__(self, g, prec):
        assert g.ndim == 2
        assert g.shape[0] == g.shape[1], 'bad shape %s' % (g.shape,)

        self.g = g
        self.prec = prec

    @property
    def N(self):
        return self.g.shape[0]

    def combine(self, BaseCorrelation2D):
        if not isinstance(other, PairCorrelation2D):
            raise TypeError("can only combine ")
        raise RuntimeError("not completed")

    def calculate_xy(self):
        return self.prec * np.mgrid[0:self.N:, 0:self.N:]

    def calculate_angular_cross_section(self, angular_precision=np.pi/32, r_min=0, r_max=None):
        x,y = self.calculate_xy()
        r2 = x**2 + y**2
        w = (r2 >= r_min**2)
        if r_max:
            w &= r2 <= r_max**2
        x = x[w].ravel()
        y = y[w].ravel()
        g = self.g[w].ravel()
        angles = np.where(x==0, np.pi/2, np.arctan(y / (x+1e-9*self.prec)))
        return Distribution.from_weighted_seq(angles, g, prec=angular_precision, naturalizer=np.round)

    def smooth(self, kernel_size):
        return self.__class__(smooth2D(self.g, kernel_size=kernel_size, mode='same'), self.prec)



class PairCorrelation2D(BaseCorrelation2D):

    @classmethod
    def from_acc(cls, ns, prec):
        assert ns.ndim == 2
        assert ns.shape[0] == ns.shape[1], 'bad shape %s' % (ns.shape,)

        bi = np.arange(ns.shape[0])
        dA = np.pi * prec * ((bi+1)**2 - bi**2)
        ns = ns / dA[..., np.newaxis]

        max_r = (1 + ns.shape[0]) * prec

        v = prec**2
        V = 0.25 * np.pi * max_r ** 2
        N = ns.sum()
        g = (V / (N * v)) * ns

        assert np.all(np.isfinite(g))
        return cls(g, prec)

    def create_1d_g(self):
        i,j = np.indices(self.g.shape)
        indices = np.floor(np.sqrt(i**2 + j**2)).astype(int).ravel()
        acc = defaultdict(list)
        for index,gi in zip(indices, self.g):
            acc[index].append(gi)
        mean = np.mean
        g = np.array(list(mean(acc[index]) if index in acc else 0.0
                          for index in np.arange(1+indices.max())))
        return PairCorrelation(g, self.prec)

    def create_1d_g(self):
        i,j = np.indices(self.g.shape)
        indices = np.floor(np.sqrt(i**2 + j**2)).astype(int).ravel()
        acc = np.ones(indices.max() + 1, dtype=float)
        for index,gi in zip(indices, self.g.ravel()):
            acc[index] += gi
        g = np.array(list(acc[index] / (indices == index).sum()
                          for index in np.arange(1+indices.max())))
        return PairCorrelation(g, self.prec)


def calculate_orient_position((positions, orientations), prec, max_r, box_size):
    n = int(np.floor(max_r / prec)) + 1
    acc = np.zeros((n,n), dtype=acc_rs_dtype)
    acc_periodic_orient_position(acc, prec, positions, orientations, box_size)
    return acc

class XStreamingPairCorrelation2D(XBaseStreamingPairCorrelationExtractor):

    global_extraction_function = staticmethod(calculate_orient_position)

    @staticmethod
    def reduce_extractions(op, acc):
        acc = acc.astype(np.float64)
        acc += op
        return acc

    def initialize_extract(self, asynchronous):
        self.extract_args = self.prec, self.max_r, self.box_size

    def wrap_extraction(self, acc):
        return PairCorrelation2D.from_acc(acc, self.prec)



class OrientationCorrelation2D(BaseCorrelation2D):

    pass

def calculate_periodic_pair_orient((positions, orientations), prec, max_r, box_size):
    n = 1 + int(max_r // prec)
    config_n_rs = np.zeros((n,n), dtype=acc_rs_dtype)
    config_orients = np.zeros((n,n), dtype=acc_orients_dtype)
    acc_periodic_pair_orient(config_orients, config_n_rs,
                             prec, positions, orientations, box_size)
    return (config_n_rs, config_orients)


class XStreamingOrientationCorrelation2D(XBaseStreamingPairCorrelationExtractor):

    global_extraction_function = staticmethod(calculate_periodic_pair_orient)

    @staticmethod
    def reduce_extractions((n_rs, orients), (acc_n_rs, acc_orients)):
        acc_n_rs = acc_n_rs.astype(np.float64)
        acc_orients = acc_orients.astype(np.float64)
        acc_n_rs += n_rs
        acc_orients += orients
        return acc_n_rs, acc_orients

    def initialize_extract(self, asynchronous):
        self.extract_args = self.prec, self.max_r, self.box_size

    def wrap_extraction(self, (n_rs, orients)):
        n_rs = n_rs.copy()
        n_rs[n_rs == 0] = 1
        return OrientationCorrelation2D(orients / n_rs, self.prec)

# old names for unpickling and deprecated class name constructors
PositionOrientationCorrelation = PairCorrelation2D
PairOrientationCorrelation = OrientationCorrelation2D

def XStreamingOrientPositionCorrelationExtractor(*args, **kwds):
    warnings.warn("XStreamingOrientPositionCorrelationExtractor is deprecated; use XStreamingPairCorrelation2D",
                  DeprecationWarning, stacklevel=2)
    return XStreamingPairCorrelation2D(*args, **kwds)

def XStreamingOrientPositionPairCorrelationExtractor(*args, **kwds):
    warnings.warn("XStreamingOrientPositionPairCorrelationExtractor is deprecated; use XStreamingOrientationCorrelation2D",
                  DeprecationWarning, stacklevel=2)
    return XStreamingOrientationCorrelation2D(*args, **kwds)
