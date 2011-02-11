'''
'''

from __future__ import division
from __future__ import absolute_import

from collections import defaultdict
import warnings

import numpy as np

from .calculated import PickelingBase, calculated
from .bases import PlottingMixin
from .plotting import make_data


class Profile(PickelingBase, PlottingMixin):
    '''establishes a mapping between uniformly distributed keys
       (at distance prec from each other), values, and uncertainty
       in those values
    '''

    def __init__(self, data, prec=1, _uncertainty=None, x_min=0):
        self.data = np.asarray(data)
        self.prec = prec
        self._uncertainty = _uncertainty
        self.x_min = x_min

    @calculated
    def length(self):
        return len(self.data)

    @calculated
    def has_uncertainty(self):
        return self._uncertainty is not None

    @calculated
    def uncertainty(self):
        return (np.asarray(self._uncertainty)
                if self._uncertainty is not None else
                np.nzeros(self.length))

    @calculated
    def indices(self):
        return (np.arange(len(self.data))+0.5) * self.prec + self.x_min

    @calculated
    def where_sparse(self):
        return np.where(self.data != 0)

    @calculated
    def sparse_data(self):
        return self.data[self.where_sparse]

    @calculated
    def sparse_indices(self):
        return self.indices[self.where_sparse]

    @calculated
    def sparse_uncertainty(self):
        return self.uncertainty[self.where_sparse]

    def coarse(self, factor=2):
        assert factor >= 1
        return self.new_prof(sum([self.data[i::factor] for i in xrange(factor)], axis=0),
                             self.prec * factor,
                             sum([self.uncertainty[i::factor]**2 for i in xrange(factor)], axis=0)**0.5
                             if self.has_uncertainty else None)

    def average2(self, other, self_weight=1, other_weight=1):
        return self.new_prof(self.combine_arrays(self.data, other.data, self_weight, other_weight),
                             self.prec,
                             self.combine_arrays(self.uncertainty, other.uncertainty,
                                                 self_weight, other_weight)
                               if self.has_uncertainty or other.has_uncertainty else
                             None)

    @staticmethod
    def combine_arrays(a1,a2,w1,w2):
        l1 = len(a1)
        l2 = len(a2)
        l = max(l1,l2)
        if l1 != l:
            a1 = np.concatenate([a1, np.zeros(l-l1)])
        if l2 != l:
            a2 = np.concatenate([a2, np.zeros(l-l2)])
        return (w1*a1 + w2*a2) / (w1+w2)

    def new_prof(self, data, prec, uncertainty, x_min=None):
        return self.__class__(data, prec, uncertainty, self.x_min if x_min is None else x_min)

    def gnuplot_item(self, title=None, sparse=True, with_=None, uncertainty=None):
        data = [self.sparse_indices if sparse else self.indices,
                self.sparse_data if sparse else self.data]
        if uncertainty is None:
            uncertainty = self.has_uncertainty
        if uncertainty:
            data.append(self.sparse_uncertainty if sparse else self.uncertainty)
        if with_ is None:
            with_ = 'yerrorlines' if uncertainty else 'lp'
        if title is None:
            title = self.__class__.__name__
        elif title is False:
            title = ''
        return make_data(title=title, with_=with_, *data)

#     @staticmethod
#     def smooth_arr(arr, offsets, weights,
#                    #cached globals
#                    len=len):
#         acc_smooth = []
#         length = len(arr)
#         assert len(offsets) == len(weights)
#         n_weights = len(weights)
#         for index in xrange(length):
#             relavent_indices = offsets+index
#             where_relavent = where((relavent_indices >= 0) &
#                                    (relavent_indices < length))
#             if len(where_relavent[0]) != n_weights:
#                 #ensure symmetic
#                 used_offsets = offsets[where_relavent]
#                 sym_offsets = set(used_offsets) & set(-used_offsets)
#                 where_relavent = array([i for i,value in enumerate(offsets)
#                                         if value in sym_offsets]),
#             relavent_indices = relavent_indices[where_relavent]
#             relavent_weights = weights[where_relavent]
#             acc_smooth.append((arr[relavent_indices] * relavent_weights).sum() /
#                               relavent_weights.sum())
#         return array(acc_smooth)

#     def smoothed(self, offsets=[-1,1,1], weights=[1,1,1]):
#         offsets = asarray(offsets, dtype=int)
#         weights = asarray(weights)
#         assert len(offsets.shape) == 1
#         assert len(weights.shape) == 1
#         assert len(offsets) == len(weights)
#         assert (weights >= 0).all()
#         return  self.new_prof(self.smooth_arr(self.data, offsets, weights),
#                               self.prec,
#                               self.smooth_arr(self.uncertainty, offsets, weights)
#                               if self.has_uncertainty else None)

#     def smoothed_guass(self, window_size=5, falloff=0.05):
#         indices = arange(-window_size, window_size+1)
#         return self.smoothed(indices, exp(-falloff * abs(indices)))

    def rms_deviation(self, other):
        scratch = self.data - other.data
        scratch *= scratch
        return scratch.mean() ** 0.5


class Distribution(Profile):
    '''In a profile data represents the frequency at which
       the index is observed
    '''

    @calculated
    def probability(self):
        return self.data / self.data.sum()

    @calculated
    def sparse_probability(self):
        return self.probability[self.where_sparse]

    def weighted_moment(self, moment=1):
        return self.probability * self.indices ** moment

    @calculated
    def weighted(self):
        return self.weighted_moment(moment=1)

    @calculated
    def mean(self):
        return self.weighted.sum()

    @calculated
    def stddev(self):
        return self.weighted_moment(moment=2).sum() ** 0.5

    @classmethod
    def fromseq(cls, items, prec=1):
        m = defaultdict(int)
        items = asarray(items)
        if prec != 1:
            items = floor(items / prec).astype(int)
        for inx in items:
            m[inx] += 1
        if not m:
            return cls([], prec)
        return cls([m[inx] for inx in xrange(max(m)+1)], prec)


class CorrelationFunction(object):

    def __new__(cls, *args, **kwds):
        warnings.warn(
            'prof.CorrelationFunction is depreicate; use paircorrelation.PairCorrelation',
            DeprecationWarning, stacklevel=2)
        from .paircorrelation import PairCorrelation
        return PairCorrelation(*args, **kwds)

    @classmethod
    def from_density_distribution(cls, *args, **kwds):
        warnings.warn(
            'prof.CorrelationFunction is depreicate; use paircorrelation.PairCorrelation',
            DeprecationWarning, stacklevel=2)
        from .paircorrelation import PairCorrelation
        return PairCorrelation.from_density_distribution(*args, **kwds)


