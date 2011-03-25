'''Time correlation functions library
'''

from __future__ import division
from __future__ import absolute_import

import numpy as np

from .util import epsilon_eq, calculate_periodic_deltas, coere_listlike
from .extractor import BaseExtractor


class BaseTCFExtractor(BaseExtractor):

    output_name = 'tcf'

    def __init__(self, seq, delta=1, longest_correlation=None, sample_rate=None, **kwds):
        super(BaseTCFExtractor, self).__init__(**kwds)
        self.seq = coere_listlike(seq)
        self.delta = delta
        self.longest_correlation = longest_correlation
        self.sample_rate = sample_rate

    def extract(self):

        N = len(self.seq)
        sample_step = 1 if self.sample_rate is None else int(round(self.sample_rate / self.delta))
        corr_lengths = np.arange(min(N-2, N if self.longest_correlation is None else
                                     int(round(self.longest_correlation / self.delta))))
        acc_corrs = []
        for inx,corr_length in enumerate(corr_lengths):
            self.provide_info(inx, len(corr_lengths), None)

            corr_samples = np.array(map(self.calculate_a_correlation,
                                        self.seq[:N-corr_length:sample_step],
                                        self.seq[corr_length::sample_step]))
            acc_corrs.append([corr_samples.mean(axis=0),
                              corr_samples.std(axis=0) / np.sqrt(len(corr_samples)),
                              len(corr_samples)])

        if not len(acc_corrs):
            return None

        mn,err,n = map(np.array, zip(*acc_corrs))
        return self.delta * np.arange(len(mn)), mn, err, n

    def calculate_correlation(self, x_i, x_f):
        raise RuntimeError("calculate_correlation not implemented")


def normalize_positions_trajectories(coms, box_size):
    '''Normalize trajectory of positions from a periodic simulation, such that periodic
       effects are removed.
    '''
    coms = coere_listlike(coms)
    box_size = np.asarray(box_size) * np.ones(3)
    deltas = np.array(list(calculate_periodic_deltas(coms_i, coms_j, box_size)
                           for coms_i, coms_j in zip(coms[:-1:], coms[1::])))
    deltas = np.concatenate([[np.zeros_like(deltas[0])], deltas])
    return np.add.accumulate(deltas, axis=0)

class MeanSquareDisplacementTCFExtractor(BaseTCFExtractor):

    def __init__(self, coms, box_size=None, **kwds):
        coms = normalize_positions_trajectories(coms, box_size)
        super(MeanSquareTCFCalcualtor, self).__init__(coms, **kwds)
        self.scratch = np.zeros_like(coms[0])
        self.n = len(self.scratch)

    def calculate_correlation(self, initial_coms, final_coms):
        # calculate the following expression efficiently
        #   sqrt(((final_coms - initial_coms) ** 2).sum(axis=1).mean())
        scratch = self.scratch
        np.subtract(final_coms, initial_coms, scratch)
        np.power(scratch, 2, scratch)
        return scratch.sum() / self.n


def ravel_vector_quantities(seq):

    def coere_vector(op):
        op = np.asarray(op)
        if len(op.shape) == 1:
            op = op.reshape(op.shape + (1,))
        elif len(op.shape) > 2:
            raise ValueError("too many dimensions in vector quantity")
        return op

    vectors = map(coere_vector, seq)

    base_shape = vectors[0].shape
    if not all(v.shape == base_shape for v in vectors):
        raise ValueError("inconsistent shape in vector quantities")

    raveled = list(v.ravel() for v in vectors)

    return raveled, base_shape


class BaseVectorTCFCalculator(BaseTCFExtractor):

    def __init__(self, orientation_vectors, **kwds):
        ravel_orientation_vectors, base_shape = ravel_vector_quantities(orientation_vectors)
        super(OrientationalTCFExtractor, self).__init__(ravel_orientation_vectors, **kwds)
        self.scratch = np.zeros_like(ravel_orientation_vectors[0])
        self.n_vectors = base_shape[0]

    def calculate_correlation(self, initial_vectors, final_vectors):
        # efficiently calculate mean(dot(v_i, v_f) for v_i, v_f in zip(initial_vectors, final_vectors))
        np.multiply(initial_vectors, final_vectors, self.output)
        return self.output.sum() / self.n_vectors


class OrientationalTCFExtractor(BaseVectorTCFCalculator):
    pass

class VelocityTCFExtractor(BaseVectorTCFCalculator):
    pass
