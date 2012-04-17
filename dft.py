'''Discrete Fourier Transform Utilities
'''

from __future__ import division

import numbers

import numpy as np
from scipy.fftpack import dst, idst


class DFTIsotropic3D(object):
    '''Utility for working with discrete isotropic (spherically symmetric) functions in
       3D space. Specifically provides methods for discrete Fourier transform and
       the corresponding inverse of discrete functions in this 3D space.

       Transformations are performed by converting the 3D integral into a
       1D integral using the Fourier-Bessel transform. Internally, this
       1D integral is then efficiently evaluated for all discrete points
       using discrete sine transformations.
    '''

    def __init__(self, n, dr):
        if not isinstance(n, numbers.Integral) or n <= 0:
            raise ValueError("n must be a positive integer")
        n = int(n)
        assert n > 0

        if not isinstance(n, numbers.Real) or dr <= 0.0:
            raise ValueError("dr must be a positive real number")
        dr = float(dr)
        assert dr > 0.0

        # defintions of finite space
        self.n = n
        self.dr = dr
        self.dk = np.pi / (n*dr)

        indices = 1+np.arange(n)
        self.r = self.dr * indices
        self.k = self.dk * indices

        # normalization constants for scipy dst and idst routines
        self.dst_normalization = 0.5 * self.dr
        self.idst_normalization = n / 2 / (n + 1) * self.dk
        assert np.allclose(self.dst_normalization * self.idst_normalization,
                           np.pi / 4 / (n + 1))

        # precompute transformation prefactor arrays
        self.dft_prefactor = 4 * np.pi / self.k * self.dst_normalization
        self.idft_prefactor = (4 * np.pi / self.r * 1/(2 * np.pi)**3 *
                               self.idst_normalization)

    dst_type = 1

    def dft(self, f):
        assert f.shape == (self.n,)
        return self.dft_prefactor * dst(f * self.r, type=self.dst_type)

    def idft(self, F):
        assert F.shape == (self.n,)
        return self.idft_prefactor * idst(F * self.k, type=self.dst_type)
