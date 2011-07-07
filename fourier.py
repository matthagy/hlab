
from __future__ import division

import numpy as np

from .batch_array_processor import batch_array_processor

@batch_array_processor(batch_size=128)
def fourier(k, t, y):
    assert len(k.shape) == 1
    assert len(t.shape) == 1
    assert len(y.shape) == 1
    assert t.shape == y.shape
    k_p = k[np.newaxis, ::]
    t_p = t[::, np.newaxis]
    y_p = y[::, np.newaxis]
    return np.trapz(np.exp(1j * k_p * t_p) * y_p,
                    x=t, axis=0)
