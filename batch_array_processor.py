
from __future__ import division

from functools import wraps

import numpy as np

default_batch_size = 128


def batch_array_processor(batch_size=None):

    def decorator(func):

        @wraps(func)
        def wrapper(arr, *args, **kwds):

            arr = np.asarray(arr)

            if not arr.size:
                return np.zeros_like(arr)

            if len(arr.shape) != 1:
                return wrapper(arr.flat, *args, **kwds).reshape(arr.shape)

            this_batch_size = kwds.pop('batch_size', batch_size)

            if arr.size > this_batch_size:
                n_batches, extra = divmod(arr.size, this_batch_size)
                if extra:
                    n_batches += 1
                return np.concatenate(list(func(arr[i     * this_batch_size:
                                                    (i+1) * this_batch_size:],
                                                *args, **kwds)
                                           for i in xrange(n_batches)))
            else:
                return func(arr, *args, **kwds)

        return wrapper

    if callable(batch_size):
        func, batch_size = batch_size, default_batch_size
        return decorator(func)

    elif batch_size is None:
        batch_size = default_batch_size

    return decorator
