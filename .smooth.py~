'''smoothing utility based off of scipy cookbook example
'''
import numpy as N

windows = '''
    flat hanning hamming bartlett blackman
'''.split()

def smooth1D(x, window_len=None, window_frac=0.05, window='hanning'):
    """smooth data using a windowing method
    """
    x = N.asarray(x)
    if x.ndim != 1:
        raise ValueError("array not 1 dimensional")
    if window_len is None:
        window_len = int(round(window_frac * x.size))
    if x.size < window_len:
        raise ValueError("input not large enough for window size")
    if window not in windows:
        raise ValueError("invalid window")
    if window_len < 3:
        return x
    s = N.r_[2*x[0]  - x[window_len:1:-1],
             x,
             2*x[-1] - x[-1:-window_len:-1]]
    if window == 'flat':
        w = N.ones(window_len, float)
    else:
        w = getattr(N, window)(window_len)
    y = N.convolve(w/w.sum(), s, mode='same')
    return y[window_len-1:-window_len+1]


