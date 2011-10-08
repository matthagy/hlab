'''smoothing utility based off of scipy cookbook example
'''
import numpy as np

windows = '''
    flat hanning hamming bartlett blackman
'''.split()

def smooth1D(x, window_len=None, window_frac=0.05, window='hanning'):
    """smooth data using a windowing method
    """
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("array not 1 dimensional")
    if window_len is None:
        window_len = int(round(window_frac * x.size))

    if not window_len % 2:
        window_len -= 1

    if x.size < window_len:
        raise ValueError("input not large enough for window size")
    if window not in windows:
        raise ValueError("invalid window")

    if window_len < 3:
        return x

    s = np.r_[2*x[0]  - x[window_len:1:-1],
             x,
             2*x[-1] - x[-1:-window_len:-1]]

    s = np.r_[np.repeat(x[0], window_len - 1),
              x,
              np.repeat(x[-1], window_len - 1)]

    s = np.r_[x[window_len-1:0:-1],
             x,
             x[-1:-window_len:-1]]


    if window == 'flat':
        w = np.ones(window_len, float)
    else:
        w = getattr(np, window)(window_len)

    y = np.convolve(w/w.sum(), s, mode='same')
    return y[window_len - 1 : -window_len + 1:]

    y = np.convolve(w/w.sum(), s, mode='same')
    return y[window_len-1:-window_len+1]

def gauss_kern2D(size, sizey=None):
    size = int(size)
    if not sizey:
        sizey = size
    else:
        sizey = int(sizey)
    x, y = np.mgrid[-size:size+1, -sizey:sizey+1]
    g = np.exp(-(x**2/float(size)+y**2/float(sizey)))
    return g / g.sum()

def smooth2D(x, kernel_size, mode='same'):
    # delay loading as scipy takes a long time to load and isn't always needed
    from scipy import signal
    return signal.convolve(x, gauss_kern2D(kernel_size), mode=mode, old_behavior=False)
