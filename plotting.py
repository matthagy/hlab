
try:
    from Gnuplot import Gnuplot, Data
except ImportError:
    pass

from numpy import *

def make_data(*args, **kwds):
    if 'with' in kwds:
        assert not 'with_' in kwds
        kwds['with_'] = kwds.pop('with')
    return Data(*args, **kwds)
