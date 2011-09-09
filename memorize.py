
from inspect import getargspec
from functools import wraps


def memorize(func):
    return make_wrapper(func, {})

def make_wrapper(func, cache):

    @wraps(func)
    def wrapper(*args, **kwds):

        args, star_args, star_kwds = parse_args(func, args, kwds)

        key = args, star_args, tuple((k,v) for k,v in sorted(star_kwds.iteritems()))

        try:
            return cache[key]
        except KeyError:
            result = cache[key] = func(*(args + star_args), **star_kwds)
            return result

    wrapper.cache = cache
    return wrapper

named_caches = {}

def named_memorize(name):
    return lambda func: make_wrapper(func, named_caches.setdefault(name, {}))

def parse_args(func, args, kwds):
    a = getargspec(func)
    n_pos = len(a.args) - len(a.defaults or ())
    pos_args = args[:n_pos:]
    star_args = args[len(a.args)::]
    star_kwds = dict(zip(a.args[n_pos::], args[n_pos::]))
    star_kwds.update(kwds)
    pos_args = pos_args + tuple(star_kwds.pop(k) if k in star_kwds else d
                                for k,d in zip(a.args[n_pos::],
                                               (a.defaults or ())))
    return pos_args, star_args, star_kwds

