
from functools import wraps

def memorize(func):
    cache = {}
    @wraps(func)
    def wrapper(*args):
        try:
            return cache[args]
        except KeyError:
            result = cache[args] = func(*args)
            return result
    wrapper.cache = cache
    return wrapper
