
from functools import wraps

@wraps
def memorize(func):
    cache = {}
    def wrapper(*args):
        try:
            return cache[args]
        except KeyError:
            result = cache[args] = func(*args)
            return result
    return wrapper
