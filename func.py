
from functools import partial

def identity(x): return x
def noop(x): return None

def returner_base(op): return op
def return_it(op):
    return partial(returner_base, op)


