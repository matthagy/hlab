
from types import GeneratorType

__all__ = '''
flatten
'''

SequenceTypes = list,tuple,GeneratorType

def sequencep(op):
    return isinstance(op, SequenceTypes)

def flatten(op, sequencep=sequencep):
    for op in op:
        if sequencep(op):
            for sop in flatten(op, sequencep):
                yield sop
        else:
            yield op
