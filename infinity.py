
__all__ = '''
oo InfinityType
'''.split()

class InfinityType(object):

    def __init__(self, negative=False):
        self.negative = not not negative

    def __eq__(self, other):
        return (isinstance(other, InfinityType) and
                self.negative == other.negative)

    def __ne__(self, other):
        return not (self==other)

    def __ge__(self, other):
        return self==other or not self.negative

    def __gt__(self, other):
        return not (self.negative or self==other)

    def __le__(self, other):
        return self==other or self.negative

    def __lt__(self, other):
        return self.negative and self!=other

    def __neg__(self):
        return InfinityType(not self.negative)

    def __str__(self):
        return (self.negative and '-' or '') + 'oo'

    def __repr__(self):
        return '%s(negative=%r)' % (
             self.__class__.__name__,
                           self.negative)

oo = InfinityType()

assert oo > 0
assert oo >= 0
assert oo == oo
assert -oo != oo
assert -oo < 0
assert -oo <= 0
assert -oo < oo
assert oo > -oo
assert -oo <= oo
assert oo >= -oo

