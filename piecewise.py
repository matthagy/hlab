
from __future__ import division
from __future__ import absolute_import

from .ihlab import IDomain

__all__ = '''NoPiece PieceWiseCollection PieceWiseFunction
'''.split()


class NoPiece(LookupError):
    def __init__(self, key):
        self.key = key
    def __str__(self):
        return 'no key for %s' % (key,)


class PieceWiseCollection(object):

    def __init__(self, seq=None):
        self.pieces = []
        if seq is not None:
            self.update(seq)

    def add(self, domain, value):
        domain = IDomain(domain)
        for piece_domain,piece_value in self.pieces:
            if not (piece_domain & domain).is_empty():
                raise ValueError("can't add piece %s; overlaps with existing %s; %s" %
                                 (domain, piece_domain,
                                  piece_domain & domain))
        self.pieces.append((domain, value))

    def get(self, key, default=None):
        for domain,value in self.pieces:
            if key in domain:
                return value
        return default

    def __getitem__(self, key):
        nop = object()
        value = self.get(key, nop)
        if value is nop:
            raise NoPiece(key)
        return value

    def __len__(self):
        return len(self.pieces)

    def __iter__(self):
        return self.itervalues()

    def itervalues(self):
        for domain,value in self.pieces:
            yield value

    def iterdomains(self):
        for domain,value in self.pieces:
            yield domain

    def iteritems(self):
        return iter(self.pieces)

    def update(self, op):
        if isinstance(op, PieceWiseCollection):
            op = op.pieces
        for domain,value in op:
            self.add(domain, value)

    def map(self, func, *args):
        return self.map_values(func, *args)

    def map_values(self, func, *args):
        return self.__class__([(domain,func(value,*args))
                               for domain,value in self.pieces])


class PieceWiseFunction(object):

    def __init__(self, funcs=None):
        self.pieces = PieceWiseCollection(funcs)
        for func in self.pieces.itervalues():
            assert callable(func)

    def add_func(self, domain, func):
        assert callable(func)
        self.pieces.add(domain, func)

    def __call__(self, x):
        return self.pieces[x](x)


