'''eXperimental physical units framework
'''

from __future__ import division

import operator
from math import log10
from decimal import Decimal
from re import Scanner
from functools import partial
from collections import defaultdict

from hlab import xalgebra as A
from hlab.sigfigs import SigFig
from hlab.bases import AutoRepr
from hlab.ratio import Ratio, as_ratio

from jamenson.runtime.collections import OrderedDict, OrderedSet, OrderedDefaultDict
from jamenson.runtime.multimethod import defmethod, MultiMethod
from jamenson.runtime.atypes import anytype, as_optimized_type, typep, Seq, TypeBase, as_type
from jamenson.runtime import atypes
from jamenson.runtime.as_string import as_string

name_type = as_optimized_type((str,unicode,type(None)))
lossless_number_type = as_optimized_type((int,long,Decimal,SigFig,Ratio))

class CompoundBase(AutoRepr, A.DivAlgebraBase):

    atom_base = None
    def __init__(self, atoms_and_powers=()):
        acc = []
        for atom,power in atoms_and_powers:
            atom = self.arg_to_atom(atom)
            assert isinstance(atom, self.atom_base)
            assert isinstance(power, (int,long))
            if power!=0:
                acc.append((atom, power))
        self.atoms_and_powers = tuple(acc)

    @staticmethod
    def arg_to_atom(arg):
        return arg

    @staticmethod
    def atom_to_arg(atom):
        return atom

    def repr_args(self):
        return [(self.atom_to_arg(atom), power)
                for atom,power in self.atoms_and_powers],

    def new_compound(self, atoms):
        return self.__class__(atoms)

    def __hash__(self):
        return hash(self.__class__) ^ hash(self.cannonicalized().atoms_and_powers)

    comparable_classes = ()
    def __eq__(self, other):
        if not isinstance(other, self.comparable_classes):
            return NotImplemented
        return self.cannonicalized().atoms_and_powers == other.cannonicalized().atoms_and_powers

    def __ne__(self, other):
        return not (self==other)

    def __rne__(self, other):
        return not (self==other)

    def cannonicalized(self):
        acc = defaultdict(int)
        for atom, power in self.atoms_and_powers:
            if isinstance(atom, CompoundBase):
                for sub_atom, sub_power in atom.cannonicalized().atoms_and_powers:
                    acc[sub_atom] += sub_power * power
            else:
                acc[atom.cannonicalized()] += power
        return self.new_compound(sorted(acc.iteritems()))

    def pow(self, p):
        assert isinstance(p, (int,long))
        return self.new_compound([(atom, power*p) for atom,power in self.atoms_and_powers])

    def invert(self):
        return self.new_compound([(atom, -power) for atom,power in self.atoms_and_powers])

    def mul(self, other):
        amap = OrderedDict(self.atoms_and_powers)
        bmap = OrderedDict(other.atoms_and_powers)
        return self.new_compound([(atom, amap.get(atom,0) + bmap.get(atom,0))
                                  for atom in OrderedSet(amap) | OrderedSet(bmap)])

    atom_text = 'get_name'
    format_single = '%s'
    format_power = '%s^%d'
    format_multiply = '%s*%s'
    format_divide = '%s/%s'

    def format(self, atom_text=None, format_single=None, format_power=None, format_multiply=None, format_divide=None):
        atom_text = self.get_attrgetter(atom_text, self.atom_text)
        format_single = self.get_formatter(format_single, self.format_single)
        format_power = self.get_formatter(format_power, self.format_power)
        format_multiply = self.get_formatter(format_multiply, self.format_multiply)
        pos,neg = [self.format_seq(seq, atom_text, format_single, format_power, format_multiply)
                   for seq in self.collect_posneg()]
        if not (neg or pos):
            return ''
        if not neg:
            return pos
        if not pos:
            pos = '1'
        format_divide = self.get_formatter(format_divide, self.format_divide)
        return format_divide(pos, neg)

    def collect_posneg(self):
        seq = self.normalized_atoms_and_powers()
        return ([[atom,power] for atom,power in seq if power>0],
                [[atom,-power] for atom,power in seq if power<0])

    def normalized_atoms_and_powers(self):
        acc = OrderedDefaultDict(int)
        for atom,power in self.atoms_and_powers:
            acc[atom] += power
        return list(acc.iteritems())

    @staticmethod
    def format_seq(seq, atom_text, format_single, format_power, format_multiply):
        seq = list(seq)
        if not seq:
            return ''
        return reduce(format_multiply, (format_power(atom_text(atom),i) if i!=1 else format_single(atom_text(atom))
                                        for atom,i in seq))

    @staticmethod
    def get_attrgetter(op, member):
        if op is None:
            op = member
        if callable(op):
            return op
        def getter(thing):
            el = getattr(thing, op)
            while callable(el):
                el = el()
            return el
        return getter

    @staticmethod
    def get_formatter(op, member):
        if op is None:
            op = member
        if callable(op):
            return op
        return lambda *args: op % args



class BaseDimensionality(AutoRepr, A.DivAlgebraBase):

    pass

as_dimensionality = MultiMethod('as_dimensionality')

@defmethod(as_dimensionality, [BaseDimensionality])
def meth(d):
    return d

class PrimitiveDimensionality(BaseDimensionality, A.DivAlgebraBase):

    names = {}
    def __init__(self, name):
        if name is not None:
            assert name not in self.names
            self.names[name] = self
        self.name = name

    def repr_args(self):
        return self.name,

    @classmethod
    def from_name(cls, name):
        try:
            return cls.names[name]
        except KeyError:
            return cls(name)

    def __str__(self):
        return self.name

    def cannonicalized(self):
        return self

    def get_name(self):
        return self.name


compound_dimensionality_names = {}

class CompoundDimensionality(BaseDimensionality, CompoundBase):

    atom_base = BaseDimensionality

    @staticmethod
    def arg_to_atom(arg):
        if isinstance(arg, str):
            return PrimitiveDimensionality.from_name(arg)
        return arg

    @staticmethod
    def atom_to_arg(atom):
        if isinstance(atom, PrimitiveDimensionality):
            return atom.name
        return atom

    def get_name(self):
        cannon = self.cannonicalized()
        try:
            return compound_dimensionality_names[cannon]
        except KeyError:
            return cannon.format('name', format_power='%s%d', format_multiply='%s*%s', format_divide='%s_%s')

    def __str__(self):
        cannon = self.cannonicalized()
        try:
            return compound_dimensionality_names[cannon]
        except KeyError:
            return cannon.format('name', format_power='%s%d', format_multiply='%s*%s', format_divide='%s/%s')

CompoundDimensionality.comparable_classes = CompoundDimensionality,

def primdim_to_compound(p):
    return CompoundDimensionality([[p, 1]])

@A.defboth_mm_eq([CompoundDimensionality, PrimitiveDimensionality])
def meth(c, p):
    return c == primdim_to_compound(p)

@defmethod(A.mm_pow, [CompoundDimensionality, (int,long)])
def meth(c, p):
    return c.pow(p)

@defmethod(A.mm_pow, [PrimitiveDimensionality, (int,long)])
def meth(d, p):
    return primdim_to_compound(d).pow(p)

@defmethod(A.mm_mul, [CompoundDimensionality, CompoundDimensionality])
def meth(a, b):
    return a.mul(b)

@A.defboth_mm_mul([CompoundDimensionality, PrimitiveDimensionality])
def meth(c, p):
    return c.mul(primdim_to_compound(p))

@defmethod(A.mm_mul, [PrimitiveDimensionality, PrimitiveDimensionality])
def meth(a, b):
    return primdim_to_compound(a).mul(primdim_to_compound(b))

@defmethod(A.mm_div, [CompoundDimensionality, CompoundDimensionality])
def meth(a, b):
    return a.mul(b.invert())

@defmethod(A.mm_div, [CompoundDimensionality, PrimitiveDimensionality])
def meth(c, p):
    return c.mul(primdim_to_compound(p).invert())

@defmethod(A.mm_div, [PrimitiveDimensionality, CompoundDimensionality])
def meth(p, c):
    return primdim_to_compound(p).mul(c.invert())

@defmethod(A.mm_div, [PrimitiveDimensionality, PrimitiveDimensionality])
def meth(a, b):
    return primdim_to_compound(a).mul(primdim_to_compound(b).invert())

def dimensionalities():
    class Dimensionalities():
        pass
    def p(name):
        d = PrimitiveDimensionality(name)
        assert not hasattr(Dimensionalities, name)
        setattr(Dimensionalities, name, d)
        return d
    #primitive dimensionalities
    length = p('length')
    mass = p('mass')
    time = p('time')
    quantity = p('quantity')
    temperature = p('temperature')

    #compound dimensionalities
    def register_compound_dimensionality(name, cd):
        assert cd not in compound_dimensionality_names
        compound_dimensionality_names[cd] = name
        assert not hasattr(Dimensionalities, name)
        setattr(Dimensionalities, name, cd)
        return cd
    rcd = register_compound_dimensionality

    dimensionless = rcd('dimensionless', CompoundDimensionality([]))
    area = rcd('area', length**2)
    volume = rcd('volume', length**3)
    velocity = rcd('velocity', length/time)
    acceleration = rcd('acceleration', velocity / time)
    force = rcd('force', mass*acceleration)
    energy = rcd('energy', force*length)
    pressure = rcd('pressure', force/area)
    entropy = rcd('entropy', energy/temperature)
    return Dimensionalities

D = dimensionalities = dimensionalities()

@defmethod(as_dimensionality, [str])
def meth(name):
    try:
        return getattr(dimensionalities, name)
    except AttributeError:
        raise ValueError("unkown dimensionality %r" % (name,))

@defmethod(as_dimensionality, [unicode])
def meth(uname):
    return as_dimensionality(str(uname))



class Prefix(A.DivAlgebraBase, AutoRepr):

    names = {}
    powers = {}

    def __init__(self, power, name, abbrev=None):
        assert name not in self.names
        self.names[name] = self
        assert isinstance(power, (int,long))
        assert power not in self.powers
        self.powers[power] = self
        self.power = power
        self.name = name
        if abbrev is None:
            abbrev = name
        if abbrev!=name:
            assert abbrev not in self.names
            self.names[abbrev] = self
        self.abbrev = abbrev


    def repr_args(self):
        return filter(lambda x: x is not None, [self.power, self.name, self.abbrev])

    @classmethod
    def from_name(self, name):
        return self.names[name]

    def get_name(self):
        return self.name

    def get_abbrev(self):
        return self.abbrev

    @classmethod
    def from_power(cls, power):
        assert isinstance(power, (int,long))
        try:
            return cls.powers[power]
        except KeyError:
            p = cls.powers[power] = Prefix(power, 'tothe%d' % (power,),'10^%d' % (power,))
            return p

    def get_factor(self, base=10):
        return base**self.power


@A.defboth_mm_mul([Prefix, (int,long,float)])
def meth(pre, factor):
    power = int(round(log10(factor)))
    if abs(10**power - factor) > 1e-3:
        raise ValueError("bad factor %r" % (factor,))
    return Prefix.from_power(pre.power + power)

@defmethod(A.mm_div, [Prefix, (int,long,float)])
def meth(pre, factor):
    return pre * factor**-1

@defmethod(A.mm_mul, [Prefix, Prefix])
def meth(a, b):
    return Prefix.from_power(a.power + b.power)

@defmethod(A.mm_div, [Prefix, Prefix])
def meth(a, b):
    return Prefix.from_power(a.power - b.power)

@defmethod(A.mm_pow, [Prefix, (int, long)])
def meth(pre, power):
    return Prefix.from_power(pre.power * power)

def prefixes():
    for line in '''Yotta 	Y	24
                   Zetta 	Z	21
                   Exa 	        E	18
                   Peta 	P	15
                   Tera 	T	12
                   Giga 	G	9
                   Mega 	M	6
                   myria 	my	4
                   kilo 	k	3
                   hecto 	h	2
                   deka 	da	1
                   deci 	d	-1
                   centi 	c	-2
                   milli 	m	-3
                   micro 	mu	-6
                   nano 	n	-9
                   pico 	p	-12
                   femto 	f	-15
                   atto 	a	-18
                   zepto 	z	-21
                   yocto 	y	-24'''.strip().split('\n'):
         line = line.strip()
         if not line:
             continue
         name,abbrev,power = line.split()
         Prefix(int(power), name, abbrev)
    class Prefixes(object):
        pass
    for n,v in Prefix.names.iteritems():
        setattr(Prefixes, n, v)
    Prefixes.no_prefix = Prefix(0, 'no_prefix', '')
    return Prefixes
P = prefixes = prefixes()


class BaseUnit(AutoRepr, A.DivAlgebraBase):

    pass

as_unit = MultiMethod('as_unit')

@defmethod(as_unit, [BaseUnit])
def meth(bu):
    return bu


class PrimitiveUnit(BaseUnit):

    names = {}

    def __init__(self, dimensionality, name, abbrev=None):
        assert isinstance(name, str)
        assert name not in self.names
        self.names[name] = self
        self.name = name
        if abbrev is None:
            abbrev = name
        if abbrev != name:
            assert abbrev not in self.names
            self.names[abbrev] = self
        self.abbrev = abbrev
        self.dimensionality = as_dimensionality(dimensionality)
        assert isinstance(self.dimensionality, BaseDimensionality)

    def repr_args(self):
        return filter(None, [self.dimensionality.get_name()
                             if isinstance(self.dimensionality, PrimitiveDimensionality)
                             else self.dimensionality, self.name, self.abbrev])

    @classmethod
    def from_name(cls, name):
        try:
            return cls.names[name]
        except KeyError:
            return cls(name)

    def cannonicalized(self):
        return self

    def get_dimensionality(self):
        return self.dimensionality

    def get_name(self):
        return self.name

    def get_abbrev(self):
        return self.abbrev

    def get_dimensionality(self):
        return self.dimensionality

    def __str__(self):
        return self.get_abbrev()

    def without_prefix(self):
        return self

    def __eq__(self, other):
        if isinstance(other, PrimitiveUnit):
            return self is other
        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, PrimitiveUnit):
            return self is not other
        return NotImplemented

    def pow(self, p):
        return primunit_to_compound(self).pow(p)


class CompoundUnit(BaseUnit, CompoundBase):

    atom_base = BaseUnit

    def __init__(self, atoms_and_powers=(), prefix=P.no_prefix, cannonical=False):
        CompoundBase.__init__(self, atoms_and_powers)
        if isinstance(prefix, str):
            prefix = Prefix.from_name(prefix)
        assert isinstance(prefix, Prefix)
        self.prefix = prefix
        self.cannonical = cannonical

    def repr_args(self):
        atoms_and_powers, = CompoundBase.repr_args(self)
        return atoms_and_powers, self.prefix.name

    @staticmethod
    def arg_to_atom(arg):
        if isinstance(arg, str):
            return PrimitiveUnit.from_name(arg)
        return arg

    @staticmethod
    def atom_to_arg(atom):
        if isinstance(atom, PrimitiveDimensionality):
            return atom.get_abbrev()
        return atom

    def new_compound(self, atoms_and_powers):
        return self.__class__(atoms_and_powers, self.prefix)

    def __hash__(self):
        return hash(self.prefix) ^ super(CompoundUnit, self).__hash__()

    def __eq__(self, other):
        if isinstance(other, PrimitiveUnit):
            other = primunit_to_compound(other)
        elif not isinstance(other, CompoundUnit):
            return NotImplemented
        a = self.cannonicalized()
        b = other.cannonicalized()
        x = super(CompoundUnit, a).__eq__(b)
        if x is NotImplemented:
            return x
        return x and a.prefix == b.prefix

    def __req__(self, other):
        return self.__eq__(other)

    def format(self, prefix_attr=None, unit_text=None, **kwds):
        return '%s%s' % (self.get_attrgetter(prefix_attr, self.prefix_attr)(self.prefix)
                         if self.prefix != P.no_prefix else '',
                         super(CompoundUnit, self).format(atom_text=unit_text, **kwds))
    prefix_attr = 'get_abbrev'
    atom_text = 'get_abbrev'

    def without_prefix(self):
        if self.prefix==P.no_prefix:
            return self
        return self.__class__(self.atoms_and_powers, prefix=P.no_prefix)

    names = {}
    @classmethod
    def register_name(cls, unit, name, abbrev=None):
        assert isinstance(unit, CompoundUnit)
        assert unit not in cls.names
        cls.names[unit.without_prefix()] = unit.prefix, name, abbrev
        return unit

    def get_name_abbrev_prefix(self):
        op = self.cannonicalized()
        try:
            prefix, name, abbrev = self.names[op.without_prefix()]
        except KeyError:
            return None, None, None
        return op.prefix/prefix, name, abbrev

    def get_name(self):
        prefix,name,abbrev = self.get_name_abbrev_prefix()
        if name is not None:
            return '%s%s' % ('' if prefix is P.no_prefix else prefix.get_name(), name)
        return self.format(format_power='%s%d', prefix_attr='get_name', unit_text='get_name')

    def get_abbrev(self):
        prefix,name,abbrev = self.get_name_abbrev_prefix()
        if name is not None:
            return '%s%s' % ('' if prefix is P.no_prefix else prefix.get_abbrev(), abbrev)
        return self.format(format_power='%s%d', prefix_attr='get_abbrev', unit_text='get_abbrev')

    def __str__(self):
        return self.get_abbrev()

    def cannonicalized(self):
        if self.cannonical:
            return self
        acc = defaultdict(int)
        prefix = self.prefix
        for unit, power in self.atoms_and_powers:
            unit = unit.cannonicalized()
            if isinstance(unit, CompoundUnit):
                prefix = prefix * unit.prefix ** power
                for sub_unit, sub_power in unit.atoms_and_powers:
                    acc[sub_unit] += sub_power * power
            else:
                acc[unit] += power
        return CompoundUnit(sorted(acc.iteritems()), prefix, True)

    def ordered(self):
        acc = OrderedDefaultDict(int)
        for unit,power in self.atoms_and_powers:
            if isinstance(unit, CompoundUnit):
                unit = unit.ordered()
            acc[unit] += power
        return CompoundUnit(sorted(acc.iteritems()), self.prefix)

    def get_dimensionality(self):
        acc = D.dimensionless
        for unit,power in self.atoms_and_powers:
            acc *= unit.get_dimensionality() ** power
        return acc

    def split_posneg(self):
        pos,neg = self.collect_posneg()
        return [CompoundUnit(pos, prefix=self.prefix, cannonical=self.cannonical),
                CompoundUnit(neg, cannonical=self.cannonical)]

CompoundUnit.comparable_classes = CompoundUnit,

@A.defboth_mm_mul([CompoundUnit, (int,long,float)])
def meth(c, factor):
    return CompoundUnit(c.atoms_and_powers, c.prefix*factor)

@defmethod(A.mm_div, [CompoundUnit, (int,long,float)])
def meth(c, factor):
    return CompoundUnit(c.atoms_and_powers, c.prefix/factor)

@defmethod(A.mm_mul, [(CompoundUnit,PrimitiveUnit), (CompoundUnit,PrimitiveUnit)])
def meth(a, b):
    return CompoundUnit([[a,1], [b,1]])

@defmethod(A.mm_div, [(CompoundUnit,PrimitiveUnit), (CompoundUnit,PrimitiveUnit)])
def meth(a, b):
    return CompoundUnit([[a,1], [b,-1]])

@defmethod(A.mm_pow, [BaseUnit, (int,long)])
def meth(c, p):
    #return c.pow(p)
    return CompoundUnit([(c, p)])
    #return CompoundUnit([(c, 1 if p>0 else -1)]*abs(p))

def primunit_to_compound(p):
    return CompoundUnit([[p, 1]])

@A.defboth_mm_mul([PrimitiveUnit, (int, long, float)])
def meth(p, factor):
    return primunit_to_compound(p) * factor

@defmethod(A.mm_div, [PrimitiveUnit, (int, long, float)])
def meth(p, factor):
    return primunit_to_compound(p) / factor

#@defmethod(A.mm_pow, [PrimitiveUnit, (int, long)])
#def meth(p, power):
#    return primunit_to_compound(p) ** power

class UnitCollection(object):
    pass

def collect_units(ns):
    col = UnitCollection()
    for n,v in ns.iteritems():
        if isinstance(v, BaseUnit):
            setattr(col, n, v)
    return col

unit_namespaces = {}

class unit_namespace(object):
    _abbrev = None
    _precedence = 10
    class __metaclass__(type):
        def __new__(cls, name, bases, dct):
            ns = type.__new__(cls, name, bases, dct)
            assert name not in unit_namespaces
            unit_namespaces[name] = ns
            if ns._abbrev:
                assert ns._abbrev not in unit_namespaces
                unit_namespaces[ns._abbrev] = ns
            units = list((n.rstrip('_'),v) for n,v in dct.iteritems()
                         if isinstance(v, BaseUnit) and not n.startswith('_'))
            ns._units = dict((n,u) for n,u in units)
            ns._units.update((u.get_name(),u) for n,u in units)
            ns._units.update((u.get_abbrev(),u) for n,u in units)
            return ns

get_unit_by_name_cache = {}
def get_unit_by_name(name, ns_name=None):
    name = name.strip()
    if ns_name:
        ns_name = ns_name.strip()
    try:
        return get_unit_by_name_cache[name, ns_name]
    except KeyError:
        unit = get_unit_by_name_cache[name, ns_name] = find_unit_by_name(name, ns_name)
        return unit

@defmethod(as_unit, [str])
def meth(name):
    return get_unit_by_name(name)

def find_unit_by_name(name, ns_name=None):
    if ns_name is None and '.' in name:
        ns_name,name = name.split('.',1)
    if ns_name:
        try:
            namespace = unit_namespaces[ns_name]
        except KeyError:
            return None
        unit = namespace._units.get(name, None)
        if not isinstance(unit, BaseUnit):
            unit = None
    else:
        for namespace in sorted(unit_namespaces.itervalues(),
                                key=lambda ns: ns._precedence):
            unit = namespace._units.get(name, None)
            if isinstance(unit, BaseUnit):
                break
    if unit is not None:
        return unit
    if len(name) == 1:
        return None
    if unit is None:
        if name.endswith('s'):
            unit = get_unit_by_name(name[:-1:], ns_name)
        if unit is None and name != name.lower():
            unit = get_unit_by_name(name.lower(), ns_name)
    if unit is None:
        for prefix in vars(prefixes).itervalues():
            if not isinstance(prefix, Prefix):
                continue
            for pre in [prefix.name, prefix.abbrev]:
                if pre and name.startswith(pre):
                    unit = get_unit_by_name(name[len(pre):], ns_name)
                    if unit is not None:
                        unit = unit * 10 ** prefix.power
                if unit is not None:
                    break
            if unit is not None:
                break
    return unit

unit_scanner = Scanner([
    (r'\s+',  lambda s, b: ('space',b)),
    (r'-?\d+',  lambda s, b: ('digits', int(b))),
    (r'[\^*/]', lambda s, b: ('operator', b)),
    (r'\(', lambda s, b: ('openp', None)),
    (r'\)', lambda s, b: ('closep', None)),
    (r'[a-zA-Z_.]+', lambda s, b: ('ident', b))])

class UnitSyntaxError(Exception):
    pass

class UnitStringParser(object):

    def __init__(self, bytes):
        self.output_queue = []
        self.operator_stack = []
        self.last_was_operand = False
        self.bytes = bytes

    def syntax_error(self, msg='poorly formatted', *args):
        if args:
            msg %= args
        raise UnitSyntaxError('error parsing %r: %s' % (self.bytes, msg))

    def parse(self):
        tokens, extra = unit_scanner.scan(self.bytes)
        l_extra = ''
        itr_tokens = iter(tokens)
        for tp,value in itr_tokens:
            if tp == 'space':
                continue
            if tp == 'ident':
                if self.last_was_operand:
                    l_extra = ' ' + value
                    break
                unit = get_unit_by_name(value)
                if not unit:
                    self.syntax_error('unkown unit %r',value)
                self.output_queue.append(unit)
                self.last_was_operand = True
            else:
                if tp == 'digits':
                    if self.last_was_operand:
                        self.process_operator('^')
                    self.output_queue.append(value)
                    self.last_was_operand = True
                elif tp == 'operator':
                    if not self.last_was_operand:
                        raise self.syntax_error('operator without operand')
                    self.process_operator(value)
                    self.last_was_operand = False
                elif tp == 'openp':
                    self.operator_stack.append('(')
                    self.last_was_operand = False
                elif tp == 'closep':
                    while True:
                        if not self.operator_stack:
                            self.syntax_error('unmatched right paranthesis')
                        if self.operator_stack[-1] == '(':
                            self.operator_stack.pop()
                            break
                        self.pop_operator()
                    self.last_was_operand = True
                else:
                    raise RuntimeError('unhandled tp %r' % (tp,))
        while self.operator_stack:
            self.pop_operator()
        if len(self.output_queue) > 1:
            self.syntax_error()
        return self.output_queue[0], l_extra + ''.join(b for t,b in itr_tokens)  + extra

    precedence_map = {'*': 2,
                      '/': 2,
                      '^': 1}
    def process_operator(self, operator):
        while (self.operator_stack and
               self.operator_stack[-1] != '(' and
               self.precedence_map[self.operator_stack[-1]] <=
               self.precedence_map[operator]):
            self.pop_operator()
        self.operator_stack.append(operator)

    def pop_operator(self):
        try:
            op = self.operator_stack.pop()
            rop = self.output_queue.pop()
            lop = self.output_queue.pop()
        except IndexError:
            self.syntax_error()
        if op == '*':
            v = lop*rop
        elif op == '/':
            v = lop/rop
        elif op == '^':
            v = lop**rop
        else:
            raise RuntimeError('unandled operator %s' % (op,))
        self.output_queue.append(v)

def ex_parse_unit(bytes):
    return UnitStringParser(bytes).parse()

def parse_unit(bytes):
    parser = UnitStringParser(bytes)
    unit,extra = parser.parse()
    if extra:
        parser.syntax_error('extra input %r', extra)
    return unit


class metric(unit_namespace):
    def name(unit, name, power):
        return CompoundUnit.register_name(unit.cannonicalized(), name, power)
    #length
    m = PrimitiveUnit(D.length, 'meter', 'm')
    km = 1e3 * m
    cm = 1e-2 * m
    mm = 1e-3 * m
    mcm = 1e-6 * m
    nm = 1e-9 *m
    #mass
    g = PrimitiveUnit(D.mass, 'gram', 'g')
    kg = 1e3 * g
    mg = 1e-3 * g
    mcg = 1e-6 * g
    ng = 1e-9 * g
    #time
    s = s = PrimitiveUnit(D.time, 'second', 's')
    #area
    m2 = m**2
    cm2 = cm**2
    #volume
    L = PrimitiveUnit(D.volume, 'liter', 'L')
    mL = 1e-3 * L
    mcL = 1e-6 * L
    nL = 1e-9 * L
    cc = cm3 = cm**3
    m3 = m**3
    #velocity
    m_s = m/s
    #acceleration
    m_s2 = m_s/s
    #force
    N = name(kg*m_s2, 'newton', 'N')
    kN = 1e3*N
    mN = 1e-3*N
    nN = 1e-9*N
    #energy
    J = name(N*m, 'joule', 'J')
    MJ = 1e6*J
    kJ = 1e3*J
    mJ = 1e-3*J
    mcJ = 1e-6*J
    nJ = 1e-9*J
    #pressure
    Pa = name(N/m2, 'pascal', 'Pa')
    GPa = 1e9*Pa
    MPa = 1e6*Pa
    kPa = 1e3*Pa
    mPa = 1e-3*Pa
    mcPa = 1e-6*Pa
    nPa = 1e-9*Pa
    del name

class imperial(unit_namespace):
    #length
    in_ = PrimitiveUnit(D.length, 'inch', 'in')
    ft = PrimitiveUnit(D.length, 'foot', 'ft')
    yd = PrimitiveUnit(D.length, 'yard', 'yd')
    furlong = PrimitiveUnit(D.length, 'furlong')
    mile = PrimitiveUnit(D.length, 'mile', 'mi')
    #mass
    oz = PrimitiveUnit(D.mass, 'ounce', 'oz')
    lb = PrimitiveUnit(D.mass, 'pound', 'lb')
    st = PrimitiveUnit(D.mass, 'stone', 'st')
    ton = PrimitiveUnit(D.mass, 'ton', 't')
    #time
    s = metric.s
    #area
    in2 = in_**2
    ft2 = ft**2
    #volume
    cubic_inch = in_**3
    ft3 = ft**3
    floz = PrimitiveUnit(D.volume, 'fluid_ounce', 'floz')
    pt = PrimitiveUnit(D.volume, 'pint', 'pt')
    qt = PrimitiveUnit(D.volume, 'quart', 'qt')
    gal = PrimitiveUnit(D.volume, 'gallon', 'gal')
    #velocity
    #acceleration
    #force
    #energy
    #pressure
    psi = PrimitiveUnit(D.pressure, 'pounds_per_a_square_inch', 'psi')


class length(unit_namespace):
    m = metric.m
    km = metric.km
    cm = metric.cm
    mm = metric.mm
    mcm = metric.mcm
    nm = metric.nm

class masses(unit_namespace):
    g = metric.g
    kg = metric.kg
    mg = metric.mg
    mcg = metric.mcg
    ng = metric.ng
    oz = imperial.oz
    lb = imperial.lb
    st = imperial.st
    ton = imperial.ton

class times(unit_namespace):
    s = metric.s
    min_ = PrimitiveUnit(D.time, 'minute', 'min')
    hour = PrimitiveUnit(D.time, 'hour', 'hr')
    day = PrimitiveUnit(D.time, 'day')
    week = PrimitiveUnit(D.time, 'week')
    year = PrimitiveUnit(D.time, 'year')

class pressures(unit_namespace):
    Pa = metric.Pa
    GPa = metric.GPa
    MPa = metric.MPa
    kPa = metric.kPa
    mPa = metric.mPa
    mcPa = metric.mcPa
    nPa = metric.nPa
    psi = imperial.psi
    atm = PrimitiveUnit(D.pressure, 'atmosphere', 'atm')
    mmHg = PrimitiveUnit(D.pressure, 'millimeters_of_mercury', 'mmHg')
    torr = PrimitiveUnit(D.pressure, 'torr')
    bar = PrimitiveUnit(D.pressure, 'bar')

class quantities(unit_namespace):
    quantity = PrimitiveUnit(D.quantity, '', '')
    mol = PrimitiveUnit(D.quantity, 'mole', 'mol')
    kmol = 1e3*mol
    mmol = 1e-3*mol
    mcmol = 1e-6*mol
    nmol = 1e-9*mol

class mws(unit_namespace):
    g_mol = metric.g / quantities.mol

class concentrations(unit_namespace):
    M = mol_L = quantities.mol / metric.L
    CompoundUnit.register_name(M.cannonicalized(), 'molarity', 'M')
    kmol_L = 1e3*mol_L
    mmol_L = 1e-3*mol_L
    mcmol_L = 1e-6*mol_L
    nmol_L = 1e-9*mol_L

class liquid_volumes(unit_namespace):
    L = metric.L
    mL = metric.mL
    mcL = metric.mcL
    nL = metric.nL
    oz = imperial.floz
    pt = imperial.pt
    qt = imperial.qt
    gal = imperial.gal

class gas_volumes(unit_namespace):
    L = metric.L
    mL = metric.mL
    cc = metric.cc
    m3 = metric.m3
    in3 = imperial.in_.pow(3)
    ft3 = imperial.ft.pow(3)
    yd3 = imperial.yd.pow(3)

class temperatures(unit_namespace):
    K = PrimitiveUnit(D.temperature, 'kelvin', 'K')
    C = PrimitiveUnit(D.temperature, 'centigrade', 'C')
    F = PrimitiveUnit(D.temperature, 'fahrenheit', 'F')

#print metric.km * metric.kg / metric.s
#print (metric.km * metric.kg / metric.s).cannonicalized()
#print (metric.km * metric.kg / metric.s).get_dimensionality()
#print metric.N, metric.kN, metric.N.get_name(), metric.kN.get_name()

dimensionless = CompoundUnit()


# # # # # # # # # #
# Physical Number #
# # # # # # # # # #



class PhysicalNumber(A.DivAlgebraBase, AutoRepr):

    def __init__(self, quantity, unit=None, name=None):
        assert typep(quantity, lossless_number_type)
        self.quantity = quantity
        if unit is None:
            unit = dimensionless
        assert isinstance(unit, BaseUnit)
        self.unit = unit
        assert typep(name, name_type)
        self.name = name

    def repr_args(self):
        yield self.quantity
        if self.unit != dimensionless:
            yield self.unit
        if self.name is not None None:
            if self.unit == dimensionless:
                yield None
            yield self.name

    def __str__(self):
        parts = [self.quantity]
        if self.unit != dimensionless:
            parts.append(self.unit)
        if self.name is not None:
            parts.append(self.name)
        return ' '.join(parts)

    def __hash__(self):
        return hash(self.quantity) ^ hash(self.unit) ^ hash(self.name)

    def __eq__(self, other):
        if isinstance(other, PhysicalNumber):
            return (self.quantity == other.quantity and
                    self.unit == other.unit and
                    self.name == other.name)
        if self.unit != dimensionless or self.name is not None:
            return NotImplemented
        return self.quantity == other

    def split_posneg(self):
        unit = self.unit
        if isinstance(unit, PrimitiveUnit):
            unit = primunit_to_compound(unit)
        u_pos,u_neg = unit.split_posneg()
        if isinstance(self.quantity, Ratio):
            q_pos,q_neg = self.quantity.num, self.quantity.den
        else:
            q_pos,q_neg = self.quantity, 1
        return [self.__class__(q_pos, u_pos, self.name),
                self.__class__(q_neg, u_neg)]

as_physical_number = MultiMethod('as_physical_number')

@defmethod(as_physical_number, [PhysicalNumber])
def meth(pn):
    return pn

@defmethod(as_physical_number, [lossless_number_type])
def meth(n):
    return PhysicalNumber(n)

@defmethod(A.mm_neg, [PhysicalNumber])
def meth(p):
    return PhysicalNumber(-p.quantity, p.unit)

@defmethod(A.mm_pow, [PhysicalNumber, (int,long)])
def meth(p,pow):
    return PhysicalNumber((as_ratio(p.quantity)**pow
                             if pow < 0 and isinstance(p, (int,long)) else
                          p.quantity) ** pow,
                       p.unit**pow)

@defmethod(A.mm_mul, [PhysicalNumber, PhysicalNumber])
def meth(a, b):
    return PhysicalNumber(a.quantity * b.quantity, a.unit * b.unit)

@A.defboth_mm_mul([PhysicalNumber, anytype])
def meth(p, a):
    return PhysicalNumber(p.quantity * a, p.unit)

@defmethod(A.mm_div, [PhysicalNumber, PhysicalNumber])
def meth(a, b):
    return PhysicalNumber(a.quantity / b.quantity, a.unit / b.unit)

@defmethod(A.mm_div, [PhysicalNumber, anytype])
def meth(p, a):
    return PhysicalNumber(p.quantity / a, p.unit)

@defmethod(A.mm_div, [anytype, PhysicalNumber])
def meth(a, p):
    return PhysicalNumber(a / p.quantity, p.unit ** -1)

def add_unit_check(verb, a, b):
    if a.unit != b.unit:
        raise ValueError("cannot %s %s and %s; units are incompatible" % (verb,a,b))
    return a.unit

def add_dimensionless_check(p, other):
    if p.unit != dimensionless:
        raise ValueError("cannot add/subtract dimensionless %s and dimensional %s" % (other, p))
    return dimensionless

@defmethod(A.mm_add, [PhysicalNumber, PhysicalNumber])
def meth(a, b):
    return PhysicalNumber(a.quantity + b.quantity, add_unit_check('add', a, b))

@A.defboth_mm_add([PhysicalNumber, anytype])
def meth(p, a):
    return PhysicalNumber(p.quantity + a, add_dimensionless_check(p, a))

@defmethod(A.mm_sub, [PhysicalNumber, PhysicalNumber])
def meth(a, b):
    return PhysicalNumber(a.quantity - b.quantity, add_unit_check('sub', a, b))

@defmethod(A.mm_sub, [PhysicalNumber, anytype])
def meth(p, a):
    return PhysicalNumber(p.quantity - a, add_dimensionless_check(p, a))

@defmethod(A.mm_sub, [anytype, PhysicalNumber])
def meth(a, p):
    return PhysicalNumber(a - p.quantity, add_dimensionless_check(p, a))


def parse_physical_number(bytes, quantity_class=None):
    parts = bytes.strip().split(None, 1)
    number, rest = (parts if len(parts)==2 else (parts[0], ''))
    unit = parse_unit(rest) if rest else dimensionless
    if number.endswith('d'):
        number = number[:-1]
        quantity_class = quantity_class or Decimal
    if number.endswith('s'):
        number = number[:-1]
        quantity_class = quantity_class or SigFig
    if quantity_class is None:
        quantity_class = int if not re.find('[.eE]', number) else Decimal
    return PhysicalNumber(quantity_class(number), unit)

# # # # # # # #
# Unit Types  #
# # # # # # # #

class UnitDimensionalityType(atypes.TypeBase):
    '''Matches units of a specific specific dimensionality
    '''
    __slots__ = ['dimensionalities']
    def __init__(self, *dimensionalities):
        self.dimensionalities = OrderedSet(map(as_dimensionality, dimensionalities))

class UnitWithoutPrefixType(atypes.TypeBase):
    '''Matches units without their prefixes
    '''
    __slots__ = ['units']
    def __init__(self, *units):
        self.units = OrderedSet(as_unit(unit).without_prefix() for unit in units)

class PhysicalNumberInnerType(atypes.TypeBase):
    '''Matches inner matchers on the quantity and unit
       of a PhysicalNumber
    '''
    __slots__ = ['quantity_inner','unit_inner']
    def __init__(self, quantity_inner=anytype, unit_inner=anytype):
        self.quantity_inner = atypes.as_type(quantity_inner)
        self.unit_inner = atypes.as_type(unit_inner)

# Type Methods
@defmethod(as_string, [UnitDimensionalityType])
def meth(op):
    return '(unit_dimensionality %s)' % ' '.join(map(str, op.dimensionalities))
@defmethod(as_string, [UnitWithoutPrefixType])
def meth(op):
    return '(unit_without_prefix %s)' % ' '.join(map(str, op.units))
@defmethod(as_string, [PhysicalNumberInnerType])
def meth(op):
    return '(physical_number_inner_type %s %s)' % (op.quantity_inner, op.unit_inner)

@defmethod(atypes.eq_types, [UnitDimensionalityType, UnitDimensionalityType])
def meth(a, b):
    return set(a.dimensionalities) == set(b.dimensionalities)
@defmethod(atypes.eq_types, [UnitWithoutPrefixType, UnitWithoutPrefixType])
def meth(a, b):
    return set(a.units) == set(b.units)
@defmethod(atypes.eq_types, [PhysicalNumberInnerType, PhysicalNumberInnerType])
def meth(a, b):
    return a.quantity_inner == b.quantity_inner and a.unit_inner == b.unit_inner

@defmethod(atypes.hash_type, [UnitDimensionalityType])
def meth(op):
    return hash(frozenset(op.dimensionalities)) ^ 7832932
@defmethod(atypes.hash_type, [UnitWithoutPrefixType])
def meth(op):
    return hash(frozenset(op.units)) ^ 324235332
@defmethod(atypes.hash_type, [PhysicalNumberInnerType])
def meth(op):
    return hash(op.quantity_inner) ^ hash(op.unit_inner) ^ 83412734

#reductions and optimizations
@atypes.defintersectionreduce([UnitDimensionalityType, UnitDimensionalityType])
def meth(a,b):
    d = a.dimensionalities & b.dimensionalities
    if not d:
        return atypes.notanytype
    return UnitDimensionalityType(d)
@atypes.defunionreduce([UnitDimensionalityType, UnitDimensionalityType])
def meth(a,b):
    return UnitDimensionalityType(a.dimensionalities | b.dimensionalities)
@defmethod(atypes.optimize_type, [UnitDimensionalityType])
def meth(ud):
    if not ud.dimensionalities:
        return atypes.notanytype
    return ud

@atypes.defintersectionreduce([UnitWithoutPrefixType, UnitWithoutPrefixType])
def meth(a,b):
    u = a.units & b.units
    if not u:
        return atypes.notanytype
    return UnitWithoutPrefixType(u)
@atypes.defunionreduce([UnitWithoutPrefixType, UnitWithoutPrefixType])
def meth(a,b):
    return UnitWithoutPrefixType(a.units | b.units)
@defmethod(atypes.optimize_type, [UnitWithoutPrefixType])
def meth(uw):
    if not uw.units:
        return atypes.notanytype
    return uw

@defmethod(atypes.optimize_type, [PhysicalNumberInnerType])
def meth(pn):
    return PhysicalNumberInnerType(atypes.optimize_type(pn.quantity_inner),
                                   atypes.optimize_type(pn.unit_inner))

# typep
@defmethod(typep, [object, UnitDimensionalityType])
def meth(op, ud):
    return isinstance(op, BaseUnit) and op.get_dimensionality() in ud.dimensionalities
@defmethod(typep, [object, UnitWithoutPrefixType])
def meth(op, uw):
    return isinstance(op, BaseUnit) and op.without_prefix() in uw.units
@defmethod(atypes.typep, [object, PhysicalNumberInnerType])
def meth(op, pn):
    return (isinstance(op, PhysicalNumber) and
            typep(pn.quantity, op.quantity_inner) and
            typep(pn.unit, op.unit_inner))

# Type Keyers

class UnitDimensionalityKeyer(atypes.KeyerBase):
    pass
class UnitWithoutPrefixKeyer(atypes.KeyerBase):
    pass
class PhysicalNumberInnerKeyer(atypes.KeyerBase):
    def __init__(self, quantity_keyer, unit_keyer):
        self.quantity_keyer = quantity_keyer
        self.unit_keyer = unit_keyer
    def __eq__(self, other):
        if not isinstance(other, PhysicalNumberInnerKeyer):
            return NotImplemented
        return (self.quantity_keyer == other.quantity_keyer and
                self.unit_keyer == other.unit_keyer)
    def __hash__(self):
        return hash(self.quantity_keyer) ^ hash(self.unit_keyer) ^ 348923432

@defmethod(atypes.get_type_keyer, [UnitDimensionalityType])
def meth(op):
    return UnitDimensionalityKeyer()
@defmethod(atypes.get_type_keyer, [UnitWithoutPrefixKeyer])
def meth(op):
    return UnitWithoutPrefixKeyer()
@defmethod(atypes.get_type_keyer, [PhysicalNumberInnerType])
def meth(op):
    return PhysicalNumberInnerKeyer(atypes.get_type_keyer(op.quantity_inner),
                                    atypes.get_type_keyer(op.unit_inner))

def unit_dimensionality_keyer_func(op):
    if not isinstance(op, BaseUnit):
        return None
    return op.get_dimensionality()
@defmethod(atypes.keyer_getfunc, [UnitDimensionalityKeyer])
def meth(op):
    return unit_dimensionality_keyer_func

def unit_without_prefix_keyer(op):
    if not isinstance(op, BaseUnit):
        return None
    return op.without_prefix()
@defmethod(atypes.keyer_getfunc, [UnitWithoutPrefixKeyer])
def meth(op):
    return unit_without_prefix_keyer

def physical_number_inner_keyer(quantity_keyer, unit_keyer, op):
    if not isinstance(op, PhysicalNumber):
        return None
    return quantity_keyer(op.quantity), unit_keyer(op.unit)
@defmethod(atypes.keyer_getfunc, [PhysicalNumberInnerKeyer])
def meth(pnk):
    return partial(physical_number_inner_keyer, pnk.quantity_keyer, pnk.unit_keyer)

def unit_dimensionality_scorer(dimensionalities, dimensionality_key):
    return atypes.best_score if dimensionality_key in dimensionalities else atypes.no_score
@defmethod(atypes.get_key_scorer, [UnitDimensionalityType])
def meth(ud):
    return partial(unit_dimensionality_scorer, ud.dimensionalities)

def unit_withoutprefix_scorer(units, unit_key):
    return atypes.best_score if unit_key in units else atypes.no_score
@defmethod(atypes.get_key_scorer, [UnitWithoutPrefixKeyer])
def meth(uw):
    return partial(unit_withoutprefix_scorer, uw.units)

def physical_number_scorer(quantity_scorer, unit_scorer, pn_key):
    if pn_key is None:
        return atypes.no_score
    quantity_key, unit_key = pn_key
    quantity_score = quantity_scorer(quantity_key)
    unit_score = unit_scorer(unit_key)
    if atypes.no_score in [quantity_score, unit_score]:
        return atypes.no_score
    return int(round((quantity_score + unit_score) / 2.0))
@defmethod(atypes.get_key_scorer, [PhysicalNumberInnerType])
def meth(pn):
    return partial(physical_number_scorer,
                   atypes.get_key_scorer(pn.quantity_inner),
                   atypes.get_key_scorer(pn.unit_inner))

print typep(parse_unit('kg*m/s'), UnitDimensionalityType('force'))

xxx = MultiMethod('xxx')
@defmethod(xxx, [UnitDimensionalityType('force')])
def meth(op):
    print 'xxx', op


# # class UnitSet(object):

# #     def __init__(self, __seq=None, **kwds):
# #         self.dimensionalities = dict()
# #         self.update(__seq, **kwds)

# #     def update(self, __seq=None, **kwds):
# #         if __seq is not None:
# #             if hasattr(__seq, 'iteritems'):
# #                 __seq = __seq.iteritems()
# #             for dim,unit in __seq:
# #                 if isinstance(dim, str):
# #                     dim = Dimensionality.names[dim.lower()]
# #                 assert isinstance(dim, Dimensionality)
# #                 assert isinstance(unit, BaseUnit)
# #                 self.dimensionality[dim] = unit
# #         if kwds:
# #             self.update(kwds)

# #     def __getattr__(self, name):
# #         return self.dimensionalities[name]

# # mks = UnitSet(lenght=metric.m, mass=metric.kg, time=metric.s)
# # cgs = UnitSet(length=metric.cm, mass=metric.g, time=metric.s)



