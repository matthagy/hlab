
from decimal import Decimal

from hlab.lexing import Lexer, LexicalError

class SigFig(object):

    def __init__(self, arg):
        if isinstance(arg, unicode):
            arg = str(arg)
        if isinstance(arg, str):
            arg = parse_string(arg)
        elif isinstance(arg, Decimal):
            arg = convert_decimal(arg)
        elif isinstance(arg, (int,long)):
            arg = convert_integer(arg)
        (self.sign,
         self.digs_pre_dot,
         self.dot,
         self.digs_post_dot,
         self.exp,
         self.exp_power
         ) = arg

    def __str__(self):
        base, exp = self.get_format_args()
        if exp is None:
            return base
        return '%se%s' % (base, exp)

    def get_format_args(self, max_exp_power=3):
        return self.as_scientific()._get_format_args(max_exp_power)

    def get_exp_format_args(self):
        return self.as_scientific()._get_exp_format_args()

    def _get_format_args(self, max_exp_power):
        if abs(self.exp_power) > max_exp_power or self.sigfigs <= self.exp_power:
            return self._get_exp_format_args()
        digs = map(str, self.digs_pre_dot + self.digs_post_dot)
        if self.exp_power >= 0:
            if not (1+self.exp_power==len(digs) and digs[-1] != '0'):
                digs.insert(1+self.exp_power, '.')
        else:
            digs = ['0.'] + ['0'] * (-1 - self.exp_power) + digs
        return ['%s%s' % ('-' if self.sign else '', ''.join(digs)),
                None]

    def _get_exp_format_args(self):
        return ['%s%d%s' % ('-' if self.sign else '',
                            (self.digs_pre_dot or [0])[0],
                            '.' + ''.join(map(str, self.digs_post_dot)) if
                            self.digs_post_dot else ''),
                self.exp_power]


    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__,
                           str(self))

    def as_decimal(self):
        x = self.as_scientific()
        return Decimal((x.sign,
                        x.digs_pre_dot + x.digs_post_dot,
                        x.exp_power - len(x.digs_post_dot)))

    @property
    def sigfigs(self):
        if list(set(self.digs_pre_dot) | set(self.digs_post_dot)) == [0]:
            return len(self.digs_pre_dot) + len(self.digs_post_dot)
        if self.dot:
            for i,d in enumerate(self.digs_pre_dot):
                if d!=0:
                    break
            else:
                i += 1
            return (len(self.digs_pre_dot)-i) + len(self.digs_post_dot)
        n = len(self.digs_pre_dot)
        while n and self.digs_pre_dot[n-1]==0:
            n -= 1
        return n

    @property
    def least_significant_place(self):
        if self.dot:
            return self.exp_power - len(self.digs_post_dot)
        n = len(self.digs_pre_dot)
        while n and self.digs_pre_dot[n-1]==0:
            n -= 1
        return len(self.digs_pre_dot) - n + self.exp_power

    def as_scientific(self):
        if len(self.digs_pre_dot)==1:
            return self
        first_dig = self.digs_pre_dot[:1]
        rest_pre = self.digs_pre_dot[1:]
        shift = len(rest_pre)
        if self.dot:
            return self.__class__((self.sign, first_dig, True,
                                   rest_pre + self.digs_post_dot, True,
                                   self.exp_power + shift))
        rest_pre = list(rest_pre)
        while rest_pre and rest_pre[-1] == 0:
            rest_pre.pop()
        return self.__class__((self.sign, first_dig, bool(rest_pre),
                               tuple(rest_pre), True,
                               self.exp_power + shift))

    def round_to_sigfigs(self, n):
        if n<1:
            raise ValueError("bad number of sigfigs to round to %r" % (n,))
        s = self.as_scientific()
        if list(s.digs_pre_dot) != [0]:
            n -= 1
        return self.round_scientific_to_index(n, s)

    def round_to_place(self, n):
        s = self.as_scientific()
        return self.round_scientific_to_index(max(0, -(n - s.exp_power)), s)

    def round_scientific_to_index(self, index, s=None):
        if s is None:
            s = self.as_scientific()
        if len(s.digs_post_dot) <= index:
            x = list(s.digs_post_dot)
            while len(x) < index:
                x.append(0)
            s.digs_post_dot = tuple(x)
            return s
        digs = list(s.digs_pre_dot + s.digs_post_dot[:index])
        round_dig = s.digs_post_dot[index]
        if round_dig >= 5:
            digs[-1] += 1
        for i in xrange(len(digs)-1, 0, -1):
            if digs[i]==10:
                digs[i] = 0
                digs[i-1] += 1
        if digs[0]==10:
            digs[0] = 0
            digs.insert(0, 1)
            s.exp_power += 1
        s.digs_pre_dot = (digs[0],)
        s.digs_post_dot = tuple(digs[1:])
        return s

    def __pos__(self):
        return self

    def __neg__(self):
        return self.__class__(int(not self.sign), self.digs_pre_dot, self.dot,
                              self.digs_post_dot, self.exp, self.exp_power)

    def __nonzero__(self):
        return self.as_decimal() != 0

    def perform_binary_operation(self, other, func, rule):
        selfd = self.as_decimal()
        otherd = other.as_decimal() if isinstance(other, SigFig) else other
        valued = func(selfd, otherd)
        value = self.__class__(valued)
        sigfigs = (min(self.sigfigs, other.sigfigs)
                   if isinstance(other, SigFig) else
                   self.sigfigs)
        if rule=='mul':
            value = value.round_to_sigfigs(min(self.sigfigs, other.sigfigs)
                                           if isinstance(other, SigFig) else
                                           self.sigfigs)
            if valued == 0:
                value.digs_pre_dot = ()
        elif rule=='add':
            lsp = (max(self.least_significant_place, other.least_significant_place)
                   if isinstance(other, SigFig) else
                   self.least_significant_place)
            value = value.round_to_place(lsp)
            if value.digs_pre_dot == (0,) and value.digs_post_dot == ():
                value.digs_post_dot = (0,) * max(0, -lsp)
                value.digs_pre_dot = (0,) * max(0, sigfigs-max(0, -lsp))
        else:
            raise ValueError("bad sigfig rule %r" % (rule,))
        return value

    def __mul__(self, other):
        return self.perform_binary_operation(other, lambda a,b: a*b, 'mul')
    def __rmul__(self, other):
        return self.perform_binary_operation(other, lambda a,b: b*a, 'mul')
    def __div__(self, other):
        return self.perform_binary_operation(other, lambda a,b: a/b, 'mul')
    def __rdiv__(self, other):
        return self.perform_binary_operation(other, lambda a,b: b/a, 'mul')
    __truediv__ = __div__
    __rtruediv__ = __rdiv__
    def __add__(self, other):
        return self.perform_binary_operation(other, lambda a,b: a+b, 'add')
    def __radd__(self, other):
        return self.perform_binary_operation(other, lambda a,b: b+a, 'add')
    def __sub__(self, other):
        return self.perform_binary_operation(other, lambda a,b: a-b, 'add')
    def __rsub__(self, other):
        return self.perform_binary_operation(other, lambda a,b: b-a, 'add')

    def __gt__(self, other):
        return (self-other).as_decimal() > 0
    def __ge__(self, other):
        return (self-other).as_decimal() >= 0
    def __eq__(self, other):
        try:
            return (self-other).as_decimal() == 0
        except TypeError:
            return False
    def __ne__(self, other):
        return (self-other).as_decimal() != 0
    def __le__(self, other):
        return (self-other).as_decimal() <= 0
    def __lt__(self, other):
        return (self-other).as_decimal() < 0


def parse_string(bytes):
    lex = Lexer(bytes.strip())
    [pm, digs_pre_dot, dot, digs_post_dot, exp, exp_power
     ] = lex.pulls(r'[+-]', r'\d+', r'\.', r'\d+', r'[eE]', r'[+-]?\d+')
    if not lex.eof or (exp and not exp_power):
        raise LexicalError("bad sigfig literal %r" % (bytes,))
    if not digs_pre_dot:
        digs_pre_dot = '0'
    sign = 1 if pm == '-' else 0
    digs_pre_dot = tuple(map(int, digs_pre_dot))
    digs_post_dot = tuple(map(int, digs_post_dot))
    dot = bool(dot)
    exp = bool(exp)
    exp_power = int(exp_power) if exp_power else 0
    return (sign, digs_pre_dot, dot, digs_post_dot, exp, exp_power)

def convert_decimal(d):
    sign, digits, exp = d.as_tuple()
    return (sign, digits, True, (), True, exp)

def convert_integer(i):
    #cheap trick, but works
    return parse_string(str(i))
