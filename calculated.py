
from __future__ import division


def calculated(func):
    func.calculate_attribute = True
    return func


class CalculatedAttributes(object):
    '''Base class for classes that have attributes that are calculated
       on demand and then cached for future access
    '''

    class __metaclass__(type):

        def __new__(cls, name, bases, dct):
            dct = dict(dct)
            calculated_attributes = {}
            for base in reversed(bases):
                try:
                    c = base.calculated_attributes
                except AttributeError:
                    pass
                else:
                    calculated_attributes.update(c)
            for n,v in dct.items():
                try:
                   is_ca = v.calculate_attribute
                except AttributeError:
                    pass
                else:
                   if is_ca:
                       calculated_attributes[n] = v
                       del dct[n]
            dct['calculated_attributes'] = calculated_attributes
            return type.__new__(cls, name, bases, dct)

    def __init__(self, dct=None):
        if dct:
           vars(self).update(dct)

    def __getattr__(self, name):
        try:
            c = self.calculated_attributes[name]
        except KeyError:
            raise AttributeError('%r object has no attribute %r'
                                   % (self.__class__.__name__, name))
        v = c(self)
        setattr(self,name,v)
        return v

    def __getitem__(self, name):
        return getattr(self,name)

    def clear_calculated_attributes(self, recursive=False):
        d = vars(self)
        for name in self.calculated_attributes:
            try:
                op = d.pop(name)
            except KeyError:
                pass
            else:
                if recursive:
                    try:
                        clear_calculated_attributes = op.clear_calculated_attributes
                    except AttributeError:
                        pass
                    else:
                        clear_calculated_attributes()


class PickelingBase(CalculatedAttributes):
    '''only pickles attributes named in __init__ method and then
       calls __init__ as the unpickler
    '''

    class __metaclass__(CalculatedAttributes.__metaclass__):

        def __init__(cls, name, bases, dict):
            CalculatedAttributes.__metaclass__.__init__(cls, name, bases, dict)
            c = cls.__init__.func_code
            args = c.co_varnames[1:c.co_argcount]
            cls.pickeling_hash = hash(args)
            cls.pickeling_attributes = ('pickeling_hash',) + args

    def __getstate__(self):
        return [getattr(self, name)
                for name in self.pickeling_attributes]

    def __setstate__(self, state):
        old_hash = state[0]
        #if old_hash != self.pickeling_hash:
        #    raise RuntimeError("bad pickeling_hash in stale data")
        self.__init__(*state[1:])
