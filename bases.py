
try:
    from Gnuplot import Gnuplot
except ImportError:
    Gnuplot = None

class PlottingMixin(object):

    def plot(self, persist=True, **kwds):
        if Gnuplot is None:
            raise RuntimeError("Gnuplot not available")
        gp = Gnuplot.Gnuplot(persist=persist)
        gp.plot(self.gnuplot_item(**kwds))
        return gp

class AutoRepr(object):

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__,
                           ', '.join(map(repr, self.repr_args())))

    def __str__(self):
        return repr(self)


