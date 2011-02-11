
import re

from hlab.pathutils import FilePath, DirPath

def treetrans(trans, indir, outdir, incfile=None, incdir=None):
    indir = DirPath(indir)
    outdir = DirPath(outdir)
    incfile = make_includer(incfile)
    incdir = make_includer(incdir)
    for child in indir:
        if child.isdir():
            if incdir(child):
                treetrans(trans, child, outdir.dchild(child.basename()), incfile, incdir)
        else:
            op = incfile(child)
            if op:
                assert isinstance(op, tuple)
                outdir.reqdir()
                outchild = outdir.child(child.basename())
                trans(*(op + (outchild,)))

def make_includer(op):
    if op is None:
        return lambda x : (x,)
    if isinstance(op, str):
        r = re.compile(op)
        def func(path):
            m = r.match(path)
            if not m:
                return ()
            return (path,) + m.groups()
        return func
    if callable(op):
        def wrap(path):
            x = op(path)
            if not x:
                return ()
            if x==True:
                return (path,)
            return x if isinstance(x, tuple) else (x,)
        return wrap
    assert 0

