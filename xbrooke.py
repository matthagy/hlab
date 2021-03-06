'''eXperimental brooke support
'''

import sys
from HH2.pathutils import FilePath

import brooke
import brooke.bompiler.load
from brooke.bompiler.context import CompilingContext

bootstrap_path = FilePath(brooke.__file__).parent().child('bootstrap/bootstrap.brk')
assert bootstrap_path.exists()

builtins = None
def get_builtins():
    global builtins, mod #XXX need to keep a reference to mod to prevents globals from dying
    if builtins is None:
        brooke.bompiler.load.defaultBuiltins.debug_enabled = False
        mod = brooke.bompiler.load.loadFile('bootstrap', bootstrap_path,
                   context=CompilingContext(optimize=False))
        builtins = mod.builtins
    return builtins

def load(name, filename, source, builtins=None, context=None):
    return brooke.bompiler.load.load(name, filename, source,
                                     builtins or get_builtins(), context)

def loadFile(name, filename, builtins=None, context=None):
    return load(name, filename, file(filename), builtins, context)

bmodules = {}
def bimport(name):
    assert '.' not in name
    try:
        return bmodules[name]
    except KeyError:
        bmodules[name] = mod = loadFile(name,
                                    FilePath(__file__).sibling(name+'.brk'))
        return mod

def main():
    import sys
    import os.path
    for fn in sys.argv[1:]:
        loadFile(os.path.basename(fn), fn)

__name__ == '__main__' and main()
