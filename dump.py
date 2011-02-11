
from __future__ import with_statement

import sys
import pickle as pickle

from HH2.pathutils import FilePath

from hlab import xbrooke
import brooke.read
from brooke.image import make_image

path = FilePath(__file__).sibling('brooke-image.p')
b = xbrooke.get_builtins()
img = make_image(b)
print path
with file(path,'w') as fp:
    pickle.dump(img, fp)

