'''Unique salt for each machine to ensure stochastic processes
   simulated on different computer clusters are not identical
'''

from __future__ import absolute_import

import os

from .pathutils import FilePath

salt_filepath = FilePath(os.path.expanduser('~/var/hlab-salt'))

def get_salt():
    if not salt_filepath.exists():
        raise RuntimeError("salt path doesn't exist on current machine")
    with open(salt_filepath) as fp:
        salt = fp.read().strip()
    if not len(salt):
        raise RuntimeError("empty salt")
    return salt

