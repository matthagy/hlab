'''Base class for extracting information from data.
   Accounts for long run time with methods to notify user of progress.
'''

from __future__ import absolute_import

from .util import msg


class BaseExtractor(object):

    verbose = True
    info_rate = 100
    msg = staticmethod(msg)
    output_name = '<unnamed>'

    def __init__(self, verbose=None, info_rate=None, msg=None, output_name=None):
        if verbose is not None:
            self.verbose = verbose
        if info_rate is not None:
            self.info_rate = info_rate
        if msg is not None:
            self.msg = msg
        if output_name is not None:
            self.output_name = output_name

    def provide_info(self, inx, N, value):
        if self.verbose and inx%self.info_rate == 0:
            self.report_info(inx, N, value)

    def report_info(self, inx, N, value):
        self.msg('%d of %s in %s', inx+1,
                 N if N is not None else '<unkown>',
                 self.output_name)

