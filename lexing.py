
import re

class LexicalError(Exception):
    pass

class Lexer(object):

    def __init__(self, bytes):
        self.bytes = bytes

    @property
    def eof(self):
        return not self.bytes

    def match(self, pattern):
        return re.match(pattern, self.bytes)

    def pull(self, pattern):
        m = self.match(pattern)
        if not m:
            return ''
        e = m.end()
        val = self.bytes[:e]
        self.bytes = self.bytes[e:]
        return val

    def pulls(self, *patterns):
        return [self.pull(pattern) for pattern in patterns]

