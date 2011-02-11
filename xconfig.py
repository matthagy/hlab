
import re

class Config(dict):

    def __init__(self, *args, **kwds):
        self.update(*args, **kwds)

    def update(self, *args, **kwds):
        if args:
            default, = args
            try:
                default = default.iteritems()
            except AttributeError:
                pass
            for k,v in default:
                self.set(k, v)
        if kwds:
            self.update(kwds)

    def set(self, k, v):
        if isinstance(v, dict) and not isinstance(v, Config):
            v = Config(v)
        config, key = self.get_attr_config(k)
        dict.__setitem__(config, key, v)

    def __setitem__(self, key, v):
        self.set(key, v)

    def get(self, k, *args):
        config, key = self.get_attr_config(k)
        return super(Config, self).get(key, *args)

    def __getitem__(self, k):
        config, key = self.get_attr_config(k)
        return dict.__getitem__(config, key)

    def get_attr_config(self, attr):
        path = self.split_path(attr)
        key = path.pop()
        config = self.get_config_by_path(*path)
        return config, key

    @staticmethod
    def split_path(attr):
         return re.split('[\._-]', attr)

    def get_config_by_path(self, *path):
        if not path:
            return self
        attr = path[0]
        try:
            config = dict.__getitem__(self, attr)
        except KeyError:
            config = Config()
            dict.__setitem__(self, attr, config)
        return config.get_config_by_path(*path[1:])
