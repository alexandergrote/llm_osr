from collections.abc import Mapping


class LazyDict(Mapping):
    def __init__(self, *args, **kw):
        self._raw_dict = dict(*args, **kw)

    def __getitem__(self, key):

        value = self._raw_dict[key]

        if isinstance(value, tuple):
            func, kwargs = value
            return func(**kwargs)
        
        return value

    def __iter__(self):
        return iter(self._raw_dict)

    def __len__(self):
        return len(self._raw_dict)