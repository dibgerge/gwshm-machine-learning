"""
This is a simple wrapper for the HDF5 library interfaced with the Pandas Library. It allows 
indexing the HD5 using user-defined objects, rather than the plain strings currently required.
"""
import pandas as pd


class Info:
    items_delim = '/'
    key_val_delim = '_'

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __repr__(self):
        out = 'Info('
        for k, v in self.__dict__.items():
            out += k + '=' + str(v) + ', '
        return out[:-2] + ')'

    @classmethod
    def from_str(cls, s):
        """
        Currently, conversion from string infers the type of value, and guesses if it is either 
        an integer or a string.
        """
        kwds = {}
        items = s.split(cls.items_delim)
        for item in items:
            parts = item.split(cls.key_val_delim)
            if len(parts) < 2:
                continue
            try:
                kwds[parts[0]] = int(parts[1])
            except ValueError:
                kwds[parts[0]] = parts[1]
        return cls(**kwds)

    def to_str(self):
        if not self.__dict__:
            raise ValueError('No user defined fields in %s.' % self.__class__)
        out_str = ''
        for key in sorted(self.__dict__):
            out_str += str(key) + self.key_val_delim + str(self.__dict__[key]) + self.items_delim
        return out_str

    def modify(self, **kwargs):
        attr = self.__dict__.keys()
        for key, value in kwargs.items():
            if key in attr:
                setattr(self, key, value)
            else:
                raise ValueError('%s not in %s.' % (key, self.__class__))

    def add(self, key, value):
        """ Add an attribute"""
        self[key] = value
        return self


def indextype(access_type=Info):
    if type(access_type) != type(Info):
        raise ValueError('type should be a derived class of Info.')

    def decorate(cls):
        cls_getitem = cls.__getitem__
        cls_setitem = cls.__setitem__
        cls_delitem = cls.__delitem__
        cls_contains = cls.__contains__
        cls_iter = cls.__iter__
        cls_keys = cls.keys

        def __getitem__(self, key):
            return cls_getitem(self, key.to_str())

        def __setitem__(self, key, value):
            return cls_setitem(self, key.to_str(), value)

        def __delitem__(self, key):
            return cls_delitem(self, key.to_str())

        def __iter__(self):
            infos = [access_type.from_str(key) for key in cls_keys(self)]
            return iter(infos)

        def __contains__(self, key):
            return cls_contains(self, key.to_str())

        def keys(self):
            return [access_type.from_str(key) for key in cls_keys(self)]

        cls.__getitem__ = __getitem__
        cls.__setitem__ = __setitem__
        cls.__delitem__ = __delitem__
        cls.__contains__ = __contains__
        cls.__iter__ = __iter__
        cls.keys = keys
        return cls
    return decorate


@indextype(access_type=Info)
class SciHDF(pd.HDFStore):
    all_keys = None

    def keys_with(self, **kwargs):
        if self.all_keys is None:
            self.all_keys = self.keys()

        return [info for info in self.all_keys if all(info[k] == v for k, v in kwargs.items())]
