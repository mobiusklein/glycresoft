
from functools import partial
from typing import Any, Callable, Optional, Union, MutableMapping
import six
from sqlalchemy import PickleType, Column
from sqlalchemy.orm import deferred
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.ext.mutable import Mutable

import dill
import pyzstd

from ms_deisotope.data_source._compression import starts_with_zstd_magic


class LazyMutableMappingWrapper(Mutable, MutableMapping[str, Any]):
    _payload: Optional[bytes]
    _unpickler: Callable[[bytes], MutableMapping[str, Any]]
    _object: MutableMapping[str, Any]

    def __init__(self, payload=None, unpickler=None, _object=None):
        if payload is None and _object is None:
            _object = PartiallySerializedMutableMapping()
        if payload is not None and not starts_with_zstd_magic(payload):
            payload = pyzstd.compress(payload)
        self._payload = payload
        self._unpickler = unpickler
        self._object = _object

    def _load(self):
        if self._object is not None:
            return self._object
        self._object = self._unpickler.loads(pyzstd.decompress(self._payload))
        self._payload = None
        return self._object

    def keys(self):
        return self._load().keys()

    def values(self):
        return self._load().values()

    def items(self):
        return self._load().items()

    def __len__(self):
        return len(self._load())

    def __contains__(self, key):
        return key in self._load()

    def __getitem__(self, key):
        return self._load()[key]

    def __iter__(self):
        return iter(self._load())

    def __setitem__(self, key, value):
        """Detect dictionary set events and emit change events."""
        self._load()[key] = value
        self.changed()

    def setdefault(self, key, value):
        result = self._load().setdefault(key, value)
        self.changed()
        return result

    def __delitem__(self, key):
        """Detect dictionary del events and emit change events."""
        self._load().__delitem__(key)
        self.changed()

    def update(self, *a, **kw):
        self._load().update(*a, **kw)
        self.changed()

    def pop(self, *arg):
        result = self._load().pop(*arg)
        self.changed()
        return result

    def popitem(self):
        result = self._load().popitem()
        self.changed()
        return result

    def clear(self):
        self._load().clear()
        self.changed()

    def __getstate__(self):
        return self._payload, self._unpickler, self._object

    def __setstate__(self, state):
        try:
            (payload, unpickler, _object) = state
        except Exception:
            payload = None
            unpickler = None
            _object = state
        self._payload = payload
        self._unpickler = unpickler
        self._object = PartiallySerializedMutableMapping()
        self.update(_object or {})

    def unload(self):
        for k, v in self._load().store.items():
            if hasattr(v, 'serialized'):
                if v.serialized:
                    v.serialize()

    @classmethod
    def coerce(cls, key, value):
        """Convert plain dictionary to instance of this class."""
        if not isinstance(value, cls):
            if isinstance(value, dict):
                return cls(None, None, PartiallySerializedMutableMapping(value))
            elif isinstance(value, PartiallySerializedMutableMapping):
                return cls(None, None, value)
            elif isinstance(value, (str, bytes)):
                return cls(value, dill.loads)
            return Mutable.coerce(key, value)
        else:
            return value

    def __repr__(self):
        return repr(self._load())

    def _ipython_key_completions_(self):
        return list(self.keys())


class MappingCell(object):
    __slots__ = ("value", "serialized")

    value: Union[bytes, Any]
    serialized: bool
    _dynamic_loading_threshold = 5e3

    def __init__(self, value, serialized=False):
        self.value = value
        self.serialized = serialized
        if serialized and isinstance(value, bytes) and not starts_with_zstd_magic(value):
            self.value = pyzstd.compress(value)

    def _view_value(self):
        if self.serialized and len(self.value) < self._dynamic_loading_threshold:
            self.deserialize()
        value = self.value if not self.serialized else "<BLOB>"
        return value

    def __repr__(self):
        template = "{self.__class__.__name__}({value}, serialized={self.serialized})"
        value = self._view_value()
        return template.format(self=self, value=value)

    def serialize(self):
        if not self.serialized:
            self.value = pyzstd.compress(dill.dumps(self.value, 2))
            self.serialized = True
        return self

    def deserialize(self):
        if self.serialized:
            if isinstance(self.value, six.text_type):
                self.value = self.value.encode('latin1')
            if starts_with_zstd_magic(self.value):
                self.value = pyzstd.decompress(self.value)
            self.value = dill.loads(self.value)
            self.serialized = False
        return self

    def copy(self):
        return self.__class__(self.value, self.serialized)

    def __reduce__(self):
        if self.serialized:
            return self.__class__, (self.value, self.serialized)
        else:
            return self.copy().serialize().__reduce__()


class PartiallySerializedMutableMapping(MutableMapping[str, MappingCell]):
    def __init__(self, arg=None, **kwargs):
        self.store = {}
        self.update(arg, **kwargs)

    def __getitem__(self, key):
        cell = self.store[key]
        if cell.serialized:
            cell.deserialize()
        return cell.value

    def __setitem__(self, key, value):
        self.store[key] = MappingCell(value, False)

    def __delitem__(self, key):
        del self.store[key]

    def __len__(self):
        return len(self.store)

    def __iter__(self):
        return iter(self.store)

    def __repr__(self):
        rep = {}
        for key, value in self.store.items():
            rep[key] = value._view_value()
        return repr(rep)

    def _ipython_key_completions_(self):
        return list(self.keys())

    def update(self, arg=None, **kwargs):
        if arg:
            if isinstance(arg, PartiallySerializedMutableMapping):
                self.store.update(arg.store)
            else:
                for key, value in arg.items():
                    if isinstance(value, MappingCell):
                        self.store[key] = value
                    else:
                        self.store[key] = MappingCell(value)
        for key, value in kwargs.items():
            if isinstance(value, MappingCell):
                self.store[key] = value
            else:
                self.store[key] = MappingCell(value)

    def unload(self):
        for k, v in self.items():
            if hasattr(v, "serialized"):
                if v.serialized:
                    v.serialize()


class _crossversion_dill(object):

    @staticmethod
    def loads(string, *args, **kwargs):
        try:
            return dill.loads(string, *args, **kwargs)
        except UnicodeDecodeError:
            if six.PY3:
                return dill.loads(string, encoding='latin1')
            else:
                raise

    @staticmethod
    def dumps(obj, *args, **kwargs):
        return dill.dumps(obj, *args, **kwargs)


DillType = partial(PickleType, pickler=_crossversion_dill)


class HasParameters(object):
    def __init__(self, **kwargs):
        self._init_parameters(kwargs)
        super(HasParameters, self).__init__(**kwargs)

    def _init_parameters(self, kwargs):
        kwargs.setdefault('parameters', PartiallySerializedMutableMapping())

    @declared_attr
    def parameters(self) -> LazyMutableMappingWrapper:
        return deferred(Column(LazyMutableMappingWrapper.as_mutable(DillType), default=LazyMutableMappingWrapper))
