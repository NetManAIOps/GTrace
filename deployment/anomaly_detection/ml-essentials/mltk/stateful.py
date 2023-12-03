import pickle as pkl
from collections import defaultdict
from typing import *

import numpy as np

__all__ = [
    'StatefulObject', 'SimpleStatefulObject', 'StatefulObjectGroup',
    'StateSaver',
]


class StatefulObject(object):
    """
    Base class for objects that persist state via state dict.

    For example::

        class YourObject(StatefulObject):

            def __init__(self, value):
                self.value = value

            def get_state_dict(self) -> StateDict:
                return {'value': self.value}

            def set_state_dict(self, state):
                self.value = state['value']
    """

    def get_state_dict(self) -> Dict[str, Any]:
        raise NotImplementedError()

    def set_state_dict(self, state: Dict[str, Any]):
        raise NotImplementedError()


class SimpleStatefulObject(StatefulObject):
    """
    Simple :class:`StatefulObject` that uses its `__dict__` as state dict.

    >>> obj = SimpleStatefulObject()
    >>> obj.value = 123
    >>> obj.get_state_dict()
    {'value': 123}
    >>> obj.set_state_dict({'value': 456, 'value2': 789})
    >>> obj.value, obj.value2
    (456, 789)
    """

    def get_state_dict(self) -> Dict[str, Any]:
        return self.__dict__

    def set_state_dict(self, state: Dict[str, Any]):
        self.__dict__.update(state)


class StatefulObjectGroup(StatefulObject, Mapping[str, StatefulObject]):
    """
    An object group contains multiple :class:`StatefulObject`, each with a
    dedicated prefix key, such that all state from these objects can be
    gathered into one state dict.

    >>> a = SimpleStatefulObject()
    >>> a.value = 123
    >>> b = SimpleStatefulObject()
    >>> b.value = 456
    >>> obj = StatefulObjectGroup({'a': a, 'b': b})

    >>> len(obj)
    2
    >>> list(obj)
    ['a', 'b']
    >>> obj['a'] is a
    True
    >>> obj['b'] is b
    True

    >>> obj.get_state_dict()
    {'a.value': 123, 'b.value': 456}
    >>> obj.set_state_dict({'a.value': 1230, 'a.value2': 1231,
    ...                     'b.value': 4560, 'b.value2': 4561})
    >>> a.value, a.value2
    (1230, 1231)
    >>> b.value, b.value2
    (4560, 4561)
    """

    def __init__(self,
                 objects: Dict[str, StatefulObject],
                 strict: bool = False):
        """
        Construct a new :class:`StatefulObjectGroup`.

        Args:
            objects: The dict of prefix key and corresponding stateful object.
                The prefix keys must not be empty, and not contain ".".
            strict: In :meth:`set_state_dict()`, whether or not to raise
                error if unexpected key appears. (default :obj:`True`)
        """
        for key, value in objects.items():
            if not isinstance(key, str) or not key or '.' in key:
                raise ValueError(f'invalid key {key!r}: must be a non-empty '
                                 f'str without "."')
            if not isinstance(value, StatefulObject):
                raise TypeError(f'item {key!r} is not a StatefulObject: '
                                f'{value!r}')

        self._objects = dict(objects)  # type: Dict[str, StatefulObject]
        self._strict = bool(strict)

    def __getitem__(self, k: str) -> StatefulObject:
        return self._objects[k]

    def __len__(self) -> int:
        return len(self._objects)

    def __iter__(self) -> Iterator[str]:
        return iter(self._objects)

    @property
    def strict(self) -> bool:
        """
        In :meth:`set_state_dict()`, whether or not to raise error if
        unexpected key appears.
        """
        return self._strict

    def add_object(self, prefix: str, obj: StatefulObject):
        """
        Add an object to this group.

        Args:
            prefix: The prefix key of this object.
            obj: The object.

        Raises:
            ValueError: If `prefix` already exists.
        """
        if prefix in self._objects:
            raise ValueError(f'`prefix` already exists: {prefix!r}')
        if not isinstance(obj, StatefulObject):
            raise TypeError(f'`obj` is not a StatefulObject: {obj!r}')
        self._objects[prefix] = obj

    def get_state_dict(self) -> Dict[str, Any]:
        """Get the state dict."""
        ret = {}
        for prefix, obj in self._objects.items():
            for key, value in obj.get_state_dict().items():
                ret[f'{prefix}.{key}'] = value
        return ret

    def set_state_dict(self, state: Dict[str, Any]):
        """
        Set the state dict.

        Args:
            state: The state dict.
        """
        decomposed = defaultdict(lambda: {})

        for prefix_key, value in state.items():
            if '.' not in prefix_key:
                raise ValueError(f'invalid state key {prefix_key!r}: no "."')
            prefix, key = prefix_key.split('.', 1)
            if prefix not in self._objects:
                if self.strict:
                    raise ValueError(f'invalid state key {prefix_key!r}: '
                                     f'does not correspond to any object')
            else:
                decomposed[prefix][key] = value

        for prefix in self._objects:
            if prefix not in decomposed:
                raise ValueError(f'state for object {prefix!r} is missing')

        for prefix, obj_state in decomposed.items():
            self._objects[prefix].set_state_dict(obj_state)


class StateSaver(object):
    """
    Class to save/load :class:`StatefulObject` to/from disk file.

    This class internally uses :func:`np.savez` to save the state dict.
    If a value of the state dict is a NumPy array, it will be directly
    saved; otherwise it will be serialized via :mod:`pickle` before saving.

    Usage::

        # save the state
        a = SimpleStatefulObject()
        a.value = (123, 'hello')
        b = SimpleStatefulObject()
        b.value = np.arange(5)

        saver = StateSaver({'a': a, 'b': b})
        saver.save('state.npz')

        # load the state
        a = SimpleStatefulObject()
        b = SimpleStatefulObject()

        saver = StateSaver({'a': a, 'b': b})
        saver.load('state.npz')

        assert(a.value == (123, 'hello'))
        np.testing.assert_equal(b.value, np.arange(5))
    """

    def __init__(self,
                 object_or_objects: Union['StatefulObjectGroup',
                                          Dict[str, 'StatefulObject']],
                 pickle_protocol=pkl.HIGHEST_PROTOCOL):
        """
        Construct a new :class:`StateSaver`.

        Args:
            object_or_objects: A :class:`StatefulObject`, or a dict of
                stateful objects
                (which can be grouped by :class:`StatefulObjectGroup`).
            pickle_protocol: The protocol for :mod:`pickle` to use.
                Default the highest possible protocol.
        """

        if isinstance(object_or_objects, StatefulObject):
            obj = object_or_objects
        else:
            obj = StatefulObjectGroup(object_or_objects)

        self._obj = obj
        self._pickle_protocol = pickle_protocol

    def save(self, file_path):
        """
        Save the state to disk file.

        Args:
            file_path: The file path.  Must end with ".npz".
        """
        if not file_path.endswith('.npz'):
            raise ValueError('`file_path` must end with ".npz"')

        npz_state = {}

        for key, value in self._obj.get_state_dict().items():
            if key.startswith('.'):
                raise ValueError(f'invalid state key {key!r}: "." cannot be '
                                 f'the first character')
            elif isinstance(value, np.ndarray):
                npz_state[key] = value
            else:
                npz_state[f'.{key}'] = np.frombuffer(
                    pkl.dumps(value, self._pickle_protocol), dtype=np.byte)

        np.savez(file_path, **npz_state)

    def load(self, file_path):
        """
        Load the state from disk file.

        Args:
            file_path: The file path.
        """
        state = {}
        for key, value in np.load(file_path).items():
            if key.startswith('.'):
                state[key[1:]] = pkl.loads(value.tobytes())
            else:
                state[key] = value
        self._obj.set_state_dict(state)
