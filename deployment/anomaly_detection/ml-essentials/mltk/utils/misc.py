import copy
import os
import threading
from contextlib import contextmanager
from typing import *

import numpy as np
from heapdict import heapdict

from ..typing_ import *

__all__ = [
    'Singleton', 'NOT_SET', 'ALL',
    'generate_random_seed',
    'optional_apply',  'validate_enum_arg', 'maybe_close', 'iter_files',
    'InheritanceDict', 'CachedInheritanceDict', 'parse_tags', 'deep_copy',
    'ContextStack', 'GeneratorIterator',
]


class Singleton(object):
    """
    Base class for singleton classes.

    >>> class Parent(Singleton):
    ...     pass

    >>> class Child(Parent):
    ...     pass

    >>> Parent() is Parent()
    True
    >>> Child() is Child()
    True
    >>> Parent() is not Child()
    True
    """

    __instances_dict = {}

    def __new__(cls, *args, **kwargs):
        if cls not in Singleton.__instances_dict:
            Singleton.__instances_dict[cls] = \
                object.__new__(cls, *args, **kwargs)
        return Singleton.__instances_dict[cls]


class NotSet(Singleton):
    """
    Class of the `NOT_SET` constant.

    >>> NOT_SET is not None
    True
    >>> bool(NOT_SET)
    False
    >>> 'empty' if not NOT_SET else 'not empty'
    'empty'
    >>> NOT_SET
    NOT_SET
    >>> NOT_SET == NOT_SET
    True
    >>> NotSet() is NOT_SET
    True
    >>> NotSet() == NOT_SET
    True
    """

    def __repr__(self):
        return 'NOT_SET'

    def __bool__(self):
        return False


NOT_SET = NotSet()


class AllConstant(Singleton):
    """
    Class of the `ALL` constant.

    >>> ALL
    ALL
    >>> ALL == ALL
    True
    >>> AllConstant() is ALL
    True
    >>> AllConstant() == ALL
    True
    """

    def __repr__(self):
        return 'ALL'


ALL = AllConstant()


def generate_random_seed():
    """
    Generate a new random seed from the default NumPy random state.

    Returns:
        int: The new random seed.
    """
    return np.random.randint(0xffffffff)


def optional_apply(f, value):
    """
    If `value` is not None, return `f(value)`, otherwise return None.

    >>> optional_apply(int, None) is None
    True
    >>> optional_apply(int, '123')
    123

    Args:
        f: The function to apply on `value`.
        value: The value, maybe None.
    """
    if value is not None:
        return f(value)


def validate_enum_arg(arg_name: str,
                      arg_value: Optional[TObject],
                      choices: Iterable[TObject],
                      nullable: bool = False) -> Optional[TObject]:
    """
    Validate the value of an enumeration argument.

    Args:
        arg_name: Name of the argument.
        arg_value: Value of the argument.
        choices: Valid choices of the argument value.
        nullable: Whether or not the argument can be None?

    Returns:
        The validated argument value.

    Raises:
        ValueError: If `arg_value` is not valid.
    """
    choices = tuple(choices)

    if not (nullable and arg_value is None) and (arg_value not in choices):
        raise ValueError('Invalid value for argument `{}`: expected to be one '
                         'of {!r}, but got {!r}.'.
                         format(arg_name, choices, arg_value))

    return arg_value


@contextmanager
def maybe_close(obj):
    """
    Enter a context, and if `obj` has ``.close()`` method, close it
    when exiting the context.

    >>> class HasClose(object):
    ...     def close(self):
    ...         print('closed')

    >>> class HasNotClose(object):
    ...     pass

    >>> with maybe_close(HasClose()) as obj:  # doctest: +ELLIPSIS
    ...     print(obj)
    <mltk.utils.misc.HasClose ...>
    closed

    >>> with maybe_close(HasNotClose()) as obj:  # doctest: +ELLIPSIS
    ...     print(obj)
    <mltk.utils.misc.HasNotClose ...>

    Args:
        obj: The object maybe to close.

    Yields:
        The specified `obj`.
    """
    try:
        yield obj
    finally:
        if hasattr(obj, 'close'):
            obj.close()


def iter_files(root_dir: str, sep: str = '/') -> Generator[str, None, None]:
    """
    Iterate through all files in `root_dir`, returning the relative paths
    of each file.  The sub-directories will not be yielded.

    Args:
        root_dir: The root directory, from which to iterate.
        sep: The separator for the relative paths.

    Yields:
        The relative paths of each file.
    """
    def f(parent_path, parent_name):
        for f_name in os.listdir(parent_path):
            f_child_path = parent_path + os.sep + f_name
            f_child_name = parent_name + sep + f_name
            if os.path.isdir(f_child_path):
                for s in f(f_child_path, f_child_name):
                    yield s
            else:
                yield f_child_name

    for name in os.listdir(root_dir):
        child_path = root_dir + os.sep + name
        if os.path.isdir(child_path):
            for x in f(child_path, name):
                yield x
        else:
            yield name


class _InheritanceNode(object):

    def __init__(self, type_: type):
        self.type = type_
        self.children = []

    def add_child(self, child: '_InheritanceNode'):
        self.children.append(child)


class InheritanceDict(Generic[TValue]):
    """
    A dict that gives the registered value of the closest known ancestor
    of a query type (`ancestor` includes the type itself).

    >>> class GrandPa(object): pass
    >>> class Parent(GrandPa): pass
    >>> class Child(Parent): pass
    >>> class Uncle(GrandPa): pass

    >>> d = InheritanceDict()
    >>> d[Child] = 1
    >>> d[GrandPa] = 2
    >>> d[Uncle] = 3
    >>> d[GrandPa]
    2
    >>> d[Parent]
    2
    >>> d[Child]
    1
    >>> d[Uncle]
    3
    >>> d[str]
    Traceback (most recent call last):
        ...
    KeyError: <class 'str'>
    """

    def __init__(self):
        self._nodes = []  # type: List[_InheritanceNode]
        self._values = {}
        self._topo_sorted = None

    def __setitem__(self, type_: type, value: TValue):
        this_node = _InheritanceNode(type_)
        if type_ not in self._values:
            for node in self._nodes:
                if issubclass(type_, node.type):
                    node.add_child(this_node)
                elif issubclass(node.type, type_):
                    this_node.add_child(node)
            self._nodes.append(this_node)
            self._topo_sorted = None
        self._values[type_] = value

    def __getitem__(self, type_: type) -> TValue:
        if self._topo_sorted is None:
            self._topo_sort()
        for t in reversed(self._topo_sorted):
            if t is type_ or issubclass(type_, t):
                return self._values[t]
        raise KeyError(type_)

    def _topo_sort(self):
        parent_count = {node: 0 for node in self._nodes}
        for node in self._nodes:
            for child in node.children:
                parent_count[child] += 1

        heap = heapdict()
        for node, pa_count in parent_count.items():
            heap[node] = pa_count

        topo_sorted = []
        while heap:
            node, priority = heap.popitem()
            topo_sorted.append(node.type)
            for child in node.children:
                heap[child] -= 1

        self._topo_sorted = topo_sorted


class CachedInheritanceDict(InheritanceDict[TValue]):
    """
    A subclass of :class:`InheritanceDict`, with an additional lookup cache.

    The cache is infinitely large, thus this class is only suitable under the
    situation where the number of queried types are not too large.
    """

    NOT_EXIST = ...

    def __init__(self):
        super().__init__()
        self._cache = {}  # type: Dict[type, TValue]

    def _topo_sort(self):
        self._cache.clear()
        super()._topo_sort()

    def __getitem__(self, type_: type) -> TValue:
        ret = self._cache.get(type_, None)
        if ret is None:
            try:
                ret = self._cache[type_] = super().__getitem__(type_)
            except KeyError:
                self._cache[type_] = self.NOT_EXIST
                raise
        elif ret is self.NOT_EXIST:
            raise KeyError(type_)
        return ret

    def __setitem__(self, type_: type, value: TValue):
        self._cache.clear()
        super().__setitem__(type_, value)


def parse_tags(s: str) -> List[str]:
    """
    Parse comma separated tags str into list of tags.

    >>> parse_tags('one tag')
    ['one tag']
    >>> parse_tags('  strip left and right ends  ')
    ['strip left and right ends']
    >>> parse_tags('two, tags')
    ['two', 'tags']
    >>> parse_tags('"quoted, string" is one tag')
    ['quoted, string is one tag']
    >>> parse_tags(', empty tags,  , will be skipped, ')
    ['empty tags', 'will be skipped']

    Args:
        s: The comma separated tags str.

    Returns:
        The parsed tags.
    """
    tags = []
    buf = []
    in_quoted = None

    for c in s:
        if in_quoted:
            if c == in_quoted:
                in_quoted = None
            else:
                buf.append(c)
        elif c == '"' or c == '\'':
            in_quoted = c
        elif c == ',':
            if buf:
                tag = ''.join(buf).strip()
                if tag:
                    tags.append(tag)
                buf.clear()
        else:
            buf.append(c)

    if buf:
        tag = ''.join(buf).strip()
        if tag:
            tags.append(tag)

    return tags


def deep_copy(value: TValue) -> TValue:
    """
    A patched deep copy function, that can handle various types cannot be
    handled by the standard :func:`copy.deepcopy`.

    Args:
        value: The value to be copied.

    Returns:
        The copied value.
    """
    def pattern_dispatcher(v, memo=None):
        return v  # we don't need to copy a regex pattern object, it's read-only

    old_dispatcher = copy._deepcopy_dispatch.get(PatternType, None)
    copy._deepcopy_dispatch[PatternType] = pattern_dispatcher
    try:
        return copy.deepcopy(value)
    finally:
        if old_dispatcher is not None:  # pragma: no cover
            copy._deepcopy_dispatch[PatternType] = old_dispatcher
        else:
            del copy._deepcopy_dispatch[PatternType]


class ContextStack(Generic[TObject]):
    """
    A thread-local context stack for general purpose.

    Usage::

        stack = ContextStack()
        stack.push(dict())  # push an object to the top of thread local stack
        stack.top()[key] = value  # use the top object
        stack.pop()  # pop an object from the top of thread local stack
    """

    def __init__(self,
                 initial_factory: Optional[Callable[[], TObject]] = None):
        """
        Construct a new instance of :class:`ContextStack`.

        Args:
            initial_factory: If specified, fill the context stack  with an
                initial object generated by this factory.
        """
        self._thread_local = threading.local()
        self._initial_factory = initial_factory

    @property
    def items(self) -> List[TObject]:
        if not hasattr(self._thread_local, 'items'):
            items = []
            if self._initial_factory is not None:
                items.append(self._initial_factory())
            setattr(self._thread_local, 'items', items)
        return self._thread_local.items

    def push(self, obj: TObject) -> None:
        """
        Push an object to the context stack.

         Args:
            obj: The object to be pushed.
        """
        self.items.append(obj)

    def pop(self) -> TObject:
        """
        Pop an object from the context stack.

        Returns:
            The poped context object.

        Raises:
            IndexError: if the context stack is empty.
        """
        return self.items.pop()

    def top(self) -> Optional[TObject]:
        """
        Get the top item of the context stack. or :obj:`None` if the
        context stack is empty.
        """
        items = self.items
        if items:
            return items[-1]


class GeneratorIterator(Iterator[TValue]):
    """
    Wraps a generator as an iterator, and provides the :meth:`close()`
    method to close the wrapped generator.

    >>> def make_generator():
    ...     try:
    ...         for i in range(3):
    ...             yield i
    ...     finally:
    ...         print('generator closed')

    >>> g = GeneratorIterator(make_generator())
    >>> next(g)
    0
    >>> iter(g) is g
    True
    >>> g.close()
    generator closed
    """

    def __init__(self, g: Generator[TValue, None, None]):
        self._g = g

    def __iter__(self) -> 'GeneratorIterator[TValue]':
        return self

    def __next__(self):
        return next(self._g)

    def close(self) -> None:
        self._g.close()
