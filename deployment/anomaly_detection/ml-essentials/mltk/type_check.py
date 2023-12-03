import base64
import collections
import copy
import dataclasses
import gzip
import inspect
import os
import pickle as pkl
import re
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, is_dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import *

import numpy as np
import yaml
from dateutil.parser import parse as parse_datetime

from .typing_ import *
from .utils import NOT_SET, Singleton, deep_copy

__all__ = [
    'TypeCheckError', 'DiscardMode', 'TypeCheckContext',
    'TYPE_INFO_MAGIC_FIELD',
    'TypeInfo', 'type_info', 'type_info_from_value', 'validate_object',
    'ObjectFieldInfo', 'ObjectFieldChecker', 'ObjectRootChecker',
    'ObjectTypeInfo',

    # various type info classes
    'AnyTypeInfo',
    'IntTypeInfo', 'FloatTypeInfo', 'BoolTypeInfo', 'StrTypeInfo',
    'BytesTypeInfo', 'PatternTypeInfo', 'PathTypeInfo', 'DateTypeInfo',
    'DateTimeTypeInfo', 'NDArrayTypeInfo',
    'NoneTypeInfo', 'EllipsisTypeInfo', 'EnumTypeInfo', 'OptionalTypeInfo',
    'UnionTypeInfo', 'ListTypeInfo', 'SequenceTypeInfo', 'TupleTypeInfo',
    'VardicTupleTypeInfo', 'DictTypeInfo', 'MappingTypeInfo',
]

TYPE_INFO_MAGIC_FIELD = '__type_info__'
"""If a class provides this magic field, :func:`type_info` will return it."""


class TypeCheckError(ValueError):
    """Represent a type check error."""

    def __init__(self,
                 path: str,
                 message: str,
                 causes: Optional[Sequence[Exception]]):
        """
        Construct a new :class:`TypeCheckError`.

        Args:
            path: Path of the object from the root, where the error occurs.
            message: Message of the error.
            causes: Underlying causes of this type check error.
        """
        causes = tuple(causes or ())
        super().__init__(path, message, causes)

    @property
    def path(self) -> str:
        return self.args[0]

    @property
    def message(self) -> str:
        return self.args[1]

    @property
    def causes(self) -> Tuple[Exception, ...]:
        return self.args[2]

    def __str__(self):
        def add_indent(cnt, indent):
            return f'\n{indent}'.join(cnt.split('\n'))

        # format the causes
        causes_buf = []
        if self.causes:
            for cause in self.causes:
                if isinstance(cause, TypeCheckError):
                    causes_buf.append(add_indent(str(cause), '  '))
                else:
                    causes_buf.append(add_indent(
                        f'{cause.__class__.__qualname__}: {cause}', '  '))
        if len(self.causes) >= 1:
            causes_text = 'caused by:\n* ' + '\n* '.join(causes_buf)
        else:
            causes_text = ''

        # get the error message
        err_buf = []
        if self.message:
            err_buf.append(self.message)
        if causes_text:
            err_buf.append(causes_text)
        err_msg = '\n'.join(err_buf)

        # concatenate with path
        if self.path:
            if err_msg:
                err_msg = f'at {self.path}: {err_msg}'
            else:
                err_msg = f'at {self.path}'

        # now return the formatted error string
        return err_msg


class DiscardMode(str, Enum):
    NO = 'no'
    """Do not discard undefined fields."""

    WARN = 'warn'
    """Discard undefined fields, but generate a warning at the same time."""

    SILENT = 'silent'
    """Silently discard undefined fields."""


class TypeCheckContext(object):
    """Maintain the context for type check."""

    __slots__ = ('strict', 'inplace', 'ignore_missing', 'discard_undefined',
                 '_errors', '_scopes')

    def __init__(self,
                 strict: bool = False,
                 inplace: bool = False,
                 ignore_missing: bool = False,
                 discard_undefined: Union[DiscardMode, str] = DiscardMode.NO):
        """
        Construct a new :class:`TypeCheckContext`.

        Args:
            strict: If :obj:`True`, disable type conversion.
                If :obj:`False`, the type checker will try its best to convert
                the input `value` into desired type.
            inplace: Whether or not to convert the input `value` in place?
            ignore_missing: Whether or not to ignore missing attribute?
                (i.e., config field with neither a default value nor a user
                specified value)
            discard_undefined: The mode to deal with undefined fields.
                Defaults to ``DiscardMode.NO``, where the undefined fields
                are not discarded.
        """
        self.strict = strict
        self.inplace = inplace
        self.ignore_missing = ignore_missing
        self.discard_undefined = DiscardMode(discard_undefined)
        self._scopes: List[str] = []

    @contextmanager
    def scoped_set_strict(self, strict: bool):
        old_strict = self.strict
        try:
            self.strict = strict
            yield
        finally:
            self.strict = old_strict

    @contextmanager
    def enter(self, scope: str):
        try:
            self._scopes.append(scope)
            yield
        finally:
            self._scopes.pop()

    def get_path(self) -> str:
        return '.'.join(self._scopes)

    def raise_error(self,
                    message: str = '',
                    causes: Optional[Sequence[Exception]] = None):
        raise TypeCheckError(self.get_path(), message=message, causes=causes)


class TypeInfo(Generic[TObject]):
    """Base class for or the type information objects."""

    def _check_value(self, o: Any, context: TypeCheckContext) -> TObject:
        raise NotImplementedError()

    def check_value(self,
                    o: Any,
                    context: Optional[TypeCheckContext] = None) -> TObject:
        if context is None:
            context = TypeCheckContext()
        try:
            return self._check_value(o, context)
        except TypeCheckError:
            raise  # do not re-generate the type check errors
        except Exception as ex:
            context.raise_error(causes=[ex])

    def _parse_string(self, s: str, context: TypeCheckContext) -> TObject:
        try:
            o = yaml.load(s, Loader=yaml.SafeLoader)
        except yaml.YAMLError:
            o = s
        with context.scoped_set_strict(False):
            return self._check_value(o, context)

    def parse_string(self,
                     s: str,
                     context: Optional[TypeCheckContext] = None) -> TObject:
        if context is None:
            context = TypeCheckContext()
        try:
            return self._parse_string(s, context)
        except TypeCheckError:
            raise  # do not re-generate the type check errors
        except Exception as ex:
            context.raise_error(causes=[ex])

    def __str__(self):
        raise NotImplementedError()

    def __repr__(self):
        return f'<TypeInfo({self})>'


generic_alias_types = (type(Tuple[int]), type(Tuple[int, ...]),
                       type(List[int]), type(Sequence[int]),
                       type(Dict[str, int]), type(Mapping[str, int]),
                       type(Union[int, float]),)
is_subclass_safe = lambda t, base: isinstance(t, type) and issubclass(t, base)


def type_info(type_) -> 'TypeInfo':
    """
    Get the compiled type information for specified Python `type_`.

    Args:
        type_: The Python type or type annotation object.

    Returns:
        The compiled type information object.
    """
    if type_ is None or type_ == type(None):
        return NoneTypeInfo()
    if type_ is ... or type_ == EllipsisType:
        return EllipsisTypeInfo()

    # try to match known types
    if isinstance(type_, type):
        # if __type_info__ is specified, use that
        if hasattr(type_, TYPE_INFO_MAGIC_FIELD):
            return getattr(type_, TYPE_INFO_MAGIC_FIELD)

        # otherwise match standard types
        elif issubclass(type_, Enum):
            return EnumTypeInfo(type_)
        elif issubclass(type_, bool):
            return BoolTypeInfo()
        elif issubclass(type_, int):
            return IntTypeInfo()
        elif issubclass(type_, float):
            return FloatTypeInfo()
        elif issubclass(type_, str):
            return StrTypeInfo()
        elif issubclass(type_, bytes):
            return BytesTypeInfo()
        elif issubclass(type_, PatternType):
            return PatternTypeInfo()
        elif issubclass(type_, Path):
            return PathTypeInfo()
        elif issubclass(type_, datetime):
            return DateTimeTypeInfo()
        elif issubclass(type_, date):
            return DateTypeInfo()
        elif issubclass(type_, np.ndarray):
            return NDArrayTypeInfo()
        elif is_dataclass(type_):
            type_fields = getattr(type_, dataclasses._FIELDS)
            fields = {}
            f = lambda v: NOT_SET if v is dataclasses.MISSING else v
            for name, field in type_fields.items():
                ti = type_info(field.type)
                default_val = f(field.default)
                default_factory = f(field.default_factory)
                if default_val is NOT_SET and \
                        default_factory is NOT_SET and \
                        isinstance(ti, OptionalTypeInfo):
                    default_val = None
                fields[name] = ObjectFieldInfo(
                    name=name, type_info=ti, default=default_val,
                    default_factory=default_factory,
                )
            return ObjectTypeInfo(type_, fields)

    # try to match generic types from typing module
    if isinstance(type_, generic_alias_types):
        # Tuple[T1, T2] or Tuple[T, ...]
        if type_.__origin__ is Tuple or \
                is_subclass_safe(type_.__origin__, tuple):
            if len(type_.__args__) == 2 and type_.__args__[1] is ...:
                return VardicTupleTypeInfo(type_info(type_.__args__[0]))
            else:
                return TupleTypeInfo([
                    type_info(t) for t in type_.__args__])

        # List[T]
        if type_.__origin__ is List or is_subclass_safe(type_.__origin__, list):
            if len(type_.__args__) == 1:
                return ListTypeInfo(type_info(type_.__args__[0]))

        # Sequence[T]
        if type_.__origin__ is Sequence or is_subclass_safe(type_.__origin__, collections.abc.Sequence):
            if len(type_.__args__) == 1:
                return SequenceTypeInfo(type_info(type_.__args__[0]))

        # Dict[T1, T2]
        if type_.__origin__ is Dict or is_subclass_safe(type_.__origin__, dict):
            if len(type_.__args__) == 2:
                return DictTypeInfo(type_info(type_.__args__[0]),
                                    type_info(type_.__args__[1]))

        # Dict[T1, T2]
        if type_.__origin__ is Mapping or is_subclass_safe(type_.__origin__, collections.abc.Mapping):
            if len(type_.__args__) == 2:
                return MappingTypeInfo(type_info(type_.__args__[0]),
                                       type_info(type_.__args__[1]))

        # Union[A, B, ...]
        if type_.__origin__ is Union:
            # parse the union types
            union_args = [type_info(t) for t in type_.__args__]
            has_none = any(isinstance(t, NoneTypeInfo) for t in union_args)
            union_args = [t for t in union_args
                          if not isinstance(t, NoneTypeInfo)]

            # construct the union base type
            if len(union_args) > 1:
                union_ti = UnionTypeInfo(union_args)
            elif len(union_args) == 1:
                union_ti = union_args[0]
            else:  # pragma: no cover
                union_ti = NoneTypeInfo()

            # wrap with Optional if required
            if has_none:
                if not isinstance(union_ti, (NoneTypeInfo, OptionalTypeInfo)):
                    union_ti = OptionalTypeInfo(union_ti)
            return union_ti

    # unrecognized type, return Any
    return AnyTypeInfo()


def type_info_from_value(value: Any) -> 'TypeInfo':
    """
    Get the type information object according to the value.

    >>> type_info_from_value(None)
    <TypeInfo(Optional[Any])>
    >>> type_info_from_value(1)
    <TypeInfo(int)>

    Args:
        value: The value to be inspected.

    Returns:
        ``type_info(type(value))`` if value is not :obj:`None`,
        or ``type_info(Optional[Any])`` otherwise.
    """
    if value is None:
        return OptionalTypeInfo(AnyTypeInfo())
    else:
        return type_info(type(value))


def validate_object(obj: TObject,
                    strict: bool = False,
                    inplace: bool = False,
                    ignore_missing: bool = False,
                    discard_undefined: Union[DiscardMode, str] = DiscardMode.NO
                    ) -> TObject:
    """
    Check the value of a given object.

    Args:
        obj: The object to be checked.
        strict: If :obj:`True`, disable type conversion.
            If :obj:`False`, the type checker will try its best to convert
            the input `value` into desired type.
        inplace: Whether or not to convert the input `value` in place?
        ignore_missing: Whether or not to ignore missing fields?
        discard_undefined: The mode to deal with undefined fields.
                Defaults to ``DiscardMode.NO``, where the undefined fields
                are not discarded.

    Returns:
        The checked object.
    """
    ctx = TypeCheckContext(
        strict=strict,
        inplace=inplace,
        ignore_missing=ignore_missing,
        discard_undefined=discard_undefined)
    return type_info_from_value(obj).check_value(obj, ctx)


class AnyTypeInfo(TypeInfo, Singleton):
    """
    Type information of ``Any``.

    >>> t = type_info(Any)
    >>> t
    <TypeInfo(Any)>
    >>> t.check_value(None)
    >>> t.check_value(123)
    123
    >>> t.parse_string('[1,2,3]')
    [1, 2, 3]
    """

    __slots__ = ()

    def _check_value(self, o: Any, context: TypeCheckContext) -> Any:
        if not context.inplace:
            o = deep_copy(o)
        return o

    def __str__(self):
        return 'Any'


class PrimitiveTypeInfo(TypeInfo[TObject], Singleton):

    __slots__ = ()

    def _parse_string(self, s: str, context: TypeCheckContext) -> TObject:
        with context.scoped_set_strict(False):
            return self._check_value(s, context)

    def __eq__(self, other):
        return isinstance(other, self.__class__)

    def __hash__(self):
        return hash(self.__class__)


class IntTypeInfo(PrimitiveTypeInfo[int]):
    """
    Type information of ``int``.

    >>> t = type_info(int)
    >>> t
    <TypeInfo(int)>
    >>> t.check_value(123)
    123
    >>> t.parse_string('456')
    456
    """

    def _check_value(self, o: Any, context: TypeCheckContext) -> int:
        if not context.strict:
            int_value = int(o)
            float_value = float(o)
            if np.abs(int_value - float_value) > np.finfo(float_value).eps:
                context.raise_error(
                    'casting a float number into integer is not allowed')
            o = int_value
        if not isinstance(o, int):
            context.raise_error('value is not an integer')
        return o

    def __str__(self):
        return 'int'


class FloatTypeInfo(PrimitiveTypeInfo[float]):
    """
    Type information of ``float``.

    >>> t = type_info(float)
    >>> t
    <TypeInfo(float)>
    >>> t.check_value(123)
    123.0
    >>> t.parse_string('456.5')
    456.5
    """

    def _check_value(self, o: Any, context: TypeCheckContext) -> float:
        if not context.strict:
            o = float(o)
        if not isinstance(o, float):
            context.raise_error('value is not a float number')
        return o

    def __str__(self):
        return 'float'


class BoolTypeInfo(PrimitiveTypeInfo[bool]):
    """
    Type information of ``bool``.

    >>> t = type_info(bool)
    >>> t
    <TypeInfo(bool)>
    >>> t.check_value(True)
    True
    >>> t.parse_string('off')
    False
    """

    def _check_value(self, o: Any, context: TypeCheckContext) -> bool:
        if not context.strict:
            if isinstance(o, (str, bytes)):
                o = str(o).lower()
                if o in ('1', 'on', 'yes', 'true'):
                    o = True
                elif o in ('0', 'off', 'no', 'false'):
                    o = False
            elif isinstance(o, int):
                if o == 1:
                    o = True
                elif o == 0:
                    o = False
            if not isinstance(o, bool):
                context.raise_error('value cannot be casted into boolean')
        else:
            if not isinstance(o, bool):
                context.raise_error('value is not a boolean')
        return o

    def __str__(self):
        return 'bool'


class StrTypeInfo(PrimitiveTypeInfo[str]):
    """
    Type information of ``str``.

    >>> t = type_info(str)
    >>> t
    <TypeInfo(str)>
    >>> t.check_value(123)
    '123'
    >>> t.parse_string('456')
    '456'
    """

    def _check_value(self, o: Any, context: TypeCheckContext) -> str:
        if not context.strict:
            if isinstance(o, bytes):
                o = o.decode('utf-8')
            elif not isinstance(o, str):
                o = str(o)
        if not isinstance(o, str):
            context.raise_error('value is not a string')
        return o

    def __str__(self):
        return 'str'


class BytesTypeInfo(PrimitiveTypeInfo[bytes]):
    """
    Type information of ``bytes``.

    >>> t = type_info(bytes)
    >>> t
    <TypeInfo(bytes)>
    >>> t.check_value(b'hello, world')
    b'hello, world'
    >>> t.check_value('hello, world')
    b'hello, world'
    """

    def _check_value(self, o: Any, context: TypeCheckContext) -> bytes:
        if not context.strict:
            if isinstance(o, str):
                o = o.encode('utf-8')
            elif not isinstance(o, bytes):
                o = bytes(o)
        if not isinstance(o, bytes):
            context.raise_error('value is not a binary string')
        return o

    def __str__(self):
        return 'bytes'


class PatternTypeInfo(PrimitiveTypeInfo[PatternType]):
    """
    Type information of ``re.Pattern``.

    >>> t = type_info(PatternType)
    >>> t
    <TypeInfo(PatternType)>
    >>> t.check_value('.*')
    re.compile('.*')
    >>> t.check_value(b'.*')
    re.compile(b'.*')
    >>> t.check_value({'regex': '.*', 'flags': 'i'})
    re.compile('.*', re.IGNORECASE)
    >>> t.parse_string('[xyz]')
    re.compile('[xyz]')
    """

    CHARS_TO_FLAGS = {
        'i': re.IGNORECASE,
        'm': re.MULTILINE,
        's': re.DOTALL,
    }

    def _check_value(self, o: Any, context: TypeCheckContext) -> PatternType:
        if not context.strict:
            if isinstance(o, (str, bytes)):
                o = re.compile(o)
            elif hasattr(o, '__contains__') and hasattr(o, '__getitem__') and \
                    'regex' in o:
                flags_str = str(o.get('flags', ''))
                flags = 0
                for c in flags_str:
                    if c not in self.CHARS_TO_FLAGS:
                        context.raise_error(f'Unknown regex flag: {c!r}')
                    flags |= self.CHARS_TO_FLAGS[c]
                o = re.compile(o['regex'], flags)
            elif not isinstance(o, PatternType):
                context.raise_error('value cannot be casted into a regex '
                                    'pattern')
        if not isinstance(o, PatternType):
            context.raise_error('value is not a regex pattern.')
        return o

    def __str__(self):
        return 'PatternType'


class PathTypeInfo(PrimitiveTypeInfo[Path]):
    """
    Type information of ``pathlib.Path``.

    >>> t = type_info(Path)
    >>> t
    <TypeInfo(Path)>
    >>> t.check_value('.') == Path('.')
    True
    >>> t.check_value(Path('.')) == Path('.')
    True
    >>> t.parse_string('.') == Path('.')
    True
    """

    def _check_value(self, o: Any, context: TypeCheckContext) -> Path:
        if not context.strict:
            if not isinstance(o, Path):
                o = Path(o)
        if not isinstance(o, Path):
            context.raise_error('value is not a Path')
        if not context.inplace:
            o = deep_copy(o)
        return o

    def __str__(self):
        return 'Path'


class DateTypeInfo(PrimitiveTypeInfo[date]):
    """
    Type information of ``date``.

    >>> t = type_info(date)
    >>> t
    <TypeInfo(date)>
    >>> t.check_value('2020-01-02')
    datetime.date(2020, 1, 2)
    >>> t.check_value(date(2020, 1, 2))
    datetime.date(2020, 1, 2)
    >>> t.parse_string('2020-01-02')
    datetime.date(2020, 1, 2)
    """

    def _check_value(self, o: Any, context: TypeCheckContext) -> Path:
        if not context.strict:
            if isinstance(o, str):
                o = parse_datetime(o)
            if isinstance(o, datetime):
                o = o.date()
        if not isinstance(o, date) or isinstance(o, datetime):
            context.raise_error('value is not a date')
        return o

    def __str__(self):
        return 'date'


class DateTimeTypeInfo(PrimitiveTypeInfo[datetime]):
    """
    Type information of ``datetime``.

    >>> t = type_info(datetime)
    >>> t
    <TypeInfo(datetime)>
    >>> t.check_value('2020-01-02')
    datetime.datetime(2020, 1, 2, 0, 0)
    >>> t.check_value('2020-01-02 13:14:15')
    datetime.datetime(2020, 1, 2, 13, 14, 15)
    >>> t.check_value(datetime(2020, 1, 2, 13, 14, 15))
    datetime.datetime(2020, 1, 2, 13, 14, 15)
    >>> t.parse_string('2020-01-02 13:14:15')
    datetime.datetime(2020, 1, 2, 13, 14, 15)
    """

    def _check_value(self, o: Any, context: TypeCheckContext) -> Path:
        if not context.strict:
            if isinstance(o, str):
                o = parse_datetime(o)
            if isinstance(o, date) and not isinstance(o, datetime):
                o = datetime(o.year, o.month, o.day, 0, 0, 0)
        if not isinstance(o, datetime):
            context.raise_error('value is not a datetime')
        return o

    def __str__(self):
        return 'datetime'


class NDArrayTypeInfo(TypeInfo[np.ndarray], Singleton):
    """
    Type information of ``numpy.ndarray``.

    >>> t = type_info(np.ndarray)
    >>> t
    <TypeInfo(numpy.ndarray)>
    >>> t.check_value(1.)
    array(1.)
    >>> t.check_value([1, 2, 3])
    array([1, 2, 3])
    >>> t.parse_string('[1, 2, 3]')
    array([1, 2, 3])
    """

    def _check_value(self, o: Any, context: TypeCheckContext) -> np.ndarray:
        if not context.strict:
            if isinstance(o, str) and o.startswith('pickle:'):
                o = pkl.loads(gzip.decompress(base64.b64decode(o[7:])))
            if not isinstance(o, np.ndarray):
                o = np.asarray(o)
        if not isinstance(o, np.ndarray):
            context.raise_error('value is not a numpy array')
        if not context.inplace:
            o = np.copy(o)
        return o

    def __str__(self):
        return 'numpy.ndarray'


class NoneTypeInfo(TypeInfo[None], Singleton):
    """
    Type information of ``None``.

    >>> t = type_info(None)
    >>> t
    <TypeInfo(None)>
    >>> t.check_value(None)
    >>> t.parse_string('')
    >>> t.parse_string('null')
    >>> t.parse_string('None')
    """

    def _check_value(self, o: Any, context: TypeCheckContext) -> None:
        if o is not None:
            context.raise_error('value is not None')
        return o

    def _parse_string(self, s: str, context: TypeCheckContext) -> None:
        if s.lower() in ('', 'null', 'none'):
            return None
        context.raise_error('value is not None')

    def __str__(self):
        return 'None'


class EllipsisTypeInfo(PrimitiveTypeInfo[EllipsisType]):
    """
    Type information of ``Ellipsis``.

    >>> t = type_info(...)
    >>> t
    <TypeInfo(...)>
    >>> t.check_value(...)
    Ellipsis
    >>> t.parse_string('Ellipsis')
    Ellipsis
    >>> t.parse_string('...')
    Ellipsis
    """

    def _check_value(self, o: Any, context: TypeCheckContext) -> None:
        if o is not ...:
            if not context.strict:
                if isinstance(o, str) and o.lower() in ('...', 'ellipsis'):
                    o = ...
        if o is not ...:
            context.raise_error('value is not Ellipsis')
        return o

    def __str__(self):
        return '...'


class EnumTypeInfo(TypeInfo[TObject]):
    """
    Type information of a specified enum class.

    >>> class MyEnum(str, Enum):
    ...     A = 'A'
    ...     B = 'B'

    >>> t = type_info(MyEnum)
    >>> t
    <TypeInfo(MyEnum)>
    >>> t.check_value('A')
    <MyEnum.A: 'A'>
    """

    __slots__ = ('enum_class',)

    def __init__(self, enum_class: Type[TObject]):
        self.enum_class = enum_class

    def _check_value(self, o: Any, context: TypeCheckContext) -> TObject:
        if not context.strict:
            if not isinstance(o, self.enum_class):
                o = self.enum_class(o)
        if not isinstance(o, self.enum_class):
            context.raise_error(f'value is not an instance of {self}')
        return o

    def __str__(self):
        return self.enum_class.__qualname__

    def __eq__(self, other):
        return isinstance(other, EnumTypeInfo) and \
            other.enum_class == self.enum_class

    def __hash__(self):
        return hash((EnumTypeInfo, self.enum_class))


class OptionalTypeInfo(TypeInfo):
    """
    Type information of ``Optional[T]``.

    >>> t = type_info(Optional[int])
    >>> t
    <TypeInfo(Optional[int])>
    >>> t.check_value(None)
    >>> t.check_value(1)
    1
    >>> t.check_value(2.0)
    2
    >>> t.check_value('3')
    3
    >>> t.parse_string('')
    >>> t.parse_string('null')
    >>> t.parse_string('None')
    >>> t.parse_string('3')
    3

    Note that, ``Optional[None]`` would induce a ``None`` type:

    >>> type_info(Optional[None])
    <TypeInfo(None)>

    Also, ``Optional[Optional[T]]`` would be equivalent to ``Optional[T]``:

    >>> type_info(Optional[Optional[int]])
    <TypeInfo(Optional[int])>
    """

    def __init__(self, base_type_info: TypeInfo[TObject]):
        self.base_type_info = base_type_info

    def _check_value(self, o: Any, context: TypeCheckContext) -> Any:
        if o is not None:
            o = self.base_type_info.check_value(o, context)
        return o

    def _parse_string(self, s: str, context: TypeCheckContext) -> Any:
        if s.lower() in ('', 'null', 'none'):
            return None
        return self.base_type_info.parse_string(s, context)

    def __str__(self):
        return f'Optional[{self.base_type_info}]'

    def __eq__(self, other):
        return isinstance(other, OptionalTypeInfo) and \
            other.base_type_info == self.base_type_info

    def __hash__(self):
        return hash((OptionalTypeInfo, self.base_type_info))


class MultiBaseTypeInfo(TypeInfo):

    __slots__ = ('_type_name', 'base_types_info')

    _type_name: str

    def __init__(self, base_types_info: Sequence[TypeInfo]):
        base_types_info = tuple(base_types_info)
        if not base_types_info:
            raise ValueError('`base_types_info` must not be empty.')
        self.base_types_info = base_types_info

    def __str__(self):
        base_types = ', '.join(map(str, self.base_types_info))
        return f'{self._type_name}[{base_types}]'

    def __eq__(self, other):
        return isinstance(other, self.__class__) and \
            self.base_types_info == other.base_types_info

    def __hash__(self):
        return hash((self.__class__,) + self.base_types_info)


class UnionTypeInfo(MultiBaseTypeInfo):
    """
    Type information of ``Union[T1, T2, ..., Tn]``.

    >>> t = type_info(Union[int, float, str])
    >>> t
    <TypeInfo(Union[int, float, str])>
    >>> t.check_value(1)
    1
    >>> t.check_value(2.0)
    2.0
    >>> t.check_value(True)
    True
    >>> t.check_value('hello')
    'hello'
    >>> t.parse_string('1')
    1
    >>> t.parse_string('2.0')
    2.0
    >>> t.parse_string('2.5')
    2.5
    >>> t.parse_string('hello')
    'hello'
    """

    _type_name = 'Union'

    def _dispatch(self, f, o, context):
        errors = []
        for base_type in self.base_types_info:
            try:
                return f(base_type, o, context)
            except TypeCheckError as ex:
                errors.append(ex)
        context.raise_error('Union type check error.', errors)

    def _check_value(self, o: Any, context: TypeCheckContext) -> Any:
        # first, try to check without type cast
        try:
            with context.scoped_set_strict(strict=True):
                return self._dispatch(
                    (lambda t, a, b: t.check_value(a, b)), o, context)
        except TypeCheckError:
            if context.strict:
                raise

        # next, try to check with type cast
        return self._dispatch(
            (lambda t, a, b: t.check_value(a, b)), o, context)

    def _parse_string(self, s: str, context: TypeCheckContext) -> Any:
        return self._dispatch(
            (lambda t, a, b: t.parse_string(a, b)), s, context)


class TupleTypeInfo(MultiBaseTypeInfo):
    """
    Type information of ``Tuple[T1, T2, ..., Tn]``.

    >>> ti = type_info(Tuple[int, float, bool])
    >>> ti
    <TypeInfo(Tuple[int, float, bool])>
    >>> ti.check_value(['1', '2.0', 'on'])
    (1, 2.0, True)
    >>> ti.parse_string('[1, 2.0, "on"]')
    (1, 2.0, True)
    """

    _type_name = 'Tuple'

    def _check_value(self, o: Any, context: TypeCheckContext) -> Any:
        if not context.strict:
            if not hasattr(o, '__iter__') or isinstance(o, (str, bytes)):
                context.raise_error('value is not a sequence, thus cannot '
                                    'be casted into a tuple')
        else:
            if not isinstance(o, tuple):
                context.raise_error('value is not a tuple')
        buf = list(o)
        if len(buf) != len(self.base_types_info):
            context.raise_error(
                f'sequence length != expected size for tuple: '
                f'got {len(buf)}, expected {len(self.base_types_info)}')
        for i, v in enumerate(buf):
            buf[i] = self.base_types_info[i].check_value(v, context)
        return tuple(buf)


class BaseSequenceTypeInfo(TypeInfo):

    __slots__ = ('base_type_info', 'sequence_type', 'coarse_type')

    def __init__(self,
                 base_type_info: TypeInfo,
                 sequence_type: Type,
                 coarse_type: Optional[Type] = None):
        self.base_type_info = base_type_info
        self.sequence_type = sequence_type
        self.coarse_type = coarse_type or sequence_type

    def _check_value(self, o: Any, context: TypeCheckContext) -> Any:
        if not context.strict:
            if o is None:
                o = []
            elif not hasattr(o, '__iter__') or isinstance(o, (str, bytes)):
                context.raise_error('value is not a sequence')
        else:
            if not isinstance(o, (self.sequence_type, self.coarse_type)):
                context.raise_error(f'value is not an instance of '
                                    f'{self.sequence_type.__qualname__}')
        buf = []
        for i, v in enumerate(o):
            with context.enter(str(i)):
                buf.append(self.base_type_info.check_value(v, context))
        if context.inplace and isinstance(o, self.coarse_type) and \
                not issubclass(self.coarse_type, tuple):
            for i, v in enumerate(buf):
                o[i] = v
            return o
        else:
            return self.coarse_type(buf)

    def __eq__(self, other):
        return isinstance(other, self.__class__) and \
            self.base_type_info == other.base_type_info and \
            self.sequence_type == other.sequence_type and \
            self.coarse_type == other.coarse_type

    def __hash__(self):
        return hash((self.__class__, self.base_type_info, self.sequence_type, self.coarse_type))


class SequenceTypeInfo(BaseSequenceTypeInfo):
    """
    Type information of ``Sequence[T]``.

    >>> ti = type_info(Sequence[int])
    >>> ti
    <TypeInfo(Sequence[int])>
    >>> ti.check_value(('1', 2.0, 3))
    [1, 2, 3]
    >>> ti.parse_string('[1, 2.0, "3"]')
    [1, 2, 3]
    """

    def __init__(self, base_type_info: TypeInfo):
        super().__init__(base_type_info=base_type_info,
                         sequence_type=Sequence,
                         coarse_type=list)

    def __str__(self):
        return f'Sequence[{self.base_type_info}]'


class ListTypeInfo(BaseSequenceTypeInfo):
    """
    Type information of ``List[T]``.

    >>> ti = type_info(List[int])
    >>> ti
    <TypeInfo(List[int])>
    >>> ti.check_value(('1', 2.0, 3))
    [1, 2, 3]
    >>> ti.parse_string('[1, 2.0, "3"]')
    [1, 2, 3]
    """

    def __init__(self, base_type_info: TypeInfo):
        super().__init__(base_type_info=base_type_info, sequence_type=list)

    def __str__(self):
        return f'List[{self.base_type_info}]'


class VardicTupleTypeInfo(BaseSequenceTypeInfo):
    """
    Type information of ``Tuple[T, ...]``.

    >>> ti = type_info(Tuple[int, ...])
    >>> ti
    <TypeInfo(Tuple[int, ...])>
    >>> ti.check_value(['1', 2.0, 3])
    (1, 2, 3)
    >>> ti.parse_string('[1, 2.0, "3"]')
    (1, 2, 3)
    """

    def __init__(self, base_type_info: TypeInfo):
        super().__init__(base_type_info=base_type_info, sequence_type=tuple)

    def __str__(self):
        return f'Tuple[{self.base_type_info}, ...]'


class BaseMappingTypeInfo(TypeInfo):

    __slots__ = ('key_type_info', 'val_type_info', 'mapping_type', 'coarse_type')

    def __init__(self,
                 key_type_info: TypeInfo,
                 val_type_info: TypeInfo,
                 mapping_type: Type,
                 coarse_type: Optional[Type] = None):
        self.key_type_info = key_type_info
        self.val_type_info = val_type_info
        self.mapping_type = mapping_type
        self.coarse_type = coarse_type or mapping_type

    def _check_value(self, o: Any, context: TypeCheckContext) -> Dict[Any, Any]:
        # ensure `o` is a dict if strict = True
        if context.strict and not isinstance(o, self.mapping_type):
            context.raise_error(f'value is not an instance of {self.mapping_type}')

        # backup the original dict instance, in case `context.inplace = True`
        origin = o if isinstance(o, self.coarse_type) and context.inplace else None
        dct = {}

        if not hasattr(o, '__getitem__') or not hasattr(o, '__iter__'):
            o = _ObjectDictProxy(o)

        # check all items
        for key in o:
            val = o[key]
            with context.enter(str(key)):
                key = self.key_type_info.check_value(key, context)
                val = self.val_type_info.check_value(val, context)
            dct[key] = val

        # if orig, then use dct to update the original dict
        if origin is not None:
            origin.clear()
            origin.update(dct)
            dct = origin

        if not isinstance(dct, self.coarse_type):
            dct = self.coarse_type(dct)
        return dct

    def __eq__(self, other):
        return isinstance(other, self.__class__) and \
            self.key_type_info == other.key_type_info and \
            self.val_type_info == other.val_type_info and \
            self.mapping_type == other.mapping_type and \
            self.coarse_type == other.coarse_type

    def __hash__(self):
        return hash((self.__class__, self.key_type_info, self.val_type_info,
                     self.mapping_type, self.coarse_type))


class DictTypeInfo(BaseMappingTypeInfo):
    """
    Type information of ``Dict[TKey, TValue]``.

    >>> ti = type_info(Dict[int, float])
    >>> ti
    <TypeInfo(Dict[int, float])>
    >>> ti.check_value({'123': '456.0'})
    {123: 456.0}
    >>> ti.parse_string('{"123": "456.0"}')
    {123: 456.0}
    """

    def __init__(self, key_type_info: TypeInfo, val_type_info: TypeInfo):
        super().__init__(
            key_type_info=key_type_info,
            val_type_info=val_type_info,
            mapping_type=dict,
        )

    def __str__(self):
        return f'Dict[{self.key_type_info}, {self.val_type_info}]'


class MappingTypeInfo(BaseMappingTypeInfo):
    """
    Type information of ``Mapping[TKey, TValue]``.

    >>> ti = type_info(Mapping[int, float])
    >>> ti
    <TypeInfo(Mapping[int, float])>
    >>> ti.check_value({'123': '456.0'})
    {123: 456.0}
    >>> ti.parse_string('{"123": "456.0"}')
    {123: 456.0}
    """

    def __init__(self, key_type_info: TypeInfo, val_type_info: TypeInfo):
        super().__init__(
            key_type_info=key_type_info,
            val_type_info=val_type_info,
            mapping_type=Mapping,
            coarse_type=dict,
        )

    def __str__(self):
        return f'Mapping[{self.key_type_info}, {self.val_type_info}]'


class ObjectFieldInfo(object):
    """Information of an object field."""

    __slots__ = (
        'name', 'type_info', 'default', 'default_factory',
        'description', 'choices', 'required', 'envvar', 'ignore_empty_env',
    )

    def __init__(self,
                 name: str,
                 type_info: TypeInfo,
                 default: Any = NOT_SET,
                 default_factory: Callable[[], Any] = NOT_SET,
                 description: Optional[str] = None,
                 choices: Optional[Sequence[Any]] = None,
                 required: bool = True,
                 envvar: Optional[str] = None,
                 ignore_empty_env: bool = True):
        if default is not NOT_SET and default_factory is not NOT_SET:
            raise ValueError('`default` and `default_factory` cannot be both '
                             'specified.')
        if choices is not None:
            choices = tuple(choices)
        self.name = name
        self.type_info = type_info
        self.default = default
        self.default_factory = default_factory
        self.description = description
        self.choices = choices
        self.required = required
        self.envvar = envvar
        self.ignore_empty_env = ignore_empty_env

    def __repr__(self):
        attributes = [f'name={self.name}, type_info={self.type_info}']
        if not self.required:
            attributes.append('required=False')
        if self.default is not NOT_SET:
            attributes.append(f'default={self.default!r}')
        if self.default_factory is not NOT_SET:
            attributes.append(f'default_factory={self.default_factory!r}')
        if self.choices is not None:
            attributes.append(f'choices={self.choices!r}')
        if self.envvar is not None:
            attributes.append(f'envvar={self.envvar}')
        attributes = ', '.join(attributes)
        return f'ObjectFieldInfo({attributes})'

    def __eq__(self, other):
        return isinstance(other, ObjectFieldInfo) and \
            all(getattr(self, k) == getattr(other, k) for k in self.__slots__)

    def __hash__(self):
        return hash(
            (ObjectFieldInfo,) +
            tuple(getattr(self, k) for k in self.__slots__)
        )

    def get_default(self):
        # from environment variable
        if self.envvar is not None:
            env_val = os.environ.get(self.envvar, None)
            if not env_val and self.ignore_empty_env:
                env_val = None
            if env_val is not None:
                return self.type_info.parse_string(env_val)

        # from configured default values
        if self.default is not NOT_SET:
            return deep_copy(self.default)
        if self.default_factory is not NOT_SET:
            return self.default_factory()
        return NOT_SET

    def copy(self, **kwargs):
        key_values = {k: getattr(self, k) for k in self.__slots__}
        key_values.update(kwargs)
        return ObjectFieldInfo(**key_values)


class ObjectFieldChecker(object):
    """Custom type checker for fields of an object."""

    __slots__ = ('fields', 'pre', 'missing', 'callback', '_wrapped_callback')

    def __init__(self, fields: Sequence[str], callback: Callable,
                 pre: bool = False, missing: bool = False):
        """
        Construct an instance of :class:`ObjectFieldChecker`.

        Args:
            fields: The fields to be checked by this checker.
            callback: The callback function to check the fields.
                It must have its first argument receiving the value of the
                fields, and return the checked value.
                It may also have two arguments to receive other information,
                i.e., ``values`` to receive the object (or the field-values
                dict if ``pre=True``), and ``field`` to receive the name
                of the field being checked.
            pre: Whether or not this checker should be applied before the
                object has been constructed?  If :obj:`True`, the ``values``
                argument that ``callback`` receives will be ``Dict[str, Any]``.
                Otherwise ``values`` will be the object.
            missing: Whether or not to call the field checker even if no
                value is assigned to the field.  In such case, the field
                value will be :obj:`NOT_SET` when the checker is called.
        """
        purified_fields = []
        for field in fields:
            if field == '*':
                purified_fields[:] = ['*']
                break
            elif field not in purified_fields:
                purified_fields.append(field)
        fields = tuple(purified_fields)
        pre = bool(pre)
        missing = bool(missing)

        # parse the argument specification of `callback`, and wrap it
        # by unified function interface ``(v, values, fields) -> None``
        spec = inspect.getfullargspec(callback)
        include_values = ('values' in spec.args) or (spec.varkw is not None)
        include_field = ('field' in spec.args) or (spec.varkw is not None)
        if not include_values and not include_field:
            wrapped_callback = \
                lambda v, values, field: callback(v)
        elif include_values and not include_field:
            wrapped_callback = \
                lambda v, values, field: callback(v, values=values)
        elif not include_values and include_field:
            wrapped_callback = \
                lambda v, values, field: callback(v, field=field)
        else:
            wrapped_callback = \
                lambda v, values, field: callback(v, values=values, field=field)

        # store parameters
        self.fields = fields
        self.pre = pre
        self.missing = missing
        self.callback = callback
        self._wrapped_callback = wrapped_callback

    def __repr__(self):
        return f'ObjectFieldChecker(fields={self.fields}, ' \
               f'callback={self.callback}, pre={self.pre}, ' \
               f'missing={self.missing})'

    def __eq__(self, other):
        return isinstance(other, ObjectFieldChecker) and \
            self.fields == other.fields and \
            self.pre == other.pre and \
            self.missing == other.missing and \
            self.callback == other.callback

    def __hash__(self):
        return hash((ObjectFieldChecker, self.fields, self.pre, self.missing,
                     self.callback))

    def __call__(self, v, values, field):
        return self._wrapped_callback(v, values, field)


class ObjectRootChecker(object):
    """Custom type checker for an object."""

    __slots__ = ('callback', 'pre')

    def __init__(self, callback: Callable, pre: bool = False):
        """
        Construct an instance of :class:`ObjectRootChecker`.

        Args:
            callback: The callback function.  It must accept the object
                values (or values dict if ``pre = True``) as its first
                and the only argument.
            pre: Whether or not to apply this checker before the object
                has been constructed?
        """
        self.callback = callback
        self.pre = bool(pre)

    def __repr__(self):
        return f'ObjectRootChecker(callback={self.callback}, pre={self.pre})'

    def __eq__(self, other):
        return isinstance(other, ObjectRootChecker) and \
            self.callback == other.callback and \
            self.pre == other.pre

    def __hash__(self):
        return hash((ObjectRootChecker, self.callback))

    def __call__(self, values):
        return self.callback(values)


class HashableDict(dict):

    def __eq__(self, other):
        return isinstance(other, self.__class__) and dict.__eq__(self, other)

    def __hash__(self):
        return hash((self.__class__,) + tuple((key, self[key]) for key in self))


class HashableList(list):

    def __eq__(self, other):
        return isinstance(other, self.__class__) and list.__eq__(self, other)

    def __hash__(self):
        return hash((self.__class__,) + tuple(self))


class ObjectFieldsDict(HashableDict):
    """A dict of object fields information."""


class _ObjectDictProxy(object):

    def __init__(self, value):
        self.value = value

    def __getitem__(self, item):
        return getattr(self.value, item)

    def __setitem__(self, item, value):
        setattr(self.value, item, value)

    def __delitem__(self, item):
        delattr(self.value, item)

    def __contains__(self, item):
        return hasattr(self.value, item)

    def __iter__(self):
        if hasattr(self.value, '__slots__'):
            return iter(self.value.__slots__)
        else:
            return iter(self.value.__dict__)


class ObjectTypeInfo(TypeInfo[TObject]):
    """
    >>> @dataclass
    ... class MyObject(object):
    ...     value: int
    ...     name: str = 'noname'

    >>> t = type_info(MyObject)
    >>> str(t)
    'MyObject'
    >>> t
    <ObjectTypeInfo(MyObject, fields={'value': ObjectFieldInfo(name=value, type_info=int), 'name': ObjectFieldInfo(name=name, type_info=str, default='noname')}, field_checkers=[], root_checkers=[])>
    >>> t.check_value({'value': 123})
    MyObject(value=123, name='noname')
    >>> t.check_value({'value': 456, 'name': 'Alice'})
    MyObject(value=456, name='Alice')
    >>> t.check_value(MyObject(value=789, name='Bob'))
    MyObject(value=789, name='Bob')
    """

    __slots__ = ('object_type', 'fields', 'field_checkers', 'root_checkers')

    def __init__(self,
                 object_type: Type[TObject],
                 fields: Mapping[str, ObjectFieldInfo],
                 field_checkers: Optional[Sequence[ObjectFieldChecker]] = None,
                 root_checkers: Optional[Sequence[ObjectRootChecker]] = None):
        self.object_type = object_type
        self.fields = ObjectFieldsDict(fields)
        self.field_checkers = HashableList(field_checkers or [])
        self.root_checkers = HashableList(root_checkers or [])

    def __str__(self):
        return self.object_type.__qualname__

    def __repr__(self):
        return f'<ObjectTypeInfo({self.object_type.__qualname__}, ' \
               f'fields={self.fields!r}, ' \
               f'field_checkers={self.field_checkers}, ' \
               f'root_checkers={self.root_checkers})>'

    def __eq__(self, other):
        return isinstance(other, ObjectTypeInfo) and \
            all(getattr(self, k) == getattr(other, k) for k in self.__slots__)

    def __hash__(self):
        return hash(
            (ObjectTypeInfo,) +
            tuple(getattr(self, k) for k in self.__slots__)
        )

    def _check_value(self, o: Any, context: TypeCheckContext) -> TObject:

        inplace_check: bool = False

        if not isinstance(o, self.object_type):
            if not context.strict:
                if hasattr(o, '__getitem__') and hasattr(o, '__iter__'):
                    o_dict = o
                elif is_dataclass(o.__class__):
                    o_dict = dataclasses.asdict(o)
                else:
                    context.raise_error(f'cannot cast value into {self}: '
                                        f'value is {o}')

                if issubclass(self.object_type, dict):
                    # if target type is dict, do not expand nested attributes
                    kv = {k: o_dict[k] for k in o_dict}

                else:
                    kv = {}
                    # break down the "a.b" nested attributes
                    # into nested dict
                    for key in o_dict:
                        val = o_dict[key]
                        parts = key.split('.', 1)
                        if len(parts) == 2:
                            left, right = parts
                            with context.enter(left):
                                if left not in kv:
                                    kv[left] = {right: val}
                                elif hasattr(kv[left], '__setitem__'):
                                    kv[left][right] = val
                                else:
                                    context.raise_error(
                                        'cannot merge a non-object value '
                                        'with an object value'
                                    )
                        else:
                            with context.enter(key):
                                if key not in kv or \
                                        not hasattr(
                                            kv[key], '__setitem__'):
                                    kv[key] = val
                                else:
                                    context.raise_error(
                                        'cannot merge an object value '
                                        'with a non-object value'
                                    )

            else:
                context.raise_error(f'value is not an instance of {self}')

        else:
            if context.inplace and hasattr(o, '__getitem__') and \
                    hasattr(o, '__setitem__') and hasattr(o, '__iter__'):
                # inplace and o is Config-like class
                kv = o
                inplace_check = True
            elif context.inplace:
                # inplace and o is dataclass
                kv = _ObjectDictProxy(o)
                inplace_check = True
            else:
                if hasattr(o, '__getitem__'):
                    kv = {k: o[k] for k in o}
                else:
                    kv = copy.copy(o.__dict__)

        # if `discard undefined` is True, delete undefined fields
        if context.discard_undefined != DiscardMode.NO:
            keys = list(kv)
            for key in keys:
                if key not in self.fields:
                    if context.discard_undefined == DiscardMode.WARN:
                        path = context.get_path()
                        prefix = f'at {path}: ' if path else ''
                        warnings.warn(
                            f'{prefix}undefined field {key!r} is discarded',
                            UserWarning
                        )
                    del kv[key]

        # run the root pre-checkers
        for checker in self.root_checkers:
            if checker.pre:
                checker(kv)

        # run the field pre-checkers
        def is_field_nullable(name):
            if name in self.fields:
                field_info: ObjectFieldInfo = self.fields[name]
                return isinstance(field_info.type_info,
                                  (OptionalTypeInfo, NoneTypeInfo))
            return True

        def check_chk_ret(chk_ret, chk_field):
            if chk_ret is None and not is_field_nullable(chk_field):
                context.raise_error(
                    f'The field checker for `{chk_field}` returns '
                    f'None, but the field is not nullable. '
                    f'Did you forget to return a value '
                    f'from the checker?'
                )
            return chk_ret

        def kv_get(name, default=NOT_SET):
            return kv[name] if name in kv else default

        kv_fields = list(kv)
        field_names_include_missing = (  # for checker.missing = True
            copy.copy(kv_fields) +
            [k for k in self.fields if k not in kv]
        )

        for checker in self.field_checkers:
            if checker.pre:
                for chk_field in checker.fields:
                    if chk_field == '*':
                        for k in (kv_fields if not checker.missing
                                  else field_names_include_missing):
                            with context.enter(k):
                                chk_ret = checker(kv_get(k), kv, k)
                                chk_ret = check_chk_ret(chk_ret, k)
                                if chk_ret is not NOT_SET:
                                    kv[k] = chk_ret
                                elif k in kv:  # we allow unset the field
                                    del kv[k]

                    elif checker.missing or chk_field in kv:
                        with context.enter(chk_field):
                            chk_ret = checker(kv_get(chk_field), kv, chk_field)
                            chk_ret = check_chk_ret(chk_ret, chk_field)
                            if chk_ret is not NOT_SET:
                                kv[chk_field] = chk_ret
                            elif chk_field in kv:  # we allow unset the field
                                del kv[chk_field]

        # check the fields by registered checkers
        for field_name, field_info in self.fields.items():
            with context.enter(field_name):
                # get the field value
                field_val = kv[field_name] \
                    if field_name in kv else NOT_SET

                # fill the field with default value, if not set
                if field_val is NOT_SET:
                    field_val = field_info.get_default()

                # further check the field, if value is set
                if field_val is not NOT_SET:
                    # check by type
                    field_type = field_info.type_info
                    field_val = field_type.check_value(field_val, context)

                    # check the choices
                    if field_info.choices is not None:
                        null_but_okay = (
                            field_val is None and
                            isinstance(field_info.type_info, OptionalTypeInfo)
                        )
                        if not null_but_okay and \
                                field_val not in field_info.choices:
                            context.raise_error(
                                f'invalid value for field {field_name!r}: '
                                f'not one of {list(field_info.choices)!r}'
                            )

                    # update the checked value
                    kv[field_name] = field_val

                # otherwise delete the attribute from key_values
                else:
                    if field_name in kv:
                        del kv[field_name]

        # construct the object
        if isinstance(kv, _ObjectDictProxy):
            o = kv.value
        elif not inplace_check:
            o = self.object_type(**kv)

        # run the root post-checkers
        for checker in self.root_checkers:
            if not checker.pre:
                checker(o)

        # run the field post-checkers
        if hasattr(o, '__getitem__') and hasattr(o, '__iter__'):
            kv_fields = list(o)

            def object_get(o, k):
                return o[k]

            def object_contains(o, k):
                return k in o

            def object_set(o, k, v):
                o[k] = v

            def object_del(o, k):
                del o[k]

        else:
            kv_fields = list(o.__dict__)
            object_get, object_contains, object_set, object_del = \
                getattr, hasattr, setattr, delattr

        field_names_include_missing = (
            copy.copy(kv_fields) +
            [k for k in self.fields if not object_contains(o, k)]
        )

        def safe_get(o, name, default=NOT_SET):
            return object_get(o, name) if object_contains(o, name) else default

        for checker in self.field_checkers:
            if not checker.pre:
                for chk_field in checker.fields:
                    if chk_field == '*':
                        for k in (kv_fields if not checker.missing
                                  else field_names_include_missing):
                            with context.enter(k):
                                chk_ret = checker(safe_get(o, k), o, k)
                                chk_ret = check_chk_ret(chk_ret, k)
                                if chk_ret is not NOT_SET:
                                    object_set(o, k, chk_ret)
                                elif object_contains(o, k):
                                    object_del(o, k)

                    elif checker.missing or object_contains(o, chk_field):
                        with context.enter(chk_field):
                            chk_ret = checker(
                                safe_get(o, chk_field), o, chk_field)
                            chk_ret = check_chk_ret(chk_ret, chk_field)
                            if chk_ret is not NOT_SET:
                                object_set(o, chk_field, chk_ret)
                            elif object_contains(o, chk_field):
                                object_del(o, chk_field)

        # Finally, we can check which required fields are missing,
        # and raise error on missing fields.
        for field_name, field_info in self.fields.items():
            with context.enter(field_name):
                if field_info.required and not context.ignore_missing and \
                        not object_contains(o, field_name):
                    context.raise_error(f'field {field_name!r} is required, '
                                        f'but its value is not specified')

        return o
