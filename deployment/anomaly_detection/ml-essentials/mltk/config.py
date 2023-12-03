import codecs
import dataclasses
import inspect
import json
import os
import warnings
from argparse import ArgumentParser, Action
from dataclasses import dataclass, is_dataclass
from enum import Enum
from typing import *

import yaml

from .utils import *
from .type_check import *
from .typing_ import *

__all__ = [
    'ConfigValidationError',
    'field_checker', 'root_checker', 'config_field', 'ConfigField',
    'config_params', 'get_config_params', 'ConfigMeta',
    'Config', 'validate_config', 'config_to_dict', 'config_defaults',
    'ConfigLoader', 'object_to_config',
    'format_config', 'print_config', 'save_config',
]

# general special attributes that is recognized by ``type_info``

# special attributes of a Config class
_FIELDS = '__mltk_config_fields__'  # fields
_UNBOUND_CHECKERS = '__mltk_config_unbound_checkers__'  # unbound field and root checker params
_PARAMS = '__mltk_config_params__'  # config parameters
_PARAMS_CLASS_NAME = '__ConfigParams__'  # nested class as config parameters

# special attributes of a Config classmethod
_CHECKER_PARAMS = '__mltk_config_checker_params__'


ConfigValidationError = TypeCheckError
"""Legacy name for the :class:`TypeCheckError`."""


@dataclass
class ObjectFieldCheckerParams(object):
    fields: Tuple[str, ...]
    method: classmethod
    pre: bool


@dataclass
class ObjectRootCheckerParams(object):
    method: classmethod
    pre: bool


ObjectCheckerParams = Union[ObjectFieldCheckerParams, ObjectRootCheckerParams]


def field_checker(*fields, pre: bool = False):
    """
    Decorator to register a class method as a field checker in :class:`Config`.

    The checker should be implemented as a class method, with `cls` as its
    first argument, and the field value as its second argument.  Besides,
    it may also accept `values` and `field` as keyword argument, with
    `values` receiving all field values (being a dict if ``pre = True``,
    or a `Config` instance if ``pre = False``), and `field` receiving the
    name of the field being checked.

    >>> class MyConfig(Config):
    ...     a: int
    ...     b: int
    ...     c: int
    ...
    ...     @field_checker('c')
    ...     def _checker(cls, v, values, field):
    ...         if v != values['a'] + values['b']:
    ...             raise ValueError('a + b != c')
    ...         return v

    >>> validate_config(MyConfig(a=1, b='2', c=3.0))
    MyConfig(a=1, b=2, c=3)
    >>> validate_config(MyConfig(a=1, b='2', c=4.0))
    Traceback (most recent call last):
       ...
    mltk.type_check.TypeCheckError: caused by:
    * ValueError: a + b != c

    Args:
        *fields: The fields to be checked.  "*" represents all fields.
        pre: Whether or not this checker should be run before the fields
            having been checked against the field definitions?  If :obj:`True`,
            `values` will be a dict, rather than an instance of `Config`.
            Defaults to :obj:`False`.
    """
    def wrapper(method):
        if not isinstance(method, classmethod):
            method = classmethod(method)
        if not hasattr(method, _CHECKER_PARAMS):
            setattr(method, _CHECKER_PARAMS, [])
        getattr(method, _CHECKER_PARAMS).append(
            ObjectFieldCheckerParams(fields=fields, method=method, pre=pre))
        return method
    return wrapper


def root_checker(pre: bool = False):
    """
    Decorator to register a class method as a root checker in :class:`Config`.

    The checker should be implemented as a class method, with `cls` as its
    first argument, and the object values as its second argument.
    When ``pre = True``, the values will be a dict; otherwise it will be
    an instance of `Config`.

    >>> class MyConfig(Config):
    ...     a: int
    ...     b: int
    ...     c: int
    ...
    ...     @root_checker()
    ...     def _checker(cls, values):
    ...         if values.c != values.a + values.b:
    ...             raise ValueError('a + b != c')

    >>> validate_config(MyConfig(a=1, b='2', c=3.0))
    MyConfig(a=1, b=2, c=3)
    >>> validate_config(MyConfig(a=1, b='2', c=4.0))
    Traceback (most recent call last):
       ...
    mltk.type_check.TypeCheckError: caused by:
    * ValueError: a + b != c

    Args:
        pre: Whether or not this checker should be run before the fields
            having been checked against the field definitions?  If :obj:`True`,
            `values` will be a dict, rather than an instance of `Config`.
            Defaults to :obj:`False`.
    """
    def wrapper(method):
        if not isinstance(method, classmethod):
            method = classmethod(method)
        if not hasattr(method, _CHECKER_PARAMS):
            setattr(method, _CHECKER_PARAMS, [])
        getattr(method, _CHECKER_PARAMS).append(
            ObjectRootCheckerParams(method=method, pre=pre))
        return method
    return wrapper


def config_field(type: Optional[Type] = None,
                 default: Any = NOT_SET,
                 default_factory: Callable[[], Any] = NOT_SET,
                 description: Optional[str] = None,
                 choices: Optional[Sequence[Any]] = None,
                 required: bool = True,
                 envvar: Optional[str] = None,
                 ignore_empty_env: bool = True,
                 # deprecated arguments
                 nullable: bool = NOT_SET):
    """
    Define a :class:`Config` field.

    Args:
        type: Type of the field.  Any type literal that can be recognized
            by :func:`mltk.utils.type_info`, e.g., ``Optional[int]``.
            If the field type is already specified via type annotation,
            then the type specified by this argument will be ignored.
        default: The default value of this field.
        default_factory: A function ``() -> Any``, which returns the
            default value of this field.  `default` and `default_factory`
            cannot be both specified.
        description: Description of this field.
        choices: Valid values for this field to take.
        required: Whether or not this field is required?
            If :obj:`False`, the object will pass type checking even
            when this field is not specified a value.
        envvar: The name of the environmental variable to read from.
        ignore_empty_env: Whether or not empty string from the environmental
            variable will be ignored, as if no value has been given?
        nullable: DEPRECATED.  Whether or not this field is nullable?
            Use ``Optional[T]`` as type instead of using this argument.

    Returns:
        The config field object.
    """
    if nullable is not NOT_SET:
        warnings.warn('`nullable` argument is deprecated.  Use `Optional[T]` '
                      'as type instead.', DeprecationWarning)

    # check the type argument.
    if type is None:
        # If the type annotation is adopted, it will be later overwritten.
        if default is not NOT_SET:
            ti = type_info_from_value(default)
        elif default_factory is not NOT_SET:
            ti = type_info_from_value(default_factory())
        else:
            ti = AnyTypeInfo()
    else:
        ti = type_info(type)

    # check nullable constraint
    if nullable is not NOT_SET and nullable:
        if not isinstance(ti, (OptionalTypeInfo, NoneTypeInfo)):
            ti = OptionalTypeInfo(ti)

    # check the choices argument
    if choices is not None:
        choices = list(choices)

    return ObjectFieldInfo(
        name=None,
        type_info=ti,
        default=default,
        default_factory=default_factory,
        description=description,
        choices=choices,
        required=required,
        envvar=envvar,
        ignore_empty_env=ignore_empty_env,
    )


ConfigField = config_field
"""Legacy name of :func:`config_field`."""


@dataclass
class ConfigParams(object):
    """The parameters of a Config class."""
    undefined_fields: bool = False


def config_params(undefined_fields: bool = False):
    """
    A decorator to set the parameters of a Config class.

    >>> @config_params(undefined_fields=True)
    ... class MyConfig(Config):
    ...    pass
    >>> get_config_params(MyConfig)
    ConfigParams(undefined_fields=True)

    Note that, the parameters defined by this method will not be inherited,
    for example:

    >>> class MyInheritedConfig(MyConfig):
    ...     pass
    >>> get_config_params(MyInheritedConfig)
    ConfigParams(undefined_fields=False)

    To define config parameters that can be inherited, you may define a
    nested class `__ConfigParams__` instead:

    >>> class MyParent(Config):
    ...     class __ConfigParams__:
    ...         undefined_fields = True

    >>> class MyChild(MyParent):
    ...     pass

    >>> get_config_params(MyParent)
    ConfigParams(undefined_fields=True)
    >>> get_config_params(MyChild)
    ConfigParams(undefined_fields=True)

    Args:
        undefined_fields: Whether or not to allow undefined attributes?
            Defaults to :obj:`False`.

    Returns:
        The decorator method.
    """
    def wrapper(cls: Type[TConfig]) -> TConfig:
        params = getattr(cls, _PARAMS)
        params.undefined_fields = undefined_fields
        return cls
    return wrapper


def get_config_params(cls: Type[TConfig]) -> ConfigParams:
    """
    Get the parameters of specified Config class `cls`.

    Args:
        cls: The config class.

    Returns:
        The config class parameters.
    """
    return getattr(cls, _PARAMS)


class ConfigMeta(type):
    """
    Meta class for :class:`Config`.

    This class collects all definitions of the fields and the checkers of
    any subclass of :class:`Config`, and compile the type information.
    """

    def __new__(cls, name, parents, dct):
        # gather the compiled fields and validators from parents
        fields = {}
        unbound_checkers: List[ObjectCheckerParams] = []

        def process_field_info(fi: ObjectFieldInfo):
            # auto set the :obj:`None` default value for Optional[T]
            if isinstance(fi.type_info, OptionalTypeInfo) and \
                    fi.default_factory is NOT_SET and \
                    fi.default is NOT_SET:
                fi = fi.copy(default=None)
            return fi

        for parent in parents:
            if not issubclass(parent, Config):
                continue

            # inherit field definitions
            parent_fields = getattr(parent, _FIELDS, {})
            for key, val in parent_fields.items():
                if key not in fields:
                    fields[key] = val

            # inherit checkers
            for cp in getattr(parent, _UNBOUND_CHECKERS, ()):
                if cp not in unbound_checkers:
                    unbound_checkers.append(cp)

        # gather the config fields defined in this class
        annotations = dct.get('__annotations__', {})
        cls_fields = {}
        dct_keys = list(dct)

        for key in dct_keys:
            val = dct[key]

            # process the checkers
            if isinstance(val, classmethod):
                for checker_params in getattr(val, _CHECKER_PARAMS, ()):
                    unbound_checkers.append(checker_params)

            # process nested config definition
            elif isinstance(val, type) and issubclass(val, Config):
                cls_fields[key] = ObjectFieldInfo(
                    name=key,
                    type_info=type_info(val),
                    default_factory=val,
                )

            # process the fields
            elif not isinstance(val, (property, staticmethod, type)) and \
                    not inspect.isfunction(val) and \
                    not inspect.ismethod(val) and \
                    not key.startswith('_'):
                # compile the type info of this field
                if key in annotations:
                    ti = type_info(annotations[key])
                elif not isinstance(val, ObjectFieldInfo):
                    ti = type_info_from_value(val)
                else:
                    ti = val.type_info

                # construct the field info object
                if isinstance(val, ObjectFieldInfo):
                    field_info = val.copy(name=key, type_info=ti)
                else:
                    field_info = ObjectFieldInfo(
                        name=key, type_info=ti, default=val)
                field_info = process_field_info(field_info)

                # add to field list
                cls_fields[key] = field_info
                if field_info.default is NOT_SET:
                    del dct[key]
                else:
                    dct[key] = field_info.default

        for key, type_ in annotations.items():
            # skip private attributes
            if key.startswith('_'):
                continue
            # skip already processed fields
            if key in cls_fields:
                continue
            # now process the field
            ti = type_info(type_)
            field_info = process_field_info(
                ObjectFieldInfo(name=key, type_info=ti))
            cls_fields[key] = field_info

        # merge the fields and validators from parents and from this class
        fields.update(cls_fields)
        fields = {k: fields[k] for k in fields}
        dct[_FIELDS] = fields
        dct[_UNBOUND_CHECKERS] = unbound_checkers

        # construct the class
        ret_cls = super(ConfigMeta, cls).__new__(cls, name, parents, dct)

        # Since this class is being constructed now, the decorator
        # `config_params` has no chance to take in place, thus the
        # nested class `__ConfigParams__` is the only way to provide
        # config class params for the time being.
        config_params = ConfigParams()
        config_params_class = getattr(ret_cls, _PARAMS_CLASS_NAME, None)
        if config_params_class is not None:
            for key in config_params.__dict__:
                if hasattr(config_params_class, key):
                    setattr(config_params, key,
                            getattr(config_params_class, key))
        setattr(ret_cls, _PARAMS, config_params)

        # bind the checkers to this class
        root_checkers = []
        field_checkers = []

        for checker_params in unbound_checkers:
            callback = checker_params.method.__get__(ret_cls, ret_cls)
            if isinstance(checker_params, ObjectRootCheckerParams):
                root_checkers.append(
                    ObjectRootChecker(
                        callback=callback,
                        pre=checker_params.pre
                    )
                )
            else:
                field_checkers.append(
                    ObjectFieldChecker(
                        fields=checker_params.fields,
                        callback=callback,
                        pre=checker_params.pre
                    )
                )

        # construct the type info
        ti = ObjectTypeInfo(
            object_type=ret_cls,
            fields=fields,
            field_checkers=field_checkers,
            root_checkers=root_checkers,
        )
        setattr(ret_cls, TYPE_INFO_MAGIC_FIELD, ti)

        # now return the class
        return ret_cls


@config_params(undefined_fields=True)
class Config(metaclass=ConfigMeta):
    """
    Base class for config classes with type checking.

    Inherit the :class:`Config` class and provide definitions for the fields:

    >>> class MyConfig(Config):
    ...     a: int = 123
    ...     b: Optional[float]
    ...     c: str = config_field(choices=['this', 'that'])

    Default values will be copied to new instances of the config, but the
    field values will not be checked immediately:

    >>> cfg = MyConfig()
    >>> cfg
    MyConfig(a=123, b=None)
    >>> cfg.a
    123
    >>> cfg.b is None
    True
    >>> cfg.c
    Traceback (most recent call last):
        ...
    AttributeError: 'MyConfig' object has no attribute 'c'

    Calling :meth:`check_value()` will check all field values.

    >>> validate_config(MyConfig(a='12', b='34.5'))
    Traceback (most recent call last):
        ...
    mltk.type_check.TypeCheckError: at c: field 'c' is required, but its value is not specified

    >>> validate_config(MyConfig(a='12', b='34.5', c='this'))
    MyConfig(a=12, b=34.5, c='this')

    >>> validate_config(MyConfig(a='12', b='34.5', c='invalid value'))
    Traceback (most recent call last):
        ...
    mltk.type_check.TypeCheckError: at c: invalid value for field 'c': not one of ['this', 'that']

    By default, a subclass of :class:`Config` does not accept undefined fields,
    unless decorated by ``@config_params(undefined_fields=True)``, or a
    sub-class ``__ConfigParams__`` with attribute ``undefined_fields=True`` is
    provided:

    >>> class MyConfig(Config):
    ...     pass

    >>> MyConfig(a=1)
    Traceback (most recent call last):
        ...
    ValueError: Field 'a' is not defined for config class: MyConfig

    >>> @config_params(undefined_fields=True)
    ... class MyConfig(Config):
    ...     pass

    >>> MyConfig(a=1)
    MyConfig(a=1)

    >>> class MyConfig(Config):
    ...     class __ConfigParams__:
    ...         undefined_fields = True

    >>> MyConfig(a=1)
    MyConfig(a=1)
    """

    def __init__(self, **kwargs):
        params = get_config_params(self.__class__)
        fields = getattr(self.__class__, _FIELDS)

        # store user specified values
        for key, value in kwargs.items():
            if not params.undefined_fields and key not in fields:
                raise ValueError(f'Field {key!r} is not defined for config '
                                 f'class: {self.__class__.__qualname__}')
            setattr(self, key, value)

        # copy default values from the field definition to this object
        # for unspecified fields
        for key in fields:
            if key not in self.__dict__:
                default_val = fields[key].get_default()
                if default_val is not NOT_SET:
                    setattr(self, key, default_val)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __delitem__(self, key: str) -> None:
        delattr(self, key)

    def __iter__(self) -> Iterator[str]:
        if hasattr(self, '__slots__'):
            return iter(self.__slots__)
        return iter(self.__dict__)

    def __len__(self) -> int:
        if hasattr(self, '__slots__'):
            return len(self.__slots__)
        return len(self.__dict__)

    def __contains__(self, item):
        if hasattr(self, '__slots__'):
            return item in self.__slots__
        return item in self.__dict__

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if len(self) != len(other):
            return False
        for key in self:
            if key not in other or self[key] != other[key]:
                return False
        return True

    def __repr__(self):
        name = self.__class__.__qualname__
        attributes = ', '.join(f'{key}={self[key]!r}' for key in sorted(self))
        return f'{name}({attributes})'

    def to_dict(self,
                flatten: bool = False,
                type_cast: Optional[Callable[[Any], Any]] = None
                ) -> Dict[str, Any]:
        """
        Cast this config instance to a dict.

        >>> cfg = Config(a=1, b=Config(value=2))
        >>> cfg.to_dict()
        {'a': 1, 'b': {'value': 2}}
        >>> cfg.to_dict(flatten=True)
        {'a': 1, 'b.value': 2}

        Args:
            flatten: Whether or not to flatten all nested objects?
            type_cast: Auxiliary type cast function, to convert a non-config object.

        Returns:
            The dict.
        """
        def f(o):
            if not isinstance(o, Config) and is_dataclass(o):
                o = Config(**dataclasses.asdict(o))

            if isinstance(o, Config):
                return o.to_dict(flatten=flatten, type_cast=type_cast)
            elif isinstance(o, dict):
                return {k: f(v) for k, v in o.items()}
            elif isinstance(o, (list, tuple)):
                return [f(v) for v in o]
            elif isinstance(o, Enum):
                return o.value
            else:
                if type_cast is not None:
                    o = type_cast(o)
                return o

        ret = {}
        for key in self:
            val = self[key]

            if isinstance(val, Config) or is_dataclass(val):
                nested = f(val)
                if flatten:
                    for sub_key, sub_val in nested.items():
                        ret[f'{key}.{sub_key}'] = sub_val
                else:
                    ret[key] = nested
            else:
                ret[key] = f(val)

        return ret


validate_config = validate_object
"""Shortcut for :func:`check_value`."""


def config_to_dict(obj,
                   flatten: bool = False,
                   type_cast: Optional[Callable[[Any], Any]] = None
                   ) -> Dict[str, Any]:
    """
    Cast a :class:`Config` instance or a dataclass object into a dict.

    >>> cfg = Config(a=1, b=Config(value=2))
    >>> config_to_dict(cfg)
    {'a': 1, 'b': {'value': 2}}
    >>> config_to_dict(cfg, flatten=True)
    {'a': 1, 'b.value': 2}

    Args:
        obj: The object to be casted.  It must be an instance of :class:`Config`
            or a dataclass object.
        flatten: Whether or not to flatten all nested objects?
            Defaults to :obj:`False`.
        type_cast: Auxiliary type cast function, to convert a non-config object.

    Returns:
        The casted dict.
    """
    if not isinstance(obj, Config) and is_dataclass(obj):
        obj = Config(**dataclasses.asdict(obj))

    if not isinstance(obj, Config):
        raise TypeError(f'`obj` is neither a Config nor a dataclass object: '
                        f'{obj!r}')

    return obj.to_dict(flatten=flatten, type_cast=type_cast)


def config_defaults(config: Union[TConfig, Type[TConfig]]) -> TConfig:
    """
    Get the default values of a specified config class.

    >>> class MyConfig(Config):
    ...     a: int = 123
    ...     b: float

    >>> config_defaults(MyConfig)
    MyConfig(a=123)
    >>> config_defaults(MyConfig(a=456, b=789.0))
    MyConfig(a=123)

    Args:
        config: An instance of a Config class, or a Config class.

    Returns:
        An instance of the Config class with all fields filled with defaults.
    """
    if isinstance(config, type):
        config_cls = config
    else:
        config_cls = config.__class__
    if not issubclass(config_cls, Config):
        raise TypeError(f'`config` is neither an instance of Config, nor a '
                        f'subclass of Config: got {config!r}')
    return config_cls()


class LeafDict(dict):
    """
    A specialized sub-class of dict, such that it will not be unflatten
    when populating the fields of a :class:`Config` object.
    """


class ConfigLoader(Generic[TConfig]):
    """
    A class to help load config attributes from multiple sources.
    """

    use_include: bool
    """Whether or not to support '!include' directive in YAML files?"""

    def __init__(self, config_or_cls: Union[Type[TConfig], TConfig],
                 use_include: bool = True):
        """
        Construct a new :class:`ConfigLoader`.

        Args:
            config_or_cls: A config object, or a config class.
        """
        if isinstance(config_or_cls, type):
            config_cls = config_or_cls
            config = config_or_cls()
        else:
            config_cls = config_or_cls.__class__
            config = config_or_cls

        if not issubclass(config_cls, Config):
            raise TypeError(f'`config_or_cls` is neither a Config class, '
                            f'nor a Config instance: {config_or_cls!r}')

        self._config_cls = config_cls
        self._config_type_info = type_info(config_cls)
        self._config = config
        self.use_include = use_include

    @property
    def config_cls(self) -> Type[TConfig]:
        return self._config_cls

    def get(self,
            inplace: bool = True,
            ignore_missing: bool = False,
            discard_undefined: Union[DiscardMode, str] = DiscardMode.NO
            ) -> TConfig:
        """
        Get the validated config object.

        Args:
            inplace: Whether or not to validate the config object inplace?
            ignore_missing: Whether or not to ignore missing attribute?
                (i.e., attribute defined by :class:`ConfigField` without
                a default value and user specified value)
            discard_undefined: The mode to deal with undefined fields.
                Defaults to ``DiscardMode.NO``, where the undefined fields
                are not discarded.
        """
        return self._config_type_info.check_value(
            self._config,
            TypeCheckContext(
                inplace=inplace,
                ignore_missing=ignore_missing,
                discard_undefined=discard_undefined
            )
        )

    def load_object(self,
                    key_values: Union[Mapping, Config],
                    no_split_key: bool = False):
        """
        Load config attributes from the specified `key_values` object.

        All nested dicts will be converted into config objects.
        Also, all "." in keys will be further parsed into nested objects.
        For example:

        >>> class ConfigNested1(Config):
        ...     a = 123
        ...     b = ConfigField(float, default=None)

        >>> @config_params(undefined_fields=True)
        ... class YourConfig(Config):
        ...     nested1: ConfigNested1
        ...
        ...     @config_params(undefined_fields=True)
        ...     class nested2(Config):
        ...         c = 789

        >>> loader = ConfigLoader(YourConfig)
        >>> loader.load_object({'nested1': Config(a=1230)})
        >>> loader.load_object({'nested2.c': '7890'})
        >>> loader.load_object(Config(nested1=Config(b=456)))
        >>> loader.load_object({'nested2.d': {'even_nested.value': 'hello'}})
        >>> loader.get()
        YourConfig(nested1=ConfigNested1(a=1230, b=456.0), nested2=YourConfig.nested2(c=7890, d=Config(even_nested=Config(value='hello'))))

        If the full name of some non-object config attribute collides with
        some object attribute in one :meth:`load_object()` call, then an
        error will be raised, for example:

        >>> loader = ConfigLoader(Config)
        >>> loader.load_object({'nested1.a': 1230, 'nested1': 'literal'})
        Traceback (most recent call last):
            ...
        ValueError: at .nested1: cannot merge a non-object attribute into an object attribute
        >>> loader.load_object({'nested1': 'literal', 'nested1.a': 1230})
        Traceback (most recent call last):
            ...
        ValueError: at .nested1.a: cannot merge an object attribute into a non-object attribute

        Args:
            key_values: The dict or config object.
            no_split_key: If True, will not split keys with internal ".".
        """
        if not isinstance(key_values, (Mapping, Config)):
            raise TypeError(f'`key_values` must be a dict or a Config object: '
                            f'got {key_values!r}')

        def copy_values(src, dst, prefix):
            for key in src:
                err_msg1 = lambda: (
                    f'at {prefix + key}: cannot merge a non-object '
                    f'attribute into an object attribute')
                err_msg2 = lambda: (
                    f'at {prefix + key}: cannot merge an object '
                    f'attribute into a non-object attribute')

                # find the target node in dst
                tmp = dst
                if no_split_key:
                    parts = [key]
                else:
                    parts = key.split('.')
                    for part in parts[:-1]:
                        if part not in tmp:
                            tmp[part] = Config()
                        elif not isinstance(tmp[part], Config):
                            raise ValueError(err_msg2())
                        tmp = tmp[part]

                # get the src and dst values
                part = parts[-1]
                src_val = src[key]
                try:
                    dst_val = getattr(tmp, part)
                except AttributeError:
                    dst_val = NOT_SET

                # now copy the values to the target node
                if isinstance(src_val, LeafDict):
                    new_val = src_val
                elif isinstance(src_val, (dict, Config)):
                    if dst_val is NOT_SET:
                        new_val = copy_values(
                            src_val, Config(), prefix=prefix + key + '.')
                    elif isinstance(dst_val, Config):
                        new_val = copy_values(
                            src_val, dst_val, prefix=prefix + key + '.')
                    else:
                        raise ValueError(err_msg2())
                else:
                    if isinstance(dst_val, Config):
                        raise ValueError(err_msg1())
                    else:
                        new_val = src_val

                tmp[part] = new_val

            return dst

        def update_config(config, source):
            for key in source:
                val = source[key]
                self_val = getattr(config, key, None)
                if isinstance(self_val, Config) and \
                        isinstance(val, (Config, Mapping)):
                    update_config(self_val, val)
                else:
                    setattr(config, key, val)

        update_config(
            self._config,
            copy_values(key_values, Config(), prefix='.')
        )

    def load_json(self, path: Union[str, bytes, os.PathLike], cls=None):
        """
        Load config from a JSON file.

        Args:
            path: Path of the JSON file.
            cls: The JSON decoder class.
        """
        with codecs.open(path, 'rb', 'utf-8') as f:
            obj = json.load(f, cls=cls)
            self.load_object(obj)

    def load_yaml(self, path: Union[str, bytes, os.PathLike],
                  Loader=NOT_SET):
        """
        Load config from a YAML file.

        Args:
            path: Path of the YAML file.
            Loader: The YAML loader class.  If not specified, will use
                :class:`mltk.utils.YAMLIncludeLoader`
                if `self.use_include` is True, or `yaml.SafeLoader` otherwise.
        """
        if Loader is NOT_SET:
            Loader = YAMLIncludeLoader if self.use_include else yaml.SafeLoader

        with codecs.open(path, 'rb', 'utf-8') as f:
            obj = yaml.load(f, Loader=Loader)
            if obj is not None:
                self.load_object(obj)

    def load_file(self, path: Union[str, bytes, os.PathLike]):
        """
        Load config from a file.

        The file will be loaded according to its extension.  Supported
        extensions are::

            *.yml, *.yaml, *.json

        Args:
            path: Path of the file.
        """
        ext = os.path.splitext(path)[-1]
        ext = ext.lower()
        if ext in ('.yml', '.yaml'):
            self.load_yaml(path)
        elif ext in ('.json',):
            self.load_json(path)
        else:
            raise IOError(f'Unsupported config file extension: {ext}')

    def build_arg_parser(self,
                         parser: Optional[ArgumentParser] = None,
                         config_file_option: Optional[str] = '--config-file'
                         ) -> ArgumentParser:
        """
        Build an argument parser.

        This method is a sub-procedure of :class:`parse_args()`.
        Un-specified options will be :obj:`NOT_SET` in the namespace
        returned by the parser.

        Args:
            parser: The parser to populate the arguments.
                If not specified, will create a new parser.
            config_file_option: If not :obj:`None`, will add an option
                to allow loading config files.  Defaults to ``--config-file``.

        Returns:
            The argument parser.
        """
        class _ConfigAction(Action):

            def __init__(self, type_info: TypeInfo, option_strings, dest,
                         **kwargs):
                super().__init__(option_strings, dest, **kwargs)
                self.type_info = type_info

            def __call__(self, parser, namespace, values,
                         option_string=None):
                try:
                    context = TypeCheckContext()
                    with context.enter(f'.{self.dest}'):
                        value = self.type_info.parse_string(values, context)

                except Exception as ex:
                    message = f'Invalid value for argument `{option_string}`'
                    if str(ex):
                        message += '; ' + str(ex)
                    if not message.endswith('.'):
                        message += '.'
                    raise ValueError(message)
                else:
                    if isinstance(value, dict):
                        value = LeafDict(value)
                    setattr(namespace, self.dest, value)

        class _LoadFileAction(Action):

            def __call__(self, parser, namespace, values, option_string=None):
                path = values
                ext = os.path.splitext(path)[-1].lower()
                if ext == '.json':
                    loader = lambda f: json.loads(f.read())
                elif ext in ('.yaml', '.yml'):
                    loader = lambda f: yaml.load(f, Loader=yaml.SafeLoader)
                else:
                    raise IOError(f'Cannot load config file {path!r}: '
                                  f'unsupported file extension.')

                with codecs.open(values, 'rb', 'utf-8') as f:
                    obj = loader(f)
                    if not isinstance(obj, dict):
                        raise ValueError(
                            f'Expected an object from config file {path!r}, '
                            f'but got: {obj!r}'
                        )
                    for key, val in obj.items():
                        setattr(namespace, key, val)

        # gather the nested config fields
        def get_field_help(field_info: ObjectFieldInfo):
            config_help = field_info.description or ''
            default_value = field_info.get_default()
            if config_help:
                config_help += ' '
            if default_value is NOT_SET:
                if field_info.required:
                    config_help += '(required'
                else:
                    config_help += '(optional'
            else:
                config_help += f'(default {default_value!r}'
            if field_info.choices:
                config_help += f'; choices {list(field_info.choices)}'
            config_help += ')'
            return config_help

        def gather_args(ti: ObjectTypeInfo, prefix: str):
            if prefix:
                prefix += '.'

            for field_name in sorted(ti.fields):
                field_info = ti.fields[field_name]
                if isinstance(field_info.type_info, ObjectTypeInfo):
                    gather_args(field_info.type_info, prefix + field_name)
                else:
                    option_string = f'--{prefix}{field_name}'
                    help_msg = get_field_help(field_info)
                    parser.add_argument(
                        option_string, help=help_msg,
                        action=_ConfigAction, type_info=field_info.type_info,
                        default=NOT_SET, metavar=str(field_info.type_info),
                    )

        if parser is None:
            parser = ArgumentParser()

        # populate the config file argument
        if config_file_option:
            parser.add_argument(
                config_file_option,
                help='Load a config file (".json" or ".yaml").',
                action=_LoadFileAction, default=NOT_SET, metavar='PATH',
            )

        # populate the config field arguments
        gather_args(type_info(self.config_cls), '')

        return parser

    def parse_args(self, args: Iterable[str]):
        """
        Parse config attributes from CLI argument.

        >>> class YourConfig(Config):
        ...     a = 123
        ...     b = ConfigField(float, description="a float number")
        ...
        ...     class nested(Config):
        ...         c = ConfigField(str, choices=['hello', 'bye'])

        >>> loader = ConfigLoader(YourConfig)
        >>> loader.parse_args([
        ...     '--a=1230',
        ...     '--b=456',
        ...     '--nested.c=hello'
        ... ])
        >>> loader.get()
        YourConfig(a=1230, b=456.0, nested=YourConfig.nested(c='hello'))

        Args:
            args: The CLI arguments.
        """
        parser = self.build_arg_parser()
        namespace = parser.parse_args(list(args))
        parsed = {key: value for key, value in vars(namespace).items()
                  if value is not NOT_SET}
        self.load_object(parsed)


def object_to_config(config_cls: Type[TConfig], obj: Any) -> TConfig:
    """
    Convert an object into config.

    Args:
        config_cls: The config class.
        obj: The object.

    Returns:
        The config object.
    """
    config_loader = ConfigLoader(config_cls)
    config_loader.load_object(obj)
    return config_loader.get()


def format_config(config: Config,
                  title: Optional[str] = 'Configurations',
                  formatter: Callable[[Any], str] = str,
                  delimiter_char: str = '=',
                  sort_keys: bool = False
                  ) -> str:
    """
    Format a config object into str.

    >>> print(format_config(Config(a=123, b=Config(value=456))))
    Configurations
    ==============
    a         123
    b.value   456

    See Also:
        :func:`format_key_values` for more details about the arguments.
    """
    from .formatting import format_key_values
    return format_key_values(
        key_values=config,
        title=title,
        formatter=formatter,
        delimiter_char=delimiter_char,
        sort_keys=sort_keys,
    )


def print_config(config: Config,
                 title: Optional[str] = 'Configurations',
                 formatter: Callable[[Any], str] = str,
                 delimiter_char: str = '=',
                 print_func: Callable[[str], None] = print) -> None:
    """
    Print a config object to stdout.

    >>> print_config(Config(a=123, b=Config(value=456)))
    Configurations
    ==============
    a         123
    b.value   456

    See Also:
        :func:`format_key_values` for more details about the arguments.
    """
    print_func(format_config(config, title, formatter, delimiter_char))


def save_config(config: Config, path: str, flatten: bool = True) -> None:
    """
    Save the specified config object into a file.

    Args:
        config: The config object.
        path: The output file path.
        flatten: Whether or not to convert the config into dict
            using ``flatten = True``?  See :func:`config_to_dict`
            for more explanations.
    """
    # convert the config object into a dict
    cfg_dict = config_to_dict(config, flatten=flatten)

    # serialize the config dict by different serializers according to extension
    ext = os.path.splitext(path)[-1].lower()
    if ext == '.json':
        cnt = json_dumps(cfg_dict, no_dollar_field=True)
    elif ext in ('.yml', '.yaml'):
        cnt = yaml.dump(cfg_dict, Dumper=yaml.SafeDumper)
    else:
        raise IOError(f'Unsupported file extension: {ext}')

    # now write to file
    with codecs.open(path, 'wb', 'utf-8') as f:
        f.write(cnt)
