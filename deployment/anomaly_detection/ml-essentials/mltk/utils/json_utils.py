import json
import math
from datetime import datetime
from enum import Enum
from typing import Any, Tuple
from uuid import UUID

import numpy as np
from bson import SON, ObjectId
from bson.json_util import JSONOptions, JSONMode, dumps, loads

__all__ = ['json_dumps', 'json_loads']

try:
    from bson import UuidRepresentation
    JSON_OPTIONS = JSONOptions(
        json_mode=JSONMode.RELAXED,
        uuid_representation=UuidRepresentation.PYTHON_LEGACY,
    )
except ImportError:
    JSON_OPTIONS = JSONOptions(json_mode=JSONMode.RELAXED)
JSON_OPTIONS.strict_uuid = False  # do not move it to the constructor above!


def _json_convert(o: Any, no_dollar_field: bool = False) -> Any:
    if isinstance(o, bool):
        # special fix: bool is subclass of int, we do not want to convert True
        # to 1, so we need to fix it.
        o = o
    elif isinstance(o, Enum):
        o = _json_convert(o.value, no_dollar_field)
    elif hasattr(o, 'items'):
        o = SON((k, _json_convert(v, no_dollar_field)) for k, v in o.items())
    elif hasattr(o, '__iter__') and not isinstance(o, (str, bytes, np.ndarray)):
        o = [_json_convert(v, no_dollar_field) for v in o]
    elif isinstance(o, str):
        o = str(o)
    elif isinstance(o, (int, np.integer, np.int, np.uint,
                        np.int8, np.int16, np.int32, np.int64,
                        np.uint8, np.uint16, np.uint32, np.uint64)):
        o = int(o)
    elif isinstance(o, (float, np.float, np.float16, np.float32, np.float64)):
        o = float(o)
    elif isinstance(o, np.ndarray):
        o = _json_convert(o.tolist(), no_dollar_field)
    else:
        o = o

    # convert special objects to str, if "no_dollar_field" is True
    if no_dollar_field:
        if isinstance(o, datetime):
            o = json.loads(dumps(o, json_options=JSON_OPTIONS))['$date']
        elif isinstance(o, UUID):
            o = str(o.hex)
        elif isinstance(o, float):
            if math.isnan(o) or math.isinf(o):
                o = str(o)
        elif isinstance(o, ObjectId):
            o = str(o)

    return o


def json_dumps(o: Any, *, separators: Tuple[str, str] = (',', ':'),
               no_dollar_field: bool = False, **kwargs) -> str:
    """
    Serialize the specified object `o` into JSON text.

    This function will convert NumPy arrays into lists, and then call
    :func:`bson.json_util.dumps` to do the remaining things.
    Thus it is compatible with MongoDB types.

    >>> json_dumps([True, False])
    '[true,false]'

    >>> json_dumps(np.array(np.nan))
    '{"$numberDouble":"NaN"}'
    >>> json_dumps(np.concatenate([np.arange(5), [np.nan, np.inf, -np.inf]], axis=0))
    '[0.0,1.0,2.0,3.0,4.0,{"$numberDouble":"NaN"},{"$numberDouble":"Infinity"},{"$numberDouble":"-Infinity"}]'
    >>> json_dumps({'values': [np.float64(0.1), np.int(2)]})
    '{"values":[0.1,2]}'
    >>> json_dumps(datetime(2019, 6, 15, 14, 50))
    '{"$date":"2019-06-15T14:50:00Z"}'
    >>> json_dumps(UUID('b8429bf5-a9c5-44ef-a8a3-f954e8aff204'))
    '{"$uuid":"b8429bf5a9c544efa8a3f954e8aff204"}'
    >>> json_dumps(ObjectId('5d04930e9dcf3fec04050251'))
    '{"$oid":"5d04930e9dcf3fec04050251"}'

    >>> json_dumps(np.array(np.nan), no_dollar_field=True)
    '"nan"'
    >>> json_dumps(
    ...     np.concatenate([np.arange(5), [np.nan, np.inf, -np.inf]], axis=0),
    ...     no_dollar_field=True
    ... )
    '[0.0,1.0,2.0,3.0,4.0,"nan","inf","-inf"]'
    >>> json_dumps(datetime(2019, 6, 15, 14, 50), no_dollar_field=True)
    '"2019-06-15T14:50:00Z"'
    >>> json_dumps(UUID('b8429bf5-a9c5-44ef-a8a3-f954e8aff204'), no_dollar_field=True)
    '"b8429bf5a9c544efa8a3f954e8aff204"'
    >>> json_dumps(ObjectId('5d04930e9dcf3fec04050251'), no_dollar_field=True)
    '"5d04930e9dcf3fec04050251"'

    >>> class MyEnum(str, Enum):
    ...     A = 'a'
    >>> json_dumps(MyEnum('a'))
    '"a"'

    Args:
        o: The object to be serialized.
        separators: The JSON separators, passed to :func:`bson.json_util.dumps`.
        no_dollar_field: Whether or not to disable MongoDB dollar field
            entries?  If True, will serialize nan, inf, date, uuid and object
            id as plain str.
        \\**kwargs: Additional arguments and named arguments passed
            to :func:`bson.json_util.dumps`.

    Returns:
        The JSON text.
    """
    return dumps(
        _json_convert(o, no_dollar_field),
        json_options=JSON_OPTIONS, separators=separators, **kwargs
    )


def json_loads(s: str, **kwargs) -> Any:
    """
    Deserialize the specified JSON text `s` into object.

    This function will call :func:`bson.json_util.loads`, thus is compatible
    with MongoDB types.

    >>> json_loads('[0.0,1.0,2.0,3.0,4.0,{"$numberDouble":"NaN"}]')
    [0.0, 1.0, 2.0, 3.0, 4.0, nan]
    >>> json_loads('{"values":[0.1,2]}')
    {'values': [0.1, 2]}
    >>> json_loads('{"$date":"2019-06-15T14:50:00Z"}')  # doctest: +ELLIPSIS
    datetime.datetime(2019, 6, 15, 14, 50...)
    >>> json_loads('{"$uuid":"b8429bf5a9c544efa8a3f954e8aff204"}')
    UUID('b8429bf5-a9c5-44ef-a8a3-f954e8aff204')
    >>> json_loads('{"$oid":"5d04930e9dcf3fec04050251"}')
    ObjectId('5d04930e9dcf3fec04050251')

    Args:
        s: The JSON text to be deserialized.
        \\**kwargs: Additional arguments and named arguments passed
            to :func:`bson.json_util.loads`.

    Returns:
        The deserialized object.
    """
    return loads(s, json_options=JSON_OPTIONS, **kwargs)
