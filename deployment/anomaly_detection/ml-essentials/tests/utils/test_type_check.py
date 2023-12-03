import base64
import collections
import copy
import gzip
import os
import pickle as pkl
import re
import unittest
import warnings
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from functools import partial
from pathlib import Path
from typing import *

import numpy as np
import pytest

from mltk import Config
from mltk.typing_ import *
from mltk.utils import *
from mltk.type_check import *
from mltk.type_check import (PrimitiveTypeInfo, MultiBaseTypeInfo,
                             BaseSequenceTypeInfo, _ObjectDictProxy)


class TypeCheckErrorTestCase(unittest.TestCase):

    def test_format(self):
        e = TypeCheckError(path='', message='', causes=None)
        self.assertEqual(str(e), '')

        e = TypeCheckError(path='.value', message='', causes=None)
        self.assertEqual(str(e), 'at .value')

        e = TypeCheckError(path='', message='something wrong', causes=None)
        self.assertEqual(str(e), 'something wrong')

        e = TypeCheckError(path='.value', message='something wrong', causes=None)
        self.assertEqual(str(e), 'at .value: something wrong')

        e = TypeCheckError(
            path='', message='', causes=[ValueError('abc\ndef')])
        self.assertEqual(str(e), 'caused by:\n* ValueError: abc\n  def')

        e = TypeCheckError(
            path='.value', message='', causes=[ValueError('abc\ndef')])
        self.assertEqual(
            str(e),
            'at .value: caused by:\n'
            '* ValueError: abc\n'
            '  def'
        )

        e = TypeCheckError(
            path='', message='something wrong', causes=[ValueError('abc\ndef')])
        self.assertEqual(
            str(e),
            'something wrong\n'
            'caused by:\n'
            '* ValueError: abc\n'
            '  def'
        )

        causes = [
            ValueError('abc\ndef'),
            TypeCheckError(path='.value.sub', message='sub value wrong',
                           causes=[ValueError('abc\ndef'), KeyError('ghi')])
        ]
        e = TypeCheckError(path='', message='', causes=causes)
        self.assertEqual(
            str(e),
            "caused by:\n"
            "* ValueError: abc\n"
            "  def\n"
            "* at .value.sub: sub value wrong\n"
            "  caused by:\n"
            "  * ValueError: abc\n"
            "    def\n"
            "  * KeyError: 'ghi'"
        )

        e = TypeCheckError(path='.value', message='', causes=causes)
        self.assertEqual(
            str(e),
            "at .value: caused by:\n"
            "* ValueError: abc\n"
            "  def\n"
            "* at .value.sub: sub value wrong\n"
            "  caused by:\n"
            "  * ValueError: abc\n"
            "    def\n"
            "  * KeyError: 'ghi'"
        )

        e = TypeCheckError(path='', message='something wrong', causes=causes)
        self.assertEqual(
            str(e),
            "something wrong\n"
            "caused by:\n"
            "* ValueError: abc\n"
            "  def\n"
            "* at .value.sub: sub value wrong\n"
            "  caused by:\n"
            "  * ValueError: abc\n"
            "    def\n"
            "  * KeyError: 'ghi'"
        )

        e = TypeCheckError(path='.value', message='something wrong', causes=causes)
        self.assertEqual(
            str(e),
            "at .value: something wrong\n"
            "caused by:\n"
            "* ValueError: abc\n"
            "  def\n"
            "* at .value.sub: sub value wrong\n"
            "  caused by:\n"
            "  * ValueError: abc\n"
            "    def\n"
            "  * KeyError: 'ghi'"
        )


class Int64TypeInfo(IntTypeInfo):

    def _check_value(self, o: Any, context: TypeCheckContext) -> int:
        return int64(super()._check_value(o, context))

    def __str__(self):
        return 'int64'


class int64(int):
    __type_info__ = Int64TypeInfo()


class TypeInfoTestCase(unittest.TestCase):

    def test_base(self):
        class MyTypeInfo(TypeInfo):

            _expect_value: Any = NOT_SET
            _expect_strict: bool = False
            _raise_error: Optional[Exception] = None

            def _check_value(self, o: Any, context: TypeCheckContext) -> Any:
                if self._expect_value is not NOT_SET:
                    this.assertEqual(o, self._expect_value)
                this.assertEqual(context.strict, self._expect_strict)
                if self._raise_error is not None:
                    err = self._raise_error
                    self._raise_error = None
                    raise err
                return o

        this = self
        ti = MyTypeInfo()

        ##############
        # type_check #
        ##############

        # test ordinary check
        ti._expect_strict = False
        self.assertEqual(ti.check_value('123'), '123')

        # test strict check
        ti._expect_strict = True
        context = TypeCheckContext(strict=True)
        self.assertEqual(ti.check_value('123', context), '123')

        # test capture and re-raise TypeCheckError
        ti._expect_strict = False
        err = TypeCheckError('.value', 'something wrong', [])
        ti._raise_error = err
        with pytest.raises(TypeCheckError) as m:
            ti.check_value('123')
        self.assertEqual(m.value, err)

        # test capture and wrap other errors
        err = TypeError('something error')
        ti._raise_error = err
        with pytest.raises(TypeCheckError) as m:
            ti.check_value('123')
        self.assertEqual(m.type, TypeCheckError)
        self.assertEqual(m.value.causes, (err,))

        ################
        # parse_string #
        ################

        # test yaml parse success
        ti._expect_strict = False
        ti._expect_value = 123
        self.assertEqual(
            ti.parse_string('123', TypeCheckContext(strict=True)),
            123
        )

        # test yaml parse error
        ti._expect_strict = False
        ti._expect_value = '[123'
        self.assertEqual(
            ti.parse_string('[123', TypeCheckContext(strict=True)),
            '[123'
        )

        # test capture and re-raise TypeCheckError
        err = TypeCheckError('.value', 'something wrong', [])
        ti._raise_error = err
        ti._expect_value = NOT_SET
        with pytest.raises(TypeCheckError) as m:
            ti.parse_string('123')
        self.assertEqual(m.value, err)

        # test capture and wrap other errors
        err = TypeError('something error')
        ti._raise_error = err
        with pytest.raises(TypeCheckError) as m:
            ti.parse_string('123')
        self.assertEqual(m.type, TypeCheckError)
        self.assertEqual(m.value.causes, (err,))

    def test_type_info(self):
        def assert_equal(ti1, ti2):
            self.assertEqual(ti1, ti2)
            self.assertEqual(hash(ti1), hash(ti2))

        def assert_not_equal(ti1, ti2):
            self.assertNotEqual(ti1, ti2)

        # override core type
        assert_equal(type_info(int64), Int64TypeInfo())

        # primitive types
        assert_equal(type_info(Any), AnyTypeInfo())
        assert_equal(type_info(int), IntTypeInfo())
        assert_equal(type_info(float), FloatTypeInfo())
        assert_equal(type_info(bool), BoolTypeInfo())
        assert_equal(type_info(str), StrTypeInfo())
        assert_equal(type_info(bytes), BytesTypeInfo())
        assert_equal(type_info(None), NoneTypeInfo())
        assert_equal(type_info(type(None)), NoneTypeInfo())
        assert_equal(type_info(...), EllipsisTypeInfo())
        assert_equal(type_info(type(...)), EllipsisTypeInfo())
        assert_equal(type_info(PatternType), PatternTypeInfo())
        assert_equal(type_info_from_value(re.compile('.*')), PatternTypeInfo())
        assert_equal(type_info(Path), PathTypeInfo())

        # np.ndarray
        assert_equal(type_info(np.ndarray), NDArrayTypeInfo())

        # enum
        class MyEnum(str, Enum):
            A = 'A'
            B = 'B'

        assert_equal(type_info(MyEnum), EnumTypeInfo(MyEnum))
        assert_not_equal(type_info(MyEnum), EnumTypeInfo(Enum))

        # Optional[T]
        assert_equal(
            type_info(Optional[int]),
            OptionalTypeInfo(IntTypeInfo()))
        assert_equal(
            type_info(Optional[Optional[int]]),
            OptionalTypeInfo(IntTypeInfo()))
        assert_equal(type_info(Optional[None]), NoneTypeInfo())
        assert_not_equal(
            type_info(Optional[int]),
            OptionalTypeInfo(FloatTypeInfo()))

        # Union[T1, T2, ..., Tn]
        assert_equal(
            type_info(Union[int, float, str]),
            UnionTypeInfo([IntTypeInfo(), FloatTypeInfo(), StrTypeInfo()]))
        assert_equal(
            type_info(Union[int, int, float, str, float]),
            UnionTypeInfo([IntTypeInfo(), FloatTypeInfo(), StrTypeInfo()]))
        assert_equal(
            type_info(Union[int, None]),
            OptionalTypeInfo(IntTypeInfo()))
        assert_equal(
            type_info(Union[Optional[int], None]),
            OptionalTypeInfo(IntTypeInfo()))
        assert_equal(type_info(Union[None]), NoneTypeInfo())
        assert_equal(type_info(Union[None, None]), NoneTypeInfo())
        assert_not_equal(
            type_info(Union[int, float, str]),
            TupleTypeInfo([IntTypeInfo(), FloatTypeInfo(), StrTypeInfo()]))
        assert_not_equal(
            type_info(Union[int, float, str]),
            UnionTypeInfo([IntTypeInfo(), FloatTypeInfo(), BoolTypeInfo()]))

        # Tuple[T1, T2, ..., Tn]
        assert_equal(
            type_info(Tuple[int, float, bool]),
            TupleTypeInfo([IntTypeInfo(), FloatTypeInfo(), BoolTypeInfo()]))
        assert_not_equal(
            type_info(Tuple[int, float, bool]),
            TupleTypeInfo([IntTypeInfo(), FloatTypeInfo(), StrTypeInfo()]))

        # List[T]
        assert_equal(type_info(List[int]), ListTypeInfo(IntTypeInfo()))
        assert_not_equal(type_info(List[int]), ListTypeInfo(FloatTypeInfo()))
        assert_not_equal(
            type_info(List[int]),
            VardicTupleTypeInfo(FloatTypeInfo()))

        # Sequence[T]
        assert_equal(type_info(Sequence[int]), SequenceTypeInfo(IntTypeInfo()))
        assert_not_equal(type_info(Sequence[int]), SequenceTypeInfo(FloatTypeInfo()))
        assert_not_equal(
            type_info(Sequence[int]),
            VardicTupleTypeInfo(FloatTypeInfo()))

        # Tuple[T, ...]
        assert_equal(
            type_info(Tuple[int, ...]),
            VardicTupleTypeInfo(IntTypeInfo()))
        assert_not_equal(
            type_info(Tuple[int, ...]),
            VardicTupleTypeInfo(FloatTypeInfo()))

        # Dict[TKey, TValue]
        assert_equal(
            type_info(Dict[str, int]),
            DictTypeInfo(StrTypeInfo(), IntTypeInfo()))
        assert_not_equal(
            type_info(Dict[str, int]),
            DictTypeInfo(StrTypeInfo(), StrTypeInfo()))
        assert_not_equal(
            type_info(Dict[str, int]),
            DictTypeInfo(IntTypeInfo(), IntTypeInfo()))

        # Mapping[TKey, TValue]
        assert_equal(
            type_info(Mapping[str, int]),
            MappingTypeInfo(StrTypeInfo(), IntTypeInfo()))
        assert_not_equal(
            type_info(Mapping[str, int]),
            MappingTypeInfo(StrTypeInfo(), StrTypeInfo()))
        assert_not_equal(
            type_info(Mapping[str, int]),
            MappingTypeInfo(IntTypeInfo(), IntTypeInfo()))

        # data class
        my_bool_factory = lambda: True
        my_list_factory =  lambda: [1, 2, 3]

        @dataclass
        class Nested(object):
            value: List[int] = field(default_factory=my_list_factory)

        @dataclass
        class MyObject(object):
            a: int
            b: Optional[float]
            c: Nested
            d: str = 'hello'
            e: bool = field(default_factory=my_bool_factory)

        ti = type_info(MyObject)
        ti2_factory = lambda: ObjectTypeInfo(
            MyObject,
            fields={
                'a': ObjectFieldInfo('a', type_info(int)),
                'b': ObjectFieldInfo('b', type_info(Optional[float]),
                                     default=None),
                'c': ObjectFieldInfo(
                    'c',
                    ObjectTypeInfo(
                        Nested,
                        fields={
                            'value': ObjectFieldInfo(
                                'value', type_info(List[int]),
                                default_factory=my_list_factory
                            )
                        }
                    )
                ),
                'd': ObjectFieldInfo('d', type_info(str),
                                     default='hello'),
                'e': ObjectFieldInfo('e', type_info(bool),
                                     default_factory=my_bool_factory)
            }
        )
        assert_equal(type_info(MyObject), ti2_factory())
        for attr in ('object_type', 'fields', 'field_checkers',
                     'root_checkers'):
            ti2 = ti2_factory()
            setattr(ti2, attr, None)
            assert_not_equal(ti, ti2)

    def test_singleton(self):
        for t in (bool, int, float, str, bytes,
                  None, Any, PatternType, Path):
            self.assertIs(type_info(t), type_info(t))

    def _check_cast(self, ti, expected, strict_inputs, loose_inputs,
                    error_inputs=(), assert_eq=None):
        if assert_eq is None:
            assert_eq = self.assertEqual

        for input_ in strict_inputs:
            assert_eq(
                ti.check_value(input_, TypeCheckContext(strict=False)),
                expected
            )
            assert_eq(
                ti.check_value(input_, TypeCheckContext(strict=True)),
                expected
            )

        for input_ in loose_inputs:
            assert_eq(
                ti.check_value(input_, TypeCheckContext(strict=False)),
                expected
            )
            with pytest.raises(TypeCheckError):
                _ = ti.check_value(input_, TypeCheckContext(strict=True))

        for input_ in error_inputs:
            with pytest.raises(TypeCheckError):
                _ = ti.check_value(input_, TypeCheckContext(strict=False))

    def test_Any(self):
        ti = type_info(Any)
        self.assertEqual(str(ti), 'Any')
        self.assertEqual(ti.check_value('true'), 'true')
        self.assertEqual(ti.parse_string('true'), True)

        # test inplace
        v = [1, 2, 3]
        v_copy = deep_copy(v)

        v_out = ti.check_value(v)
        self.assertIsNot(v_out, v)
        self.assertEqual(v_out, v_copy)

        v_out = ti.check_value(v, TypeCheckContext(inplace=True))
        self.assertIs(v_out, v)
        self.assertEqual(v_out, v_copy)

    def test_primitive(self):
        class MyPrimitive(PrimitiveTypeInfo):
            _expected_value = None
            def _check_value(self, o: Any, context: TypeCheckContext) -> Any:
                this.assertEqual(o, self._expected_value)
                return o

        this = self
        ti = MyPrimitive()
        # Primitive types should directly parse input strings, not yaml
        # decoded values.  This is especially important for StrTypeInfo.
        ti._expected_value = '123'
        self.assertEqual(ti.parse_string('123'), '123')

    def test_int(self):
        ti = type_info(int)
        self.assertEqual(str(ti), 'int')
        self._check_cast(ti, 123, [123], ['123', 123.0], [123.5, 'xxx'])

    def test_int64(self):
        ti = type_info(int64)
        self.assertEqual(str(ti), 'int64')
        self._check_cast(ti, 123, [123], ['123', 123.0], [123.5, 'xxx'])

    def test_float(self):
        ti = type_info(float)
        self.assertEqual(str(ti), 'float')
        self._check_cast(ti, 123.0, [123.0], ['123', 123], ['xxx'])
        self._check_cast(ti, 123.5, [123.5], ['123.5'])

    def test_bool(self):
        ti = type_info(bool)
        self.assertEqual(str(ti), 'bool')
        self._check_cast(ti, True, [True], ['TRUE', 'On', 'yes', 1], ['xxx'])
        self._check_cast(ti, False, [False], ['false', 'OFF', 'No', 0])

    def test_str(self):
        ti = type_info(str)
        self.assertEqual(str(ti), 'str')
        self._check_cast(ti, '123', ['123'], [b'123', 123])
        self.assertEqual(ti.parse_string('true'), 'true')

    def test_bytes(self):
        ti = type_info(bytes)
        self.assertEqual(str(ti), 'bytes')
        self._check_cast(ti, b'123', [b'123'], ['123'], [object()])

    def test_pattern_type(self):
        ti = type_info(PatternType)
        self.assertEqual(str(ti), 'PatternType')
        self._check_cast(
            ti,
            re.compile(b'[xyz]'),
            [re.compile(b'[xyz]')],
            [b'[xyz]', {'regex': b'[xyz]'}],
            [object(), b'[xyz']
        )
        self._check_cast(
            ti,
            re.compile('[xyz]', re.I),
            [re.compile('[xyz]', re.I)],
            [{'regex': '[xyz]', 'flags': 'i'}]
        )
        # the following sentence checks whether or not the pattern string
        # is parsed by the PatternTypeInfo directly without being parsed
        # by YAML decoder.
        self.assertEqual(ti.parse_string('[xyz]'), re.compile('[xyz]'))
        with pytest.raises(TypeCheckError, match='Unknown regex flag: \'x\''):
            _ = ti.check_value({'regex': '.*', 'flags': 'x'})

    def test_path(self):
        ti = type_info(Path)
        self.assertEqual(str(ti), 'Path')
        self._check_cast(ti, Path('.'), [Path('.')], ['.'], [object()])

        # test inplace = False
        path = Path('.')
        self.assertIsNot(ti.check_value(path), path)

    def test_date(self):
        ti = type_info(date)
        self.assertEqual(str(ti), 'date')
        self._check_cast(
            ti,
            date(2020, 1, 2),
            [date(2020, 1, 2)],
            ['2020-01-02', datetime(2020, 1, 2, 13, 14, 15), '2020-01-02 13:14:15'],
            [object()]
        )

    def test_datetime(self):
        ti = type_info(datetime)
        self.assertEqual(str(ti), 'datetime')
        self._check_cast(
            ti,
            datetime(2020, 1, 2, 13, 14, 15),
            [datetime(2020, 1, 2, 13, 14, 15)],
            ['2020-01-02 13:14:15'],
            [object()]
        )
        self._check_cast(
            ti,
            datetime(2020, 1, 2, 0, 0, 0),
            [datetime(2020, 1, 2, 0, 0, 0)],
            ['2020-01-02', date(2020, 1, 2), '2020-01-02'],
            [object()]
        )

    def test_ndarray(self):
        serialized_array = base64.b64encode(
            gzip.compress(pkl.dumps(np.array([1, 2, 3])))).decode('utf-8')

        ti = type_info(np.ndarray)
        self.assertEqual(str(ti), 'numpy.ndarray')
        self._check_cast(
            ti, np.array([1, 2, 3]),
            [np.array([1, 2, 3])],
            [[1, 2, 3], (1, 2, 3), f'pickle:{serialized_array}'],
            [],
            assert_eq=np.testing.assert_equal
        )

        # test parse_string
        arr2 = ti.parse_string('[1, 2, 3]')
        self.assertIsInstance(arr2, np.ndarray)
        np.testing.assert_equal(arr2, [1, 2, 3])

        # test inplace = False
        arr = np.array([1, 2, 3])
        arr2 = ti.check_value(arr)
        self.assertIsNot(arr2, arr)
        arr[:] = 0
        np.testing.assert_equal(arr2, [1, 2, 3])

    def test_none(self):
        ti = type_info(None)
        self.assertEqual(str(ti), 'None')
        self._check_cast(ti, None, [None], [], ['xxx', 'null', 'NULL', 'None'])
        self.assertEqual(ti.parse_string(''), None)
        self.assertEqual(ti.parse_string('null'), None)
        self.assertEqual(ti.parse_string('NULL'), None)
        self.assertEqual(ti.parse_string('none'), None)
        self.assertEqual(ti.parse_string('None'), None)
        with pytest.raises(TypeCheckError, match='value is not None'):
            _ = ti.parse_string('xxx')

    def test_Ellipsis(self):
        ti = type_info(...)
        self.assertEqual(str(ti), '...')
        self._check_cast(ti, ..., [...], ['...', 'Ellipsis', 'ellipsis'], [0, None, 'null'])

    def test_enum(self):
        # typed enum
        class MyEnum(str, Enum):
            A = 'A'
            B = 'B'

        ti = type_info(MyEnum)
        self.assertEqual(str(ti), MyEnum.__qualname__)
        self.assertEqual(ti, EnumTypeInfo(MyEnum))
        self.assertIsInstance(ti.check_value('A'), MyEnum)
        self.assertIsInstance(ti.check_value('A'), str)
        self._check_cast(ti, MyEnum.A, [MyEnum.A], ['A'], ['xxx'])
        self._check_cast(ti, MyEnum.B, [MyEnum.B], ['B'])

        # untyped enum
        class MyEnum(Enum):
            A = 1
            B = 2

        ti = type_info(MyEnum)
        self.assertEqual(str(ti), MyEnum.__qualname__)
        self.assertEqual(ti, EnumTypeInfo(MyEnum))
        self.assertIsInstance(ti.check_value(1), MyEnum)
        self._check_cast(ti, MyEnum.A, [MyEnum.A], [1], [3])
        self._check_cast(ti, MyEnum.B, [MyEnum.B], [2])

    def test_optional(self):
        ti = type_info(Optional[int])
        self.assertEqual(str(ti), 'Optional[int]')

        self._check_cast(ti, None, [None], [], ['null', 'NULL', 'None'])
        self._check_cast(ti, 123, [123], ['123', 123.0], [123.5, 'xxx'])
        self.assertEqual(ti.parse_string(''), None)
        self.assertEqual(ti.parse_string('Null'), None)
        self.assertEqual(ti.parse_string('NONE'), None)

    def test_multi_base(self):
        with pytest.raises(ValueError,
                           match='`base_types_info` must not be empty'):
            _ = MultiBaseTypeInfo([])

    def test_union(self):
        # simple test case
        ti = type_info(Union[float, int])
        self.assertEqual(str(ti), 'Union[float, int]')
        self._check_cast(ti, 123.5, [123.5], ['123.5'], ['xxx'])
        self._check_cast(ti, 123, [123], ['123', '123.0'])
        self.assertEqual(ti.parse_string('123'), 123)
        self.assertEqual(ti.parse_string('123.5'), 123.5)

        # integrated test case: args
        ti = type_info(Union[List[Union[str, bytes]],
                             Tuple[Union[str, bytes], ...],
                             str,
                             bytes])
        self.assertEqual(str(ti), 'Union[List[Union[str, bytes]], '
                                  'Tuple[Union[str, bytes], ...], str, bytes]')
        self.assertEqual(ti.check_value(['a', b'b']), ['a', b'b'])
        self.assertEqual(ti.check_value(('a', b'b')), ('a', b'b'))
        self.assertEqual(ti.check_value(['a', b'b', 123]), ['a', b'b', '123'])
        self.assertEqual(ti.check_value('a'), 'a')
        self.assertEqual(ti.check_value(b'b'), b'b')
        self.assertEqual(ti.check_value(123), '123')

    def test_tuple(self):
        ti = type_info(Tuple[float, int, bool])
        self.assertEqual(str(ti), 'Tuple[float, int, bool]')

        self._check_cast(
            ti,
            (123.5, 123, True),
            [(123.5, 123, True)],
            [
                ('123.5', 123.0, 'ON'),
                ['123.5', '123', 'true']
            ],
            [123, '123', ('123.5', 123.0)]
        )

    def test_base_sequence(self):
        class MyList(list):
            def __eq__(self, other):
                return isinstance(other, MyList) and list.__eq__(self, other)

        ti = BaseSequenceTypeInfo(IntTypeInfo(), MyList)
        self._check_cast(ti, MyList([]), [MyList([])], [None, []], [False])
        self._check_cast(
            ti,
            MyList([123, 456]),
            [MyList([123, 456])],
            [
                (123.0, '456'),
            ],
            [123, '123']
        )

        # test specialized sequence types
        ti = type_info(List[int])
        self.assertEqual(str(ti), 'List[int]')
        self.assertEqual(ti.check_value((1, 2, 3)), [1, 2, 3])

        ti = type_info(Tuple[int, ...])
        self.assertEqual(str(ti), 'Tuple[int, ...]')
        self.assertEqual(ti.check_value([1, 2, 3]), (1, 2, 3))

    def test_list(self):
        ti = ListTypeInfo(IntTypeInfo())
        self.assertEqual(str(ti), 'List[int]')

        v = [1, '2', 3.0]
        v_orig = deep_copy(v)
        v_ans = [1, 2, 3]

        v_out = ti.check_value(v)
        self.assertIsNot(v_out, v)
        self.assertEqual(v_out, v_ans)
        self.assertEqual(v, v_orig)

        v_out = ti.check_value(v, TypeCheckContext(inplace=True))
        self.assertIs(v_out, v)
        self.assertEqual(v_out, v_ans)

    def test_sequence(self):
        ti = SequenceTypeInfo(IntTypeInfo())
        self.assertEqual(str(ti), 'Sequence[int]')

        # list -> list
        v = [1, '2', 3.0]
        v_orig = deep_copy(v)
        v_ans = [1, 2, 3]

        v_out = ti.check_value(v)
        self.assertIsNot(v_out, v)
        self.assertEqual(v_out, v_ans)
        self.assertEqual(v, v_orig)

        v_out = ti.check_value(v, TypeCheckContext(inplace=True))
        self.assertIs(v_out, v)
        self.assertEqual(v_out, v_ans)

        # tuple -> list is strict under Sequence[int]
        v = (1, 2, 3)
        self.assertEqual(ti.check_value(v, TypeCheckContext(strict=True)), v_ans)

    def test_dict(self):
        class MyDictLike(Mapping):
            def __init__(self, wrapped):
                self.wrapped = wrapped

            def __getitem__(self, k):
                return self.wrapped[k]

            def __len__(self) -> int:
                return len(self.wrapped)

            def __iter__(self):
                return iter(self.wrapped)

        ti = type_info(Dict[str, float])
        self.assertEqual(str(ti), 'Dict[str, float]')

        self._check_cast(
            ti,
            {'12': 34.0, '56': 78.5},
            [{'12': 34.0, '56': 78.5}],
            [
                {12: 34, '56': '78.5'},
                MyDictLike({'12': 34, '56': '78.5'})
            ],
            [
                object(),
                [1, 2, 3]
            ]
        )

        @dataclass
        class MyDataClass(object):
            a: int = 12
            b: float = 34.5

        self._check_cast(
            ti,
            {'a': 12.0, 'b': 34.5},
            [{'a': 12.0, 'b': 34.5}],
            [
                MyDataClass()
            ]
        )

        # test inplace
        v = {12: 34, '56': '78.5'}
        v_orig = deep_copy(v)
        v_ans = {'12': 34.0, '56': 78.5}

        v_out = ti.check_value(v)
        self.assertIsNot(v_out, v)
        self.assertEqual(v_out, v_ans)
        self.assertEqual(v, v_orig)

        v_out = ti.check_value(v, TypeCheckContext(inplace=True))
        self.assertIs(v_out, v)
        self.assertEqual(v_out, v_ans)

    def test_mapping(self):
        ti = type_info(Mapping[str, float])
        self.assertEqual(str(ti), 'Mapping[str, float]')

        v = {'a': '12', 'b': 13.5}
        v_orig = deep_copy(v)
        v_ans = {'a': 12.0, 'b': 13.5}

        v_out = ti.check_value(v)
        self.assertIsNot(v_out, v)
        self.assertEqual(v_out, v_ans)
        self.assertEqual(v, v_orig)

        v_out = ti.check_value(v, TypeCheckContext(inplace=True))
        self.assertIs(v_out, v)
        self.assertEqual(v_out, v_ans)

        # strict should recognize OrderedDict
        v = collections.OrderedDict({'a': 12.0, 'b': 13.5})
        v_out = ti.check_value(v, TypeCheckContext(strict=True))
        self.assertEqual(v_out, v_ans)

    def test_object_field_info(self):
        with pytest.raises(ValueError,
                           match='`default` and `default_factory` cannot be '
                                 'both specified'):
            _ = ObjectFieldInfo(name='abc', type_info=IntTypeInfo(),
                                default=123, default_factory=lambda: 456)

        # test with minimum configuration
        factory = lambda: ObjectFieldInfo(name='abc', type_info=IntTypeInfo())
        fi = factory()
        self.assertEqual(repr(fi), 'ObjectFieldInfo(name=abc, type_info=int)')
        self.assertEqual(fi, factory())
        self.assertEqual(hash(fi), hash(factory()))
        self.assertEqual(fi.get_default(), NOT_SET)

        # test copy and default factory
        default_factory = lambda: 456
        fi = fi.copy(default_factory=default_factory)
        self.assertEqual(
            repr(fi),
            f'ObjectFieldInfo(name=abc, type_info=int, '
            f'default_factory={default_factory!r})'
        )

        # test with default and other fields
        envvar = 'MLTK_TEST_ABC'
        factory = lambda: ObjectFieldInfo(
            name='abc', type_info=type_info(Optional[float]), default=123.5,
            description='a float field', choices=(123.5, 124.0),
            required=False, envvar=envvar, ignore_empty_env=True,
        )
        fi = factory()
        self.assertEqual(
            repr(fi),
            f'ObjectFieldInfo(name=abc, type_info=Optional[float], '
            f'required=False, default=123.5, choices=(123.5, 124.0), '
            f'envvar={envvar})'
        )
        self.assertEqual(fi, factory())
        self.assertEqual(hash(fi), hash(factory()))
        if envvar in os.environ:
            os.environ.pop(envvar)
        self.assertEqual(fi.get_default(), 123.5)
        os.environ[envvar] = ''
        self.assertEqual(fi.get_default(), 123.5)
        os.environ[envvar] = '124.5'
        self.assertEqual(fi.get_default(), 124.5)

        # test ignore_empty_env is False
        fi = ObjectFieldInfo(
            name='abc', type_info=type_info(Optional[float]), default=123.5,
            description='a float field', choices=[123.5, 124.0],
            required=False, envvar=envvar, ignore_empty_env=False,
        )
        os.environ[envvar] = ''
        self.assertEqual(fi.get_default(), None)

        # test default value should be copied
        fi_default = [1, 2, 3]
        fi = ObjectFieldInfo(name='abc', type_info=type_info(List[int]),
                             default=fi_default)
        self.assertIsNot(fi.get_default(), fi_default)
        self.assertEqual(fi.get_default(), fi_default)

    def test_object_field_checker(self):
        watchers = []

        def f_v(v):
            watchers.append({'name': 'f_v', 'v': v})

        def f_v_values(v, values):
            watchers.append({'name': 'f_v_values', 'v': v, 'values': values})

        def f_v_field(v, field):
            watchers.append({'name': 'f_v_field', 'v': v, 'field': field})

        def f_v_values_field(v, values, field):
            watchers.append(
                {'name': 'f_v_values_field', 'v': v, 'values': values,
                 'field': field}
            )

        def f_v_kwargs(v, **kwargs):
            watchers.append({'name': 'f_v_kwargs', 'v': v, 'kwargs': kwargs})

        def f_v_values_kwargs(v, values, **kwargs):
            watchers.append(
                {'name': 'f_v_values_kwargs', 'v': v, 'values': values,
                 'kwargs': kwargs}
            )

        def f_v_field_kwargs(v, field, **kwargs):
            watchers.append(
                {'name': 'f_v_field_kwargs', 'v': v, 'field': field,
                 'kwargs': kwargs}
            )

        def f_v_values_field_kwargs(v, values, field, **kwargs):
            watchers.append(
                {'name': 'f_v_values_field_kwargs', 'v': v, 'values': values,
                 'field': field, 'kwargs': kwargs}
            )

        fields = ('a', 'b', 'c')
        handlers = {k: v for k, v in locals().items() if k.startswith('f_')}
        self.assertEqual(len(handlers), 8)

        for _, handler in handlers.items():
            # test props
            fc = ObjectFieldChecker(fields, handler)
            self.assertEqual(fc.fields, fields)
            self.assertFalse(fc.pre)
            self.assertFalse(fc.missing)
            self.assertEqual(fc, ObjectFieldChecker(fields, handler))
            self.assertEqual(hash(fc),
                             hash(ObjectFieldChecker(fields, handler)))
            self.assertNotEqual(
                fc,
                ObjectFieldChecker(fields, handler, pre=True)
            )
            self.assertNotEqual(
                fc,
                ObjectFieldChecker(fields, handler, missing=True)
            )
            self.assertEqual(
                repr(fc),
                f'ObjectFieldChecker(fields={fields}, callback={handler}, '
                f'pre=False, missing=False)'
            )

            # test call
            for _, handler2 in handlers.items():
                if handler2 != handler:
                    self.assertNotEqual(
                        fc, ObjectFieldChecker(fields, handler2))
            fc('v', 'values', 'field')

            # test field simplifier
            fc = ObjectFieldChecker(['a', 'c', 'a', 'b'], f_v)
            self.assertEqual(fc.fields, ('a', 'c', 'b',))
            fc = ObjectFieldChecker(['a', '*', 'a', 'b'], f_v)
            self.assertEqual(fc.fields, ('*',))

        for w, k in zip(watchers, handlers):
            self.assertEqual(w['name'], k)
            self.assertEqual(w['v'], 'v')
            expected_kwargs = {}

            if '_kwargs' in k:
                self.assertIn('kwargs', w)
                kwargs = w['kwargs']
            else:
                kwargs = None

            if '_values' in k:
                self.assertEqual(w['values'], 'values')
            else:
                expected_kwargs['values'] = 'values'

            if '_field' in k:
                self.assertEqual(w['field'], 'field')
            else:
                expected_kwargs['field'] = 'field'

            if kwargs is not None:
                self.assertEqual(kwargs, expected_kwargs)

    def test_object_root_checker(self):
        watcher = []

        def f(values):
            watcher.append(values)

        def f1(values):
            pass

        # test pre = False
        fc = ObjectRootChecker(f)
        self.assertFalse(fc.pre)
        self.assertEqual(fc, ObjectRootChecker(f))
        self.assertEqual(hash(fc), hash(ObjectRootChecker(f)))
        self.assertNotEqual(fc, ObjectRootChecker(f1))
        self.assertEqual(
            repr(fc), f'ObjectRootChecker(callback={f}, pre=False)')

        fc('values')
        self.assertEqual(watcher, ['values'])

        # test pre = True
        fc = ObjectRootChecker(f, pre=True)
        self.assertTrue(fc.pre)
        self.assertNotEqual(ObjectRootChecker(f), fc)
        self.assertEqual(
            repr(fc), f'ObjectRootChecker(callback={f}, pre=True)')

    def test_ObjectDictProxy(self):
        def chk(o):
            p = _ObjectDictProxy(o)
            self.assertIs(p.value, o)
            self.assertEqual(p['a'], 123)
            self.assertTrue('a' in p)
            self.assertFalse('b' in p)
            self.assertEqual(list(p), ['a'])
            p['a'] = 456
            self.assertEqual(o.a, 456)

        # test by slots
        class MyClass1(object):
            __slots__ = ('a',)

            def __init__(self, a):
                self.a = a

            def __eq__(self, other):
                return isinstance(other, MyClass1) and other.a == self.a

        chk(MyClass1(a=123))

        # test by dict
        class MyClass2(object):
            def __init__(self, a):
                self.a = a

            def __eq__(self, other):
                return isinstance(other, MyClass2) and other.a == self.a

        chk(MyClass2(a=123))

    def test_object_type_info(self):
        class Sink(object):
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

            def __repr__(self):
                attrs = ', '.join(f'{k}={v!r}'
                                  for k, v in self.__dict__.items())
                return f'Sink({attrs})'

            def __eq__(self, other):
                return isinstance(other, Sink) and \
                    other.__dict__ == self.__dict__

        # test to str
        ti = ObjectTypeInfo(Sink, fields={})
        self.assertEqual(str(ti), Sink.__qualname__)
        self.assertEqual(
            repr(ti),
            f'<ObjectTypeInfo({Sink.__qualname__}, fields={{}}, '
            f'field_checkers=[], root_checkers=[])>'
        )

        ##################
        # test type cast #
        ##################
        @dataclass
        class MyDataClass(object):
            a: float = 123.5

        @dataclass
        class MyDataClass2(MyDataClass):
            b: Optional[int] = None

        def my_test(target_type):
            ti = ObjectTypeInfo(target_type, fields={
                'a': ObjectFieldInfo('a', type_info=type_info(float),
                                     default=123.5)
            })

            # homogenuous type conversion
            for strict in [True, False]:
                for kwargs in [{'a': 456.0}, {'b': 456}]:
                    origin = target_type(**kwargs)
                    origin_copy = deep_copy(origin)

                    # test inplace = False
                    target = ti.check_value(
                        origin, TypeCheckContext(strict=strict))
                    self.assertIsNot(target, origin)
                    target_kw = copy.copy(kwargs)
                    target_kw['a'] = float(target_kw.get('a', 123.5))
                    self.assertEqual(target, target_type(**target_kw))
                    self.assertEqual(origin, target_type(**kwargs))
                    self.assertIsNot(target, origin)

                    # origin should not be changed when inplace = False
                    self.assertEqual(origin, origin_copy)

                    # test inplace = True
                    if isinstance(origin, target_type):
                        target = ti.check_value(
                            origin,
                            TypeCheckContext(strict=strict, inplace=True)
                        )
                        self.assertIs(target, origin)
                        target_kw = copy.copy(kwargs)
                        target_kw['a'] = float(target_kw.get('a', 123.5))
                        self.assertEqual(target, target_type(**target_kw))

                    else:
                        target = ti.check_value(
                            origin,
                            TypeCheckContext(strict=strict, inplace=True)
                        )
                        self.assertIsNot(target, origin)
                        target_kw = copy.copy(kwargs)
                        target_kw['a'] = float(target_kw.get('a', 123.5))
                        self.assertEqual(target, target_type(**target_kw))

            # hetergenuous type conversion
            self.assertEqual(ti.check_value({'a': 456}),
                             target_type(a=456.0))
            self.assertEqual(ti.check_value({'b': 456}),
                             target_type(a=123.5, b=456))
            self.assertEqual(ti.check_value(MyDataClass(a=456)),
                             target_type(a=456.0))
            self.assertEqual(ti.check_value(MyDataClass()),
                             target_type(a=123.5))
            self.assertEqual(ti.check_value(Config(a=456)),
                             target_type(a=456.0))
            self.assertEqual(ti.check_value(Config(b=456)),
                             target_type(a=123.5, b=456))

            # error source type
            with pytest.raises(TypeCheckError,
                               match=f'cannot cast value into {ti}'):
                _ = ti.check_value(object())

            with pytest.raises(TypeCheckError,
                               match=f'value is not an instance of {ti}'):
                _ = ti.check_value(MyDataClass(), TypeCheckContext(strict=True))

        my_test(dict)
        my_test(Sink)
        my_test(MyDataClass2)
        my_test(Config)

        #######################
        # test missing fields #
        #######################
        # test required is True
        ti = ObjectTypeInfo(Sink, fields={
            'a': ObjectFieldInfo('a', type_info=type_info(int))
        })

        with pytest.raises(TypeCheckError,
                           match='at a: field \'a\' is required, but its value '
                                 'is not specified'):
            _ = ti.check_value({})

        with pytest.raises(TypeCheckError,
                           match='at a: field \'a\' is required, but its value '
                                 'is not specified'):
            _ = ti.check_value({'a': NOT_SET})

        self.assertEqual(
            ti.check_value({}, TypeCheckContext(ignore_missing=True)),
            Sink()
        )
        self.assertEqual(
            ti.check_value(
                {'a': NOT_SET},
                TypeCheckContext(ignore_missing=True)),
            Sink()
        )

        # test required is False
        ti = ObjectTypeInfo(Sink, fields={
            'a': ObjectFieldInfo('a', type_info=type_info(int), required=False)
        })
        self.assertEqual(ti.check_value({}), Sink())
        self.assertEqual(ti.check_value({'a': NOT_SET}), Sink())

        ##############################
        # test nested attribute name #
        ##############################
        @dataclass
        class NestedC(object):
            name: str
            value: int

        @dataclass
        class Nested(object):
            b: int
            c: NestedC

        # test without nested definition
        ti = ObjectTypeInfo(Sink, fields={})
        target = ti.check_value({
            'a': 123, 'nested.b': 456,
            'nested.c.name': 'Alice', 'nested.c.value': 999,
        })
        expected = Sink(
            a=123,
            nested={'b': 456, 'c.name': 'Alice', 'c.value': 999}
        )
        self.assertEqual(target, expected)

        # test with nested definition
        ti = ObjectTypeInfo(Sink, fields={
            'nested': ObjectFieldInfo('nested', type_info(Nested))
        })
        target = ti.check_value({
            'a': 123, 'nested.b': 456,
            'nested.c.name': 'Alice', 'nested.c.value': 999,
        })
        expected = Sink(
            a=123,
            nested=Nested(b=456, c=NestedC(name='Alice', value=999))
        )
        self.assertEqual(target, expected)

        # test nested attribute assignment error
        ti = ObjectTypeInfo(Sink, fields={})
        with pytest.raises(TypeCheckError,
                           match='at a: cannot merge a non-object value with '
                                 'an object value'):
            _ = ti.check_value({'a': 123, 'a.value': 456})
        with pytest.raises(TypeCheckError,
                           match='at a: cannot merge an object value '
                                 'with a non-object value'):
            _ = ti.check_value({'a.value': 456, 'a': 123})

        ######################
        # test default value #
        ######################
        # test default
        ti = ObjectTypeInfo(
            Sink, fields={
                'a': ObjectFieldInfo('a', type_info(int), default=123)
            }
        )
        self.assertEqual(ti.check_value({}), Sink(a=123))

        # test default factory
        ti = ObjectTypeInfo(
            Sink, fields={
                'a': ObjectFieldInfo('a', type_info(int),
                                     default_factory=lambda: 123)
            }
        )
        self.assertEqual(ti.check_value({}), Sink(a=123))

        # test envvar
        env_key = 'MLTK_TEST_A_VALUE'
        ti = ObjectTypeInfo(
            Sink, fields={
                'a': ObjectFieldInfo('a', type_info(int), default=123,
                                     envvar=env_key, ignore_empty_env=True)
            }
        )
        if env_key in os.environ:
            os.environ.pop(env_key)
        self.assertEqual(ti.check_value({}), Sink(a=123))
        os.environ[env_key] = ''
        self.assertEqual(ti.check_value({}), Sink(a=123))
        os.environ[env_key] = '456'
        self.assertEqual(ti.check_value({}), Sink(a=456))
        self.assertEqual(ti.check_value({'a': 789}), Sink(a=789))

        ################
        # test choices #
        ################
        # test not nullable
        ti = ObjectTypeInfo(
            Sink, fields={
                'a': ObjectFieldInfo('a', type_info(int), choices=[1, 2, 3])
            }
        )
        self.assertEqual(ti.check_value({'a': 1}), Sink(a=1))
        with pytest.raises(TypeCheckError,
                           match='at a: invalid value for field \'a\': '
                                 'not one of \\[1, 2, 3\\]'):
            _ = ti.check_value({'a': 4})

        # test nullable
        ti = ObjectTypeInfo(
            Sink, fields={
                'a': ObjectFieldInfo('a', type_info(Optional[int]),
                                     choices=[1, 2, 3])
            }
        )
        self.assertEqual(ti.check_value({'a': None}), Sink(a=None))

        ###################
        # test validators #
        ###################
        watchers = []

        def field_checker(v, values, field, name=NOT_SET):
            watchers.append({'name': name, 'v': v, 'values': copy.copy(values),
                             'field': field})
            if v is not NOT_SET:
                if 'pre' in name:
                    self.assertEqual(v, values[field])
                else:
                    self.assertEqual(v, getattr(values, field))
                return v + 1
            else:
                if 'pre' in name:
                    self.assertNotIn(field, values)
                else:
                    self.assertFalse(hasattr(values, field))
                return NOT_SET

        def root_checker(values, name=NOT_SET):
            watchers.append({'name': name, 'values': copy.copy(values)})
            if 'pre' in name:
                for k in values:
                    values[k] += 1
            else:
                for k in values.__dict__:
                    setattr(values, k, getattr(values, k) + 1)

        ti = ObjectTypeInfo(
            Sink, fields={
                'a': ObjectFieldInfo('a', type_info(int)),
                'b': ObjectFieldInfo('b', type_info(int), default=20),
            },
            field_checkers=[
                ObjectFieldChecker(
                    fields=['a', '*'],
                    callback=partial(field_checker, name='a_any_pre_checker'),
                    pre=True,
                    missing=True,
                ),
                ObjectFieldChecker(
                    fields=['*', 'b'],
                    callback=partial(field_checker, name='any_b_post_checker'),
                    pre=False,
                    missing=True,
                ),
                ObjectFieldChecker(
                    fields=['a', 'b'],
                    callback=partial(field_checker, name='ab_pre_checker'),
                    pre=True,
                    missing=True,
                ),
                ObjectFieldChecker(
                    fields=['a', 'b'],
                    callback=partial(field_checker, name='ab_post_checker'),
                    pre=False,
                    missing=True,
                ),
                ObjectFieldChecker(
                    fields=['b'],
                    callback=partial(field_checker, name='b_pre_checker'),
                    pre=True,
                ),
            ],
            root_checkers=[
                ObjectRootChecker(
                    callback=partial(root_checker, name='root_pre_checker'),
                    pre=True,
                ),
                ObjectRootChecker(
                    callback=partial(root_checker, name='root_post_checker'),
                    pre=False,
                ),
            ]
        )

        # test missing values
        watchers.clear()
        ret = ti.check_value({'c': 99}, TypeCheckContext(ignore_missing=True))
        self.assertListEqual(watchers, [
            {'name': 'root_pre_checker', 'values': {'c': 99}},
            {'name': 'a_any_pre_checker', 'v': 100, 'values': {'c': 100}, 'field': 'c'},
            {'name': 'a_any_pre_checker', 'v': NOT_SET, 'values': {'c': 101}, 'field': 'a'},
            {'name': 'a_any_pre_checker', 'v': NOT_SET, 'values': {'c': 101}, 'field': 'b'},
            {'name': 'ab_pre_checker', 'v': NOT_SET, 'values': {'c': 101}, 'field': 'a'},
            {'name': 'ab_pre_checker', 'v': NOT_SET, 'values': {'c': 101}, 'field': 'b'},
            {'name': 'root_post_checker', 'values': Sink(c=101, b=20)},
            {'name': 'any_b_post_checker', 'v': 102, 'values': Sink(c=102, b=21), 'field': 'c'},
            {'name': 'any_b_post_checker', 'v': 21, 'values': Sink(c=103, b=21), 'field': 'b'},
            {'name': 'any_b_post_checker', 'v': NOT_SET, 'values': Sink(c=103, b=22), 'field': 'a'},
            {'name': 'ab_post_checker', 'v': NOT_SET, 'values': Sink(c=103, b=22), 'field': 'a'},
            {'name': 'ab_post_checker', 'v': 22, 'values': Sink(c=103, b=22), 'field': 'b'},
        ])
        self.assertEqual(ret, Sink(b=23, c=103))

        # test all values provided
        watchers.clear()
        ret = ti.check_value({'a': 1, 'b': 2, 'c': 99})
        self.assertListEqual(watchers, [
            {'name': 'root_pre_checker', 'values': {'a': 1, 'b': 2, 'c': 99}},
            {'name': 'a_any_pre_checker', 'v': 2, 'values': {'a': 2, 'b': 3, 'c': 100}, 'field': 'a'},
            {'name': 'a_any_pre_checker', 'v': 3, 'values': {'a': 3, 'b': 3, 'c': 100}, 'field': 'b'},
            {'name': 'a_any_pre_checker', 'v': 100, 'values': {'a': 3, 'b': 4, 'c': 100}, 'field': 'c'},
            {'name': 'ab_pre_checker', 'v': 3, 'values': {'a': 3, 'b': 4, 'c': 101}, 'field': 'a'},
            {'name': 'ab_pre_checker', 'v': 4, 'values': {'a': 4, 'b': 4, 'c': 101}, 'field': 'b'},
            {'name': 'b_pre_checker', 'v': 5, 'values': {'a': 4, 'b': 5, 'c': 101}, 'field': 'b'},
            {'name': 'root_post_checker', 'values': Sink(a=4, b=6, c=101)},
            {'name': 'any_b_post_checker', 'v': 5, 'values': Sink(a=5, b=7, c=102), 'field': 'a'},
            {'name': 'any_b_post_checker', 'v': 7, 'values': Sink(a=6, b=7, c=102), 'field': 'b'},
            {'name': 'any_b_post_checker', 'v': 102, 'values': Sink(a=6, b=8, c=102), 'field': 'c'},
            {'name': 'ab_post_checker', 'v': 6, 'values': Sink(a=6, b=8, c=103), 'field': 'a'},
            {'name': 'ab_post_checker', 'v': 8, 'values': Sink(a=7, b=8, c=103), 'field': 'b'},
        ])
        self.assertEqual(ret, Sink(a=7, b=9, c=103))

        ################################
        # test validators: return None #
        ################################

        def field_checker(v, values, field):
            return None

        # test not nullable
        ti = ObjectTypeInfo(
            Sink, fields={
                'a': ObjectFieldInfo('a', type_info(int), required=False),
            },
            field_checkers=[
                ObjectFieldChecker(fields=['a', 'b'], callback=field_checker)
            ],
            root_checkers=[]
        )

        self.assertEqual(ti.check_value({'b': 123}), Sink(b=None))
        with pytest.raises(TypeCheckError,
                           match='The field checker for `a` returns None'):
            _ = ti.check_value({'a': 123})

        # test not nullable by wildcard field checker
        ti = ObjectTypeInfo(
            Sink, fields={
                'a': ObjectFieldInfo('a', type_info(int), required=False),
            },
            field_checkers=[
                ObjectFieldChecker(fields=['*'], callback=field_checker)
            ],
            root_checkers=[]
        )

        self.assertEqual(ti.check_value({'b': 123}), Sink(b=None))
        with pytest.raises(TypeCheckError,
                           match='The field checker for `a` returns None, '
                                 'but the field is not nullable.'):
            _ = ti.check_value({'a': 123})

        # test nullable
        ti = ObjectTypeInfo(
            Sink, fields={
                'a': ObjectFieldInfo('a', type_info(Optional[int])),
            },
            field_checkers=[
                ObjectFieldChecker(fields=['a', 'b'], callback=field_checker)
            ],
            root_checkers=[]
        )
        self.assertEqual(ti.check_value({'a': 123}), Sink(a=None))

        ###################################
        # test validators: return NOT_SET #
        ###################################

        def field_checker(v, values, field):
            return NOT_SET

        # test via non-wildcard field checker
        ti = ObjectTypeInfo(
            Sink, fields={
                'a': ObjectFieldInfo('a', type_info(int), required=False),
                'b': ObjectFieldInfo('b', type_info(int), required=False),
            },
            field_checkers=[
                ObjectFieldChecker(fields=['a'], callback=field_checker),
                ObjectFieldChecker(fields=['b'], callback=field_checker,
                                   pre=True),
            ],
            root_checkers=[]
        )

        self.assertEqual(ti.check_value({'a': 12, 'b': 34}), Sink())

        # test via wildcard field checker
        ti = ObjectTypeInfo(
            Sink, fields={
                'a': ObjectFieldInfo('a', type_info(int), required=False),
                'b': ObjectFieldInfo('b', type_info(int), required=False),
            },
            field_checkers=[
                ObjectFieldChecker(fields=['*'], callback=field_checker),
            ],
            root_checkers=[]
        )

        self.assertEqual(ti.check_value({'a': 12, 'b': 34}), Sink())

        # test via wildcard field checker, pre = True
        ti = ObjectTypeInfo(
            Sink, fields={
                'a': ObjectFieldInfo('a', type_info(int), required=False),
                'b': ObjectFieldInfo('b', type_info(int), required=False),
            },
            field_checkers=[
                ObjectFieldChecker(fields=['*'], callback=field_checker,
                                   pre=True),
            ],
            root_checkers=[]
        )

        self.assertEqual(ti.check_value({'a': 12, 'b': 34}), Sink())

        # test unset should cause error if required = True
        ti = ObjectTypeInfo(
            Sink, fields={
                'a': ObjectFieldInfo('a', type_info(int)),
            },
            field_checkers=[
                ObjectFieldChecker(fields=['*'], callback=field_checker),
            ],
            root_checkers=[]
        )

        with pytest.raises(TypeCheckError,
                           match='at a: field \'a\' is required, but its '
                                 'value is not specified'):
            _ = ti.check_value({'a': 12})

        #########################
        # test undefined fields #
        #########################
        def root_pre_checker(values):
            logs.append(dict(values))

        logs = []
        ti = ObjectTypeInfo(
            Sink,
            fields={
                'a': ObjectFieldInfo('a', type_info=type_info(int))
            },
            root_checkers=[
                ObjectRootChecker(root_pre_checker, pre=True),
            ]
        )
        parent_ti = ObjectTypeInfo(
            Sink, fields={'value': ObjectFieldInfo('value', ti)}
        )

        # discard_undefined = NO
        self.assertEqual(
            ti.check_value({'a': 123, 'b': 456}, TypeCheckContext()),
            Sink(a=123, b=456)
        )
        self.assertEqual(logs, [{'a': 123, 'b': 456}])
        logs.clear()

        # discard_undefined = SILENT
        self.assertEqual(
            ti.check_value(
                {'a': 123, 'b': 456},
                TypeCheckContext(discard_undefined=DiscardMode.SILENT)),
            Sink(a=123)
        )
        self.assertEqual(logs, [{'a': 123}])
        logs.clear()

        # discard_undefined = 'warn'
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            self.assertEqual(
                ti.check_value(
                    {'a': 123, 'b': 456},
                    TypeCheckContext(discard_undefined='warn')),
                Sink(a=123)
            )
            self.assertEqual(logs, [{'a': 123}])
            logs.clear()
        self.assertEqual(len(w), 1)
        self.assertTrue(issubclass(w[-1].category, UserWarning))
        self.assertRegex(
            str(w[-1].message),
            "undefined field 'b' is discarded"
        )

        # discard_undefined = 'warn', nested
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            self.assertEqual(
                parent_ti.check_value(
                    {'value': {'a': 123, 'b': 456}, 'c': 789},
                    TypeCheckContext(discard_undefined='warn')),
                Sink(value=Sink(a=123))
            )
        self.assertEqual(len(w), 2)
        self.assertTrue(issubclass(w[-2].category, UserWarning))
        self.assertRegex(
            str(w[-2].message),
            "undefined field 'c' is discarded"
        )
        self.assertTrue(issubclass(w[-1].category, UserWarning))
        self.assertRegex(
            str(w[-1].message),
            "at value: undefined field 'b' is discarded"
        )
