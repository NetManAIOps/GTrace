import unittest
import warnings

import pytest

from mltk.utils import *


class DeprecatedArgTestCase(unittest.TestCase):

    def test_basic(self):
        @deprecated_arg('a')
        def f(*, a=None):
            pass

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')

            f()
            self.assertEqual(len(w), 0)

            f(a='123')
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[-1].category, DeprecationWarning))
            self.assertRegex(
                str(w[-1].message),
                r'The argument `a` is deprecated.'
            )

    def test_with_message(self):
        @deprecated_arg('a', message='xyz')
        def f(*, a=None):
            pass

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')

            f()
            self.assertEqual(len(w), 0)

            f(a='123')
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[-1].category, DeprecationWarning))
            self.assertRegex(
                str(w[-1].message),
                r'The argument `a` is deprecated: xyz.'
            )

    def test_with_new_name(self):
        @deprecated_arg('a', 'b')
        def f(*, b, a=None):
            return b

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')

            self.assertEqual(f(b=123), 123)
            self.assertEqual(len(w), 0)

            self.assertEqual(f(a=123), 123)
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[-1].category, DeprecationWarning))
            self.assertRegex(
                str(w[-1].message),
                r'The argument `a` is deprecated, use `b` instead.'
            )

            with pytest.raises(ValueError,
                               match='Values are specified to both the '
                                     'deprecated argument `a` and the new '
                                     'argument `b` to replace it.'):
                f(a=123, b=456)
            self.assertEqual(len(w), 2)
            self.assertTrue(issubclass(w[-1].category, DeprecationWarning))
            self.assertRegex(
                str(w[-1].message),
                r'The argument `a` is deprecated, use `b` instead.'
            )
