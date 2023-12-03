import os
import unittest
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from mltk import *


class StatefulObjectTestCase(unittest.TestCase):

    def test_object_group(self):
        a = SimpleStatefulObject()
        b = SimpleStatefulObject()
        c = SimpleStatefulObject()

        # test argument validation
        with pytest.raises(ValueError,
                           match='invalid key \'\': must be a non-empty str '
                                 'without "."'):
            _ = StatefulObjectGroup({'': a})

        with pytest.raises(ValueError,
                           match='invalid key \'a\\.\': must be a non-empty '
                                 'str without "\\."'):
            _ = StatefulObjectGroup({'a.': a})

        with pytest.raises(TypeError,
                           match='item \'a\' is not a StatefulObject: '
                                 '<object.*>'):
            _ = StatefulObjectGroup({'a': object()})

        # test set_state_dict
        obj = StatefulObjectGroup({'a': a, 'b': b})
        a.value = 123
        b.value = 456

        with pytest.raises(ValueError,
                           match='invalid state key \'a\': no "."'):
            obj.set_state_dict({'a': 0xdeadbeef})

        # test add object
        obj = StatefulObjectGroup({'a': a, 'b': b})
        a.value = 123
        b.value = 456
        c.value = 789

        with pytest.raises(ValueError,
                           match='`prefix` already exists: \'a\''):
            obj.add_object('a', c)

        with pytest.raises(TypeError,
                           match='`obj` is not a StatefulObject: <object.*>'):
            obj.add_object('c', object())

        obj.add_object('c', c)
        self.assertEqual(obj.get_state_dict(),
                         {'a.value': 123, 'b.value': 456, 'c.value': 789})

        # strict mode, should raise error in the following two cases
        obj = StatefulObjectGroup({'a': a, 'b': b}, strict=True)
        with pytest.raises(ValueError,
                           match='invalid state key \'c\\.value\': does not '
                                 'correspond to any object'):
            obj.set_state_dict({'c.value': 0xdeadbeef})

        with pytest.raises(ValueError,
                           match='state for object \'a\' is missing'):
            obj.set_state_dict({'b.value': 4560})

        self.assertDictEqual(a.get_state_dict(), {'value': 123})
        self.assertDictEqual(b.get_state_dict(), {'value': 456})

        # non-strict mode
        obj = StatefulObjectGroup({'a': a, 'b': b})
        obj.set_state_dict({
            'a.value': 1230,
            'b.value': 4560,
            'c.value': 0xdeadbeef,
        })
        self.assertDictEqual(a.get_state_dict(), {'value': 1230})
        self.assertDictEqual(b.get_state_dict(), {'value': 4560})

    def test_state_saver(self):
        with TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, 'state.npz')

            # save the state
            a = SimpleStatefulObject()
            a.value = (123, 'hello')
            b = SimpleStatefulObject()
            b.value = np.arange(5)

            saver = StateSaver({'a': a, 'b': b})
            saver.save(path)

            # load the state
            a = SimpleStatefulObject()
            b = SimpleStatefulObject()
            obj = StatefulObjectGroup({'a': a, 'b': b})

            saver = StateSaver(obj)
            saver.load(path)

            self.assertEqual(a.value, (123, 'hello'))
            np.testing.assert_equal(b.value, np.arange(5))

            # test invalid filename
            with pytest.raises(ValueError,
                               match='`file_path` must end with ".npz"'):
                saver.save(path + '.dat')

            # test invalid state key
            with pytest.raises(ValueError,
                               match='invalid state key \'.value\': "." '
                                     'cannot be the first character'):
                obj = SimpleStatefulObject()
                saver = StateSaver(obj)
                setattr(obj, '.value', 123)
                saver.save(os.path.join(temp_dir, 'invalid.npz'))
