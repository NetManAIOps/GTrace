import copy
import random
import unittest

import pytest
from mltk import *


class FormatKeyValuesTestCase(unittest.TestCase):

    def test_format_key_values(self):
        with pytest.raises(ValueError,
                           match='`delimiter_char` must be one character: '
                                 'got \'xx\''):
            format_key_values({'a': 1}, delimiter_char='xx')


class MetricsFormatterTestCase(unittest.TestCase):

    def test_sorted_names(self):
        array = [
            # non-timer metrics
            'acc', 'loss',
            'train_acc', 'train_loss',
            'val_acc',  'val_loss',
            'valid_acc', 'valid_loss',
            'test_acc', 'test_loss',
            'pred_acc', 'pred_loss',
            'predict_acc', 'predict_loss',
            'epoch_acc', 'epoch_loss',
            'batch_acc', 'batch_loss',

            # timer metrics
            'train_time', 'val_time', 'valid_time', 'test_time',
            'pred_time', 'predict_time', 'epoch_time', 'batch_time',
            'train_timer', 'val_timer', 'valid_timer', 'test_timer',
            'pred_timer', 'predict_timer', 'epoch_timer', 'batch_timer',
        ]

        fmt = MetricsFormatter()
        self.assertEqual(fmt.sorted_names(array), array)
        self.assertEqual(fmt.sorted_names(reversed(array)), array)
        array2 = copy.copy(array)
        random.shuffle(array2)
        self.assertEqual(fmt.sorted_names(array2), array)
