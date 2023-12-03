import unittest
from functools import partial

import numpy as np
import pytest

from mltk import *


class BatchAggregatorTestCase(unittest.TestCase):

    def test_construct(self):
        # check default axis for different modes
        agg = BatchAggregator(mode='CONCAT')
        self.assertEqual(agg.mode, BatchAggregationMode.CONCAT)
        self.assertEqual(agg.axis, 0)
        self.assertEqual(repr(agg), 'BatchAggregator(mode=CONCAT, axis=0)')

        agg = BatchAggregator(mode='SUM')
        self.assertEqual(agg.mode, BatchAggregationMode.SUM)
        self.assertEqual(agg.axis, None)
        self.assertEqual(repr(agg), 'BatchAggregator(mode=SUM, axis=None)')

        agg = BatchAggregator(mode='AVERAGE')
        self.assertEqual(agg.mode, BatchAggregationMode.AVERAGE)
        self.assertEqual(agg.axis, None)

        # manually specifying axis
        agg = BatchAggregator(mode=BatchAggregationMode.AVERAGE,
                              axis=1)
        self.assertEqual(agg.axis, 1)
        agg = BatchAggregator(mode=BatchAggregationMode.AVERAGE,
                              axis=[-1])
        self.assertEqual(agg.axis, -1)
        agg = BatchAggregator(mode=BatchAggregationMode.AVERAGE,
                              axis=[0, 1])
        self.assertEqual(agg.axis, (0, 1))
        self.assertEqual(repr(agg), 'BatchAggregator(mode=AVERAGE, axis=(0, 1))')

        # invalid axis for CONCAT
        with pytest.raises(TypeError,
                           match='`axis` must be a int when `mode` is CONCAT'):
            _ = BatchAggregator(mode='CONCAT', axis=())
        with pytest.raises(TypeError,
                           match='`axis` must be a int when `mode` is CONCAT'):
            _ = BatchAggregator(mode='CONCAT', axis=None)

    def test_concat(self):
        # test empty get
        agg = BatchAggregator('CONCAT')
        self.assertIsNone(agg.get())

        # test concat on axis 0
        batch_arrays = [
            np.random.normal(size=[5, 2]),
            np.random.normal(size=[5, 2]),
            np.random.normal(size=[3, 2]),
        ]
        full_array = np.concatenate(batch_arrays, axis=0)
        agg = BatchAggregator('CONCAT', 0)
        for arr in batch_arrays:
            agg.add(arr)
        np.testing.assert_equal(agg.get(), full_array)

        # test concat on axis -1
        batch_arrays = [
            np.random.normal(size=[5, 2]),
            np.random.normal(size=[5, 3]),
        ]
        full_array = np.concatenate(batch_arrays, axis=-1)
        agg = BatchAggregator('CONCAT', -1)
        for arr in batch_arrays:
            agg.add(arr)
        np.testing.assert_equal(agg.get(), full_array)

    def test_sum(self):
        agg = BatchAggregator('SUM')
        self.assertIsNone(agg.get())

        def sum_up(arrays, axis, weights=None):
            arrays = [np.sum(arr, axis=axis) for arr in arrays]
            if weights:
                for i in range(len(arrays)):
                    arrays[i] = arrays[i] * weights[i]
            return sum(arrays)

        # test sum on axis 0
        batch_arrays = [
            np.random.normal(size=[5, 4, 2]),
            np.random.normal(size=[5, 4, 2]),
            np.random.normal(size=[3, 4, 2]),
        ]
        summed_array = sum_up(batch_arrays, axis=0)
        agg = BatchAggregator('SUM', 0)
        for arr in batch_arrays:
            agg.add(arr)
        np.testing.assert_allclose(agg.get(), summed_array)

        # test sum on axis -1
        batch_arrays = [
            np.random.normal(size=[5, 4, 2]),
            np.random.normal(size=[5, 4, 3]),
        ]
        summed_array = sum_up(batch_arrays, axis=-1)
        agg = BatchAggregator('SUM', -1)
        for arr in batch_arrays:
            agg.add(arr)
        np.testing.assert_allclose(agg.get(), summed_array)

        # test sum on axis [0, -1]
        batch_arrays = [
            np.random.normal(size=[5, 4, 2]),
            np.random.normal(size=[6, 4, 3]),
        ]
        summed_array = sum_up(batch_arrays, axis=(0, -1))
        agg = BatchAggregator('SUM', [0, -1])
        for arr in batch_arrays:
            agg.add(arr)
        np.testing.assert_allclose(agg.get(), summed_array)

        # test sum on all axis
        batch_arrays = [
            np.random.normal(size=[5, 4, 2]),
            np.random.normal(size=[6, 4, 3]),
        ]
        summed_array = sum_up(batch_arrays, axis=None)
        agg = BatchAggregator('SUM', None)
        for arr in batch_arrays:
            agg.add(arr)
        np.testing.assert_allclose(agg.get(), summed_array)

        # sum should ignore the weights
        batch_arrays = [
            np.random.normal(size=[5, 4, 2]),
            np.random.normal(size=[3, 4, 2]),
        ]
        weights = [3, 5]
        summed_array = sum_up(batch_arrays, axis=0)
        agg = BatchAggregator('SUM', 0)
        for arr, weight in zip(batch_arrays, weights):
            agg.add(arr, weight)
        np.testing.assert_allclose(agg.get(), summed_array)

    def test_average(self):
        agg = BatchAggregator('AVERAGE')
        self.assertIsNone(agg.get())

        def average_up(arrays, axis, weights=None):
            def get_size(arr):
                ret = 1.
                if axis is None:
                    for s in arr.shape:
                        ret *= s
                elif isinstance(axis, int):
                    ret = float(arr.shape[axis])
                else:
                    for a in axis:
                        ret *= arr.shape[a]
                return ret

            sizes = [get_size(arr) for arr in arrays]
            arrays = [np.sum(arr, axis=axis) for arr in arrays]
            if weights:
                for i in range(len(arrays)):
                    arrays[i] *= weights[i]
                    sizes[i] *= weights[i]
            return sum(arrays) / sum(sizes)

        # test average on axis 0
        batch_arrays = [
            np.random.normal(size=[5, 4, 2]),
            np.random.normal(size=[5, 4, 2]),
            np.random.normal(size=[3, 4, 2]),
        ]
        averaged_array = average_up(batch_arrays, axis=0)
        agg = BatchAggregator('AVERAGE', 0)
        for arr in batch_arrays:
            agg.add(arr)
        np.testing.assert_allclose(agg.get(), averaged_array)

        # test average on axis -1
        batch_arrays = [
            np.random.normal(size=[5, 4, 2]),
            np.random.normal(size=[5, 4, 3]),
        ]
        averaged_array = average_up(batch_arrays, axis=-1)
        agg = BatchAggregator('AVERAGE', -1)
        for arr in batch_arrays:
            agg.add(arr)
        np.testing.assert_allclose(agg.get(), averaged_array)

        # test average on axis [0, -1]
        batch_arrays = [
            np.random.normal(size=[5, 4, 2]),
            np.random.normal(size=[6, 4, 3]),
        ]
        averaged_array = average_up(batch_arrays, axis=(0, -1))
        agg = BatchAggregator('AVERAGE', [0, -1])
        for arr in batch_arrays:
            agg.add(arr)
        np.testing.assert_allclose(agg.get(), averaged_array)

        # test average on all axis
        batch_arrays = [
            np.random.normal(size=[5, 4, 2]),
            np.random.normal(size=[6, 4, 3]),
        ]
        averaged_array = average_up(batch_arrays, axis=None)
        agg = BatchAggregator('AVERAGE', None)
        for arr in batch_arrays:
            agg.add(arr)
        np.testing.assert_allclose(agg.get(), averaged_array)

        # weighted average
        batch_arrays = [
            np.random.normal(size=[5, 4, 2]),
            np.random.normal(size=[3, 4, 2]),
        ]
        weights = [3, 5]
        averaged_array = average_up(batch_arrays, axis=0, weights=weights)
        agg = BatchAggregator('AVERAGE', 0)
        for arr, weight in zip(batch_arrays, weights):
            agg.add(arr, weight)
        np.testing.assert_allclose(agg.get(), averaged_array)


class BatchAggregatorDictTestCase(unittest.TestCase):

    def test_instances(self):
        a = BatchAggregator('SUM')
        b = BatchAggregator('AVERAGE')

        # test without factory
        agg_dict = BatchAggregatorDict({'a': a, 'b': b})
        self.assertEqual(len(agg_dict), 2)
        self.assertEqual(list(agg_dict), ['a', 'b'])

        self.assertIs(agg_dict['a'], a)
        self.assertIs(agg_dict['b'], b)
        with pytest.raises(KeyError, match='c'):
            _ = agg_dict['c']

        self.assertIs(agg_dict.get('a'), a)
        self.assertIs(agg_dict.get('b'), b)
        self.assertIsNone(agg_dict.get('c'))

        # test with factory
        agg_dict = BatchAggregatorDict(
            {'a': a, 'b': b},
            default_factory=partial(BatchAggregator, 'SUM', (0, 1))
        )
        self.assertEqual(len(agg_dict), 2)
        self.assertEqual(list(agg_dict), ['a', 'b'])

        self.assertIs(agg_dict['a'], a)
        self.assertIs(agg_dict['b'], b)
        c = agg_dict['c']
        self.assertEqual(repr(c), 'BatchAggregator(mode=SUM, axis=(0, 1))')
        self.assertEqual(len(agg_dict), 3)
        self.assertEqual(list(agg_dict), ['a', 'b', 'c'])

        self.assertIs(agg_dict.get('a'), a)
        self.assertIs(agg_dict.get('b'), b)
        self.assertIs(agg_dict.get('c'), c)
        d = agg_dict.get('d')
        self.assertEqual(repr(d), 'BatchAggregator(mode=SUM, axis=(0, 1))')
        self.assertEqual(len(agg_dict), 4)
        self.assertEqual(list(agg_dict), ['a', 'b', 'c', 'd'])

        # test excludes
        agg_dict = BatchAggregatorDict(
            {'a': a, 'b': b},
            default_factory=partial(BatchAggregator, 'SUM', (0, 1)),
            excludes=('a', 'c')
        )
        for key in ('a', 'c'):
            self.assertIsNone(agg_dict.get(key))
            with pytest.raises(KeyError, match=key):
                _ = agg_dict[key]

        # test type error
        with pytest.raises(TypeError,
                           match='Item \'a\' is not an instance of '
                                 'BatchAggregator: 123'):
            _ = BatchAggregatorDict({'a': 123})

    def test_new(self):
        def summarize(agg_dict: BatchAggregatorDict):
            aggregators = {k: str(v) for k, v in agg_dict.items()}
            excludes = (sorted(agg_dict._excludes)
                        if agg_dict._excludes else [])
            default = str(
                agg_dict._default_factory and agg_dict._default_factory())
            return aggregators, excludes, default

        self.assertEqual(
            summarize(BatchAggregatorDict.new()),
            ({}, [], 'BatchAggregator(mode=AVERAGE, axis=None)')
        )
        self.assertEqual(
            summarize(BatchAggregatorDict.new(metrics=())),
            ({}, [], 'None')
        )
        self.assertEqual(
            summarize(BatchAggregatorDict.new(metrics=(), outputs=ALL)),
            ({}, [], 'BatchAggregator(mode=CONCAT, axis=0)')
        )
        self.assertEqual(
            summarize(BatchAggregatorDict.new(
                metrics=['a', 'b', 'c'],
                outputs=['a', 'b'],
                aggregators={
                    'a': BatchAggregator('SUM', axis=(0, 1))
                }
            )),
            (
                {
                    'a': 'BatchAggregator(mode=SUM, axis=(0, 1))',
                    'b': 'BatchAggregator(mode=CONCAT, axis=0)',
                    'c': 'BatchAggregator(mode=AVERAGE, axis=None)',
                },
                [],
                'None'
            )
        )
        self.assertEqual(
            summarize(BatchAggregatorDict.new(
                outputs=['a', 'b'],
                aggregators={
                    'a': BatchAggregator('SUM', axis=(0, 1))
                }
            )),
            (
                {
                    'a': 'BatchAggregator(mode=SUM, axis=(0, 1))',
                    'b': 'BatchAggregator(mode=CONCAT, axis=0)',
                },
                [],
                'BatchAggregator(mode=AVERAGE, axis=None)'
            )
        )
        self.assertEqual(
            summarize(BatchAggregatorDict.new(
                outputs=['a', 'b'],
                aggregators={
                    'a': BatchAggregator('SUM', axis=(0, 1))
                },
                excludes=['b', 'd']
            )),
            (
                {
                    'a': 'BatchAggregator(mode=SUM, axis=(0, 1))',
                },
                ['b', 'd'],
                'BatchAggregator(mode=AVERAGE, axis=None)'
            )
        )
        self.assertEqual(
            summarize(BatchAggregatorDict.new(
                metrics=['c'],
                outputs=['test_a', 'b'],
                aggregators={
                    'a': BatchAggregator('SUM', axis=(0, 1))
                },
                stage_type=StageType.TEST,
            )),
            (
                {
                    'test_a': 'BatchAggregator(mode=SUM, axis=(0, 1))',
                    'test_b': 'BatchAggregator(mode=CONCAT, axis=0)',
                    'test_c': 'BatchAggregator(mode=AVERAGE, axis=None)',
                },
                [],
                'None'
            )
        )
        self.assertEqual(
            summarize(BatchAggregatorDict.new(
                metrics=['c'],
                outputs=['test_a', 'b'],
                aggregators={
                    'a': BatchAggregator('SUM', axis=(0, 1))
                },
                excludes=['a', 'b', 'c'],
                stage_type=StageType.TEST,
            )),
            (
                {},
                ['test_a', 'test_b', 'test_c'],
                'None'
            )
        )
