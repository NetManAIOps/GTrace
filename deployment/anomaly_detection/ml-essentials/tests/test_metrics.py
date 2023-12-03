import unittest

import numpy as np
import pytest

from mltk.metrics import *


class MetricCollectorTestCase(unittest.TestCase):

    def test_empty_scalar(self):
        collector = GeneralMetricCollector()
        self.assertEqual(collector.shape, ())
        self.assertIsNone(collector.stats)
        self.assertFalse(collector.has_stats)
        self.assertEqual(collector.counter, 0)
        self.assertAlmostEqual(collector.mean, 0.)
        self.assertAlmostEqual(collector.second_order_moment, 0.)
        self.assertEqual(collector.weight_sum, 0.)

    def test_empty_vector(self):
        collector = GeneralMetricCollector(shape=(3, 2))
        self.assertEqual(collector.shape, (3, 2))
        self.assertFalse(collector.has_stats)
        self.assertEqual(collector.counter, 0)
        np.testing.assert_almost_equal(collector.mean, np.zeros([3, 2]))
        np.testing.assert_almost_equal(collector.second_order_moment, np.zeros([3, 2]))
        self.assertEqual(collector.weight_sum, 0.)

    def test_scalar_collect(self):
        collector = GeneralMetricCollector()
        collector.update([2, 1, 7, 6])
        self.assertTrue(collector.has_stats)
        self.assertEqual(collector.counter, 4)
        self.assertAlmostEqual(collector.mean, 4.)
        self.assertAlmostEqual(collector.stats.mean, 4.)
        self.assertAlmostEqual(collector.second_order_moment, 22.5)
        self.assertAlmostEqual(collector.weight_sum, 4.)

    def test_var_std(self):
        collector = GeneralMetricCollector()
        collector.update([2, 1, 7, 6])
        self.assertEqual(collector.counter, 4)
        self.assertAlmostEqual(collector.stats.var, 6.5)
        self.assertAlmostEqual(collector.stats.std, 2.549509756796)

    def test_scalar_multi_collect(self):
        collector = GeneralMetricCollector()
        collector.update(2)
        collector.update(1, weight=3)
        collector.update(7, weight=6)
        self.assertTrue(collector.has_stats)
        self.assertEqual(collector.counter, 3)
        self.assertAlmostEqual(collector.mean, 4.7)
        self.assertAlmostEqual(collector.second_order_moment, 30.1)
        self.assertAlmostEqual(collector.weight_sum, 10.)

    def test_reset(self):
        collector = GeneralMetricCollector()
        collector.update([2, 1, 7, 6])
        collector.reset()
        self.assertFalse(collector.has_stats)
        self.assertEqual(collector.counter, 0)
        self.assertAlmostEqual(collector.mean, 0.)
        self.assertAlmostEqual(collector.second_order_moment, 0.)
        self.assertAlmostEqual(collector.weight_sum, 0.)
        self.assertIsNone(collector.stats)

    def test_scalar_collect_batch(self):
        collector = GeneralMetricCollector()
        collector.update([2, 1, 7], weight=[1, 3, 6])
        self.assertTrue(collector.has_stats)
        self.assertEqual(collector.counter, 3)
        self.assertAlmostEqual(collector.mean, 4.7)
        self.assertAlmostEqual(collector.stats.mean, 4.7)
        self.assertAlmostEqual(collector.second_order_moment, 30.1)
        self.assertAlmostEqual(collector.weight_sum, 10.)

    def test_scalar_collect_batch_weight_broadcast(self):
        collector = GeneralMetricCollector()
        collector.update([2, 1, 7, 6], weight=1.)
        self.assertTrue(collector.has_stats)
        self.assertEqual(collector.counter, 4)
        self.assertAlmostEqual(collector.mean, 4.)
        self.assertAlmostEqual(collector.stats.mean, 4.)
        self.assertAlmostEqual(collector.second_order_moment, 22.5)
        self.assertAlmostEqual(collector.weight_sum, 4.)

    def test_vector_collect(self):
        collector = GeneralMetricCollector(shape=(3, 2))
        arr = np.arange(6).reshape([3, 2])
        collector.update(arr)
        self.assertTrue(collector.has_stats)
        self.assertEqual(collector.counter, 1)
        np.testing.assert_almost_equal(collector.mean, arr)
        np.testing.assert_almost_equal(collector.stats.mean, arr)
        np.testing.assert_almost_equal(collector.second_order_moment, arr ** 2)
        self.assertAlmostEqual(collector.weight_sum, 1.)

    def test_vector_multi_collect(self):
        collector = GeneralMetricCollector(shape=(3, 2))
        arr = np.arange(24).reshape([4, 3, 2])
        collector.update(arr[0])
        collector.update(arr[1], weight=2)
        collector.update(arr[2], weight=3)
        collector.update(arr[3], weight=4)
        self.assertTrue(collector.has_stats)
        self.assertEqual(collector.counter, 4)
        np.testing.assert_almost_equal(
            collector.mean,
            [[12, 13], [14, 15], [16, 17]]
        )
        np.testing.assert_almost_equal(
            collector.second_order_moment,
            [[180., 205.], [232., 261.], [292., 325.]]
        )
        np.testing.assert_almost_equal(
            collector.stats.var,
            np.maximum(collector.second_order_moment - collector.mean ** 2, 0.)
        )
        np.testing.assert_almost_equal(
            collector.stats.std,
            np.sqrt(np.maximum(collector.second_order_moment - collector.mean ** 2, 0.))
        )
        self.assertAlmostEqual(collector.weight_sum, 10.)

    def test_vector_collect_batch(self):
        collector = GeneralMetricCollector(shape=(3, 2))
        arr = np.arange(24).reshape([4, 3, 2])
        collector.update(arr, weight=[1, 2, 3, 4])
        self.assertTrue(collector.has_stats)
        self.assertEqual(collector.counter, 4)
        np.testing.assert_almost_equal(
            collector.mean,
            [[12, 13], [14, 15], [16, 17]]
        )
        np.testing.assert_almost_equal(
            collector.second_order_moment,
            [[180., 205.], [232., 261.], [292., 325.]]
        )
        self.assertAlmostEqual(collector.weight_sum, 10.)

    def test_vector_collect_batch_weight_broadcast(self):
        collector = GeneralMetricCollector(shape=(3, 2))
        arr = np.arange(24).reshape([4, 3, 2])
        collector.update(arr, weight=1.)
        self.assertTrue(collector.has_stats)
        self.assertEqual(collector.counter, 4)
        np.testing.assert_almost_equal(
            collector.mean,
            [[9, 10], [11, 12], [13, 14]]
        )
        np.testing.assert_almost_equal(
            collector.second_order_moment,
            [[126., 145.], [166., 189.], [214., 241.]]
        )
        self.assertAlmostEqual(collector.weight_sum, 4.)

    def test_collect_empty(self):
        collector = GeneralMetricCollector()
        collector.update([])
        self.assertEqual(collector.shape, ())
        self.assertFalse(collector.has_stats)
        self.assertEqual(collector.counter, 0)
        self.assertAlmostEqual(collector.mean, 0.)
        self.assertAlmostEqual(collector.second_order_moment, 0.)
        self.assertEqual(collector.weight_sum, 0.)
        self.assertIsNone(collector.stats)

    def test_collect_empty_weight(self):
        collector = GeneralMetricCollector()
        collector.update([2, 1, 7, 6], weight=[])
        self.assertTrue(collector.has_stats)
        self.assertEqual(collector.counter, 4)
        self.assertAlmostEqual(collector.mean, 4.)
        self.assertAlmostEqual(collector.stats.mean, 4.)
        self.assertAlmostEqual(collector.second_order_moment, 22.5)
        self.assertAlmostEqual(collector.weight_sum, 4.)

    def test_shape_mismatch(self):
        collector = GeneralMetricCollector(shape=(3, 2))
        with pytest.raises(
                ValueError,
                match=r'Shape mismatch: \(3,\) not ending with \(3, 2\)'):
            collector.update([1, 2, 3])


