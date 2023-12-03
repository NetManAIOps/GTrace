import unittest

import numpy as np
import pytest

from mltk.data import *
from tests.helpers import remote_test


class DataLoadersTestCase(unittest.TestCase):

    @remote_test
    def test_mnist_like(self):
        # test x_shape
        (train_x, train_y), (test_x, test_y) = load_mnist(x_shape=(784,))
        self.assertTupleEqual(train_x.shape, (60000, 784))
        self.assertTupleEqual(test_x.shape, (10000, 784))

        with pytest.raises(ValueError,
                           match='`x_shape` does not product to 784'):
            _ = load_mnist(x_shape=(1, 2, 3))

        # test MNIST
        (train_x, train_y), (test_x, test_y) = load_mnist()
        self.assertTupleEqual(train_x.shape, (60000, 28, 28))
        self.assertTupleEqual(train_y.shape, (60000,))
        self.assertTupleEqual(test_x.shape, (10000, 28, 28))
        self.assertTupleEqual(test_y.shape, (10000,))
        self.assertGreater(np.max(train_x), 128.)

        # test Fashion MNIST
        (train_x, train_y), (test_x, test_y) = load_fashion_mnist()
        self.assertTupleEqual(train_x.shape, (60000, 28, 28))
        self.assertTupleEqual(train_y.shape, (60000,))
        self.assertTupleEqual(test_x.shape, (10000, 28, 28))
        self.assertTupleEqual(test_y.shape, (10000,))
        self.assertGreater(np.max(train_x), 128.)

    @remote_test
    def test_cifar10(self):
        # test x_shape
        (train_x, train_y), (test_x, test_y) = load_cifar10(x_shape=(1024, 3))
        self.assertTupleEqual(train_x.shape, (50000, 1024, 3))
        self.assertTupleEqual(test_x.shape, (10000, 1024, 3))

        with pytest.raises(ValueError,
                           match='`x_shape` does not product to 3072'):
            _ = load_cifar10(x_shape=(1, 2, 3))

            # test fetch
        (train_x, train_y), (test_x, test_y) = load_cifar10()
        self.assertTupleEqual(train_x.shape, (50000, 32, 32, 3))
        self.assertTupleEqual(train_y.shape, (50000,))
        self.assertTupleEqual(test_x.shape, (10000, 32, 32, 3))
        self.assertTupleEqual(test_y.shape, (10000,))

        self.assertGreater(np.max(train_x), 128.)
        self.assertEqual(np.max(train_y), 9)

    @remote_test
    def test_cifar100(self):
        # test x_shape
        (train_x, train_y), (test_x, test_y) = load_cifar100(x_shape=(1024, 3))
        self.assertTupleEqual(train_x.shape, (50000, 1024, 3))
        self.assertTupleEqual(test_x.shape, (10000, 1024, 3))

        with pytest.raises(ValueError,
                           match='`x_shape` does not product to 3072'):
            _ = load_cifar100(x_shape=(1, 2, 3))

        # test fetch
        (train_x, train_y), (test_x, test_y) = load_cifar100()
        self.assertTupleEqual(train_x.shape, (50000, 32, 32, 3))
        self.assertTupleEqual(train_y.shape, (50000,))
        self.assertTupleEqual(test_x.shape, (10000, 32, 32, 3))
        self.assertTupleEqual(test_y.shape, (10000,))

        self.assertGreater(np.max(train_x), 128.)
        self.assertEqual(np.max(train_y), 99)
