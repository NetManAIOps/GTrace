"""
Simple dataset loaders.

For more datasets and more comprehensive loaders, you may turn to dedicated
libraries like `fuel`.
"""

import gzip
import hashlib
import os
import pickle
from typing import *

import idx2numpy
import numpy as np

from ..typing_ import *
from ..utils import CacheDir, validate_enum_arg

__all__ = ['load_mnist', 'load_fashion_mnist', 'load_cifar10', 'load_cifar100']

_MNIST_LIKE_FILE_NAMES = {
    'train_x': 'train-images-idx3-ubyte.gz',
    'train_y': 'train-labels-idx1-ubyte.gz',
    'test_x': 't10k-images-idx3-ubyte.gz',
    'test_y': 't10k-labels-idx1-ubyte.gz',
}
_MNIST_URI_PREFIX = 'http://yann.lecun.com/exdb/mnist/'
_MNIST_FILE_MD5 = {
    'train_x': 'f68b3c2dcbeaaa9fbdd348bbdeb94873',
    'train_y': 'd53e105ee54ea40749a09fcbcd1e9432',
    'test_x': '9fb629c4189551a2d022fa330f9573f3',
    'test_y': 'ec29112dd5afa0611ce80d1b7f02629c',
}
_FASHION_MNIST_URI_PREFIX = 'http://fashion-mnist.s3-website.eu-central-1.' \
                            'amazonaws.com/'
_FASHION_MNIST_FILE_MD5 = {
    'train_x': '8d4fb7e6c68d591d4c3dfef9ec88bf0d',
    'train_y': '25c81989df183df01b3e8a0aad5dffbe',
    'test_x': 'bef4ecab320f06d8554ea6380940ec79',
    'test_y': 'bb300cfdad3c16e7a12a480ee83cd310',
}

_CIFAR_10_URI = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
_CIFAR_10_MD5 = 'c58f30108f718f92721af3b95e74349a'
_CIFAR_10_CONTENT_DIR = 'cifar-10-batches-py'
_CIFAR_100_URI = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
_CIFAR_100_MD5 = 'eb9058c3a382ffc7106e4002c42a8d85'
_CIFAR_100_CONTENT_DIR = 'cifar-100-python'


def _validate_x_shape(shape, default_shape):
    shape = tuple(int(v) for v in shape)
    default_shape = tuple(int(v) for v in default_shape)
    value_size = int(np.prod(default_shape))

    if np.prod(shape) != value_size:
        raise ValueError(f'`x_shape` does not product to {value_size}: {shape}')
    return shape


def load_mnist_like(uri_prefix: str,
                    file_md5: Dict[str, str],
                    cache_name: str,
                    x_shape: Sequence[int] = (28, 28),
                    x_dtype: ArrayDType = np.uint8,
                    y_dtype: ArrayDType = np.int32
                    ) -> Tuple[XYArrayTuple, XYArrayTuple]:
    """
    Load an MNIST-like dataset as NumPy arrays.

    Args:
        uri_prefix: Common prefix of the URIs in `remote_files`.
        file_md5: The remote file MD5 hash sums, a dict of
            `{'train_x': ..., 'train_y': ..., 'test_x': ..., 'test_y': ...}`,
            where each value is the md5 sum.
        cache_name: Name of the cache directory.
        x_shape: Reshape each digit into this shape.
        x_dtype: Cast each digit into this data type.
        y_dtype: Cast each label into this data type.

    Returns:
        The ``(train_x, train_y), (test_x, test_y)`` arrays.
    """

    def _fetch_array(array_name):
        uri = uri_prefix + _MNIST_LIKE_FILE_NAMES[array_name]
        md5 = file_md5[array_name]
        path = CacheDir(cache_name).download(
            uri, hasher=hashlib.md5(), expected_hash=md5)
        with gzip.open(path, 'rb') as f:
            return idx2numpy.convert_from_file(f)

    # check arguments
    x_shape = _validate_x_shape(x_shape, (28, 28))

    # load data
    train_x = _fetch_array('train_x').astype(x_dtype)
    train_y = _fetch_array('train_y').astype(y_dtype)
    test_x = _fetch_array('test_x').astype(x_dtype)
    test_y = _fetch_array('test_y').astype(y_dtype)

    assert(len(train_x) == len(train_y) == 60000)
    assert(len(test_x) == len(test_y) == 10000)

    # change shape
    train_x = train_x.reshape([len(train_x)] + list(x_shape))
    test_x = test_x.reshape([len(test_x)] + list(x_shape))

    return (train_x, train_y), (test_x, test_y)


def load_mnist(x_shape: Sequence[int] = (28, 28),
               x_dtype: ArrayDType = np.uint8,
               y_dtype: ArrayDType = np.int32
               ) -> Tuple[XYArrayTuple, XYArrayTuple]:
    """
    Load an MNIST dataset as NumPy arrays.

    Args:
        x_shape: Reshape each digit into this shape.
        x_dtype: Cast each digit into this data type.
        y_dtype: Cast each label into this data type.

    Returns:
        The ``(train_x, train_y), (test_x, test_y)`` arrays.
    """
    return load_mnist_like(
        _MNIST_URI_PREFIX, _MNIST_FILE_MD5, 'mnist', x_shape, x_dtype, y_dtype)


def load_fashion_mnist(x_shape: Sequence[int] = (28, 28),
                       x_dtype: ArrayDType = np.uint8,
                       y_dtype: ArrayDType = np.int32
                       ) -> Tuple[XYArrayTuple, XYArrayTuple]:
    """
    Load an MNIST dataset as NumPy arrays.

    Args:
        x_shape: Reshape each digit into this shape.
        x_dtype: Cast each digit into this data type.
        y_dtype: Cast each label into this data type.

    Returns:
        The ``(train_x, train_y), (test_x, test_y)`` arrays.
    """
    return load_mnist_like(
        _FASHION_MNIST_URI_PREFIX, _FASHION_MNIST_FILE_MD5, 'fashion_mnist',
        x_shape, x_dtype, y_dtype)


def _cifar_load_batch(path, x_shape, x_dtype, y_dtype, expected_batch_label,
                      labels_key='labels'):
    # load from file
    with open(path, 'rb') as f:
        d = {
            k.decode('utf-8'): v
            for k, v in pickle.load(f, encoding='bytes').items()
        }
        d['batch_label'] = d['batch_label'].decode('utf-8')
    assert(d['batch_label'] == expected_batch_label)

    data = np.asarray(d['data'], dtype=x_dtype)
    labels = np.asarray(d[labels_key], dtype=y_dtype)

    # change shape
    data = data.reshape((data.shape[0], 3, 32, 32))
    data = np.transpose(data, (0, 2, 3, 1))
    if x_shape:
        data = data.reshape([data.shape[0]] + list(x_shape))

    return data, labels


def load_cifar10(x_shape: Sequence[int] = (32, 32, 3),
                 x_dtype: ArrayDType = np.float32,
                 y_dtype: ArrayDType = np.int32) -> Tuple[XYArrayTuple, XYArrayTuple]:
    """
    Load the CIFAR-10 dataset as NumPy arrays.

    Args:
        x_shape: Reshape each digit into this shape.
        x_dtype: Cast each digit into this data type.
        y_dtype: Cast each label into this data type.

    Returns:
        The ``(train_x, train_y), (test_x, test_y)`` arrays.
    """
    # check the arguments
    x_shape = _validate_x_shape(x_shape, (32, 32, 3))

    # fetch data
    path = CacheDir('cifar').download_and_extract(
        _CIFAR_10_URI, hasher=hashlib.md5(), expected_hash=_CIFAR_10_MD5)
    data_dir = os.path.join(path, _CIFAR_10_CONTENT_DIR)

    # load the data
    train_num = 50000
    train_x = np.zeros((train_num,) + x_shape, dtype=x_dtype)
    train_y = np.zeros((train_num,), dtype=y_dtype)

    for i in range(1, 6):
        path = os.path.join(data_dir, 'data_batch_{}'.format(i))
        x, y = _cifar_load_batch(
            path, x_shape=x_shape, x_dtype=x_dtype, y_dtype=y_dtype,
            expected_batch_label='training batch {} of 5'.format(i)
        )
        (train_x[(i - 1) * 10000: i * 10000, ...],
         train_y[(i - 1) * 10000: i * 10000]) = x, y

    path = os.path.join(data_dir, 'test_batch')
    test_x, test_y = _cifar_load_batch(
        path, x_shape=x_shape, x_dtype=x_dtype, y_dtype=y_dtype,
        expected_batch_label='testing batch 1 of 1'
    )
    assert(len(test_x) == len(test_y) == 10000)

    return (train_x, train_y), (test_x, test_y)


def load_cifar100(label_mode: str = 'fine',
                  x_shape: Sequence[int] = (32, 32, 3),
                  x_dtype: ArrayDType = np.float32,
                  y_dtype: ArrayDType = np.int32) -> Tuple[XYArrayTuple, XYArrayTuple]:
    """
    Load the CIFAR-100 dataset as NumPy arrays.

    Args:
        label_mode: One of {"fine", "coarse"}.
        x_shape: Reshape each digit into this shape.
        x_dtype: Cast each digit into this data type.
        y_dtype: Cast each label into this data type.

    Returns:
        The ``(train_x, train_y), (test_x, test_y)`` arrays.
    """
    # check the arguments
    label_mode = validate_enum_arg('label_mode', label_mode, ('fine', 'coarse'))
    x_shape = _validate_x_shape(x_shape, (32, 32, 3))

    # fetch data
    path = CacheDir('cifar').download_and_extract(
        _CIFAR_100_URI, hasher=hashlib.md5(), expected_hash=_CIFAR_100_MD5)
    data_dir = os.path.join(path, _CIFAR_100_CONTENT_DIR)

    # load the data
    path = os.path.join(data_dir, 'train')
    train_x, train_y = _cifar_load_batch(
        path, x_shape=x_shape, x_dtype=x_dtype, y_dtype=y_dtype,
        expected_batch_label='training batch 1 of 1',
        labels_key='{}_labels'.format(label_mode)
    )
    assert(len(train_x) == len(train_y) == 50000)

    path = os.path.join(data_dir, 'test')
    test_x, test_y = _cifar_load_batch(
        path, x_shape=x_shape, x_dtype=x_dtype, y_dtype=y_dtype,
        expected_batch_label='testing batch 1 of 1',
        labels_key='{}_labels'.format(label_mode)
    )
    assert(len(test_x) == len(test_y) == 10000)

    return (train_x, train_y), (test_x, test_y)
