from typing import *

import numpy as np

from ..typing_ import *
from .misc import generate_random_seed

__all__ = [
    'get_array_shape', 'to_number_or_numpy', 'minibatch_slices_iterator',
    'arrays_minibatch_iterator', 'split_numpy_arrays', 'split_numpy_array',
]


def get_array_shape(arr) -> ArrayShape:
    """
    Inspect the shape of an array-like object.

    >>> get_array_shape(1)
    ()
    >>> get_array_shape(2.5)
    ()
    >>> get_array_shape(np.zeros([1, 2, 3]))
    (1, 2, 3)

    Args:
        arr: The array-like object.

    Returns:
        The shape of the array.
    """
    if isinstance(arr, (float, int)):
        return ()
    if hasattr(arr, 'shape'):
        # TensorFlow, PyTorch and NumPy tensors should all have ``.shape``
        return tuple(arr.shape)
    return np.shape(arr)


def to_number_or_numpy(arr) -> Union[np.ndarray, float, int]:
    """
    Convert an array-like object into NumPy array or int/float number.

    Args:
        arr: The array-like object.

    Returns:
        The converted NumPy array.
    """
    if isinstance(arr, (float, int, np.ndarray)):
        return arr
    elif hasattr(arr, 'numpy'):
        # TensorFlow and PyTorch tensor has ``.numpy()``.
        if hasattr(arr, 'detach'):  # PyTorch further requires ``.detach()`` before calling ``.numpy()``
            arr = arr.detach()
        if hasattr(arr, 'cpu'):  # PyTorch further requires ``.cpu()`` before calling ``.numpy()``
            arr = arr.cpu()
        return arr.numpy()
    elif hasattr(arr, 'eval'):
        return arr.eval()
    else:
        return np.array(arr)


def minibatch_slices_iterator(length: int,
                              batch_size: int,
                              skip_incomplete: bool = False
                              ) -> Generator[slice, None, None]:
    """
    Iterate through all the mini-batch slices.

    Args:
        length: Total length of data in an epoch.
        batch_size: Size of each mini-batch.
        skip_incomplete: If :obj:`True`, discard the final
            batch if it contains less than `batch_size` number of items.
            (default :obj:`False`)

    Yields
        Slices of each mini-batch.  The last mini-batch may contain less
        indices than `batch_size`.
    """
    start = 0
    stop1 = (length // batch_size) * batch_size
    while start < stop1:
        yield slice(start, start + batch_size, 1)
        start += batch_size
    if not skip_incomplete and start < length:
        yield slice(start, length, 1)


def arrays_minibatch_iterator(arrays: Sequence[Array],
                              batch_size: int,
                              skip_incomplete: bool = False
                              ) -> Generator[slice, None, None]:
    """
    Iterate through all the mini-batches in the arrays.

    Args:
        arrays: Total length of data in an epoch.
        batch_size: Size of each mini-batch.
        skip_incomplete: If :obj:`True`, discard the final
            batch if it contains less than `batch_size` number of items.
            (default :obj:`False`)

    Yields
        Tuple of arrays of each mini-batch.  The last mini-batch may contain
        less indices than `batch_size`.
    """
    length = len(arrays[0])
    for slc in minibatch_slices_iterator(
            length, batch_size=batch_size, skip_incomplete=skip_incomplete):
        yield tuple(a[slc] for a in arrays)


def split_numpy_arrays(arrays: Sequence[np.ndarray],
                       portion: Optional[float] = None,
                       size: Optional[int] = None,
                       shuffle: bool = True,
                       random_state: Optional[np.random.RandomState] = None
                       ) -> Tuple[Tuple[np.ndarray, ...],
                                  Tuple[np.ndarray, ...]]:
    """
    Split numpy arrays into two halves, by portion or by size.

    Args:
        arrays: Numpy arrays to be splitted.
        portion: Portion of the second half.  Ignored if `size` is specified.
        size: Size of the second half.
        shuffle: Whether or not to shuffle before splitting?
        random_state: Optional numpy RandomState for shuffling data. (default
            :obj:`None`, construct a new :class:`np.random.RandomState`).

    Returns:
        The two halves of the arrays after splitting.
    """
    # check the arguments
    if size is None and portion is None:
        raise ValueError('At least one of `portion` and `size` should '
                         'be specified.')

    # zero arrays should return empty tuples
    arrays = tuple(arrays)
    if not arrays:
        return (), ()

    # check the length of provided arrays
    data_count = len(arrays[0])
    for array in arrays[1:]:
        if len(array) != data_count:
            raise ValueError('The length of specified arrays are not equal.')

    # determine the size for second half
    if size is None:
        if portion < 0.0 or portion > 1.0:
            raise ValueError('`portion` must range from 0.0 to 1.0.')
        elif portion < 0.5:
            size = data_count - int(data_count * (1.0 - portion))
        else:
            size = int(data_count * portion)

    # shuffle the data if necessary
    if shuffle:
        random_state = \
            random_state or np.random.RandomState(generate_random_seed())
        indices = np.arange(data_count)
        random_state.shuffle(indices)
        arrays = tuple(a[indices] for a in arrays)

    # return directly if each side remains no data after splitting
    if size <= 0:
        return arrays, tuple(a[:0] for a in arrays)
    elif size >= data_count:
        return tuple(a[:0] for a in arrays), arrays

    # split the data according to demand
    return (
        tuple(v[: -size, ...] for v in arrays),
        tuple(v[-size:, ...] for v in arrays)
    )


def split_numpy_array(array: np.ndarray,
                      portion: Optional[float] = None,
                      size: Optional[int] = None,
                      shuffle: bool = True) -> Tuple[np.ndarray, ...]:
    """
    Split numpy array into two halves, by portion or by size.

    Args:
        array: A numpy array to be splitted.
        portion: Portion of the second half.  Ignored if `size` is specified.
        size: Size of the second half.
        shuffle: Whether or not to shuffle before splitting?

    Returns:
        The two halves of the array after splitting.
    """
    (a,), (b,) = split_numpy_arrays((array,), portion=portion, size=size,
                                    shuffle=shuffle)
    return a, b
