import warnings
from logging import getLogger
from queue import Queue
from threading import Thread, Semaphore
from typing import *

import numpy as np

from ..typing_ import *
from ..utils import (minibatch_slices_iterator, AutoInitAndCloseable, NOT_SET,
                     GeneratorIterator, to_number_or_numpy)

__all__ = [
    'DataStream', 'UserGeneratorDataStream',
    'ArraysDataStream', 'IntSeqDataStream',
    'GeneratorFactoryDataStream', 'GatherDataStream',
    'MapperDataStream', 'ThreadingDataStream',
]


def map_to_tuple(fn: Callable[[Any], TObject], seq: Iterable[Any]):
    return tuple(fn(s) for s in seq)


def to_data_shapes(data_shapes) -> Tuple[ArrayShape, ...]:
    return map_to_tuple(lambda x: map_to_tuple(int, x), data_shapes)


def to_readonly_array(arr: Array) -> Array:
    arr = np.asarray(arr)
    arr.setflags(write=False)
    return arr


def ensure_batch_is_tuple(batch: Union[Array, ArrayTupleOrList]
                          ) -> ArrayTuple:
    if not isinstance(batch, (tuple, list)):
        batch = (batch,)
    else:
        batch = tuple(batch)
    return batch


class DataStream(object):
    """
    Class to construct mini-batch data iterators.

    Constructing Data Streams
    =========================

    All :class:`DataStream` subclasses shipped by `ml_essentials` can be
    constructed via factory methods of this base class.

    To construct a data stream from numpy arrays, you may:

    >>> x = np.arange(5, dtype=np.int32)
    >>> y = x ** 2
    >>> stream = DataStream.arrays([x, y], batch_size=3)
    >>> for [a, b] in stream:
    ...     print(a, b)
    [0 1 2] [0 1 4]
    [3 4] [ 9 16]

    To construct a integer sequence data stream, you may:

    >>> stream = DataStream.int_seq(start=1, stop=10, step=2, batch_size=3)
    >>> for [a] in stream:
    ...     print(a)
    [1 3 5]
    [7 9]

    To gather multiple data streams into one, you may:

    >>> stream_1 = DataStream.int_seq(5, batch_size=3)
    >>> stream_2 = DataStream.int_seq(-5, step=-1, batch_size=3)
    >>> for [a] in stream_1:
    ...     print(a)
    [0 1 2]
    [3 4]
    >>> for [b] in stream_2:
    ...     print(b)
    [ 0 -1 -2]
    [-3 -4]
    >>> stream = DataStream.gather([stream_1, stream_2])
    >>> for [a, b] in stream:
    ...     print(a, b)
    [0 1 2] [ 0 -1 -2]
    [3 4] [-3 -4]

    To turn an arbitrary mini-batch generator factory function into a data
    stream, you may:

    >>> def data_generator():
    ...     for i in range(2):
    ...         yield np.arange(i * 3, (i + 1) * 3, dtype=np.int32)

    >>> stream = DataStream.generator(data_generator)
    >>> for [a] in stream:
    ...     print(a)
    [0 1 2]
    [3 4 5]

    or you may generate a tuple / list of arrays:

    >>> def data_generator():
    ...     for i in range(2):
    ...         arr = np.arange(i * 3, (i + 1) * 3, dtype=np.int32)
    ...         yield arr, arr ** 2  # or return [x + y, x * y]

    >>> stream = DataStream.generator(data_generator)
    >>> for [a, b] in stream:
    ...     print(a, b)
    [0 1 2] [0 1 4]
    [3 4 5] [ 9 16 25]

    Transforming Data Streams
    =========================

    A :class:`DataStream` instance can be transformed into another data stream.

    To select a subset of the arrays within each mini-batch, or re-order the
    arrays, you may:

    >>> x = np.arange(0, 5, dtype=np.int32)
    >>> y = np.arange(5, 10, dtype=np.int32)
    >>> z = np.arange(10, 15, dtype=np.int32)
    >>> # note we shall select [x, z, x]
    >>> stream = DataStream.arrays([x, y, z], batch_size=3).select([0, 2, 0])
    >>> for [a, b, c] in stream:
    ...     print(a, b, c)
    [0 1 2] [10 11 12] [0 1 2]
    [3 4] [13 14] [3 4]

    To transform the arrays within each mini-batch by a mapper function,
    you may:

    >>> def mapper(x, y):
    ...     return x + y

    >>> x = np.arange(0, 5, dtype=np.int32)
    >>> y = np.arange(5, 10, dtype=np.int32)
    >>> stream = DataStream.arrays([x, y], batch_size=3).map(mapper)
    >>> for [a] in stream:
    ...     print(a)
    [5 7 9]
    [11 13]

    or you may return a tuple / list of arrays:

    >>> def mapper(x, y):
    ...     return x + y, x * y  # or return [x + y, x * y]

    >>> x = np.arange(0, 5, dtype=np.int32)
    >>> y = np.arange(5, 10, dtype=np.int32)
    >>> stream = DataStream.arrays([x, y], batch_size=3).map(mapper)
    >>> for [a, b] in stream:
    ...     print(a, b)
    [5 7 9] [ 0  6 14]
    [11 13] [24 36]

    To pre-fetch from a time-consuming data stream in background thread
    (which is necessary when using a slow mapper), you may:

    >>> stream = DataStream.int_seq(5, batch_size=3)
    >>> with stream.threaded(prefetch=2) as prefetch_stream:
    ...     for [x] in prefetch_stream:
    ...         print(x)
    [0 1 2]
    [3 4]
    """

    def __init__(self,
                 batch_size: Optional[int] = None,
                 array_count: Optional[int] = None,
                 data_shapes: Optional[Tuple[ArrayShape, ...]] = None,
                 data_length: Optional[int] = None,
                 random_state: Optional[np.random.RandomState] = None):
        """
        Construct a :class:`DataStream`.

        Args:
            batch_size: The number of data within each mini-batch.
            array_count: The number of arrays within each mini-batch.
            data_shapes: The data shapes (excluding the batch axis).
            data_length: The total number of data.
            random_state: The NumPy random state instance.

        Raises:
            ValueError: If `len(data_shapes) != array_count`.

                >>> stream = DataStream(data_shapes=((), (3, 5)), array_count=3)
                Traceback (most recent call last):
                    ...
                ValueError: len(data_shapes) != array_count: data_shapes ((), (3, 5)) vs array_count 3
        """
        if batch_size is not None:
            batch_size = int(batch_size)
        if array_count is not None:
            array_count = int(array_count)
        if data_shapes is not None:
            data_shapes = to_data_shapes(data_shapes)
            if array_count is None:
                array_count = len(data_shapes)
            elif array_count != len(data_shapes):
                raise ValueError(f'len(data_shapes) != array_count: '
                                 f'data_shapes {data_shapes} vs '
                                 f'array_count {array_count}')
        if data_length is not None:
            data_length = int(data_length)
        if random_state is not None and not \
                isinstance(random_state, np.random.RandomState):
            raise TypeError(f'`random_state` is not np.random.RandomState: '
                            f'{random_state!r}')

        if data_length is not None and batch_size is not None:
            batch_count = int((data_length + batch_size - 1) // batch_size)
        else:
            batch_count = None

        self._batch_size = batch_size
        self._batch_count = batch_count
        self._array_count = array_count
        self._data_shapes = data_shapes
        self._data_length = data_length
        self._random_state = random_state
        self._active_iterator = None
        self._auto_close_iterator_warning_printed = False

    def __iter__(self) -> GeneratorIterator[ArrayTuple]:
        """
        Iterate through the mini-batches.

        Note if a previous iterator is not closed before obtaining a new one,
        the previous iterator will be closed automatically, and a warning will
        be printed to the console (for only once).
        """
        if self._active_iterator is not None:
            self._active_iterator.close()
            self._active_iterator = None
            if not self._auto_close_iterator_warning_printed:
                warnings.warn(
                    f'Another iterator of the DataStream {self!r} is still '
                    f'active, will close it automatically.  If you did not '
                    f'exhaust the iterator, remember to call `close()` on it.',
                    UserWarning,
                )
                self._auto_close_iterator_warning_printed = True

        def make_generator():
            g = self._minibatch_iterator()
            try:
                yield from g
            finally:
                self._active_iterator = None

        self._active_iterator = GeneratorIterator(make_generator())
        return self._active_iterator

    def __len__(self):
        """
        Get the total number of data.

        If a data stream reports this number (i.e., being not None), then it
        equals to the sum of array lengths from all mini-batches in one epoch.

        >>> stream = DataStream.int_seq(5, batch_size=3)
        >>> len(stream)
        5
        >>> stream = DataStream.int_seq(5, batch_size=3, skip_incomplete=True)
        >>> len(stream)
        3

        Raises:
            RuntimeError: If a data stream cannot report this number,
                i.e., `data_length` is None.


                >>> def g():
                ...     yield np.arange(3)

                >>> stream = DataStream.generator(g)
                >>> stream.data_length is None
                True
                >>> len(stream)
                Traceback (most recent call last):
                    ...
                RuntimeError: stream data length is not available
        """
        ret = self.data_length
        if ret is None:
            raise RuntimeError(f'stream data length is not available')
        return ret

    @property
    def batch_size(self) -> Optional[int]:
        """
        Get the batch size of this data stream.

        If a data stream reports this number (i.e., being not None), then the
        actual length of each mini-batch is guaranteed to be NO MORE THAN this.

        >>> x = np.random.normal(size=[5, 4])
        >>> stream = DataStream.arrays([x], batch_size=3)
        >>> stream.batch_size
        3
        """
        return self._batch_size

    @property
    def array_count(self) -> Optional[int]:
        """
        Get the count of arrays within each mini-batch.

        >>> x = np.random.normal(size=[5, 4])
        >>> y = np.random.normal(size=[5, 3, 2])
        >>> stream = DataStream.arrays([x, y], batch_size=3)
        >>> stream.array_count
        2
        """
        return self._array_count

    @property
    def data_shapes(self) -> Optional[Tuple[ArrayShape, ...]]:
        """
        Get the data shapes.

        Data shapes are shapes if mini-batch array without the batch axis.

        >>> x = np.random.normal(size=[5, 4])
        >>> y = np.random.normal(size=[5, 3, 2])
        >>> stream = DataStream.arrays([x, y], batch_size=3)
        >>> stream.data_shapes
        ((4,), (3, 2))
        """
        return self._data_shapes

    @property
    def data_length(self) -> Optional[int]:
        """
        Get the total number of data.

        If a data stream reports this number (i.e., being not None), then it
        equals to the sum of array lengths from all mini-batches in one epoch.

        >>> stream = DataStream.int_seq(5, batch_size=3)
        >>> stream.data_length
        5
        >>> stream = DataStream.int_seq(5, batch_size=3, skip_incomplete=True)
        >>> stream.data_length
        3
        """
        return self._data_length

    @property
    def batch_count(self) -> Optional[int]:
        """
        Get the total number of batches in an epoch.

        >>> stream = DataStream.int_seq(5, batch_size=3)
        >>> stream.batch_count
        2
        >>> stream = DataStream.int_seq(5, batch_size=3, skip_incomplete=True)
        >>> stream.batch_count
        1
        """
        return self._batch_count

    @property
    def random_state(self) -> Optional[np.random.RandomState]:
        """Get the NumPy random state associated with this data stream."""
        return self._random_state

    def copy(self, **kwargs):
        """
        Get a copy of this data stream.

        You may override some of the construction arguments by specifying
        named arguments via :param:`kwargs`.  However, some argument may
        not be overridable (depends on the implementation of subclasses).

        >>> x = np.arange(5, dtype=np.int32)
        >>> stream = DataStream.arrays([x], batch_size=3)
        >>> for [a] in stream:
        ...     print(a)
        [0 1 2]
        [3 4]
        >>> stream2 = stream.copy(batch_size=4)
        >>> isinstance(stream2, ArraysDataStream)
        True
        >>> for [a] in stream2:
        ...     print(a)
        [0 1 2 3]
        [4]

        Args:
            \\**kwargs: The overrided construction arguments.

        Returns:
            The copied data stream.
        """
        raise NotImplementedError()

    def _copy_helper(self, attrs: Iterable[str], **kwargs):
        for attr in attrs:
            kwargs.setdefault(attr, getattr(self, attr))
        return self.__class__(**kwargs)

    def _minibatch_iterator(self) -> Generator[ArrayTuple, None, None]:
        raise NotImplementedError()

    def get_arrays(self, max_batch: Optional[int] = None) -> Tuple[np.ndarray, ...]:
        """
        Collecting mini-batches into NumPy arrays.

        >>> x = np.arange(0, 5, dtype=np.int32)
        >>> stream = DataStream.arrays([x], batch_size=3).map(lambda t: t ** 2)

        >>> arrays = stream.get_arrays()
        >>> len(arrays)
        1
        >>> print(arrays[0])
        [ 0  1  4  9 16]

        >>> arrays = stream.get_arrays(max_batch=1)
        >>> len(arrays)
        1
        >>> print(arrays[0])
        [0 1 4]

        >>> arrays = stream.get_arrays(max_batch=0)
        >>> len(arrays)
        1
        >>> print(arrays[0])
        []

        Args:
            max_batch: If specified, will take at most this number of batches.

        Returns:
            The collected arrays.

        Raises:
            RuntimeError: If this data-flow is empty.

                >>> def g():
                ...     if False:
                ...         yield ()

                >>> stream = DataStream.generator(g)
                >>> stream.get_arrays()
                Traceback (most recent call last):
                    ...
                RuntimeError: empty data stream cannot be converted to arrays
        """
        arrays_buf = []
        g = iter(self)
        try:
            try:
                batch = next(g)
            except StopIteration:
                raise RuntimeError(
                    'empty data stream cannot be converted to arrays')
            try:
                arrays_buf = [[to_number_or_numpy(arr)] for arr in batch]
                batch_index = 1
                while max_batch is None or batch_index < max_batch:
                    batch = next(g)
                    for i, arr in enumerate(batch):
                        arrays_buf[i].append(to_number_or_numpy(arr))
                    batch_index += 1
                if max_batch == 0:
                    arrays_buf = [[array_buf[0][:0]]
                                  for array_buf in arrays_buf]
            except StopIteration:
                pass
            return tuple(np.concatenate(array_buf) for array_buf in arrays_buf)
        finally:
            g.close()

    def to_arrays_stream(self,
                         batch_size: int = NOT_SET,
                         shuffle: bool = False,
                         skip_incomplete: bool = False,
                         random_state: Optional[np.random.RandomState] = NOT_SET
                         ) -> 'ArraysDataStream':
        """
        Convert this data-flow to an arrays stream.

        By default, the original batch size will be preserved:

        >>> stream = DataStream.int_seq(5, batch_size=3).map(lambda x: x ** 2)
        >>> isinstance(stream, MapperDataStream)
        True
        >>> stream2 = stream.to_arrays_stream()
        >>> isinstance(stream2, ArraysDataStream)
        True
        >>> for [a] in stream2:
        ...     print(a)
        [0 1 4]
        [ 9 16]

        You may also override the batch size:

        >>> stream3 = stream.to_arrays_stream(batch_size=4)
        >>> for [a] in stream3:
        ...     print(a)
        [0 1 4 9]
        [16]

        Args:
            batch_size: The number of data within each mini-batch.
                If not specified, will use the original batch size if possible.
            shuffle: Whether or not to shuffle data?
            skip_incomplete: Whether or not to exclude the last mini-batch
                if it is incomplete?
            random_state : The NumPy random state instance.
                If not specified, will use the original random state instance.

        Returns:
            The constructed array stream.

        Raises:
            ValueError: If the batch size is neither specified, nor can it
                be determined according to the original batch size.

                >>> def g():
                ...     yield np.arange(3)

                >>> stream = DataStream.generator(g)
                >>> stream.to_arrays_stream()
                Traceback (most recent call last):
                    ...
                ValueError: `batch_size` must be specified
        """
        if batch_size is NOT_SET:
            batch_size = self.batch_size
        if batch_size is None:
            raise ValueError('`batch_size` must be specified')

        if random_state is NOT_SET:
            random_state = self.random_state

        return ArraysDataStream(
            self.get_arrays(), batch_size=batch_size, shuffle=shuffle,
            skip_incomplete=skip_incomplete, random_state=random_state
        )

    # -------- here starts the factory methods --------
    @staticmethod
    def arrays(arrays: Iterable[Array],
               batch_size: int,
               shuffle: bool = False,
               skip_incomplete: bool = False,
               random_state: Optional[np.random.RandomState] = None
               ) -> 'ArraysDataStream':
        """
        Construct an arrays stream, i.e., :class:`ArraysDataStream`.

        >>> x = np.arange(5, dtype=np.int32)
        >>> y = x ** 2
        >>> stream = DataStream.arrays([x, y], batch_size=3)
        >>> for [a, b] in stream:
        ...     print(a, b)
        [0 1 2] [0 1 4]
        [3 4] [ 9 16]

        You may shuffle the data by setting `shuffle = True`:

        >>> np.random.seed(1234)
        >>> stream = DataStream.arrays([x, y], batch_size=3, shuffle=True)
        >>> for [a, b] in stream:
        ...     print(a, b)
        [4 0 1] [16  0  1]
        [2 3] [4 9]

        You may discard the last incomplete mini-batch by setting
        `skip_incomplete = True`:

        >>> stream = DataStream.arrays(
        ...     [x, y], batch_size=3, skip_incomplete=True)
        >>> for [a, b] in stream:
        ...     print(a, b)
        [0 1 2] [0 1 4]

        Args:
            arrays: A sequence of numpy-like arrays.
                These arrays should be at least 1-d, and the size of the
                first axis must be identical.
            batch_size: The number of data within each mini-batch.
            shuffle: Whether or not to shuffle data?
            skip_incomplete: Whether or not to exclude the last mini-batch
                if it is incomplete?
            random_state: The numpy random state instance.

        Returns:
            The arrays stream.
        """
        return ArraysDataStream(
            arrays=arrays,
            batch_size=batch_size,
            shuffle=shuffle,
            skip_incomplete=skip_incomplete,
            random_state=random_state
        )

    @staticmethod
    def int_seq(start: int,
                stop: int = None,
                step: int = None,
                *,
                dtype=np.int32,
                batch_size: int = NOT_SET,
                shuffle: bool = False,
                skip_incomplete: bool = False,
                random_state: Optional[np.random.RandomState] = None
                ) -> 'IntSeqDataStream':
        """
        Construct a integer sequence stream, i.e., :class:`IntSeqStream`.

        To construct various integer sequences:

        >>> stream = DataStream.int_seq(5, batch_size=3)
        >>> for [a] in stream:
        ...     print(a)
        [0 1 2]
        [3 4]

        >>> stream = DataStream.int_seq(2, 11, 2, batch_size=3)
        >>> for [a] in stream:
        ...     print(a)
        [2 4 6]
        [ 8 10]

        >>> stream = DataStream.int_seq(-5, step=-1, batch_size=3)
        >>> for [a] in stream:
        ...     print(a)
        [ 0 -1 -2]
        [-3 -4]

        >>> stream = DataStream.int_seq(-2, -11, -2, batch_size=3)
        >>> for [a] in stream:
        ...     print(a)
        [-2 -4 -6]
        [ -8 -10]

        You may shuffle the sequence by setting `shuffle = True`:

        >>> np.random.seed(1234)
        >>> stream = DataStream.int_seq(5, batch_size=3, shuffle=True)
        >>> for [a] in stream:
        ...     print(a)
        [4 0 1]
        [2 3]

        You may discard the last incomplete mini-batch by setting
        `skip_incomplete = True`:

        >>> stream = DataStream.int_seq(5, batch_size=3, skip_incomplete=True)
        >>> for [a] in stream:
        ...     print(a)
        [0 1 2]

        Args:
            start: If `stop` is specified, this is the starting number.
                Otherwise this is the ending number, and the starting
                number is 0.
            stop: The ending number.
            step: The sequence incremental step.
            dtype: The NumPy data type.
            batch_size: The number of data within each mini-batch.
            shuffle: Whether or not to shuffle data?
            skip_incomplete: Whether or not to exclude the last mini-batch
                if it is incomplete?
            random_state: The numpy random state instance.

        Returns:
            The integer sequence stream.
        """
        return IntSeqDataStream(
            start=start, stop=stop, step=step, dtype=dtype,
            batch_size=batch_size, shuffle=shuffle,
            skip_incomplete=skip_incomplete, random_state=random_state,
        )

    @staticmethod
    def gather(streams: Iterable['DataStream'],
               random_state: Optional[np.random.RandomState] = None
               ) -> 'GatherDataStream':
        return GatherDataStream(streams=streams, random_state=random_state)

    @staticmethod
    def generator(f: Callable[[], ArraysOrArrayGenerator]
                  ) -> 'GeneratorFactoryDataStream':
        return GeneratorFactoryDataStream(f)

    # -------- here starts the transforming methods --------
    def map(self,
            mapper: Callable[..., ArraysOrArray],
            preserve_shapes: bool = False
            ) -> 'MapperDataStream':
        """
        Transform this data stream via a mapper function.

        To return a single array:

        >>> def mapper(x, y):
        ...     return x + y

        >>> x = np.arange(0, 5, dtype=np.int32)
        >>> y = np.arange(5, 10, dtype=np.int32)
        >>> stream = DataStream.arrays([x, y], batch_size=3).map(mapper)
        >>> for [a] in stream:
        ...     print(a)
        [5 7 9]
        [11 13]

        To return a tuple / list of arrays:

        >>> def mapper(x, y):
        ...     return x + y, x * y  # or return [x + y, x * y]

        >>> x = np.arange(0, 5, dtype=np.int32)
        >>> y = np.arange(5, 10, dtype=np.int32)
        >>> stream = DataStream.arrays([x, y], batch_size=3).map(mapper)
        >>> for [a, b] in stream:
        ...     print(a, b)
        [5 7 9] [ 0  6 14]
        [11 13] [24 36]

        Args:
            mapper: The mapper function.
            preserve_shapes: User specified hint, whether or not the
                `mapper` preserves the array count and shapes within
                each mini-batch?  This hint might benefit further
                transformation.  By default :obj:`False`.

                >>> def mapper(x, y):
                ...     return x ** 2, y - 1

                >>> x = np.random.normal(size=[5, 4])
                >>> y = np.random.normal(size=[5, 3, 2])
                >>> stream = DataStream.arrays([x, y], batch_size=3)
                >>> stream.array_count, stream.data_shapes
                (2, ((4,), (3, 2)))

                >>> stream2 = stream.map(mapper)
                >>> stream2.array_count, stream2.data_shapes
                (None, None)

                >>> stream3 = stream.map(mapper, preserve_shapes=True)
                >>> stream3.array_count, stream3.data_shapes
                (2, ((4,), (3, 2)))

        Returns:
            The transformed data stream.
        """
        return MapperDataStream(
            source=self, mapper=mapper, preserve_shapes=preserve_shapes)

    def threaded(self, prefetch: int = 5) -> 'ThreadingDataStream':
        """
        Construct a data stream that prefetches this data stream in a
        background thread.

        >>> stream = DataStream.int_seq(5, batch_size=3)
        >>> with stream.threaded() as prefetch_stream:
        ...     for [x] in prefetch_stream:
        ...         print(x)
        [0 1 2]
        [3 4]

        Args:
            prefetch: Number of mini-batches to prefetch in background.

        Returns:
            The background data stream.
        """
        return ThreadingDataStream(self, prefetch=prefetch)

    def select(self, indices: Iterable[int]) -> 'MapperDataStream':
        """
        Construct a data stream that selects a subset of the arrays within
        each mini-batch, or re-order the arrays.

        Given the following source data stream:

        >>> x = np.arange(0, 5, dtype=np.int32)
        >>> y = np.arange(5, 10, dtype=np.int32)
        >>> z = np.arange(10, 15, dtype=np.int32)
        >>> source = DataStream.arrays([x, y, z], batch_size=3)

        We shall select [x, z, x] from source:

        >>> stream = source.select([0, 2, 0])
        >>> for [a, b, c] in stream:
        ...     print(a, b, c)
        [0 1 2] [10 11 12] [0 1 2]
        [3 4] [13 14] [3 4]

        The various data stream properties are also properly inherited:

        >>> x = np.random.normal(size=[5, 4])
        >>> y = np.random.normal(size=[5, 2, 3])
        >>> source = DataStream.arrays([x, y], batch_size=3)
        >>> stream = source.select([-1, 0, 1])
        >>> stream.array_count
        3
        >>> stream.data_shapes
        ((2, 3), (4,), (2, 3))
        >>> stream.data_length
        5

        Args:
            indices: The indices of the arrays to select within each mini-batch.

        Returns:
            The transformed data stream.

        Raises:
            IndexError: If `self.array_count` is reported, and any index
                in `indices` out of this range.

                >>> x = np.arange(0, 5, dtype=np.int32)
                >>> y = np.arange(5, 10, dtype=np.int32)
                >>> stream = DataStream.arrays([x, y], batch_size=3)
                >>> stream.select([0, 1, 2])
                Traceback (most recent call last):
                    ...
                IndexError: array index out of range

                Note if `self.array_count` is not reported (i.e., is None),
                then :class:`IndexError` will not be raised until iterated.

                >>> def mapper(x, y, z):
                ...     return x + y, y - z

                >>> x = np.arange(0, 5, dtype=np.int32)
                >>> y = np.arange(5, 10, dtype=np.int32)
                >>> z = np.arange(10, 15, dtype=np.int32)
                >>> stream = DataStream.arrays([x, y, z], batch_size=3). \
                        map(mapper).select([0, 1, 2])
                >>> for batch in stream:
                ...     print(batch)
                Traceback (most recent call last):
                    ...
                IndexError: tuple index out of range
        """
        # validate the argument
        indices = tuple(indices)
        if self.array_count is not None:
            for i in indices:
                if i < -self.array_count or i >= self.array_count:
                    raise IndexError(f'array index out of range')

        # prepare for the mapper
        def mapper(*arrays):
            return tuple(arrays[j] for j in indices)

        # construct the mapper data stream
        if self.data_shapes is not None:
            data_shapes = tuple(self.data_shapes[i] for i in indices)
        else:
            data_shapes = None

        array_count = len(indices)

        return MapperDataStream(
            source=self, mapper=mapper, data_shapes=data_shapes,
            array_count=array_count
        )


class ArraysDataStream(DataStream):
    """NumPy arrays data stream."""

    def __init__(self,
                 arrays: Iterable[Array],
                 batch_size: int,
                 shuffle: bool,
                 skip_incomplete: bool,
                 random_state: Optional[np.random.RandomState] = None):
        # validate parameters
        arrays = tuple(arrays)
        if not arrays:
            raise ValueError('`arrays` must not be empty.')
        for a in arrays:
            if not hasattr(a, 'shape'):
                raise ValueError('`arrays` must be arrays.')
            if len(a.shape) < 1:
                raise ValueError('`arrays` must be at least 1-d arrays.')

        data_shapes = to_data_shapes(arr.shape[1:] for arr in arrays)

        array_length = len(arrays[0])
        for a in arrays[1:]:
            if len(a) != array_length:
                raise ValueError('`arrays` must have the same length.')
        if skip_incomplete:
            data_length = array_length // batch_size * batch_size
        else:
            data_length = array_length

        # construct the instance
        super().__init__(
            batch_size=batch_size,
            array_count=len(data_shapes),
            data_shapes=data_shapes,
            data_length=data_length,
            random_state=random_state,
        )
        self._arrays = map_to_tuple(to_readonly_array, arrays)
        self._indices_buffer = None  # type: Array
        self._shuffle = bool(shuffle)
        self._skip_incomplete = bool(skip_incomplete)

    @property
    def the_arrays(self):
        """Get the underlying NumPy arrays without copy."""
        return self._arrays

    @property
    def shuffle(self) -> bool:
        """Whether or not to shuffle data?"""
        return self._shuffle

    @property
    def skip_incomplete(self) -> bool:
        """Whether or not to exclude the last mini-batch if it is incomplete?"""
        return self._skip_incomplete

    def _minibatch_iterator(self) -> Generator[ArrayTuple, None, None]:
        # shuffle the source arrays if necessary
        if self.shuffle:
            if self._indices_buffer is None:
                indices_count = len(self._arrays[0])
                t = np.int32 if indices_count < (1 << 31) else np.int64
                self._indices_buffer = np.arange(indices_count, dtype=t)

            rng = self._random_state or np.random
            rng.shuffle(self._indices_buffer)

            def get_slice(s):
                return tuple(
                    a[self._indices_buffer[s]]
                    for a in self.the_arrays
                )
        else:
            def get_slice(s):
                return tuple(a[s] for a in self.the_arrays)

        # now iterator through the mini-batches
        for batch_s in minibatch_slices_iterator(
                length=self.data_length,
                batch_size=self.batch_size,
                skip_incomplete=self.skip_incomplete):
            yield get_slice(batch_s)

    def copy(self, **kwargs):
        return self._copy_helper(
            ('batch_size', 'shuffle', 'skip_incomplete', 'random_state'),
            arrays=self._arrays,
            **kwargs
        )


class IntSeqDataStream(DataStream):
    """Integer sequence data stream."""

    def __init__(self,
                 start: int,
                 stop: int = None,
                 step: int = None,
                 *,
                 dtype=np.int32,
                 batch_size: int = NOT_SET,
                 shuffle: bool = False,
                 skip_incomplete: bool = False,
                 random_state: Optional[np.random.RandomState] = None):
        # validate the arguments
        start = int(start)

        if stop is None:
            stop = start
            start = 0
        else:
            stop = int(stop)

        if step is None:
            step = 1
        else:
            step = int(step)

        dtype = np.dtype(dtype)

        if batch_size is NOT_SET:
            raise ValueError('`batch_size` is required.')

        # construct the int sequence
        seq = np.arange(start=start, stop=stop, step=step, dtype=dtype)

        if skip_incomplete:
            data_length = len(seq) // batch_size * batch_size
        else:
            data_length = len(seq)

        # construct the instance
        super().__init__(
            batch_size=batch_size,
            array_count=1,
            data_shapes=((),),
            data_length=data_length,
            random_state=random_state,
        )
        self._start = start
        self._stop = stop
        self._step = step
        self._dtype = dtype
        self._seq = seq
        self._shuffle = bool(shuffle)
        self._skip_incomplete = bool(skip_incomplete)

    @property
    def start(self) -> int:
        """Get the starting number."""
        return self._start

    @property
    def stop(self) -> int:
        """Get the ending number."""
        return self._stop

    @property
    def step(self) -> int:
        """Get the sequence incremental step."""
        return self._step

    @property
    def dtype(self) -> np.dtype:
        """Get the NumPy data type."""
        return self._dtype

    @property
    def shuffle(self) -> bool:
        """Whether or not to shuffle data?"""
        return self._shuffle

    @property
    def skip_incomplete(self) -> bool:
        """Whether or not to exclude the last mini-batch if it is incomplete?"""
        return self._skip_incomplete

    def _minibatch_iterator(self):
        if self.shuffle:
            rng = self._random_state or np.random
            rng.shuffle(self._seq)

        for batch_s in minibatch_slices_iterator(
                length=self.data_length,
                batch_size=self.batch_size,
                skip_incomplete=self.skip_incomplete):
            yield (to_readonly_array(self._seq[batch_s]),)

    def copy(self, **kwargs):
        return self._copy_helper(
            ('dtype', 'batch_size', 'shuffle', 'skip_incomplete',
             'random_state'),
            start=self.start, stop=self.stop, step=self.step,
            **kwargs
        )


class UserGeneratorDataStream(DataStream):
    """Base class for data streams with user generated data."""

    def _validate_batch(self, batch):
        batch = ensure_batch_is_tuple(batch)

        if self.batch_size is not None and batch:
            batch_size = len(batch[0])
            if batch_size > self.batch_size:
                raise ValueError(
                    f'batch size of the mapper output is not '
                    f'valid: expected <= {self.batch_size}, '
                    f'got {batch_size}'
                )
            for i, b in enumerate(batch[1:], 1):
                if len(b) != batch_size:
                    raise ValueError(
                        f'batch size of the {i}-th mapper output != '
                        f'the first output'
                    )
        if self.array_count is not None and len(batch) != self.array_count:
            raise ValueError(f'user generator returned invalid number of '
                             f'arrays: expected {self.array_count}, got '
                             f'{len(batch)}')
        if self.data_shapes is not None:
            for i, (x, y) in enumerate(zip(batch, self.data_shapes)):
                if x.shape[1:] != y:
                    raise ValueError(
                        f'data shape of the {i}-th mapper output is not '
                        f'valid: expected {y}, got {x.shape[1:]}'
                    )

        return batch


class GeneratorFactoryDataStream(UserGeneratorDataStream):
    """Data stream that turns a generator factory function into a stream."""

    def __init__(self, factory: Callable[[], ArraysOrArrayGenerator]):
        super().__init__()
        self._factory = factory

    @property
    def factory(self) -> Callable[[], Generator[Sequence[Array], None, None]]:
        """
        Get the generator factory function (i.e., function that returns a
        mini-batch arrays generator).
        """
        return self._factory

    def _minibatch_iterator(self):
        g = self._factory()
        try:
            for batch in g:
                yield self._validate_batch(batch)
        finally:
            if hasattr(g, 'close'):  # pragma: no cover
                g.close()

    def copy(self, **kwargs):
        return self._copy_helper((), factory=self.factory, **kwargs)


class GatherDataStream(DataStream):
    """Data stream that gathers multiple streams into one."""

    def __init__(self,
                 streams: Iterable[DataStream],
                 random_state: Optional[np.random.RandomState] = NOT_SET):
        # validate the streams
        streams = tuple(streams)
        if not streams:
            raise ValueError('At least one data stream should be specified.')

        for i, stream in enumerate(streams):
            if not isinstance(stream, DataStream):
                raise TypeError(f'The {i}-th element of `streams` is not an '
                                f'instance of DataStream: {stream}.')

        # inspect the properties of the data streams
        batch_size = NOT_SET
        array_count = 0
        data_shapes = []
        data_length = NOT_SET

        for i, stream in enumerate(streams):
            # check the batch size
            if stream.batch_size is not None:
                if batch_size is NOT_SET:
                    batch_size = stream.batch_size
                elif batch_size != stream.batch_size:
                    raise ValueError(
                        f'Inconsistent batch size among the specified streams: '
                        f'encountered {stream.batch_size} at the {i}-th '
                        f'stream, but has already encountered {batch_size} '
                        f'before.'
                    )

            # check the array count
            if array_count is not None:
                if stream.array_count is not None:
                    array_count += stream.array_count
                else:
                    array_count = None

            # check the data shapes
            if data_shapes is not None:
                if stream.data_shapes is not None:
                    data_shapes.extend(stream.data_shapes)
                else:
                    data_shapes = None

            # check the data length
            if stream.data_length is not None:
                if data_length is NOT_SET:
                    data_length = stream.data_length
                elif data_length != stream.data_length:
                    raise ValueError(
                        f'Inconsistent data length among the specified '
                        f'streams: encountered {stream.data_length} at '
                        f'the {i}-th stream, but has already encountered '
                        f'{data_length} before.'
                    )

            # check the random state
            if stream.random_state is not None and random_state is NOT_SET:
                random_state = stream.random_state

        if batch_size is NOT_SET:
            batch_size = None

        if data_shapes is not None:
            data_shapes = tuple(data_shapes)

        if data_length is NOT_SET:
            data_length = None

        if random_state is NOT_SET:
            random_state = None

        # construct the instance
        super().__init__(
            batch_size=batch_size,
            array_count=array_count,
            data_shapes=data_shapes,
            data_length=data_length,
            random_state=random_state
        )

        self._streams = streams

    @property
    def streams(self) -> Tuple[DataStream, ...]:
        """Get the gathered data streams."""
        return self._streams

    def _minibatch_iterator(self):
        iterators = [iter(s) for s in self._streams]
        try:
            for batches in zip(*iterators):
                yield sum([tuple(b) for b in batches], ())
        finally:
            for i in iterators:
                if hasattr(i, 'close'):  # pragma: no cover
                    i.close()

    def copy(self, **kwargs):
        return self._copy_helper(('random_state',), streams=self.streams, **kwargs)


class MapperDataStream(UserGeneratorDataStream):
    """Data stream that transforms the source stream via a mapper function."""

    def __init__(self,
                 source: DataStream,
                 mapper: Callable[..., ArraysOrArray],
                 batch_size: Optional[int] = NOT_SET,
                 array_count: Optional[int] = NOT_SET,
                 data_shapes: Optional[Tuple[ArrayShape, ...]] = NOT_SET,
                 data_length: Optional[int] = NOT_SET,
                 random_state: Optional[np.random.RandomState] = NOT_SET,
                 preserve_shapes: bool = False):
        # validate the arguments
        if not isinstance(source, DataStream):
            raise TypeError(f'`source` is not a DataStream: {source!r}')

        if batch_size is NOT_SET:
            batch_size = source.batch_size

        if array_count is NOT_SET:
            if preserve_shapes:
                array_count = source.array_count
            else:
                array_count = None

        if data_shapes is NOT_SET:
            if preserve_shapes:
                data_shapes = source.data_shapes
            else:
                data_shapes = None

        if data_length is NOT_SET:
            data_length = source.data_length

        if random_state is NOT_SET:
            random_state = source.random_state

        super().__init__(
            batch_size=batch_size,
            array_count=array_count,
            data_shapes=data_shapes,
            data_length=data_length,
            random_state=random_state
        )

        self._source = source
        self._mapper = mapper

    @property
    def source(self) -> DataStream:
        """Get the source data stream."""
        return self._source

    def _minibatch_iterator(self):
        g = iter(self._source)
        try:
            for batch in g:
                yield self._validate_batch(
                    self._mapper(*ensure_batch_is_tuple(batch)))
        finally:
            g.close()

    def copy(self, **kwargs):
        return self._copy_helper(
            ('batch_size', 'array_count', 'data_shapes', 'data_length',
             'random_state'),
            source=self._source,
            mapper=self._mapper,
            **kwargs
        )


class ThreadingDataStream(DataStream, AutoInitAndCloseable):
    """
    Data stream that prefetches mini-batches from the source stream
    in a background thread.
    """

    EPOCH_END = object()
    """Object to mark an ending position of an epoch."""

    class ErrorBox(object):
        """Class to carry an error."""
        def __init__(self, error):
            self.error = error

    def __init__(self,
                 source: DataStream,
                 prefetch: int):
        # validate the parameters
        if not isinstance(source, DataStream):
            raise TypeError(f'`source` is not a DataStream: {source!r}')
        prefetch = int(prefetch)
        if prefetch < 1:
            raise ValueError('`prefetch` must be at least 1')

        # construct the instance
        super().__init__(
            batch_size=source.batch_size,
            array_count=source.array_count,
            data_shapes=source.data_shapes,
            data_length=source.data_length,
            random_state=source.random_state,
        )
        self._source = source
        self._prefetch = prefetch

        # internal states for background worker
        self._worker = None  # type: Thread
        self._batch_queue = None  # type: Queue
        self._epoch_counter = None  # counter for tracking the active epoch
        self._stopping = None
        self._worker_alive = None
        self._worker_ready_sem = None

    @property
    def source(self) -> DataStream:
        """Get the source data stream."""
        return self._source

    @property
    def prefetch(self) -> int:
        """Get the number of mini-batches to prefetch in background."""
        return self._prefetch

    def _worker_func(self):
        active_epoch = self._epoch_counter
        self._worker_alive = True
        self._worker_ready_sem.release()

        try:
            while not self._stopping:
                # iterate through the mini-batches in the current epoch
                g = iter(self.source)
                try:
                    for batch in g:
                        if self._stopping or active_epoch < self._epoch_counter:
                            break
                        self._batch_queue.put((active_epoch, batch))
                finally:
                    g.close()

                # put the epoch ending mark into the queue
                if not self._stopping:
                    self._batch_queue.put((active_epoch, self.EPOCH_END))

                # move to the next epoch
                active_epoch += 1
        except Exception as ex:  # pragma: no cover
            getLogger(__name__).warning(
                f'{self.__class__.__qualname__} exited because of error',
                exc_info=True
            )
            self._batch_queue.put((active_epoch, self.ErrorBox(ex)))
            raise
        finally:
            self._worker_alive = False

    def _init(self):
        # prepare for the worker states
        self._batch_queue = Queue(self.prefetch)
        self._epoch_counter = 0
        self._stopping = False
        self._worker_ready_sem = Semaphore(value=0)

        # create and start the worker
        self._worker = Thread(target=self._worker_func)
        self._worker.daemon = True
        self._worker.start()

        # wait for the thread to show up
        self._worker_ready_sem.acquire()

    def _close(self):
        try:
            # prevent the worker thread from further work
            self._stopping = True
            # exhaust all remaining queue items to notify the background worker
            while not self._batch_queue.empty():
                self._batch_queue.get()
            # wait until the worker exit
            self._worker.join()
        finally:
            self._worker = None
            self._batch_queue = None
            self._worker_ready_sem = None
            self._initialized = False

    def _minibatch_iterator(self):
        self.init()

        try:
            # iterate through one epoch
            while self._worker_alive or not self._batch_queue.empty():
                epoch, payload = self._batch_queue.get()
                if epoch < self._epoch_counter:
                    # we've got a remaining item from the last epoch, skip it
                    pass
                elif epoch > self._epoch_counter:  # pragma: no cover
                    # we've accidentally got an item from the future epoch
                    # it should be a bug, and we shall report it
                    raise RuntimeError('Unexpected entry from future epoch.')
                elif payload is self.EPOCH_END:
                    # we've got the epoch ending mark for the current epoch,
                    # so we should break the loop
                    break
                elif isinstance(payload, self.ErrorBox):
                    # we've got an error, re-raise it
                    self.close()
                    raise payload.error
                else:
                    # we've got a normal batch for the current epoch,
                    # so yield it
                    yield payload
        finally:
            self._epoch_counter += 1

    def copy(self, **kwargs):
        return self._copy_helper(('prefetch',), source=self.source, **kwargs)
