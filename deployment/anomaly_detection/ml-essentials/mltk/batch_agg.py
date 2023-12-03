import operator
from enum import Enum
from functools import reduce
from typing import *

import numpy as np

from .stage import StageType
from .utils import ALL, NOT_SET

__all__ = [
    'BatchAggregationMode',
    'BatchAggregator', 'BatchAggregatorDict',
]


class BatchAggregationMode(str, Enum):

    CONCAT = 'CONCAT'
    """To concat the batch arrays along specified axis."""

    SUM = 'SUM'
    """To sum the batch arrays along specified axis."""

    AVERAGE = 'AVERAGE'
    """To average the batch arrays along specified axis."""


class BatchAggregator(object):
    """
    Class to aggregate batch arrays.

    >>> agg = BatchAggregator(BatchAggregationMode.CONCAT)
    >>> agg
    BatchAggregator(mode=CONCAT, axis=0)
    >>> agg.add(np.array([1, 2, 3, 4]))
    >>> agg.add(np.array([5, 6]))
    >>> agg.get()
    array([1, 2, 3, 4, 5, 6])

    >>> agg = BatchAggregator(BatchAggregationMode.AVERAGE)
    >>> agg
    BatchAggregator(mode=AVERAGE, axis=None)
    >>> agg.add(np.array([1, 2, 3, 4]))
    >>> agg.add(np.array([5, 6]))
    >>> agg.get()
    3.5

    >>> agg = BatchAggregator(BatchAggregationMode.SUM)
    >>> agg
    BatchAggregator(mode=SUM, axis=None)
    >>> agg.add(np.array([1, 2, 3, 4]))
    >>> agg.add(np.array([5, 6]))
    >>> agg.get()
    21
    """

    mode: BatchAggregationMode
    axis: Union[int, Tuple[int, ...]]

    def __init__(self,
                 mode: Union[str, BatchAggregationMode],
                 axis: Optional[Union[int, Tuple[int, ...], List[int]]] = NOT_SET):
        """
        Construct a new :class:`BatchAggregator`.

        Args:
            mode: Aggregation mode.
            axis: The axis to aggregate.  Defaults to `0` for `CONCAT` mode,
                while :obj:`None` for `SUM` and `AVERAGE` mode.
        """
        mode = BatchAggregationMode(mode)

        if axis is NOT_SET:
            axis = 0 if mode == BatchAggregationMode.CONCAT else None
        if mode == BatchAggregationMode.CONCAT:
            if not isinstance(axis, int):
                raise TypeError('`axis` must be a int when `mode` is CONCAT.')
        if axis is not None:
            if hasattr(axis, '__iter__'):
                axis = tuple(int(v) for v in axis)
                if len(axis) == 1:
                    axis = axis[0]
            else:
                axis = int(axis)

        self.mode = mode
        self.axis = axis
        self._buf = None
        self._weight_sum = 0.

    def __repr__(self):
        return f'{self.__class__.__qualname__}' \
               f'(mode={self.mode.value}, axis={self.axis})'

    def get(self) -> Optional[np.ndarray]:
        """
        Get the aggregation result.

        Returns:
            The result, or :obj:`None` if no value has been collected.
        """
        if self._buf is not None:
            if self.mode == BatchAggregationMode.CONCAT:
                return np.concatenate(self._buf, axis=self.axis)
            else:
                return self._buf

    def add(self,
            values: np.ndarray,
            weight: Optional[float] = 1.):
        """
        Add a batch array to the aggregator.

        Args:
            values: The batch array.
            weight: The batch weight, used only in `AVERAGE` mode.
        """
        # CONCAT: append the values to the buf
        if self.mode == BatchAggregationMode.CONCAT:
            if self._buf is None:
                self._buf = []
            self._buf.append(values)

        # SUM
        elif self.mode == BatchAggregationMode.SUM:
            batch_sum = np.sum(values, axis=self.axis)
            if self._buf is None:
                self._buf = batch_sum
            else:
                self._buf += batch_sum

        # AVERAGE: maintain the `total_weight` state and update the buf
        else:
            # infer the batch size and weight
            batch_shape = np.shape(values)
            if self.axis is None:
                batch_size = float(reduce(operator.mul, np.shape(values), 1.))
            elif isinstance(self.axis, tuple):
                batch_size = 1.
                for a in self.axis:
                    batch_size *= batch_shape[a]
            else:
                batch_size = batch_shape[self.axis]
            batch_weight = weight * batch_size

            # do update the weight
            self._weight_sum += batch_weight
            r1 = weight / self._weight_sum
            batch_sum = np.sum(values, axis=self.axis)
            if self._buf is None:
                self._buf = r1 * batch_sum
            else:
                r2 = batch_weight / self._weight_sum
                self._buf += r1 * batch_sum - r2 * self._buf


class BatchAggregatorDict(Mapping[str, BatchAggregator]):
    """
    Maintain a dict of :class:`BatchAggregator` instances, maybe with
    a default factory to construct :class:`BatchAggregator` instance
    for new keys.

    >>> agg_dict = BatchAggregatorDict.new()
    >>> agg_dict['acc'].add(np.array([0.75, 0.875]))
    >>> agg_dict['loss'].add(np.array([0.125, 0.2]))
    >>> len(agg_dict)
    2
    >>> list(agg_dict)
    ['acc', 'loss']
    >>> agg_dict['acc'].get()
    0.8125
    >>> agg_dict['loss'].get()
    0.1625
    """

    @staticmethod
    def new(metrics: Union[Sequence[str], type(ALL)] = ALL,
            outputs: Union[Sequence[str], type(ALL)] = (),
            aggregators: Optional[Mapping[str, BatchAggregator]] = None,
            excludes: Sequence[str] = (),
            stage_type: Optional[StageType] = None) -> 'BatchAggregatorDict':
        """
        Construct a new :class:`BatchAggregatorDict` according to the field
        settings `metrics`, `outputs` and `aggregators`.

        Args:
            metrics: The names of the batch arrays, which should be aggregated
                by ``BatchAggregator('AVERAGE', axis=None)``.  :obj:`ALL`
                indicates that an array is by default a metric if it is neither
                specified in `outputs` nor in `aggregator`.
            outputs: The names of the batch arrays, which should be aggregated
                by ``BatchAggregator('CONCAT', axis=0)``.  :obj:`ALL`
                indicates that an array is by default an output if it is neither
                specified in `outputs` nor in `aggregator`.
            aggregators: The dict of names and their corresponding aggregators.
            excludes: The names to exclude.  If a name is excluded, no
                aggregator will be designated to this name, i.e., ``get(name)``
                returns None, and ``__getitem__(name)`` raises `KeyError`.
            stage_type: If specified, will add stage metric prefix to the keys
                of `metrics`, `outputs` and `aggregators`.

        Returns:
            The aggregator dict.

        Notes:
            :obj:`ALL` could be specified to at most one of `metrics`
            and `outputs`.  The argument `aggregators` has higher priority
            than `outputs`, and so does `outputs` have higher priority than
            `metrics`.  That is to say, if a name is specified in both
            `aggregators` and `outputs`, then the aggregator specified in
            `aggregators` will be chosen; this is also true if a name is
            specified in both `outputs` and `metrics`.
        """
        # the aggregator factories
        average_aggregator_factory = lambda: \
            BatchAggregator(mode=BatchAggregationMode.AVERAGE, axis=None)
        concat_aggregator_factory = lambda: \
            BatchAggregator(mode=BatchAggregationMode.CONCAT, axis=0)

        # determine the default factory
        if metrics == ALL and outputs == ALL:
            raise ValueError('Only one of `metrics` and `outputs` can be '
                             '`ALL`.')
        elif metrics == ALL:
            default_factory = average_aggregator_factory
        elif outputs == ALL:
            default_factory = concat_aggregator_factory
        else:
            default_factory = None

        # build the aggregator instances
        agg_dict = {}
        if metrics != ALL and metrics:
            for key in metrics:
                if stage_type is not None:
                    key = stage_type.add_metric_prefix(key)
                agg_dict[key] = average_aggregator_factory()
        if outputs != ALL and outputs:
            for key in outputs:
                if stage_type is not None:
                    key = stage_type.add_metric_prefix(key)
                agg_dict[key] = concat_aggregator_factory()
        if aggregators:
            for key, agg in aggregators.items():
                if stage_type is not None:
                    key = stage_type.add_metric_prefix(key)
                agg_dict[key] = agg

        # build the excludes names
        if excludes and stage_type is not None:
            excludes = [stage_type.add_metric_prefix(n) for n in excludes]

        # now construct the `BatchAggregatorDict` instance
        return BatchAggregatorDict(
            agg_dict, excludes=excludes, default_factory=default_factory)

    def __init__(self,
                 aggregators: Mapping[str, BatchAggregator],
                 excludes: Sequence[str] = (),
                 default_factory: Optional[
                     Callable[[], BatchAggregator]] = None):
        """
        Construct a new :class:`BatchAggregatorDict`.

        Args:
            aggregators: The mapping from names to aggregators.
            excludes: The names to exclude from this dict.  If a name is
                excluded, no aggregator will be designated to this name,
                i.e., ``get(name)`` returns None, and ``__getitem__(name)``
                raises :class:`KeyError`.
            default_factory: The default factory, which is used to create
                new :class:`BatchAggregator` instances if the aggregator
                to a requested name does not exist.  If not specified,
                accessing non-existing name will raise an error.
        """
        self._aggregators = {}
        self._excludes = set(excludes or ())
        self._default_factory = default_factory

        for key in aggregators:
            if key not in self._excludes:
                agg = aggregators[key]
                if not isinstance(agg, BatchAggregator):
                    raise TypeError(f'Item {key!r} is not an instance of '
                                    f'{BatchAggregator.__qualname__}: '
                                    f'{agg!r}')
                self._aggregators[key] = agg

    def get(self, item: str, default: Any = None) -> Optional[BatchAggregator]:
        if item not in self._excludes:
            if item not in self._aggregators:
                if self._default_factory is not None:
                    self._aggregators[item] = self._default_factory()
                else:
                    return default
            return self._aggregators[item]

    def __getitem__(self, item: str) -> BatchAggregator:
        ret = self.get(item)
        if ret is None:
            raise KeyError(item)
        return ret

    def __len__(self) -> int:
        return len(self._aggregators)

    def __iter__(self) -> Iterator[str]:
        return iter(self._aggregators)
