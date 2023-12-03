# -*- coding: utf-8 -*-
import math
import operator
from dataclasses import dataclass
from functools import reduce
from typing import *

import numpy as np

from .typing_ import *

__all__ = [
    'MetricStats',
    'MetricCollector', 'GeneralMetricCollector',
    'ScalarMetricCollector', 'ScalarMetricsLogger',
]


@dataclass
class MetricStats(object):
    """Mean and std(dev) of a metric."""

    __slots__ = ('mean', 'var', 'std')

    mean: MetricValue
    var: Optional[MetricValue]
    std: Optional[MetricValue]

    def to_json(self, mean_only: bool = False
                ) -> Union[MetricValue, Dict[str, MetricValue]]:
        """
        Get the JSON representation of this metric stats object.

        >>> MetricStats(mean=1.0, var=None, std=None).to_json()
        1.0
        >>> MetricStats(mean=1.0, var=4.0, std=2.0).to_json()
        {'mean': 1.0, 'std': 2.0}
        >>> MetricStats(mean=1.0, var=4.0, std=2.0).to_json(mean_only=True)
        1.0

        Args:
            mean_only: Whether or not to include only the mean of the metric.

        Returns:
            The JSON metric value.
        """
        if mean_only or self.std is None:
            return self.mean
        return {'mean': self.mean, 'std': self.std}


class MetricCollector(object):
    """Base class of a metric statistics collector."""

    def reset(self):
        """Reset the collector to initial state."""
        raise NotImplementedError()

    @property
    def has_stats(self) -> bool:
        """Whether or not any value has been collected?"""
        raise NotImplementedError()

    @property
    def stats(self) -> Optional[MetricStats]:
        """
        Get the metric object.

        Returns:
             The statistics, or :obj:`None` if no value has been collected.
        """
        raise NotImplementedError()

    def update(self, values: MetricValue, weight: MetricValue = 1.):
        """
        Update the metric statistics from values.

        This method uses the following equation to update `mean` and `square`:

        .. math::
            \\frac{\\sum_{i=1}^n w_i f(x_i)}{\\sum_{j=1}^n w_j} =
                \\frac{\\sum_{i=1}^m w_i f(x_i)}{\\sum_{j=1}^m w_j} +
                \\frac{\\sum_{i=m+1}^n w_i}{\\sum_{j=1}^n w_j} \\Bigg(
                    \\frac{\\sum_{i=m+1}^n w_i f(x_i)}{\\sum_{j=m+1}^n w_j} -
                    \\frac{\\sum_{i=1}^m w_i f(x_i)}{\\sum_{j=1}^m w_j}
                \\Bigg)

        Args:
            values: Values to be collected in batch, numpy array or scalar
                whose shape ends with ``self.shape``. The leading shape in
                front of ``self.shape`` is regarded as the batch shape.
            weight: Weights of the `values`, should be broadcastable against
                the batch shape. (default is 1)

        Raises:
            ValueError: If the shape of `values` does not end with `self.shape`.
        """
        raise NotImplementedError()


class GeneralMetricCollector(MetricCollector):
    """
    Class to collect statistics of metric values.

    To collect statistics of a scalar:

    >>> collector = GeneralMetricCollector()
    >>> collector.stats is None
    True
    >>> collector.update(1.)
    >>> collector.stats  # doctest: +ELLIPSIS
    MetricStats(mean=1.0, var=None, std=None)
    >>> for value in [2., 3., 4.]:
    ...     collector.update(value)
    >>> collector.stats  # doctest: +ELLIPSIS
    MetricStats(mean=2.5, var=1.25, std=1.11803...)
    >>> collector.update(np.array([5., 6., 7., 8.]))
    >>> collector.stats  # doctest: +ELLIPSIS
    MetricStats(mean=4.5, var=5.25, std=2.29128...)

    weighted statistics:

    >>> collector = GeneralMetricCollector()
    >>> for value in [1., 2., 3., 4.]:
    ...     collector.update(value, weight=value)
    >>> collector.stats  # doctest: +ELLIPSIS
    MetricStats(mean=3.0, var=1.0, std=1.0)
    >>> collector.update(np.array([5., 6., 7., 8.]),
    ...                  weight=np.array([5., 6., 7., 8.]))
    >>> collector.stats  # doctest: +ELLIPSIS
    MetricStats(mean=5.66666..., var=3.88888..., std=1.97202...)

    To collect element-wise statistics of a vector:

    >>> collector = GeneralMetricCollector(shape=[3])
    >>> x = np.arange(12).reshape([4, 3])
    >>> for value in x:
    ...     collector.update(value)
    >>> collector.stats  # doctest: +ELLIPSIS
    MetricStats(mean=array([4.5, 5.5, 6.5]), var=array([11.25, 11.25, 11.25]), std=array([3.35410..., 3.35410..., 3.35410...]))
    """

    __slots__ = ('shape', 'dtype', 'mean', 'second_order_moment',
                 'counter', 'weight_sum', '_array_to_value_type',
                 '_value')

    shape: ArrayShape
    dtype: np.dtype
    mean: np.ndarray
    second_order_moment: np.ndarray
    counter: int  # count the number of times where `add` is called
    weight_sum: float  # sum of all weights of added values
    _array_to_value_type: Callable[[Any], MetricValue]
    _value: Optional[MetricStats]

    def __init__(self,
                 shape: Sequence[int] = (),
                 dtype: np.dtype = np.float64):

        self.shape = tuple(map(int, shape))
        self.dtype = np.dtype(dtype)
        self.reset()
        if self.shape == ():
            self._array_to_value_type = float
        else:
            self._array_to_value_type = lambda v: v

    def reset(self):
        self.mean = np.zeros(shape=self.shape, dtype=self.dtype)  # E[X]
        self.second_order_moment = np.zeros(shape=self.shape, dtype=self.dtype)  # E[X^2]
        self.counter = 0
        self.weight_sum = 0.
        self._value = None

    @property
    def has_stats(self) -> bool:
        return self.counter > 0

    def _make_value(self):
        if self.counter > 0:
            mean = self._array_to_value_type(self.mean)
            if self.counter > 1:
                var = self._array_to_value_type(
                    np.maximum(self.second_order_moment - self.mean ** 2, 0.))
                std = self._array_to_value_type(np.sqrt(var))
            else:
                var = std = None
            self._value = MetricStats(mean=mean, var=var, std=std)

    @property
    def stats(self) -> Optional[MetricStats]:
        if self._value is None:
            self._make_value()
        return self._value

    def update(self, values: MetricValue, weight: MetricValue = 1.):
        values = np.asarray(values)
        if not values.size:
            return
        weight = np.asarray(weight)
        if not weight.size:
            weight = np.asarray(1.)

        if self.shape:
            if values.shape[-len(self.shape):] != self.shape:
                raise ValueError(
                    'Shape mismatch: {} not ending with {}'.format(
                        values.shape, self.shape
                    )
                )
            batch_shape = values.shape[:-len(self.shape)]
        else:
            batch_shape = values.shape

        batch_weight = np.ones(shape=batch_shape, dtype=np.float64) * weight
        batch_weight = np.reshape(batch_weight,
                                  batch_weight.shape + (1,) * len(self.shape))
        batch_weight_sum = np.sum(batch_weight)
        normed_batch_weight = batch_weight / batch_weight_sum

        self.weight_sum += batch_weight_sum
        discount = batch_weight_sum / self.weight_sum

        def update_array(arr, update):
            reduce_axis = tuple(range(len(batch_shape)))
            update_reduced = normed_batch_weight * update
            if reduce_axis:
                update_reduced = np.sum(update_reduced, axis=reduce_axis)
            arr += discount * (update_reduced - arr)

        update_array(self.mean, values)
        update_array(self.second_order_moment, values ** 2)
        self.counter += batch_weight.size
        self._value = None


class ScalarMetricCollector(MetricCollector):
    """
    Class to collect statistics of scalar metric values.

    This class should be faster than the general :class:`MetricCollector`.

    >>> collector = ScalarMetricCollector()
    >>> collector.has_stats
    False
    >>> collector.stats is None
    True
    >>> collector.update(1.)
    >>> collector.has_stats
    True
    >>> collector.stats  # doctest: +ELLIPSIS
    MetricStats(mean=1.0, var=None, std=None)
    >>> for value in [2., 3., 4.]:
    ...     collector.update(value)
    >>> collector.stats  # doctest: +ELLIPSIS
    MetricStats(mean=2.5, var=1.25, std=1.11803...)
    >>> collector.update(np.array([5., 6., 7., 8.]))
    >>> collector.stats  # doctest: +ELLIPSIS
    MetricStats(mean=4.5, var=5.25, std=2.29128...)

    weighted statistics:

    >>> collector = ScalarMetricCollector()
    >>> for value in [1., 2., 3., 4.]:
    ...     collector.update(value, weight=value)
    >>> collector.stats  # doctest: +ELLIPSIS
    MetricStats(mean=3.0, var=1.0, std=1.0)
    """

    __slots__ = ('mean', 'second_order_moment', 'counter', 'weight_sum')

    mean: float
    second_order_moment: float
    counter: int  # number of times `add` is called
    weight_sum: float  # sum of all weights of added values

    def __init__(self):
        self.reset()

    def reset(self):
        self.counter = 0
        self.mean = self.second_order_moment = self.weight_sum = 0.

    @property
    def has_stats(self) -> bool:
        return self.counter > 0

    @property
    def stats(self) -> Optional[MetricStats]:
        if self.counter > 1:
            var = max(self.second_order_moment - self.mean ** 2, 0.)
            return MetricStats(mean=self.mean, var=var, std=math.sqrt(var))
        elif self.counter > 0:
            return MetricStats(mean=self.mean, var=None, std=None)

    def update(self, values: MetricValue, weight: MetricValue = 1.):
        if np.shape(weight) == ():
            batch_size = float(reduce(operator.mul, np.shape(values), 1.))
            batch_weight = float(weight * batch_size)

            self.weight_sum = self.weight_sum + batch_weight
            r1 = weight / self.weight_sum
            r2 = batch_weight / self.weight_sum

            self.mean += r1 * float(np.sum(values)) - r2 * self.mean
            self.second_order_moment += (r1 * float(np.sum(values ** 2)) -
                                         r2 * self.second_order_moment)

            self.counter += 1
        else:  # pragma: no cover
            raise RuntimeError(
                '`ScalarMetricCollector` only supports scalar `weight` '
                'argument.  Please use `GeneralMetricCollector` if you need '
                'per-element weight on `values`.'
            )


class ScalarMetricsLogger(Mapping[str, GeneralMetricCollector]):
    """
    Class to log named scalar metrics.

    Metrics can be collected via :meth:`update()`, and a weight can be
    granted to each update:

    >>> logger = ScalarMetricsLogger()
    >>> logger.update({'train_loss': 5, 'train_acc': 90})
    >>> logger.update({'train_loss': 6}, weight=3)
    >>> len(logger)
    2
    >>> logger['train_loss'].stats  # doctest: +ELLIPSIS
    MetricStats(mean=5.75, var=0.1875, std=0.433012...)
    >>> list(logger)
    ['train_loss', 'train_acc']
    >>> for key, val in logger.items():  # doctest: +ELLIPSIS
    ...     print(key, val.stats)
    train_loss MetricStats(mean=5.75, var=0.1875, std=0.433012...)
    train_acc MetricStats(mean=90.0, var=None, std=None)

    :meth:`replace()` can discard some of the collected metrics, and
    replace with new values.  For example, we replace `train_loss`
    with a new value `5` (also we add a new metric `train_time`):

    >>> logger.replace({'train_loss': 5, 'train_time': 3})
    >>> for key, val in logger.items():  # doctest: +ELLIPSIS
    ...     print(key, val.stats)
    train_loss MetricStats(mean=5.0, var=None, std=None)
    train_acc MetricStats(mean=90.0, var=None, std=None)
    train_time MetricStats(mean=3.0, var=None, std=None)

    :meth:`clear()` can clear all collected metrics:

    >>> logger.clear()
    >>> list(logger)
    []
    """

    __slots__ = ('_metrics',)

    _metrics: Dict[str, ScalarMetricCollector]

    def __init__(self):
        self._metrics = {}

    def __iter__(self) -> Iterator[str]:
        return iter(self._metrics)

    def __len__(self) -> int:
        return len(self._metrics)

    def __getitem__(self, item) -> ScalarMetricCollector:
        return self._metrics[item]

    def items(self) -> ItemsView[str, ScalarMetricCollector]:
        return self._metrics.items()

    def clear(self) -> None:
        """Clear all collected metrics."""
        self._metrics.clear()

    def update(self, metrics: Mapping[str, MetricValue], weight: float = 1.):
        """
        Update metrics with new values.

        Args:
            metrics: The new metric values.
            weight: The weight of the metric values.
        """
        for key, val in metrics.items():
            if key not in self._metrics:
                self._metrics[key] = ScalarMetricCollector()
            self._metrics[key].update(val, weight)

    def replace(self, metrics: Mapping[str, MetricValue], weight: float = 1.):
        """
        Replace metrics with new values.

        The behaviour of this method can be briefly described as::

            for key, val in metrics:
                self[key].reset()
                self[key].update(val, weight)

        Args:
            metrics: The new metrics values.
            weight: The weight of the metric values.
        """
        for key, val in metrics.items():
            if key not in self._metrics:
                self._metrics[key] = ScalarMetricCollector()
            else:
                self._metrics[key].reset()
            self._metrics[key].update(val, weight)

    def to_json(self, mean_only: bool = False
                ) -> Dict[str, Union[MetricValue, Dict[str, MetricValue]]]:
        """
        Get the JSON representation of collected metrics.

        >>> logger = ScalarMetricsLogger()
        >>> logger.update({'train_loss': 5, 'train_acc': 90})
        >>> logger.update({'train_loss': 6}, weight=3)
        >>> logger.to_json()  # doctest: +ELLIPSIS
        {'train_loss': {'mean': 5.75, 'std': 0.433012...}, 'train_acc': 90.0}
        >>> logger.to_json(mean_only=True)
        {'train_loss': 5.75, 'train_acc': 90.0}

        Args:
            mean_only: Whether or not to include only the mean of the metrics.

        Returns:
            The JSON metrics dict.
        """
        return {k: v.stats.to_json(mean_only=mean_only)
                for k, v in self._metrics.items()}
