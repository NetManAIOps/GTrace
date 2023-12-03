import time
from enum import Enum
from typing import *

from .utils import NOT_SET
from .stateful import StatefulObject

__all__ = [
    'CycleCounter', 'TimedCycleCounter', 'StageType', 'Stage',
]

FIRST_CYCLE_INDEX = 1


class CycleCounter(object):
    """
    Base cycle counter.

    A cycle is one step in a certain type of loop.  For example, in a train
    loop, there are typically multiple epoch cycles; and in each epoch,
    there are multiple batch cycles.

    When cycles are nested, one step of the external cycle is called
    a whole loop of the internal cycle.
    """

    __slots__ = ('index', 'total', 'original_total', 'avg_total', 'is_active',
                 'is_first_loop', '_avg_total_n_estimates')

    index: int  # index of the current cycle, start from 1.
    total: Optional[int]  # total number of cycles to run in each loop.
    original_total: Optional[int]  # the original `total`, specified in the constructor
    avg_total: Optional[float]  # average total number of cycles in each loop.
    is_active: bool  # is this cycle entered but not exited now?
    is_first_loop: int  # is this the first loop entered since the counter is created?
    _avg_total_n_estimates: int  # number of times to estimate `avg_total`

    def __init__(self,
                 index: int = FIRST_CYCLE_INDEX - 1,
                 total: Optional[int] = None):
        self.index = index
        self.total = total
        self.original_total = total
        self.avg_total = None
        self.is_active = False
        self.is_first_loop = True
        self._avg_total_n_estimates = 0

    def enter(self, index: Optional[int] = None):
        """
        Enter one cycle.

        Args:
            index: The index of this cycle.  If not specified, will
                increase ``self.index`` by 1.
        """
        if index is not None:
            self.index = index
        else:
            self.index += 1
        self.is_active = True

    def exit(self):
        """Exit this cycle."""
        self.is_active = False

    def enter_loop(self):
        """
        Enter the current loop.

        This will increase `loop_index` by 1, and set `index` to
        `FIRST_CYCLE_INDEX - 1` if it is not the first loop.
        """
        if not self.is_first_loop:
            self.index = FIRST_CYCLE_INDEX - 1

    def exit_loop(self):
        """
        Enter the next loop.

        This will update the `avg_total` estimation by the current `index`.
        """
        # set `is_first_loop` to False
        self.is_first_loop = False

        # update the `avg_total` counter
        this_total = self.index
        self._avg_total_n_estimates += 1
        n = self._avg_total_n_estimates
        if self.avg_total is None:
            self.avg_total = this_total
        elif abs(self.avg_total - this_total) > 1e-7:
            self.avg_total = ((n - 1.) / n * self.avg_total +
                              float(this_total) / n)

        # reset `total` to `original_total`
        self.total = self.original_total

    def estimated_cycles_ahead(self,
                               count_this_cycle: bool = NOT_SET
                               ) -> Optional[float]:
        """
        Get the estimated cycles ahead in the current loop.

        This method will use `avg_total` prior than `total`.  If `is_open`
        is :obj:`True`, then the current cycle will also be counted.

        Args:
            count_this_cycle: Whether or not to count the current cycle?
                If :obj:`True`, the current cycle (i.e., ``self.index``)
                will also be counted as cycles ahead.  Otherwise the current
                cycle will not be counted.  If not specified, will count
                this cycle only if ``self.is_active == True``.

        Returns:
            The estimated cycles ahead, may be a float number.
            Will be :obj:`None` if the total cycles is not known.
        """
        if count_this_cycle is NOT_SET:
            count_this_cycle = self.is_active
        total = self.avg_total if self.avg_total is not None else self.total
        if total is not None:
            ahead = total - self.index
            if count_this_cycle:
                ahead += 1
            return ahead

    def __str__(self):
        s = f'/{self.total}' if self.total is not None else ''
        return f'{self.index}{s}'


class TimedCycleCounter(CycleCounter):
    """A cycle counter with timer."""

    __slots__ = (
        'index', 'total', 'avg_total', 'is_active', 'loop_index',
        'start_timestamp', 'end_timestamp', 'last_cycle_time', 'avg_cycle_time',
        '_avg_cycle_time_n_estimates'
    )

    def __init__(self,
                 index: int = FIRST_CYCLE_INDEX - 1,
                 total: Optional[int] = None):
        super().__init__(index=index, total=total)
        self.start_timestamp: Optional[float] = None
        self.end_timestamp: Optional[float] = None
        self.avg_cycle_time: Optional[float] = None
        self.last_cycle_time: Optional[float] = None  # execution time of the last finished cycle
        self._avg_cycle_time_n_estimates: int = 0

    def enter(self, index: Optional[int] = None):
        super().enter(index)
        self.start_timestamp = time.time()

    def pre_exit(self):
        self.end_timestamp = time.time()
        self.last_cycle_time = self.end_timestamp - self.start_timestamp

        # update the `avg_cycle_time` estimation
        self._avg_cycle_time_n_estimates += 1
        if self.avg_cycle_time is None:
            self.avg_cycle_time = self.last_cycle_time
        else:
            n = self._avg_cycle_time_n_estimates
            self.avg_cycle_time = (
                (n - 1.) / n * self.avg_cycle_time +
                float(self.last_cycle_time) / n
            )

    def estimated_time_ahead(self,
                             count_this_cycle: bool = NOT_SET
                             ) -> Optional[float]:
        """
        Get the estimated time ahead (ETA).

        Args:
            count_this_cycle: Whether or not to count the current cycle?
                If :obj:`True`, the current cycle (i.e., ``self.index``)
                will also be counted as cycles ahead.  Otherwise the current
                cycle will not be counted.  If not specified, will count
                this cycle only if ``self.is_active == True``.

        Returns:
            The estimated time ahead, in seconds.
            Will be :obj:`None` if the total cycles is not known.
        """
        if self.avg_cycle_time is not None:
            cycles_ahead = self.estimated_cycles_ahead(count_this_cycle)
            if cycles_ahead is not None:
                return cycles_ahead * self.avg_cycle_time


class StageType(str, Enum):
    """Machine learning experiment stage types."""

    TRAIN = 'train'
    VALIDATION = 'validation'
    TEST = 'test'
    PREDICT = 'predict'

    @property
    def metric_prefixes(self) -> Tuple[str, ...]:
        """
        Get the allowed prefixes of the metric names for this stage.

        The correspondence between the stage type and the allowed prefixes
        is listed as follows:

        *  StageType.TRAIN: "", "train_"
        *  StageType.VALIDATION: "val_", "valid_"
        *  StageType.TEST: "test_"
        *  StageType.PREDICT: "pred_", "predict_"
        """
        return _METRIC_PREFIXES_FOR_STAGES[self]

    @property
    def metric_prefix(self) -> str:
        """Get the preferred metric name prefix."""
        return self.metric_prefixes[0]

    def add_metric_prefix(self, name: str) -> str:
        """
        Add stage metric prefix to the metric name, if absent.

        Args:
            name: The original metric name.

        Returns:
            The processed metric name.
        """
        if not any(name.startswith(pfx) for pfx in self.metric_prefixes):
            name = f'{self.metric_prefixes[0]}{name}'
        return name


_METRIC_PREFIXES_FOR_STAGES = {
    StageType.TRAIN: ('', 'train_'),
    StageType.VALIDATION: ('val_', 'valid_'),
    StageType.TEST: ('test_',),
    StageType.PREDICT: ('pred_', 'predict_',)
}


class _StageCounterState(StatefulObject):

    stage: 'Stage'

    def __init__(self, stage: 'Stage'):
        self.stage = stage

    def get_state_dict(self) -> Dict[str, Any]:
        ret = {}

        def get_counter(counter: CycleCounter,
                        index: Optional[int] = None) -> int:
            if index is None:
                index = counter.index
            if counter.is_active:
                index -= 1
            return index

        if self.stage.epoch is not None:
            ret['epoch'] = get_counter(self.stage.epoch)
        ret['batch'] = get_counter(self.stage.batch)
        if self.stage.global_step is not None:
            ret['global_step'] = get_counter(
                self.stage.batch, self.stage.global_step)
        ret['memo'] = self.stage.memo
        return ret

    def set_state_dict(self, state: Dict[str, Any]):
        if self.stage.epoch is not None:
            self.stage.epoch.index = state['epoch']
        if self.stage.batch is not None:
            self.stage.batch.index = state['batch']
        self.stage.global_step = state.get('global_step') or None
        self.stage.memo = dict(state.get('memo', {}))


class Stage(object):
    """
    Base class of a machine learning stage.

    A :class:`Stage` represents a certain (large) step of a machine learning
    experiment, which uses a given dataset for one or more epochs.
    The :class:`Stage` class maintains the epoch and batch counters, and
    organizes callbacks for the stage.

    Ordinary users should barely use this class, unless they are writing
    their own :class:`Callback` classes.
    """

    type: StageType
    """Type of the stage."""

    epoch: Optional[TimedCycleCounter]
    """The epoch counter."""

    batch: TimedCycleCounter
    """The batch counter."""

    batch_size: Optional[int]
    """Batch size, i.e., maximum number of samples in each batch."""

    data_length: Optional[int]
    """The total number of data samples."""

    global_step: Optional[int]
    """The global step counter."""

    memo: Dict[str, Any]
    """
    If anything is assigned to this dict, it will be saved in checkpoints
    and restored after recovered from the checkpoints.
    """

    callbacks: 'CallbackList'
    """The list of callbacks that should be called on various events."""

    known_metrics: Tuple[str, ...]
    """
    The names of known metrics.  Only stored but not used in :class:`Stage`,
    but may be useful for other callbacks, e.g., :class:`LoggerCallback`
    uses this field to determine the order of formatted metrics.
    """

    best_validation_mark: bool = False
    """
    Whether or not we've encountered the best validation metric in the last
    batch or epoch?  will be cleared at the begin of each epoch and batch.
    """

    _current_batch_size: Optional[int] = None
    """The size (i.e., batch size) of the current active batch."""

    _current_epoch_size: Optional[int] = None
    """The size (i.e., batch count) of the current active epoch."""

    is_active: bool = False
    """Whether or not this stage is entered but not exited?"""

    termination_requested: bool = False
    """Whether or not a callback has requested the train loop to terminate."""

    start_timestamp: Optional[float] = None
    """The timestamp when the last time :meth:`enter()` was called."""

    end_timestamp: Optional[float] = None
    """The timestamp when the last time :meth:`exit()` was called."""

    def __init__(self,
                 type: StageType,
                 epoch: int = FIRST_CYCLE_INDEX - 1,
                 max_epoch: Optional[int] = None,
                 batch: int = FIRST_CYCLE_INDEX - 1,
                 max_batch: Optional[int] = None,
                 batch_size: Optional[int] = None,
                 data_length: Optional[int] = None,
                 global_step: Optional[int] = None,
                 callbacks: Sequence['Callback'] = (),
                 known_metrics: Optional[Sequence[str]] = None):
        epoch = TimedCycleCounter(epoch, total=max_epoch) \
            if type == StageType.TRAIN else None
        batch = TimedCycleCounter(batch, total=max_batch)
        callbacks = CallbackList(callbacks)

        self.type = type
        self.epoch = epoch
        self.batch = batch
        self.batch_size = batch_size
        self.data_length = data_length
        self.global_step = global_step
        self.memo = {}
        self.callbacks = callbacks

        if known_metrics is not None:
            known_metrics = tuple([self.type.add_metric_prefix(k)
                                   for k in known_metrics])
        self.known_metrics = known_metrics

    @property
    def name(self) -> str:
        """
        Get the name of the stage.

        Returns:
            One of: {"train", "validation", "test", "predict"}.
        """
        return self.type.value

    def state_proxy(self) -> _StageCounterState:
        """
        Get the proxy object that helps to save and restore states of this
        stage object.
        """
        return _StageCounterState(self)

    def add_callback(self, callback: 'Callback'):
        """Add a callback to this stage."""
        self.callbacks.add(callback)

    def remove_callback(self, callback: 'Callback'):
        """Remove a callback from this stage."""
        self.callbacks.remove(callback)

    def get_eta(self) -> Optional[float]:
        if self.epoch is not None:
            # get the total batches
            max_batch = self.batch.avg_total or self.batch.total

            # estimate the epoch time
            epoch_time = self.epoch.avg_cycle_time
            if epoch_time is None:
                batch_time = self.batch.avg_cycle_time
                if batch_time is None or max_batch is None:
                    return None  # no way to estimate epoch time, return None
                epoch_time = batch_time * max_batch

            # estimate the epoch ahead
            epoch_ahead = self.epoch.estimated_cycles_ahead()
            if epoch_ahead is None:
                return None

            if max_batch:
                batch_ahead = self.batch.estimated_cycles_ahead()
                if batch_ahead is not None:
                    epoch_ahead = epoch_ahead - 1 + \
                        float(batch_ahead) / max_batch

            # now compute eta
            return epoch_ahead * epoch_time

        else:
            return self.batch.estimated_time_ahead()

    def push_metrics(self, metrics) -> None:
        if self.end_timestamp is not None:
            exc_time = self.end_timestamp - self.start_timestamp
        else:
            exc_time = None

        event_data = CallbackData(
            stage=self,
            index=None,
            size=None,
            start_timestamp=self.start_timestamp,
            end_timestamp=self.end_timestamp,
            exc_time=exc_time,
            metrics=metrics,
        )
        for cb in self.callbacks:
            cb.on_metrics(event_data)

    def enter(self) -> None:
        if self.start_timestamp:
            raise RuntimeError('`Stage` is neither re-entrant, nor reusable.')

        # initialize the statuses
        self.best_validation_mark = False
        self.is_active = True
        self.termination_requested = False
        self.start_timestamp = time.time()
        self.end_timestamp = None

        # call the callbacks
        event_name = f'on_{self.name}_begin'
        event_data = CallbackData(
            stage=self,
            index=None,
            size=None,
            start_timestamp=self.start_timestamp,
            end_timestamp=None,
            exc_time=None,
            metrics=None
        )
        for cb in self.callbacks:
            cb.on_stage_begin(event_data)
            getattr(cb, event_name)(event_data)

    def exit(self, metrics: Optional[Dict[str, Any]] = None):
        try:
            self.end_timestamp = time.time()

            # call the callbacks
            event_name = f'on_{self.name}_end'
            event_data = CallbackData(
                stage=self,
                index=None,
                size=None,
                start_timestamp=self.start_timestamp,
                end_timestamp=self.end_timestamp,
                exc_time=self.end_timestamp - self.start_timestamp,
                metrics=metrics
            )
            for cb in self.callbacks:
                cb.on_metrics(event_data)
                getattr(cb, event_name)(event_data)
                cb.on_stage_end(event_data)
        finally:
            self.is_active = False

    def enter_epoch(self,
                    epoch: Optional[int] = None,
                    epoch_size: Optional[int] = None):
        if self.epoch is None:
            raise RuntimeError(f'Stage {self!r} does not have an epoch '
                               f'counter.')
        self.epoch.enter(epoch)
        self.batch.enter_loop()
        self._current_epoch_size = epoch_size
        self.best_validation_mark = False

        # call the callbacks
        event_name = f'on_{self.name}_epoch_begin'
        event_data = CallbackData(
            stage=self,
            index=self.epoch.index,
            size=self._current_epoch_size,
            start_timestamp=self.epoch.start_timestamp,
            end_timestamp=None,
            exc_time=None,
            metrics=None
        )
        for cb in self.callbacks:
            cb.on_epoch_begin(event_data)
            getattr(cb, event_name)(event_data)

    def exit_epoch(self,
                   metrics: Optional[Dict[str, Any]] = None):
        if self.epoch is None:
            raise RuntimeError(f'Stage {self!r} does not have an epoch '
                               f'counter.')
        self.epoch.pre_exit()

        try:
            # call the callbacks
            event_name = f'on_{self.name}_epoch_end'
            event_data = CallbackData(
                stage=self,
                index=self.epoch.index,
                size=self._current_epoch_size,
                start_timestamp=self.epoch.start_timestamp,
                end_timestamp=self.epoch.end_timestamp,
                exc_time=self.epoch.last_cycle_time,
                metrics=metrics
            )
            for cb in self.callbacks:
                cb.on_metrics(event_data)
                getattr(cb, event_name)(event_data)
                cb.on_epoch_end(event_data)

        finally:
            self.epoch.exit()
            self.batch.exit_loop()
            self._current_epoch_size = None

    def enter_batch(self,
                    batch: Optional[int] = None,
                    batch_size: Optional[int] = None):
        self.batch.enter(batch)
        if self.global_step is not None:
            self.global_step += 1
        self._current_batch_size = batch_size
        self.best_validation_mark = False

        # call the callbacks
        event_name = f'on_{self.name}_batch_begin'
        event_data = CallbackData(
            stage=self,
            index=self.batch.index,
            size=self._current_batch_size,
            start_timestamp=self.batch.start_timestamp,
            end_timestamp=None,
            exc_time=None,
            metrics=None
        )
        for cb in self.callbacks:
            cb.on_batch_begin(event_data)
            getattr(cb, event_name)(event_data)

    def exit_batch(self,
                   metrics: Optional[Dict[str, Any]] = None):
        self.batch.pre_exit()
        try:
            # call the callbacks
            event_name = f'on_{self.name}_batch_end'
            event_data = CallbackData(
                stage=self,
                index=self.batch.index,
                size=self._current_batch_size,
                start_timestamp=self.batch.start_timestamp,
                end_timestamp=self.batch.end_timestamp,
                exc_time=self.batch.last_cycle_time,
                metrics=metrics
            )
            for cb in self.callbacks:
                cb.on_metrics(event_data)
                getattr(cb, event_name)(event_data)
                cb.on_batch_end(event_data)

        finally:
            self.batch.exit()
            self._current_batch_size = None

    def request_termination(self):
        self.termination_requested = True


# imported for type annotation on `CallbackData` and `Callback`
from .callbacks import CallbackData, Callback, CallbackList
