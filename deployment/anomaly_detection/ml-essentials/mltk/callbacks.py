import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from enum import IntFlag
from logging import getLogger
from typing import *

import numpy as np

from .checkpoint import BaseCheckpoint, CheckpointManager
from .errors import NaNMetricError
from .formatting import MetricsFormatter, format_duration, format_as_asctime
from .logging_ import print_with_time
from .metrics import ScalarMetricsLogger, ScalarMetricCollector
from .mlstorage import ExperimentDoc
from .stateful import StatefulObjectGroup, StatefulObject
from .utils import NOT_SET

__all__ = [
    'CallbackData', 'Callback', 'CallbackList',
    'LoggerMode', 'LoggerCallback', 'StopOnNaN',
    'BaseTrainCallback', 'BaseCheckpointCallback',
    'AutoCheckpoint', 'EarlyStopping',
]


@dataclass
class CallbackData(object):
    """
    Data carried by a cycle begin/end event from :class:`Callback`.
    """

    __slots__ = ('stage', 'index', 'size', 'start_timestamp',
                 'end_timestamp', 'exc_time', 'metrics')

    stage: 'Stage'
    """The stage that calls the callback."""

    index: Optional[int]
    """Index of the epoch or batch, start from 1."""

    size: Optional[int]
    """The size of the batch."""

    start_timestamp: float
    """Start timestamp of the stage/epoch/batch."""

    end_timestamp: Optional[float]
    """End timestamp of the stage/epoch/batch, available at the cycle end."""

    exc_time: Optional[float]
    """Execution time of the stage/epoch/batch, available at the cycle end."""

    metrics: Optional[Dict[str, Any]]
    """Metrics dict, available at the cycle end."""


class Callback(object):
    """Base class of a callback for a machine learning stage."""

    priority: int = 0
    """
    The priority of the callback.  Smaller priority indicates the callback
    should be called earlier than other callbacks with larger priorities.
    """

    ###########
    # metrics #
    ###########
    def on_metrics(self, data: CallbackData):
        pass  # pragma: no cover

    ##################
    # general events #
    ##################
    def on_stage_begin(self, data: CallbackData):
        pass  # pragma: no cover

    def on_stage_end(self, data: CallbackData):
        pass  # pragma: no cover

    def on_epoch_begin(self, data: CallbackData):
        pass  # pragma: no cover

    def on_epoch_end(self, data: CallbackData):
        pass  # pragma: no cover

    def on_batch_begin(self, data: CallbackData):
        pass  # pragma: no cover

    def on_batch_end(self, data: CallbackData):
        pass  # pragma: no cover

    ################
    # train events #
    ################
    def on_train_begin(self, data: CallbackData):
        pass  # pragma: no cover

    def on_train_end(self, data: CallbackData):
        pass  # pragma: no cover

    def on_train_epoch_begin(self, data: CallbackData):
        pass  # pragma: no cover

    def on_train_epoch_end(self, data: CallbackData):
        pass  # pragma: no cover

    def on_train_batch_begin(self, data: CallbackData):
        pass  # pragma: no cover

    def on_train_batch_end(self, data: CallbackData):
        pass  # pragma: no cover

    #####################
    # validation events #
    #####################
    def on_validation_begin(self, data: CallbackData):
        pass  # pragma: no cover

    def on_validation_end(self, data: CallbackData):
        pass  # pragma: no cover

    def on_validation_batch_begin(self, data: CallbackData):
        pass  # pragma: no cover

    def on_validation_batch_end(self, data: CallbackData):
        pass  # pragma: no cover

    ###############
    # test events #
    ###############
    def on_test_begin(self, data: CallbackData):
        pass  # pragma: no cover

    def on_test_end(self, data: CallbackData):
        pass  # pragma: no cover

    def on_test_batch_begin(self, data: CallbackData):
        pass  # pragma: no cover

    def on_test_batch_end(self, data: CallbackData):
        pass  # pragma: no cover

    ##################
    # predict events #
    ##################
    def on_predict_begin(self, data: CallbackData):
        pass  # pragma: no cover

    def on_predict_end(self, data: CallbackData):
        pass  # pragma: no cover

    def on_predict_batch_begin(self, data: CallbackData):
        pass  # pragma: no cover

    def on_predict_batch_end(self, data: CallbackData):
        pass  # pragma: no cover


class CallbackList(Sequence[Callback]):
    """
    A callback list, which maintains the orders of callbacks according to
    their priority.
    """

    _SORTED = object()

    def __init__(self,
                 callbacks: Optional[Iterator[Callback]] = None,
                 *,
                 _sorted=None):
        if callbacks is not None:
            if _sorted is not self._SORTED:
                callbacks = sorted(callbacks, key=lambda cb: cb.priority)
            else:
                callbacks = list(callbacks)
        self._callbacks = callbacks

    def __len__(self) -> int:
        return len(self._callbacks)

    def __iter__(self):
        return iter(self._callbacks)

    def __eq__(self, other):
        return isinstance(other, CallbackList) and \
            self._callbacks == other._callbacks

    def __getitem__(self, item):
        return self._callbacks[item]

    def __delitem__(self, item):
        del self._callbacks[item]

    def __copy__(self):
        return self.clone()

    def clone(self) -> 'CallbackList':
        return CallbackList(self._callbacks, _sorted=self._SORTED)

    def add(self, callback: Callback):
        """
        Add a callback to this list, respecting the `priority`.

        Args:
            callback: The callback object.
        """
        i = len(self._callbacks) - 1
        while i > -1:
            if self._callbacks[i].priority <= callback.priority:
                break
            i -= 1
        self._callbacks.insert(i + 1, callback)

    def remove(self, callback: Callback):
        """
        Remove a callback from this list.

        Args:
            callback: The callback to be removed.

        Raises:
            ValueError: If `callback` is not present.
        """
        self._callbacks.remove(callback)


@dataclass
class _LoggerContext(object):
    """The context of an open stage in :class:`LoggerCallback`."""

    __slots__ = ('stage', 'progress', 'metrics_collector', 'batch_metrics',
                 'last_console_log_time', 'last_remote_push_time')

    stage: 'Stage'
    progress: Dict[str, Any]
    metrics_collector: ScalarMetricsLogger
    """
    Metrics logger to accumulate the mean and std of metrics.  This logger
    will be cleared at the beginning when `on_epoch_begin` is called.

    For validation, test and predict, this should effectively accumulate
    the metrics throughout the whole stage, since the `on_epoch_begin`
    callback will never be called.
    """
    batch_metrics: Dict[str, Any]
    """
    The current batch metrics.  Will be cleared after each batch.
    """
    last_console_log_time: float
    """Last time that the logs have been written to console."""
    last_remote_push_time: float
    """Last time that the logs have been pushed to remote."""

    @staticmethod
    def new_context(stage) -> '_LoggerContext':
        now_time = time.time()
        return _LoggerContext(
            stage=stage,
            progress={},
            metrics_collector=ScalarMetricsLogger(),
            batch_metrics={},
            # set these two log times as the current time, such that these
            # logs will not be written immediately after the stage begins.
            last_console_log_time=now_time,
            last_remote_push_time=now_time,
        )

    def update_metrics(self,
                       metrics: Mapping[str, Any],
                       replace: bool = False,
                       batch_size: Optional[float] = None) -> None:
        """
        Update the epoch metrics logger and batch metrics dict (if a batch
        is currently active) according to `metrics`.

        Args:
            metrics: The batch, epoch or stage metrics from stage callback.
            replace: Whether to replace the epoch/stage metrics
                instead of updating them.
            batch_size: The batch size information from stage callback.
        """
        # We expect the metrics to be scalars.  If not, we shall take average.
        raw_metrics = {}
        averaged_metrics = {}
        if metrics:
            for key, val in metrics.items():
                key = self.stage.type.add_metric_prefix(key)
                if np.shape(val) == ():
                    averaged_metrics[key] = val
                else:
                    raw_metrics[key] = val

            updater = self.metrics_collector.replace \
                if replace else self.metrics_collector.update
            updater(raw_metrics)
            updater(averaged_metrics, weight=batch_size or 1.)

        # if inside a batch, update the batch metrics dict
        if self.stage.batch.is_active:
            self.batch_metrics.update(averaged_metrics)
            self.batch_metrics.update(
                {k: np.mean(v) for k, v in raw_metrics.items()})

    def copy_metrics_from_nested_context(self, ctx: '_LoggerContext'):
        """
        Copy the final metrics from nested stage.

        Args:
            ctx: The nested stage context.
        """
        # obtain the final metrics from the nested context
        nested_metrics = ctx.metrics_collector.to_json(mean_only=True)

        # if currently a batch is active, update the batch metrics
        if self.stage.batch.is_active:
            self.batch_metrics.update(nested_metrics)

        # update the final metrics
        self.metrics_collector.update(nested_metrics)

    def next_epoch(self):
        """Reset the internal states and enter the next epoch."""
        self.metrics_collector.clear()
        self.batch_metrics.clear()

    def next_batch(self):
        """Reset the internal states and enter the next batch."""
        self.batch_metrics.clear()


def _console_writer(s: str) -> None:
    sys.stdout.write(s)
    sys.stdout.flush()


def _print_log(console_writer: Optional[Callable[[str], None]],
               text: str,
               nl: bool = True,
               show_time: bool = True):
    if console_writer is not None:
        if show_time:
            time_str = format_as_asctime(datetime.now())
            text = f'[{time_str}] {text}'
        if nl:
            text += '\n'
        console_writer(text)


class LoggerMode(IntFlag):
    """Integer flags of logger mode."""

    NONE = 0x0

    LOG_START_END = 0x1
    """Log at the stage start/end."""

    LOG_EVERY_EPOCH = 0x2
    """Log at every epoch."""

    LOG_MAJOR = LOG_START_END | LOG_EVERY_EPOCH
    """Log at the stage start/end and at every epoch."""

    LOG_EVERY_FEW_BATCHES = 0x4
    """Log after every few batches."""

    LOG_EVERY_FEW_SECONDS = 0x8
    """Log every few seconds."""

    DEFAULT = LOG_MAJOR | LOG_EVERY_FEW_SECONDS
    """Default log mode."""

    def check_integrity(self):
        if LoggerMode.LOG_EVERY_FEW_SECONDS in self and \
                LoggerMode.LOG_EVERY_FEW_BATCHES in self:
            raise ValueError(
                '`LOG_EVERY_FEW_SECONDS` and `LOG_EVERY_FEW_BATCHES` '
                'cannot be both enabled.'
            )


class LoggerCallback(Callback):
    """
    Callback that logs training/testing/predicting progress and metrics
    to console and to MLStorage server.

    For performance considerations, batch metrics and progress information
    will be written to console every ``console_log_interval`` seconds,
    and sent to server every ``remote_log_interval`` seconds.

    The progress info will be stored as `progress.<stage.type>` field, and
    the batch metrics will be stored in `progress.<stage.type>.batch_metrics`
    field.  Stages with different types thus will not override the progress
    information and batch metrics of each other.

    Batch metrics will be accumulated by :class:`MetricsLogger`, and reported
    at the end of the epoch.  If the epoch callback provides metrics with the
    same names as the batch metrics, the epoch metrics will override the batch
    metrics.  These metrics are the epoch metrics.

    Moreover, metrics provided by the stage end callback are the stage metrics.
    Epoch metrics and stage metrics will be stored in the `result` field.
    For epoch and stage metrics of the train stage, the metrics will be saved
    as-is; but for other stages, the metrics names will be enforced to have
    the following prefix:

    *  validation stage: "val_" or "valid_".
    *  test stage: "test_"
    *  predict stage: "pred_" or "predict_"

    For nested stages (e.g., validation stage inside a train stage), the
    progress and metrics of the inner stages will not be written to the
    console, but will indeed be sent to the server.
    """

    # a sufficiently large value, should run after almost all callbacks
    priority = 999999

    _ctx_stack: List[_LoggerContext]
    """The stack of :class:`_LoggerContext`, one for every :class:`Stage`."""

    console_mode: LoggerMode
    """The console logger mode."""

    console_writer: Optional[Callable[[str], None]]
    """The console writer."""

    console_log_batch_freq: int
    """Write batch progress and metrics every this number of batches."""

    console_log_interval: float
    """Write batch progress and metrics every this number of seconds."""

    remote_doc: Optional[ExperimentDoc]
    """The :class:`ExperimentDoc`, where to push progress and metrics."""

    remote_push_interval: float
    """Push updates to the remote every this number of seconds."""

    enabled: bool
    """Whether or not this :class:`LoggerCallback` is enabled?"""

    def __init__(self,
                 console_mode: LoggerMode = LoggerMode.DEFAULT,
                 console_writer: Optional[
                     Callable[[str], None]] = _console_writer,
                 console_log_batch_freq: int = 100,
                 console_log_interval: float = 10.,
                 remote_doc: Optional[ExperimentDoc] = NOT_SET,
                 remote_push_interval: float = 60.,
                 metrics_formatter: MetricsFormatter = MetricsFormatter()):
        """
        Construct a new :class:`LoggerCallback`.

        Args:
            console_mode: Mode of console log.
            console_writer: The console writer.
            console_log_batch_freq: Log to console every this number of batches,
                if `LOG_EVERY_FEW_BATCHES` is enabled in `console_mode`.
            console_log_interval: Log to console every this number of seconds,
                if `LOG_EVERY_FEW_SECONDS` is enabled in `console_mode`.
            remote_doc: The remote doc object, where to push updates.
            remote_push_interval: Push to remote every this number of seconds.
            metrics_formatter: The metrics formatter.
        """
        # check the argument
        console_mode.check_integrity()

        # get the remote document according to the context and the environment
        if remote_doc is NOT_SET:
            remote_doc = ExperimentDoc.default_doc()

        self._ctx_stack = []
        self.console_mode = console_mode
        self.console_writer = console_writer
        self.console_log_batch_freq = console_log_batch_freq
        self.console_log_interval = console_log_interval
        self.remote_doc = remote_doc
        self.remote_push_interval = remote_push_interval
        self.metrics_formatter = metrics_formatter
        self._enabled = self.remote_doc is not None or bool(self.console_mode)

    @property
    def enabled(self) -> bool:
        """Whether or not this logger callback is enabled?"""
        return self._enabled

    @property
    def in_nested_stage(self) -> bool:
        """Whether or not this logger callback is in nested stage?"""
        return len(self._ctx_stack) > 1

    @property
    def ctx(self) -> _LoggerContext:
        """Get the current active context (at the top of context stack)."""
        return self._ctx_stack[-1]

    @property
    def stage(self) -> 'Stage':
        """Get the current active stage (at the top of context stack)."""
        return self.ctx.stage

    def _should_write_start_end_console_log(self) -> bool:
        return (not self.in_nested_stage and
                LoggerMode.LOG_START_END in self.console_mode)

    def _should_write_epoch_console_log(self) -> bool:
        return (not self.in_nested_stage and
                LoggerMode.LOG_EVERY_EPOCH in self.console_mode)

    def _should_write_batch_console_log(self,
                                        batch_id: int,
                                        end_timestamp: float) -> bool:
        # if we are now in a nested stage, we shall never write batch log
        if self.in_nested_stage:
            return False

        # if the epoch log is enabled, and this is the final batch, we
        # shall write epoch log instead of the batch log.
        if (LoggerMode.LOG_EVERY_EPOCH in self.console_mode and
                self.stage.epoch is not None and
                batch_id == self.stage.batch.total):
            return False

        # if the best validation mark is True, write batch log
        if self.stage.best_validation_mark:
            return True

        # ordinary checks for the batch
        if (LoggerMode.LOG_EVERY_FEW_BATCHES in self.console_mode and
                batch_id % self.console_log_batch_freq == 0):
            return True

        if (LoggerMode.LOG_EVERY_FEW_SECONDS in self.console_mode and
                end_timestamp - self.ctx.last_console_log_time >=
                self.console_log_interval):
            return True

        # no log is required to be written now
        return False

    def _should_push_batch_remote_log(self,
                                      batch_id: int,
                                      end_timestamp: float) -> bool:
        return (end_timestamp - self.ctx.last_remote_push_time >=
                self.remote_push_interval)

    def _push_to_remote(self, result: Optional[Dict[str, Any]] = None):
        payload = {
            f'progress.{self.stage.name}': self.ctx.progress
        }
        if result:
            payload['result'] = result
        self.remote_doc.update(payload)
        self.ctx.last_remote_push_time = time.time()

    def _write_stage_or_epoch_end_console_log(
            self,
            result_dict: Optional[Dict[str, Any]],
            prefix: str = '',
            suffix: str = '',
            show_time: bool = False,
            is_stage_end: bool = False) -> None:
        # first, compose the log line
        buf = []
        # - <prefix>
        if prefix:
            buf.append(prefix)
        # - <metrics>
        if result_dict:
            result_str = self.metrics_formatter.format(
                result_dict,
                sep=(': ', ' - '),
                known_names=self.stage.known_metrics
            )
            buf.append(result_str)
        # - <suffix>
        if suffix:
            buf.append(suffix)
        log_line = ' - '.join(buf)
        # - " (*)" mark
        if not is_stage_end and self.stage.best_validation_mark:
            log_line += ' (*)'

        # then, print the log
        _print_log(self.console_writer, log_line, show_time=show_time)
        self.ctx.last_console_log_time = time.time()

    def _batch_console_head(self, batch=None) -> str:
        # the batch counter
        max_batch = str(self.ctx.progress.get('max_batch', ''))
        if batch is None:
            batch = str(self.ctx.progress.get('batch', ''))
        if max_batch:
            return f'{batch:>{len(max_batch)}s}/{max_batch}'
        return batch

    def _update_progress_time_info(self, end_time: Optional[float]):
        # update elapsed
        if end_time is not None:
            self.ctx.progress['elapsed'] = end_time - self.stage.start_timestamp

        # update eta
        eta = self.stage.get_eta()
        if eta is not None and eta > 1e-7:
            self.ctx.progress['eta'] = eta
        else:
            self.ctx.progress.pop('eta', None)

    def on_metrics(self, data: CallbackData):
        if not data.stage.is_active and self.remote_doc is not None:
            # some immediate metrics outside a loop context
            m_logger = ScalarMetricsLogger()
            m_logger.update(data.metrics)
            payload = {'result': m_logger.to_json()}
            self.remote_doc.update(payload)

    def on_stage_begin(self, data: CallbackData):
        self._ctx_stack.append(_LoggerContext.new_context(data.stage))

        # write console log
        if self._should_write_start_end_console_log():
            _print_log(
                self.console_writer,
                f'{self.stage.name.capitalize()} started',
                show_time=True
            )

        # start the remote doc worker if this is the first stage
        if len(self._ctx_stack) == 1 and self.remote_doc is not None:
            self.remote_doc.start_worker()

    def on_stage_end(self, data: CallbackData):
        try:
            # set the progress info
            self._update_progress_time_info(data.end_timestamp)

            # replace the epoch metrics with stage metrics, if provided
            if data.metrics:
                self.ctx.update_metrics(data.metrics, replace=True)

            # obtain the stage result dict
            stage_result = self.ctx.metrics_collector.to_json()

            # write the console log
            if self._should_write_start_end_console_log():
                log_prefix = f'{self.stage.name.capitalize()} finished'
                if data.exc_time is not None:
                    elapsed_str = format_duration(
                        data.exc_time, precision=1, count_down=True)
                    log_prefix += f' in {elapsed_str}'

                self._write_stage_or_epoch_end_console_log(
                    result_dict=stage_result,
                    prefix=log_prefix,
                    suffix='',
                    show_time=True,
                    is_stage_end=True,
                )

            # push to remote
            if self.remote_doc is not None:
                self._push_to_remote(stage_result)

        finally:
            # pop this stage
            if len(self._ctx_stack) > 1:
                self._ctx_stack[-2].copy_metrics_from_nested_context(self.ctx)

            self._ctx_stack.pop()

            # stop the remote doc worker if there is no context left
            if not self._ctx_stack and self.remote_doc is not None:
                self.remote_doc.stop_worker()

    def on_epoch_begin(self, data: CallbackData):
        # set the progress info
        self.ctx.progress['epoch'] = data.index
        if data.stage.epoch.total is not None:
            self.ctx.progress['max_epoch'] = data.stage.epoch.total

        # set the context to enter next epoch
        self.ctx.next_epoch()

        # write epoch beginning log
        if self._should_write_epoch_console_log():
            _print_log(self.console_writer, f'Epoch {data.stage.epoch}',
                       show_time=False)

    def on_epoch_end(self, data: CallbackData):
        # set the progress info
        self._update_progress_time_info(data.end_timestamp)
        if data.exc_time is not None:
            self.ctx.progress['epoch_time'] = data.exc_time

        # We use the metric values provided in `data.metrics` as the final
        # metric values for the epoch, to replace any batch metrics.
        self.ctx.update_metrics(data.metrics, replace=True)
        epoch_result = self.ctx.metrics_collector.to_json()

        # write the console log
        if self._should_write_epoch_console_log():
            # log_prefix: total number of executed batches + exc_time
            batch = self.ctx.progress.get('batch', None)
            log_prefix = f'{batch} iters'
            if data.exc_time:
                elapsed_str = format_duration(
                    data.exc_time, precision=1, count_down=True)
                log_prefix += f' in {elapsed_str}'

            # eta
            eta = self.stage.get_eta()
            if eta is not None:
                # just to be consistent with the format of batch logs
                log_prefix += f' - eta {format_duration(eta, count_down=True)}'

            self._write_stage_or_epoch_end_console_log(
                result_dict=epoch_result,
                prefix=log_prefix,
                suffix='',
            )

        # push to remote log
        if self.remote_doc is not None:
            self._push_to_remote(epoch_result)

    def on_batch_begin(self, data: CallbackData):
        self.ctx.progress['batch'] = data.index
        if data.stage.batch.total is not None:
            self.ctx.progress['max_batch'] = data.stage.batch.total
        self.ctx.progress.pop('batch_metrics', None)

        # set the context to enter next batch
        self.ctx.next_batch()

    def on_batch_end(self, data: CallbackData):
        # update the progress info
        self._update_progress_time_info(data.end_timestamp)
        if data.exc_time is not None:
            self.ctx.progress['batch_time'] = data.exc_time

        # update the metrics
        self.ctx.update_metrics(data.metrics, batch_size=data.size)

        # obtain the results of the batch
        batch_result = self.ctx.batch_metrics

        # Copy the batch metrics to the progress dict.
        # This assignment will be cleared at the beginning of the next batch.
        self.ctx.progress['batch_metrics'] = batch_result

        # write logs to console
        if self._should_write_batch_console_log(data.index,
                                                data.end_timestamp):
            buf = [self._batch_console_head()]
            if 'eta' in self.ctx.progress:
                eta_str = format_duration(self.ctx.progress["eta"],
                                          count_down=True)
                buf.append(f'eta {eta_str}')
            if batch_result:
                result_str = self.metrics_formatter.format(
                    batch_result,
                    sep=(': ', ' - '),
                    known_names=self.stage.known_metrics,
                )
                buf.append(result_str)
            log_line = ' - '.join(buf)
            if self.stage.best_validation_mark:
                log_line += ' (*)'
            _print_log(self.console_writer, log_line, show_time=False)
            self.ctx.last_console_log_time = time.time()

        # push the logs to remote
        if self.remote_doc is not None and \
                self._should_push_batch_remote_log(data.index,
                                                   data.end_timestamp):
            self._push_to_remote(batch_result)


class StopOnNaN(Callback):
    """
    Callback that raises :class:`NaNMetricError` whenever an NaN metric
    has been encountered.
    """

    # its priority should be even larger than the LoggerCallback, such that
    # the NaN metrics would be printed before exiting on NaNs
    priority = LoggerCallback.priority + 1

    def _check_metrics(self, metrics: Optional[Mapping[str, Any]]):
        if metrics:
            for key, val in metrics.items():
                if np.isnan(val):
                    raise NaNMetricError(key)

    def on_batch_end(self, data: CallbackData):
        self._check_metrics(data.metrics)

    def on_epoch_end(self, data: CallbackData):
        self._check_metrics(data.metrics)

    def on_stage_end(self, data: CallbackData):
        self._check_metrics(data.metrics)


class BaseTrainCallback(Callback):
    """
    Base callback class for train stages.

    Binds to the first train stage.  If the first stage that this callback
    encounters is not a train stage, then an error will be raised.

    If a subclass need to override :meth:`on_stage_begin()` or
    :meth:`on_stage_end()`, they should call the parent's method.
    For other event callbacks, they need to verify whether or not ``data.stage``
    equals to ``self.stage``.
    """

    stage: Optional['Stage'] = None
    """The current active train stage."""

    def on_stage_begin(self, data: CallbackData):
        if self.stage is None:
            if data.stage.type != StageType.TRAIN:
                raise RuntimeError(
                    f'The outer stage of `{self.__class__.__qualname__}` must '
                    f'be a train stage: got {data.stage.name} stage '
                    f'{data.stage!r}')
            self.stage = data.stage  # bind to this train stage

    def on_stage_end(self, data: CallbackData):
        if data.stage == self.stage:
            self.stage = None  # unbind from the current stage


class BaseCheckpointCallback(BaseTrainCallback):
    """
    Base class for checkpoint callbacks.

    Checkpoint callbacks are train callbacks, which will only work for a
    train stage.  Sub-classes should check ``if self.stage == data.stage``
    in any overrided method.
    """

    STAGE_STATE_KEY: str = '__stage'
    """State key that stores the stage states."""

    checkpoint: BaseCheckpoint
    """The checkpoint object."""

    root_dir: str
    """The root directory, where to save checkpoints."""

    state_objects: Dict[str, StatefulObject]
    """The state objects to be saved along with checkpoints."""

    max_checkpoints_to_keep: Optional[int]
    """
    Maximum number of checkpoints to keep.
    :obj:`None` means that all checkpoints will be kept.
    """

    save_stage_state: bool
    """Whether or not to save the stage state?"""

    checkpoint_manager: Optional[CheckpointManager] = None
    """The checkpoint manager instance."""

    def __init__(self,
                 checkpoint: BaseCheckpoint,
                 root_dir: str,
                 state_objects: Optional[Mapping[str, StatefulObject]] = None,
                 max_checkpoints_to_keep: Optional[int] = None,
                 save_stage_state: bool = True):
        """
        Construct a new :class:`BaseCheckpointCallback`.

        Args:
            checkpoint: The checkpoint object.
            root_dir: The root directory, where to save checkpoints.
            state_objects: The state objects to be saved along with checkpoints.
            max_checkpoints_to_keep: Maximum number of checkpoints to keep.
                Defaults to :obj:`None`, where all checkpoints will be kept.
            save_stage_state: Whether or not to save stage state?
        """
        # check the argument `state_objects`
        state_objects = {k: state_objects[k] for k in (state_objects or ())}
        for k in state_objects:
            if k == self.STAGE_STATE_KEY:
                raise ValueError(f'State object key {k!r} is reserved.')
            v = state_objects[k]
            if not isinstance(v, StatefulObject):
                raise ValueError(f'The item {k!r} in `state_objects` is not '
                                 f'a StatefulObject: got {v!r}')

        # memorize the argument
        self.checkpoint = checkpoint
        self.root_dir = os.path.abspath(root_dir)
        self.state_objects = state_objects
        self.max_checkpoints_to_keep = max_checkpoints_to_keep
        self.save_stage_state = save_stage_state

    def on_stage_begin(self, data: CallbackData):
        super().on_stage_begin(data)
        if data.stage == self.stage:
            if self.save_stage_state:
                self.state_objects[self.STAGE_STATE_KEY] = \
                    data.stage.state_proxy()
            self.checkpoint_manager = CheckpointManager(
                checkpoint=self.checkpoint,
                state_objects=StatefulObjectGroup(self.state_objects),
                root_dir=self.root_dir,
                max_to_keep=self.max_checkpoints_to_keep,
            )

    def on_stage_end(self, data: CallbackData):
        super().on_stage_end(data)
        if self.stage is None and self.checkpoint_manager is not None:
            self.checkpoint_manager = None
            self.state_objects.pop(self.STAGE_STATE_KEY, None)

    def make_checkpoint(self):
        epoch = self.stage.epoch.index
        batch = self.stage.batch.index
        ckpt_name = f'epoch-{epoch}-batch-{batch}'
        ckpt_path = self.checkpoint_manager.save(ckpt_name)
        getLogger(__name__).debug('Saved to checkpoint: %s', ckpt_path)


class AutoCheckpoint(BaseCheckpointCallback):
    """
    Callback to save train checkpoints automatically.
    """

    # priority is one larger than that of `EarlyStopping`, should run after it
    priority = 1000

    interval: Optional[float]
    """If not :obj:`None`, will save checkpoint every this number of seconds."""

    epoch_freq: Optional[int]
    """If not :obj:`None`, will save checkpoint every this number of epochs."""

    batch_freq: Optional[int]
    """If not :obj:`None`, will save checkpoint every this number of batches."""

    restore_checkpoint: Union[str, bool]
    """
    If :obj:`True`, restore the latest saved checkpoint from `root_dir` when
    train begins.  If a str, treat it as the path of a checkpoint, and restore
    it when train begins.
    """

    last_checkpoint_time: float
    """The timestamp when the last checkpoint was saved."""

    def __init__(self,
                 checkpoint: BaseCheckpoint,
                 root_dir: str,
                 *,
                 interval: Optional[float] = None,
                 epoch_freq: Optional[int] = None,
                 batch_freq: Optional[int] = None,
                 state_objects: Optional[Mapping[str, StatefulObject]] = None,
                 max_checkpoints_to_keep: Optional[int] = None,
                 restore_checkpoint: Union[str, bool] = True):
        """
        Construct a new :class:`AutoCheckpoint`.

        Args:
            checkpoint: The checkpoint object.
            root_dir: The root directory, where to save checkpoints.
            interval: If not :obj:`None`, will save checkpoint every this
                number of seconds.  One and only one of `interval`, `epoch_freq`
                and `batch_freq` can be not :obj:`None`.
            epoch_freq: If not :obj:`None`, will save checkpoint every this
                number of epochs.
            batch_freq: If not :obj:`None`, will save checkpoint every this
                number of batches.
            state_objects: The state objects to be saved along with checkpoints.
            max_checkpoints_to_keep: Maximum number of checkpoints to keep.
                Defaults to :obj:`None`, where all checkpoints will be kept.
            restore_checkpoint: If :obj:`True`, restore the latest saved
                checkpoint from `root_dir` when train begins.
                If a str, treat it as the path of a checkpoint, and restore
                it when train begins.
        """
        not_none_count = (
            int(interval is not None) + int(epoch_freq is not None) +
            int(batch_freq is not None)
        )
        if not_none_count != 1:
            raise ValueError('One and only one of `interval`, `epoch_freq` '
                             'and `batch_freq` should be specified.')
        if not isinstance(restore_checkpoint, str) and \
                restore_checkpoint not in (True, False):
            raise TypeError(f'`restore_checkpoint` must be a str or a bool: '
                            f'got {restore_checkpoint!r}')

        super().__init__(
            checkpoint=checkpoint,
            root_dir=root_dir,
            state_objects=state_objects,
            max_checkpoints_to_keep=max_checkpoints_to_keep,
        )
        self.interval = interval
        self.epoch_freq = epoch_freq
        self.batch_freq = batch_freq
        self.restore_checkpoint = restore_checkpoint
        self.last_checkpoint_time = 0.

    def on_train_begin(self, data: CallbackData):
        if data.stage == self.stage:
            # restore the checkpoint
            if isinstance(self.restore_checkpoint, str):
                ckpt_path = self.restore_checkpoint
            elif self.restore_checkpoint is True:
                ckpt_path = self.checkpoint_manager.latest_checkpoint()
            else:
                ckpt_path = None

            if ckpt_path is not None:
                self.checkpoint_manager.restore(ckpt_path)
                print_with_time(f'Restored from the previous checkpoint: '
                                f'{ckpt_path}')

            # set `last_checkpoint_time` to the current timestamp, such that
            # the first checkpoint will not be saved immediately after the
            # train begins
            self.last_checkpoint_time = time.time()

    def on_train_epoch_end(self, data: CallbackData):
        if data.stage == self.stage:
            need_checkpoint = (
                (self.epoch_freq is not None and
                 data.index % self.epoch_freq == 0) or
                (self.interval is not None and
                 data.end_timestamp - self.last_checkpoint_time >= self.interval)
            )
            if need_checkpoint:
                self.make_checkpoint()
                self.last_checkpoint_time = time.time()

    def on_train_batch_end(self, data: CallbackData):
        if data.stage == self.stage:
            need_checkpoint = (
                (self.batch_freq is not None and
                 data.index % self.batch_freq == 0) or
                (data.index != self.stage.batch.total and
                 # if the last batch in an epoch, better to make the checkpoint at
                 # the end of the epoch
                 self.interval is not None and
                 data.end_timestamp - self.last_checkpoint_time >= self.interval)
            )
            if need_checkpoint:
                self.make_checkpoint()
                self.last_checkpoint_time = time.time()


def _es_state_property(name: str, default: Any = None):
    name = '__mltk.callbacks.EarlyStopping.' + name

    def _getter(self):
        return self.stage.memo.get(name, default)

    def _setter(self, value):
        self.stage.memo[name] = value

    def _deleter(self):
        del self.stage.memo[name]

    return property(_getter, _setter, _deleter)


class EarlyStopping(BaseCheckpointCallback):

    # should run before the AutoCheckpoint
    priority = AutoCheckpoint.priority - 1

    metric_name: str
    """Name of the validation metric."""

    smaller_is_better: bool
    """Whether or not smaller values of the metrics indicates better models?"""

    update_at_equal_metric: bool
    """
    Whether or not to update the recorded parameters when encountering the
    same validation metric as the best ever recorded parameters?
    """

    max_no_improvement_epochs: Optional[int]
    """
    The maximum number of epochs to run when there is no improvement in the
    validation metric.
    """

    max_no_improvement_batches: Optional[int]
    """
    The maximum number of batches to run when there is no improvement in the
    validation metric.
    """

    restore_on_error: bool
    """
    Whether or not to restore from the early-stopping saved checkpoint even
    if error occurs?
    """

    _metric_stats: ScalarMetricCollector
    """The collected validation metrics."""

    _is_metric_better: Callable[[Any, Any], bool]
    """The function that checks whether a given metric value is better."""

    best_metric_value: float = _es_state_property('best_metric_value')
    """The best validation metric that has been ever encountered."""

    no_improvement_epochs: int = _es_state_property('no_improvement_epochs', 0)
    """
    Number of epochs executed with no improvement in the validation metric.
    """

    no_improvement_batches: int = _es_state_property('no_improvement_batches', 0)
    """
    Number of batches executed with no improvement in the validation metric.
    """

    def __init__(self,
                 checkpoint: BaseCheckpoint,
                 root_dir: str,
                 metric_name: str,
                 smaller_is_better: bool = True,
                 update_at_equal_metric: bool = True,
                 max_no_improvement_epochs: Optional[int] = None,
                 max_no_improvement_batches: Optional[int] = None,
                 restore_on_error: bool = False,
                 state_objects: Optional[Mapping[str, StatefulObject]] = None,
                 max_checkpoints_to_keep: int = 1):
        """
        Construct a new :class:`EarlyStopping`.

        Args:
            checkpoint: The checkpoint object.
            root_dir: The root directory, where to save checkpoints.
            metric_name: Name of the validation metric.
            smaller_is_better: Whether or not smaller values of the metrics
                indicates better models?
            update_at_equal_metric: Whether or not to update the recorded
                parameters when encountering the same validation metric
                as the best ever recorded parameters?
            max_no_improvement_epochs: The maximum number of epochs to run
                when there is no improvement in the validation metric.
            max_no_improvement_batches: The maximum number of batches to run
                when there is no improvement in the validation metric.
            restore_on_error: Whether or not to restore from the early-stopping
                saved checkpoint even if error occurs?
            state_objects: The state objects to be saved along with checkpoints.
            max_checkpoints_to_keep: The maximum number of early-stopping
                checkpoints to keep on disk.  Defaults to 1.
        """
        super().__init__(
            checkpoint=checkpoint,
            root_dir=root_dir,
            state_objects=state_objects,
            max_checkpoints_to_keep=max_checkpoints_to_keep,
            save_stage_state=False
        )
        metric_name = str(metric_name)

        if not any(metric_name.startswith(pfx)
                   for pfx in StageType.VALIDATION.metric_prefixes):
            raise ValueError(f'Early-stopping metric name must start with '
                             f'any of the following prefixes: '
                             f'{list(StageType.VALIDATION.metric_prefixes)}; '
                             f'got metric name {metric_name!r}')

        self.metric_name = metric_name
        self.smaller_is_better = smaller_is_better
        self.update_at_equal_metric = update_at_equal_metric
        self.max_no_improvement_epochs = max_no_improvement_epochs
        self.max_no_improvement_batches = max_no_improvement_batches
        self.restore_on_error = restore_on_error

        self._metric_stats = ScalarMetricCollector()

        if smaller_is_better:
            self._is_metric_better = lambda new, old: \
                old is None or new < old
        else:
            self._is_metric_better = lambda new, old: \
                old is None or new > old

        if self.update_at_equal_metric:
            def wrap_accept_equal(fn):
                return lambda new, old: fn(new, old) or (new == old)
            self._is_metric_better = wrap_accept_equal(self._is_metric_better)

    def _need_termination(self):
        return (
            (self.max_no_improvement_epochs is not None and
             self.no_improvement_epochs >= self.max_no_improvement_epochs) or
            (self.max_no_improvement_batches is not None and
             self.no_improvement_batches >= self.max_no_improvement_batches)
        )

    # update the validation metric
    def _update_valid_metric(self, metric_value: Union[float, np.ndarray]):
        if not np.isnan(metric_value) and \
                (self.best_metric_value is None or
                 self._is_metric_better(metric_value, self.best_metric_value)):
            self.stage.best_validation_mark = True
            self.best_metric_value = metric_value
            self.no_improvement_batches = 0
            self.no_improvement_epochs = 0
            self.make_checkpoint()

        if self._need_termination():
            self.stage.request_termination()

    # validation events
    def on_validation_begin(self, data: CallbackData):
        if self.stage is not None:  # ensure we are in a train stage
            self._metric_stats.reset()

    def on_validation_batch_end(self, data: CallbackData):
        if self.stage is not None:  # ensure we are in a train stage
            if data.metrics and self.metric_name in data.metrics:
                self._metric_stats.update(data.metrics[self.metric_name],
                                          weight=data.size or 1.)

    def on_validation_end(self, data: CallbackData):
        if self.stage is not None:  # ensure we are in a train stage
            if data.metrics and self.metric_name in data.metrics:
                # if the stage end callback reports the metric, use it
                metric_value = data.metrics[self.metric_name]
            elif self._metric_stats.has_stats:
                # otherwise get the averaged batch metric value
                metric_value = self._metric_stats.mean
            else:
                metric_value = None

            # check the new validation metric
            if metric_value is not None:
                self._update_valid_metric(metric_value)

    # train events
    def on_train_epoch_begin(self, data: CallbackData):
        if data.stage == self.stage:
            self.no_improvement_epochs += 1

    def on_train_batch_begin(self, data: CallbackData):
        if data.stage == self.stage:
            self.no_improvement_batches += 1

    def on_train_end(self, data: CallbackData):
        if self.stage is not None:
            from .errors import UserTermination

            # first, check whether or not we're interrupted by exception
            err_type = sys.exc_info()[0]
            has_error = (
                err_type is not None and
                not issubclass(err_type, (SystemExit, KeyboardInterrupt,
                                          UserTermination))
            )

            # restore from the best checkpoint, if any checkpoint is saved,
            # and there is no error (or `restore_on_error = True`)
            if not has_error or self.restore_on_error:
                latest_checkpoint = self.checkpoint_manager.latest_checkpoint()
                if latest_checkpoint is None:
                    getLogger(__name__).warning(
                        'No checkpoint has been saved for early-stopping.  '
                        'Did you forget to update the validation metric %r?',
                        self.metric_name
                    )
                else:
                    self.checkpoint_manager.restore(latest_checkpoint)
                    print_with_time(f'Restored early-stopping checkpoint from: '
                                    f'{latest_checkpoint}')


# imported for type annotation on `Stage`
from .stage import Stage, StageType
