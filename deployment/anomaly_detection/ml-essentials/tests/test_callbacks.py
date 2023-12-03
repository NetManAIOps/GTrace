import copy
import os
import random
import unittest
from datetime import datetime
from tempfile import TemporaryDirectory

import mock
import numpy as np
import pytest

from mltk import *
from mltk.callbacks import *
from mltk.callbacks import (_LoggerContext, _console_writer, _print_log,
                            BaseTrainCallback, BaseCheckpointCallback)
from mltk.stage import _StageCounterState
from mltk.utils import deep_copy
from tests.helpers import new_callback_data


class CallbackListTestCase(unittest.TestCase):

    def test_CallbackList(self):
        def make_callback(priority: int):
            cb = Callback()
            cb.priority = priority
            return cb

        # test construct and copy
        orig_list = [make_callback(i) for i in range(-10, 10, 2)]
        cb_list = CallbackList(orig_list)
        self.assertEqual(len(cb_list), len(orig_list))
        self.assertTrue(all(a == b for a, b in zip(cb_list, orig_list)))
        self.assertTrue(all(cb_list[i] == orig_list[i] for i in range(len(orig_list))))

        self.assertEqual(copy.copy(cb_list), cb_list)
        self.assertEqual(CallbackList(cb_list), cb_list)
        self.assertEqual(
            CallbackList(reversed(cb_list)),
            cb_list
        )

        temp_list = list(cb_list)
        random.shuffle(temp_list)
        self.assertEqual(CallbackList(temp_list), cb_list)

        # test add
        new_cb = make_callback(6)
        orig_list = orig_list[:9] + [new_cb] + orig_list[9:]
        cb_list.add(new_cb)
        self.assertEqual(list(cb_list), orig_list)

        new_cb = make_callback(5)
        orig_list = orig_list[:8] + [new_cb] + orig_list[8:]
        cb_list.add(new_cb)
        self.assertEqual(list(cb_list), orig_list)

        # test del
        del orig_list[3]
        del cb_list[3]
        self.assertEqual(list(cb_list), orig_list)

        cb = orig_list[2]
        del orig_list[2]
        cb_list.remove(cb)
        self.assertEqual(list(cb_list), orig_list)

    def test_builtin_priority(self):
        orders = [
            Callback,  # == 0
            EarlyStopping, AutoCheckpoint, LoggerCallback,
            StopOnNaN,
        ]
        self.assertEqual(Callback.priority, 0)
        for a, b in zip(orders[:-1], orders[1:]):
            self.assertLess(a.priority, b.priority)


class LoggerCallbackTestCase(unittest.TestCase):

    def test_LoggerContext(self):
        #################
        # new_context() #
        #################
        stage = Stage(StageType.TRAIN)
        with mock.patch('time.time', return_value=123.):
            ctx = _LoggerContext.new_context(stage)

        self.assertEqual(ctx.stage, stage)
        self.assertEqual(ctx.progress, {})
        self.assertIsInstance(ctx.metrics_collector, ScalarMetricsLogger)
        self.assertEqual(ctx.batch_metrics, {})
        self.assertEqual(ctx.last_console_log_time, 123.)
        self.assertEqual(ctx.last_remote_push_time, 123.)

        ####################
        # update_metrics() #
        ####################
        ctx = _LoggerContext.new_context(Stage(StageType.TEST))

        # batch.is_active = False
        ctx.stage.batch.is_active = False
        ctx.update_metrics({
            'loss': 0.25,
            'acc': np.array([0.1, 0.2, 0.3]),
        })
        self.assertEqual(ctx.batch_metrics, {})
        loss_metric = ctx.metrics_collector['test_loss']
        self.assertEqual(loss_metric.weight_sum, 1.)
        self.assertAlmostEqual(loss_metric.mean, 0.25)
        acc_metric = ctx.metrics_collector['test_acc']
        self.assertEqual(acc_metric.weight_sum, 3.)
        self.assertAlmostEqual(acc_metric.mean, 0.2)

        # batch.is_active = True
        ctx.stage.batch.is_active = True
        ctx.update_metrics(
            {'test_loss': 0.5, 'test_acc': np.array([0.4, 0.5, 0.6])},
            batch_size=3.
        )
        self.assertEqual(ctx.batch_metrics, {
            'test_loss': 0.5, 'test_acc': 0.5,
        })
        loss_metric = ctx.metrics_collector['test_loss']
        self.assertEqual(loss_metric.weight_sum, 4.)
        self.assertAlmostEqual(loss_metric.mean, 0.4375)
        acc_metric = ctx.metrics_collector['test_acc']
        self.assertEqual(acc_metric.weight_sum, 6.)
        self.assertAlmostEqual(acc_metric.mean, 0.35)

        # batch.is_active = True, replace = True
        ctx.stage.batch.is_active = True
        ctx.update_metrics(
            {'test_loss': 0.75, 'test_acc': 0.875, 'test_ll': 1.25},
            replace=True,
        )
        self.assertEqual(ctx.batch_metrics, {
            'test_loss': 0.75, 'test_acc': 0.875, 'test_ll': 1.25,
        })
        loss_metric = ctx.metrics_collector['test_loss']
        self.assertEqual(loss_metric.weight_sum, 1.)
        self.assertAlmostEqual(loss_metric.mean, 0.75)
        acc_metric = ctx.metrics_collector['test_acc']
        self.assertEqual(acc_metric.weight_sum, 1.)
        self.assertAlmostEqual(acc_metric.mean, 0.875)
        ll_metric = ctx.metrics_collector['test_ll']
        self.assertEqual(ll_metric.weight_sum, 1.)
        self.assertAlmostEqual(ll_metric.mean, 1.25)

        ######################################
        # copy_metrics_from_nested_context() #
        ######################################
        ctx.stage.batch.is_active = True
        ctx.update_metrics({'ll': 0.75})  # ensure to have std
        self.assertIsNotNone(ctx.metrics_collector['test_ll'].stats.std)

        # batch.is_active = False
        ctx2 = _LoggerContext.new_context(Stage(StageType.TEST))
        ctx2.stage.batch.is_active = False
        ctx2.copy_metrics_from_nested_context(ctx)
        self.assertEqual(ctx2.batch_metrics, {})
        loss_metric = ctx2.metrics_collector['test_loss']
        self.assertEqual(loss_metric.weight_sum, 1.)
        self.assertAlmostEqual(loss_metric.mean, 0.75)
        acc_metric = ctx2.metrics_collector['test_acc']
        self.assertEqual(acc_metric.weight_sum, 1.)
        self.assertAlmostEqual(acc_metric.mean, 0.875)
        ll_metric = ctx2.metrics_collector['test_ll']
        self.assertEqual(ll_metric.weight_sum, 1.)
        self.assertAlmostEqual(ll_metric.mean, 1.)

        # batch.is_active = True
        ctx2 = _LoggerContext.new_context(Stage(StageType.TEST))
        ctx2.stage.batch.is_active = True
        ctx2.copy_metrics_from_nested_context(ctx)
        self.assertEqual(ctx2.batch_metrics, {
            'test_loss': 0.75,
            'test_acc': 0.875,
            'test_ll': 1.,
        })

        ################
        # next_epoch() #
        ################
        ctx = _LoggerContext.new_context(Stage(StageType.TRAIN))
        ctx.metrics_collector.update({'acc': 0.875})
        ctx.batch_metrics.update({'acc': 0.875})
        self.assertEqual(list(ctx.metrics_collector), ['acc'])
        self.assertEqual(list(ctx.batch_metrics), ['acc'])
        ctx.next_epoch()
        self.assertEqual(list(ctx.metrics_collector), [])
        self.assertEqual(list(ctx.batch_metrics), [])

        ################
        # next_batch() #
        ################
        ctx = _LoggerContext.new_context(Stage(StageType.TRAIN))
        ctx.metrics_collector.update({'acc': 0.875})
        ctx.batch_metrics.update({'acc': 0.875})
        self.assertEqual(list(ctx.metrics_collector), ['acc'])
        self.assertEqual(list(ctx.batch_metrics), ['acc'])
        ctx.next_batch()
        self.assertEqual(list(ctx.metrics_collector), ['acc'])
        self.assertEqual(list(ctx.batch_metrics), [])

    def test_default_console_writer(self):
        class FakeStdout(object):
            logs = []

            def write(self, s: str):
                self.logs.append(('write', s))

            def flush(self):
                self.logs.append('flush')

        my_stdout = FakeStdout()
        with mock.patch('sys.stdout', my_stdout):
            _console_writer('hello, world!')
        self.assertEqual(my_stdout.logs, [
            ('write', 'hello, world!'),
            'flush',
        ])

    def test_print_log(self):
        class FakeDateTime(object):
            def now(self):
                return dt

        dt = datetime.utcfromtimestamp(1576755571.662434)
        dt_str = format_as_asctime(dt)
        with mock.patch('mltk.callbacks.datetime', FakeDateTime()):
            logs = []

            # test nl = True, show_time = True
            _print_log(logs.append, 'hello, world!')
            self.assertEqual(logs, [f'[{dt_str}] hello, world!\n'])
            logs.clear()

            # test nl = True, show_time = False
            _print_log(logs.append, 'hello, world!', show_time=False)
            self.assertEqual(logs, ['hello, world!\n'])
            logs.clear()

            # test nl = False, show_time = True
            _print_log(logs.append, 'hello, world!', nl=False)
            self.assertEqual(logs, [f'[{dt_str}] hello, world!'])
            logs.clear()

            # test console_writer is None
            _print_log(None, 'hello, world!')  # nothing should happen

    def test_LoggerMode(self):
        self.assertFalse(LoggerMode.NONE)
        self.assertEqual(LoggerMode.NONE, 0)
        self.assertEqual(
            LoggerMode.DEFAULT,
            (LoggerMode.LOG_START_END | LoggerMode.LOG_EVERY_EPOCH |
             LoggerMode.LOG_EVERY_FEW_SECONDS)
        )
        self.assertIn(LoggerMode.LOG_START_END, LoggerMode.DEFAULT)
        self.assertNotIn(LoggerMode.LOG_START_END,
                         LoggerMode.DEFAULT & ~LoggerMode.LOG_START_END)

        with pytest.raises(ValueError,
                           match='`LOG_EVERY_FEW_SECONDS` and '
                                 '`LOG_EVERY_FEW_BATCHES` cannot be both '
                                 'enabled'):
            m = (LoggerMode.LOG_EVERY_FEW_SECONDS |
                 LoggerMode.LOG_EVERY_FEW_BATCHES)
            m.check_integrity()

    def test_construct(self):
        # default args
        os.environ.pop('MLSTORAGE_EXPERIMENT_ID', None)

        cb = LoggerCallback()
        self.assertEqual(cb.console_mode, LoggerMode.DEFAULT)
        self.assertEqual(cb.console_writer, _console_writer)
        self.assertEqual(cb.console_log_batch_freq, 100)
        self.assertEqual(cb.console_log_interval, 10.)
        self.assertEqual(cb.remote_doc, None)
        self.assertEqual(cb.remote_push_interval, 60.)
        self.assertIsInstance(cb.metrics_formatter, MetricsFormatter)
        self.assertTrue(cb.enabled)

        self.assertEqual(cb._ctx_stack, [])
        self.assertFalse(cb.in_nested_stage)

        # none console log
        cb = LoggerCallback(LoggerMode.NONE)
        self.assertEqual(cb.console_mode, LoggerMode.NONE)
        self.assertFalse(cb.enabled)

        # none console log and remote doc specified
        remote_doc = ExperimentDoc()
        cb = LoggerCallback(LoggerMode.NONE, remote_doc=remote_doc)
        self.assertIs(cb.remote_doc, remote_doc)
        self.assertTrue(cb.enabled)

    def test_log_conditions(self):
        cb = LoggerCallback()
        stage = Stage(StageType.TRAIN)

        # add a context
        with mock.patch('time.time', return_value=0.):
            ctx = _LoggerContext.new_context(stage)
        self.assertEqual(ctx.last_console_log_time, 0.)
        self.assertEqual(ctx.last_remote_push_time, 0.)
        cb._ctx_stack.append(ctx)
        self.assertIs(cb.ctx, ctx)
        self.assertIs(cb.stage, stage)

        #########################################
        # _should_write_start_end_console_log() #
        #########################################
        cb.console_mode = LoggerMode.DEFAULT
        self.assertTrue(cb._should_write_start_end_console_log())

        # nested stage
        cb._ctx_stack.append(ctx)
        self.assertFalse(cb._should_write_start_end_console_log())
        cb._ctx_stack.pop()

        # disable by console_mode
        cb.console_mode = LoggerMode.DEFAULT & ~LoggerMode.LOG_START_END
        self.assertFalse(cb._should_write_start_end_console_log())

        #####################################
        # _should_write_epoch_console_log() #
        #####################################
        cb.console_mode = LoggerMode.DEFAULT
        self.assertTrue(cb._should_write_epoch_console_log())

        # nested stage
        cb._ctx_stack.append(ctx)
        self.assertFalse(cb._should_write_epoch_console_log())
        cb._ctx_stack.pop()

        # disable by console_mode
        cb.console_mode = LoggerMode.DEFAULT & ~LoggerMode.LOG_EVERY_EPOCH
        self.assertFalse(cb._should_write_epoch_console_log())

        #####################################
        # _should_write_batch_console_log() #
        #####################################
        cb.console_log_interval = 1.
        cb.console_log_batch_freq = 2
        stage.batch.total = 4
        stage.best_validation_mark = False

        # test log every few seconds
        cb.console_mode = LoggerMode.LOG_EVERY_FEW_SECONDS

        self.assertFalse(cb._should_write_batch_console_log(
            batch_id=1, end_timestamp=0.5))
        self.assertFalse(cb._should_write_batch_console_log(
            batch_id=2, end_timestamp=0.5))
        self.assertFalse(cb._should_write_batch_console_log(
            batch_id=3, end_timestamp=0.5))
        self.assertFalse(cb._should_write_batch_console_log(
            batch_id=4, end_timestamp=0.5))
        self.assertTrue(cb._should_write_batch_console_log(
            batch_id=1, end_timestamp=1.))
        self.assertTrue(cb._should_write_batch_console_log(
            batch_id=2, end_timestamp=1.))
        self.assertTrue(cb._should_write_batch_console_log(
            batch_id=3, end_timestamp=1.))
        self.assertTrue(cb._should_write_batch_console_log(
            batch_id=4, end_timestamp=1.))

        stage.best_validation_mark = True  # log on best validation
        self.assertTrue(cb._should_write_batch_console_log(
            batch_id=1, end_timestamp=0.5))
        stage.best_validation_mark = False

        cb._ctx_stack.append(ctx)  # do not log in nested stage
        self.assertFalse(cb._should_write_batch_console_log(
            batch_id=2, end_timestamp=1.))
        cb._ctx_stack.pop()

        cb.console_mode |= LoggerMode.LOG_EVERY_EPOCH  # do not log at the end of epoch if epoch log is enabled
        stage.best_validation_mark = True
        self.assertFalse(cb._should_write_batch_console_log(
            batch_id=4, end_timestamp=1.))
        stage.best_validation_mark = False

        epoch = stage.epoch  # epoch is None, LOG_EVERY_EPOCH is not enabled
        stage.epoch = None
        self.assertTrue(cb._should_write_batch_console_log(
            batch_id=4, end_timestamp=1.))
        stage.epoch = epoch

        # test log every few batches
        cb.console_mode = LoggerMode.LOG_EVERY_FEW_BATCHES

        self.assertFalse(cb._should_write_batch_console_log(
            batch_id=1, end_timestamp=.5))
        self.assertTrue(cb._should_write_batch_console_log(
            batch_id=2, end_timestamp=.5))
        self.assertFalse(cb._should_write_batch_console_log(
            batch_id=3, end_timestamp=.5))
        self.assertTrue(cb._should_write_batch_console_log(
            batch_id=4, end_timestamp=.5))
        self.assertFalse(cb._should_write_batch_console_log(
            batch_id=1, end_timestamp=1.))
        self.assertFalse(cb._should_write_batch_console_log(
            batch_id=3, end_timestamp=1.))

        stage.best_validation_mark = True  # log on best validation
        self.assertTrue(cb._should_write_batch_console_log(
            batch_id=1, end_timestamp=.5))
        stage.best_validation_mark = False

        cb._ctx_stack.append(ctx)  # do not log in nested stage
        self.assertFalse(cb._should_write_batch_console_log(
            batch_id=2, end_timestamp=1.))
        cb._ctx_stack.pop()

        cb.console_mode |= LoggerMode.LOG_EVERY_EPOCH  # do not log at the end of epoch if epoch log is enabled
        stage.best_validation_mark = True
        self.assertFalse(cb._should_write_batch_console_log(
            batch_id=4, end_timestamp=1.))
        stage.best_validation_mark = False

        epoch = stage.epoch  # epoch is None, LOG_EVERY_EPOCH is not enabled
        stage.epoch = None
        self.assertTrue(cb._should_write_batch_console_log(
            batch_id=4, end_timestamp=1.))
        stage.epoch = epoch

        ###################################
        # _should_push_batch_remote_log() #
        ###################################
        cb.remote_push_interval = 1.
        self.assertFalse(cb._should_push_batch_remote_log(2, .5))
        self.assertTrue(cb._should_push_batch_remote_log(2, 1.))
        self.assertTrue(cb._should_push_batch_remote_log(2, 1.5))

    def test_push_to_remote(self):
        class _MyRemoteDoc(object):
            logs = []

            def update(self, result_dict):
                self.logs.append(copy.copy(result_dict))

        # construct the objects
        remote_doc = _MyRemoteDoc()
        cb = LoggerCallback(remote_doc=remote_doc)
        stage = Stage(StageType.TRAIN)
        with mock.patch('time.time', return_value=0.):
            ctx = _LoggerContext.new_context(stage)
        self.assertEqual(ctx.last_remote_push_time, 0.)
        cb._ctx_stack.append(ctx)

        # prepare data for the test
        ctx.progress['epoch'] = 2
        ctx.progress['max_epoch'] = 23

        # test without result
        with mock.patch('time.time', return_value=123.):
            cb._push_to_remote()
        self.assertEqual(remote_doc.logs, [{
            'progress.train': {'epoch': 2, 'max_epoch': 23}
        }])
        self.assertEqual(ctx.last_remote_push_time, 123.)
        remote_doc.logs.clear()

        # test with result
        with mock.patch('time.time', return_value=456.):
            cb._push_to_remote({'acc': .875})
        self.assertEqual(remote_doc.logs, [{
            'progress.train': {'epoch': 2, 'max_epoch': 23},
            'result': {'acc': .875}
        }])
        self.assertEqual(ctx.last_remote_push_time, 456.)

    def test_write_stage_or_epoch_end_console_log(self):
        logs = []
        cb = LoggerCallback(console_writer=logs.append)
        stage = Stage(StageType.TRAIN)
        with mock.patch('time.time', return_value=0.):
            ctx = _LoggerContext.new_context(stage)
        cb._ctx_stack.append(ctx)

        # empty log
        with mock.patch('time.time', return_value=123.):
            cb._write_stage_or_epoch_end_console_log(None)
        self.assertEqual(logs, ['\n'])
        self.assertEqual(ctx.last_console_log_time, 123.)
        logs.clear()

        # with prefix, suffix and metrics
        with mock.patch('time.time', return_value=456.):
            cb._write_stage_or_epoch_end_console_log(
                result_dict={'acc': .875, 'loss': {'mean': .125, 'std': .25}},
                prefix='<<<',
                suffix='>>>',
            )
        self.assertEqual(
            logs,
            ['<<< - acc: 0.875 - loss: 0.125 (±0.25) - >>>\n']
        )
        self.assertEqual(ctx.last_console_log_time, 456.)
        logs.clear()

        # also set `known_metrics`
        stage.known_metrics = ('loss',)
        cb._write_stage_or_epoch_end_console_log(
            result_dict={'acc': .875, 'loss': {'mean': .125, 'std': .25}},
            prefix='<<<',
            suffix='>>>',
        )
        self.assertEqual(
            logs,
            ['<<< - loss: 0.125 (±0.25) - acc: 0.875 - >>>\n']
        )
        logs.clear()
        stage.known_metrics = ()

        # set `best_validation_mark`
        stage.best_validation_mark = True
        cb._write_stage_or_epoch_end_console_log(
            result_dict={'acc': .875, 'loss': {'mean': .125, 'std': .25}},
            prefix='<<<',
            suffix='>>>',
        )
        self.assertEqual(
            logs,
            ['<<< - acc: 0.875 - loss: 0.125 (±0.25) - >>> (*)\n']
        )
        logs.clear()
        stage.best_validation_mark = False

        # set `show_time`
        class FakeDateTime(object):
            def now(self):
                return datetime.utcfromtimestamp(1576755571.662434)

        with mock.patch('mltk.callbacks.datetime', FakeDateTime()):
            cb._write_stage_or_epoch_end_console_log(
                result_dict={'acc': .875, 'loss': {'mean': .125, 'std': .25}},
                prefix='<<<',
                suffix='>>>',
                show_time=True,
            )
        self.assertEqual(
            logs,
            ['[2019-12-19 11:39:31,662] <<< - acc: 0.875 - '
             'loss: 0.125 (±0.25) - >>>\n']
        )
        logs.clear()

    def test_batch_console_head(self):
        cb = LoggerCallback()
        stage = Stage(StageType.TRAIN)
        with mock.patch('time.time', return_value=0.):
            ctx = _LoggerContext.new_context(stage)
        cb._ctx_stack.append(ctx)

        # empty batch counter
        self.assertEqual(cb._batch_console_head(), '')

        # with batch progress
        ctx.progress['batch'] = 12
        self.assertEqual(cb._batch_console_head(), '12')

        # with total batch
        ctx.progress['max_batch'] = 3
        self.assertEqual(cb._batch_console_head(), '12/3')
        ctx.progress['max_batch'] = 34
        self.assertEqual(cb._batch_console_head(), '12/34')
        ctx.progress['max_batch'] = 345
        self.assertEqual(cb._batch_console_head(), ' 12/345')

    def test_update_progress_time_info(self):
        cb = LoggerCallback()
        stage = Stage(StageType.TRAIN)
        ctx = _LoggerContext.new_context(stage)
        cb._ctx_stack.append(ctx)

        # update without end_time
        stage.start_timestamp = .1
        cb._update_progress_time_info(None)
        self.assertEqual(ctx.progress, {})

        # update with end_time
        cb._update_progress_time_info(1.5)
        self.assertAlmostEqual(ctx.progress['elapsed'], 1.4)

        # update eta
        self.assertNotIn('eta', ctx.progress)

        stage.get_eta = mock.Mock(return_value=123.)
        cb._update_progress_time_info(1.6)
        self.assertEqual(ctx.progress['eta'], 123.)

        stage.get_eta = mock.Mock(return_value=1e-8)
        cb._update_progress_time_info(1.6)
        self.assertNotIn('eta', ctx.progress)

    def test_life_cycle(self):
        class _MyRemoteDoc(object):
            logs = []

            def start_worker(self):
                self.logs.append('start_worker')

            def update(self, metrics):
                self.logs.append(('update', deep_copy(metrics)))

            def stop_worker(self):
                self.logs.append('stop_worker')

        logs = []
        remote_doc = _MyRemoteDoc()
        cb = LoggerCallback(
            console_mode=(LoggerMode.LOG_MAJOR |
                          LoggerMode.LOG_EVERY_FEW_BATCHES),
            console_writer=logs.append,
            remote_doc=remote_doc,
            console_log_batch_freq=2,
            remote_push_interval=.1,
        )

        stage = Stage(StageType.TRAIN, max_epoch=13, max_batch=78)
        stage.start_timestamp = 0.5
        stage.get_eta = mock.Mock(return_value=61)

        ################
        # on_metrics() #
        ################
        data = new_callback_data(stage=stage, metrics={'a': 123.5})
        cb.on_metrics(data)
        self.assertEqual(remote_doc.logs, [('update', {'result': {'a': 123.5}})])
        remote_doc.logs.clear()

        ####################
        # on_stage_begin() #
        ####################
        data = new_callback_data(stage=stage, start_timestamp=0.5)
        cb.on_stage_begin(data)
        self.assertEqual(cb.stage, data.stage)
        self.assertEqual(len(logs), 1)
        self.assertRegex(
            logs[-1],
            r'^\[[^\[\]]+\] Train started\n$',
        )
        self.assertEqual(remote_doc.logs, ['start_worker'])
        logs.clear()
        remote_doc.logs.clear()

        ####################
        # on_epoch_begin() #
        ####################
        stage.epoch.index = 3
        stage.epoch.is_active = True
        cb.ctx.metrics_collector.update({'xxx': 123.})
        cb.ctx.batch_metrics.update({'yyy': 456.})
        data = new_callback_data(
            stage=stage,
            index=3,
            size=567,
            start_timestamp=1.0,
        )
        cb.on_epoch_begin(data)
        self.assertEqual(cb.ctx.progress, {
            'epoch': 3,  # i.e., ``index``
            'max_epoch': 13,
        })
        self.assertEqual(cb.ctx.metrics_collector.to_json(), {})  # should be cleared
        self.assertEqual(cb.ctx.batch_metrics, {})  # should be cleared
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[-1], 'Epoch 3/13\n')
        self.assertEqual(remote_doc.logs, [])
        logs.clear()
        remote_doc.logs.clear()

        ####################
        # on_batch_begin() #
        ####################
        stage.batch.index = 4
        stage.batch.is_active = True
        cb.ctx.batch_metrics.update({'yyy': 456.})
        cb.ctx.progress['batch_metrics'] = {}
        data = new_callback_data(
            stage=stage,
            index=4,
            size=32,
            start_timestamp=1.5,
        )
        cb.on_batch_begin(data)
        self.assertEqual(cb.ctx.progress, {
            'epoch': 3,
            'max_epoch': 13,
            'batch': 4,
            'max_batch': 78,
        })
        self.assertEqual(cb.ctx.batch_metrics, {})
        self.assertEqual(len(logs), 0)
        self.assertEqual(remote_doc.logs, [])

        ##################################
        # nested stage within this batch #
        ##################################
        def nested_stage():
            stage2 = Stage(StageType.VALIDATION)
            stage2.start_timestamp = 2.0

            # enter stage
            data2 = new_callback_data(stage=stage2, start_timestamp=2.0)
            cb.on_stage_begin(data2)
            self.assertEqual(len(logs), 0)
            self.assertEqual(remote_doc.logs, [])

            # exit stage
            data2 = new_callback_data(
                stage=stage2,
                start_timestamp=2.0,
                end_timestamp=2.5,
                exc_time=0.5,
                metrics={'val_acc': 0.75},
            )
            cb.on_stage_end(data2)

        nested_stage()
        self.assertEqual(len(logs), 0)
        self.assertEqual(remote_doc.logs, [
            ('update', {
                'progress.validation': {'elapsed': 0.5},
                'result': {'val_acc': 0.75}
            })
        ])
        self.assertEqual(cb.ctx.metrics_collector.to_json(), {'val_acc': 0.75})
        self.assertEqual(cb.ctx.batch_metrics, {'val_acc': 0.75})
        remote_doc.logs.clear()

        ##################
        # on_batch_end() #
        ##################
        cb.ctx.last_remote_push_time = 0.
        stage.batch.end_timestamp = 3.0
        stage.best_validation_mark = True
        data = new_callback_data(
            stage=stage,
            index=4,
            size=32,
            start_timestamp=1.5,
            end_timestamp=3.0,
            exc_time=1.5,
            metrics={'acc': 0.5, 'loss': 0.25},
        )
        cb.on_batch_end(data)
        self.assertEqual(
            cb.ctx.metrics_collector.to_json(),
            {'acc': 0.5, 'loss': 0.25, 'val_acc': 0.75},
        )
        self.assertEqual(
            cb.ctx.batch_metrics,
            {'acc': 0.5, 'loss': 0.25, 'val_acc': 0.75},
        )
        stage.batch.is_active = False
        self.assertEqual(len(logs), 1)
        self.assertRegex(
            logs[-1],
            r'^\s4/78 - eta 1:01 - acc: 0.5 - loss: 0.25 - val_acc: 0.75 \(\*\)\n$',
        )
        self.assertEqual(remote_doc.logs, [
            ('update',
             {'progress.train': {
                 'batch': 4,
                 'batch_metrics': {'acc': 0.5, 'loss': 0.25, 'val_acc': 0.75},
                 'batch_time': 1.5,
                 'elapsed': 2.5,
                 'epoch': 3,
                 'eta': 61,
                 'max_batch': 78,
                 'max_epoch': 13},
              'result': {'acc': 0.5, 'loss': 0.25, 'val_acc': 0.75}})
        ])
        logs.clear()
        remote_doc.logs.clear()

        ##################
        # on_epoch_end() #
        ##################
        stage.epoch.is_active = False
        stage.epoch.end_timestamp = 3.5
        data = new_callback_data(
            stage=stage,
            index=3,
            size=567,
            start_timestamp=1.0,
            end_timestamp=3.5,
            exc_time=2.5,
            metrics={'acc': 0.125},
        )
        cb.on_epoch_end(data)
        self.assertEqual(
            cb.ctx.metrics_collector.to_json(),
            {'acc': 0.125, 'loss': 0.25, 'val_acc': 0.75},
        )
        self.assertEqual(
            cb.ctx.batch_metrics,
            {'acc': 0.5, 'loss': 0.25, 'val_acc': 0.75},
        )
        stage.epoch.is_active = False
        self.assertEqual(len(logs), 1)
        self.assertRegex(
            logs[-1],
            r'^4 iters in 2.5s - eta 1:01 - acc: 0.125 - loss: 0.25 - val_acc: 0.75 \(\*\)\n$',
        )
        self.assertEqual(remote_doc.logs, [
            ('update',
             {'progress.train': {
                 'batch': 4,
                 'batch_metrics': {'acc': 0.5, 'loss': 0.25, 'val_acc': 0.75},
                 'batch_time': 1.5,
                 'elapsed': 3.0,
                 'eta': 61,
                 'epoch': 3,
                 'epoch_time': 2.5,
                 'max_batch': 78,
                 'max_epoch': 13},
                 'result': {'acc': 0.125, 'loss': 0.25, 'val_acc': 0.75}})
        ])
        logs.clear()
        remote_doc.logs.clear()

        ##################
        # on_stage_end() #
        ##################
        stage.end_timestamp = 10.
        data = new_callback_data(
            stage=stage,
            start_timestamp=0.5,
            end_timestamp=10.,
            exc_time=9.5,
            metrics={'acc': 0.875},
        )
        ctx = cb.ctx
        cb.on_stage_end(data)
        self.assertEqual(len(cb._ctx_stack), 0)
        self.assertEqual(ctx.progress, {
            'elapsed': 9.5,
            'eta': 61,
            'epoch': 3,
            'max_epoch': 13,
            'epoch_time': 2.5,
            'batch': 4,
            'max_batch': 78,
            'batch_time': 1.5,
            'batch_metrics': {'acc': 0.5, 'loss': 0.25, 'val_acc': 0.75},
        })
        self.assertEqual(ctx.metrics_collector.to_json(), {
            'acc': 0.875,
            'loss': 0.25,
            'val_acc': 0.75,
        })
        self.assertEqual(len(logs), 1)
        self.assertRegex(
            logs[-1],
            r'^\[[^\[\]]+\] Train finished in 9.5s - acc: 0.875 - loss: 0.25 - '
            r'val_acc: 0.75\n$',
        )
        self.assertEqual(remote_doc.logs, [
            ('update', {
                'progress.train': {
                    'elapsed': 9.5,
                    'eta': 61,
                    'epoch': 3,
                    'epoch_time': 2.5,
                    'max_epoch': 13,
                    'batch': 4,
                    'max_batch': 78,
                    'batch_time': 1.5,
                    'batch_metrics': {
                        'acc': 0.5, 'loss': 0.25, 'val_acc': 0.75},
                },
                'result': {'acc': 0.875, 'loss': 0.25, 'val_acc': 0.75},
            }),
            'stop_worker'
        ])
        remote_doc.logs.clear()

        ################
        # on_metrics() #
        ################
        data = new_callback_data(stage=stage, metrics={'b': 789.5})
        cb.on_metrics(data)
        self.assertEqual(remote_doc.logs, [('update', {'result': {'b': 789.5}})])
        remote_doc.logs.clear()


class StopOnNaNTestCase(unittest.TestCase):

    def test_stop_on_nan(self):
        metrics = {'a': 1., 'b': 2.}
        data = new_callback_data(metrics=metrics)
        cb = StopOnNaN()

        # no nan metric should be okay
        cb.on_batch_end(data)
        cb.on_epoch_end(data)
        cb.on_stage_end(data)

        # nan metric should raise error
        data.metrics['b'] = np.nan
        with pytest.raises(NaNMetricError,
                           match='NaN metric encountered: \'b\''):
            cb.on_batch_end(data)
        with pytest.raises(NaNMetricError,
                           match='NaN metric encountered: \'b\''):
            cb.on_epoch_end(data)
        with pytest.raises(NaNMetricError,
                           match='NaN metric encountered: \'b\''):
            cb.on_stage_end(data)


class _MyTrainCallback(BaseTrainCallback):
    pass


class TrainStageCallbackTestCase(unittest.TestCase):

    def test_life_cycle(self):
        def make_data(stage):
            return new_callback_data(stage=stage, start_timestamp=0.)

        cb = _MyTrainCallback()
        train_stage = Stage(StageType.TRAIN)
        train_stage2 = Stage(StageType.TRAIN)
        test_stage = Stage(StageType.TEST)

        self.assertIsNone(cb.stage)
        cb.on_stage_begin(make_data(train_stage))
        self.assertIs(cb.stage, train_stage)
        cb.on_stage_begin(make_data(train_stage2))
        self.assertIs(cb.stage, train_stage)
        cb.on_stage_begin(make_data(test_stage))
        self.assertIs(cb.stage, train_stage)

        cb.on_stage_end(make_data(test_stage))
        self.assertIs(cb.stage, train_stage)
        cb.on_stage_end(make_data(train_stage2))
        self.assertIs(cb.stage, train_stage)
        cb.on_stage_end(make_data(train_stage))
        self.assertIsNone(cb.stage)

        with pytest.raises(
                RuntimeError,
                match=f'The outer stage of `{_MyTrainCallback.__qualname__}` '
                      f'must be a train stage: got test stage .*'):
            cb.on_stage_begin(make_data(test_stage))


class CheckpointCallbackTestCase(unittest.TestCase):

    def test_base(self):
        stage = Stage(StageType.TRAIN)
        data = new_callback_data(stage=stage, start_timestamp=0.)

        checkpoint = BaseCheckpoint()
        a = SimpleStatefulObject()

        with TemporaryDirectory() as temp_dir:
            root_dir = os.path.join(temp_dir, 'ckpt')

            # test invalid state objects
            with pytest.raises(ValueError,
                               match='State object key \'__stage\' is '
                                     'reserved.'):
                _ = BaseCheckpointCallback(checkpoint, root_dir, {'__stage': a})
            with pytest.raises(ValueError,
                               match='The item \'a\' in `state_objects` is not '
                                     'a StatefulObject: got .*'):
                _ = BaseCheckpointCallback(checkpoint, root_dir, {'a': 123})

            # test construct
            cb = BaseCheckpointCallback(
                checkpoint=checkpoint,
                root_dir=root_dir,
                state_objects=StatefulObjectGroup({'a': a}),
                max_checkpoints_to_keep=3
            )
            self.assertIsNone(cb.checkpoint_manager)
            self.assertEqual(list(cb.state_objects), ['a'])
            self.assertIs(cb.state_objects['a'], a)

            # on_stage_begin()
            cb.on_stage_begin(data)
            self.assertEqual(cb.checkpoint_manager.checkpoint, checkpoint)
            self.assertEqual(cb.checkpoint_manager.root_dir, root_dir)
            self.assertEqual(cb.checkpoint_manager.max_to_keep, 3)
            self.assertEqual(
                list(cb.checkpoint_manager.state_objects),
                ['a', '__stage']
            )
            self.assertIs(cb.checkpoint_manager.state_objects['a'], a)
            stage_state = cb.checkpoint_manager.state_objects['__stage']
            self.assertIsInstance(stage_state, _StageCounterState)
            self.assertEqual(stage_state.stage, stage)

            # make checkpoint
            cb.checkpoint_manager.save = mock.Mock()
            stage.epoch.index = 4
            stage.batch.index = 5
            cb.make_checkpoint()
            self.assertEqual(
                cb.checkpoint_manager.save.call_args,
                (
                    ('epoch-4-batch-5',), {}
                )
            )

            # on_stage_end()
            cb.on_stage_end(data)
            self.assertNotIn('__stage', cb.state_objects)
            self.assertIsNone(cb.checkpoint_manager)

    def test_auto_checkpoint(self):
        class _MyCheckpoint(BaseCheckpoint):
            logs = []

            def _save(self, checkpoint_path: str) -> None:
                os.makedirs(checkpoint_path)
                self.logs.append(('save', checkpoint_path))

            def _restore(self, checkpoint_path: str) -> None:
                self.logs.append(('restore', checkpoint_path))

        checkpoint = _MyCheckpoint()
        a = SimpleStatefulObject()
        state_objects = {'a': a}
        stage = Stage(StageType.TRAIN)
        data = new_callback_data(stage=stage, start_timestamp=0.)
        stage2 = Stage(StageType.TRAIN)
        data2 = new_callback_data(stage=stage2, start_timestamp=0.)

        with TemporaryDirectory() as temp_dir:
            root_dir = os.path.join(temp_dir, 'ckpt')

            # test construct
            exclusive_args = ('interval', 'epoch_freq', 'batch_freq')
            for mode in (0b000, 0b011, 0b101, 0b110, 0b111):
                kwargs = {k: None if (mode & (1 << i)) == 0 else 1
                          for i, k in enumerate(exclusive_args)}
                with pytest.raises(ValueError,
                                   match='One and only one of `interval`, '
                                         '`epoch_freq` and `batch_freq` should '
                                         'be specified'):
                    _ = AutoCheckpoint(checkpoint=checkpoint, root_dir=root_dir,
                                       **kwargs)
            with pytest.raises(TypeError,
                               match='`restore_checkpoint` must be a str or a '
                                     'bool: got .*'):
                _ = AutoCheckpoint(checkpoint=checkpoint, root_dir=root_dir,
                                   interval=1., restore_checkpoint=123)

            for arg in exclusive_args:
                cb = AutoCheckpoint(checkpoint=checkpoint, root_dir=root_dir,
                                    **{arg: 123})
                self.assertEqual(cb.checkpoint, checkpoint)
                self.assertEqual(cb.root_dir, root_dir)
                self.assertEqual(cb.restore_checkpoint, True)
                self.assertEqual(cb.last_checkpoint_time, 0.)
                self.assertEqual(getattr(cb, arg), 123)
                for arg2 in exclusive_args:
                    if arg2 != arg:
                        self.assertIsNone(getattr(cb, arg2))

            ####################
            # on_train_begin() #
            ####################
            cb = AutoCheckpoint(checkpoint=checkpoint, root_dir=root_dir,
                                interval=0., state_objects=state_objects)
            cb.make_checkpoint = mock.Mock()
            cb.on_stage_begin(data)
            self.assertIs(cb.checkpoint_manager.state_objects['a'], a)
            self.assertIs(cb.checkpoint_manager.state_objects['__stage'].stage,
                          stage)

            a.value = 123
            ckpt_path_1 = cb.checkpoint_manager.save('ckpt_1')
            a.value = 456
            ckpt_path_2 = cb.checkpoint_manager.save('ckpt_2')
            checkpoint.logs.clear()

            # restore_checkpoint is True
            cb.restore_checkpoint = True
            a.value = 789
            with mock.patch('time.time', return_value=111.):
                cb.on_train_begin(data)
            self.assertEqual(cb.last_checkpoint_time, 111.)
            self.assertEqual(a.value, 456)
            self.assertEqual(
                checkpoint.logs,
                [('restore', os.path.join(ckpt_path_2, 'ckpt'))]
            )
            checkpoint.logs.clear()

            # restore_checkpoint is False
            cb.restore_checkpoint = False
            a.value = 789
            with mock.patch('time.time', return_value=222.):
                cb.on_train_begin(data)
            self.assertEqual(cb.last_checkpoint_time, 222.)
            self.assertEqual(a.value, 789)
            self.assertEqual(checkpoint.logs, [])

            # restore_checkpoint is str
            cb.restore_checkpoint = ckpt_path_1
            a.value = 789
            with mock.patch('time.time', return_value=333.):
                cb.on_train_begin(data)
            self.assertEqual(cb.last_checkpoint_time, 333.)
            self.assertEqual(a.value, 123)
            self.assertEqual(
                checkpoint.logs,
                [('restore', os.path.join(ckpt_path_1, 'ckpt'))]
            )
            checkpoint.logs.clear()

            # a new stage should not trigger action
            cb.restore_checkpoint = True
            with mock.patch('time.time', return_value=444.):
                cb.on_train_begin(data2)
            self.assertEqual(cb.last_checkpoint_time, 333.)
            self.assertEqual(checkpoint.logs, [])

            #################################################
            # on_train_epoch_end() and on_batch_epoch_end() #
            #################################################
            # on_train_epoch_end() by interval
            cb.interval = 1.
            cb.last_checkpoint_time = 0.
            cb.epoch_freq = cb.batch_freq = None
            data.index = 3

            data.end_timestamp = .5
            with mock.patch('time.time', return_value=555.):
                cb.on_train_epoch_end(data)
            self.assertFalse(cb.make_checkpoint.called)
            self.assertEqual(cb.last_checkpoint_time, 0.)
            cb.make_checkpoint.reset_mock()

            data.end_timestamp = 1.
            with mock.patch('time.time', return_value=555.):
                cb.on_train_epoch_end(data)
            self.assertTrue(cb.make_checkpoint.called)
            self.assertEqual(cb.last_checkpoint_time, 555.)
            cb.make_checkpoint.reset_mock()

            # on_batch_epoch_end() by interval
            cb.interval = 1.
            cb.last_checkpoint_time = 0.
            cb.epoch_freq = cb.batch_freq = None
            data.index = 3

            data.end_timestamp = .5
            with mock.patch('time.time', return_value=555.):
                cb.on_train_batch_end(data)
            self.assertFalse(cb.make_checkpoint.called)
            self.assertEqual(cb.last_checkpoint_time, 0.)
            cb.make_checkpoint.reset_mock()

            data.end_timestamp = 1.
            with mock.patch('time.time', return_value=555.):
                cb.on_train_batch_end(data)
            self.assertTrue(cb.make_checkpoint.called)
            self.assertEqual(cb.last_checkpoint_time, 555.)
            cb.make_checkpoint.reset_mock()

            cb.stage.batch.total = 3  # do not save checkpoint at the last batch
            cb.last_checkpoint_time = 0.
            data.end_timestamp = 1.
            with mock.patch('time.time', return_value=555.):
                cb.on_train_batch_end(data)
            self.assertFalse(cb.make_checkpoint.called)
            self.assertEqual(cb.last_checkpoint_time, 0.)
            cb.make_checkpoint.reset_mock()

            # on_train_epoch_end() by epoch_freq
            cb.epoch_freq = 3
            cb.last_checkpoint_time = 0.
            cb.interval = cb.batch_freq = None
            data.end_timestamp = 1.

            data.index = 2
            with mock.patch('time.time', return_value=555.):
                cb.on_train_epoch_end(data)
            self.assertFalse(cb.make_checkpoint.called)
            self.assertEqual(cb.last_checkpoint_time, 0.)
            cb.make_checkpoint.reset_mock()

            data.index = 3
            with mock.patch('time.time', return_value=555.):
                cb.on_train_epoch_end(data)
            self.assertTrue(cb.make_checkpoint.called)
            self.assertEqual(cb.last_checkpoint_time, 555.)
            cb.make_checkpoint.reset_mock()

            # on_train_batch_end() by batch_freq
            cb.batch_freq = 3
            cb.last_checkpoint_time = 0.
            cb.interval = cb.epoch_freq = None
            data.end_timestamp = 1.

            data.index = 2
            with mock.patch('time.time', return_value=555.):
                cb.on_train_batch_end(data)
            self.assertFalse(cb.make_checkpoint.called)
            self.assertEqual(cb.last_checkpoint_time, 0.)
            cb.make_checkpoint.reset_mock()

            data.index = 3
            cb.stage.batch.total = 3  # do save at the last batch if batch_freq matches
            with mock.patch('time.time', return_value=555.):
                cb.on_train_batch_end(data)
            self.assertTrue(cb.make_checkpoint.called)
            self.assertEqual(cb.last_checkpoint_time, 555.)
            cb.make_checkpoint.reset_mock()

    def test_early_stopping(self):
        class _MyCheckpoint(BaseCheckpoint):
            logs = []

            def _save(self, checkpoint_path: str) -> None:
                os.makedirs(checkpoint_path)
                self.logs.append(('save', checkpoint_path))

            def _restore(self, checkpoint_path: str) -> None:
                self.logs.append(('restore', checkpoint_path))

        checkpoint = _MyCheckpoint()
        a = SimpleStatefulObject()
        state_objects = {'a': a}
        stage = Stage(StageType.TRAIN)
        data = new_callback_data(stage=stage, start_timestamp=0.)
        stage2 = Stage(StageType.TRAIN)
        data2 = new_callback_data(stage=stage2, start_timestamp=0.)

        with TemporaryDirectory() as temp_dir:
            root_dir = os.path.join(temp_dir, 'ckpt')

            #############
            # construct #
            #############
            # default args
            cb = EarlyStopping(
                checkpoint=checkpoint,
                root_dir=root_dir,
                metric_name='val_loss',
            )
            self.assertIs(cb.checkpoint, checkpoint)
            self.assertEqual(cb.root_dir, root_dir)
            self.assertEqual(cb.metric_name, 'val_loss')
            self.assertTrue(cb.smaller_is_better)
            self.assertTrue(cb.update_at_equal_metric)
            self.assertIsNone(cb.max_no_improvement_epochs)
            self.assertIsNone(cb.max_no_improvement_batches)
            self.assertFalse(cb.restore_on_error)
            self.assertTrue(cb._is_metric_better(1.0, 2.0))
            self.assertTrue(cb._is_metric_better(1.0, 1.0))
            self.assertEqual(cb.max_checkpoints_to_keep, 1)
            self.assertFalse(cb.save_stage_state)

            def latest_saved_flag():
                temp_state = StatefulObjectGroup({
                    'a': SimpleStatefulObject()
                })
                StateSaver(temp_state).load(
                    os.path.join(cb.checkpoint_manager.latest_checkpoint(),
                                 'state.npz'))
                return temp_state['a'].flag

            # custom args
            cb = EarlyStopping(
                checkpoint=checkpoint,
                root_dir=root_dir,
                metric_name='val_acc',
                smaller_is_better=False,
                update_at_equal_metric=False,
                max_no_improvement_epochs=12,
                max_no_improvement_batches=23,
                restore_on_error=True,
                state_objects=state_objects,
                max_checkpoints_to_keep=2,
            )
            self.assertEqual(cb.metric_name, 'val_acc')
            self.assertFalse(cb.smaller_is_better)
            self.assertEqual(cb.max_no_improvement_epochs, 12)
            self.assertEqual(cb.max_no_improvement_batches, 23)
            self.assertTrue(cb.restore_on_error)
            self.assertTrue(cb._is_metric_better(2.0, 1.0))
            self.assertFalse(cb._is_metric_better(1.0, 1.0))
            self.assertEqual(cb.max_checkpoints_to_keep, 2)
            self.assertIs(cb.state_objects['a'], a)
            self.assertFalse(cb.save_stage_state)

            # error args
            with pytest.raises(ValueError,
                               match=r"Early-stopping metric name must start "
                                     r"with any of the following prefixes: "
                                     r"\['val_', 'valid_'\]; got metric "
                                     r"name 'loss'"):
                _ = EarlyStopping(checkpoint=checkpoint, root_dir=root_dir,
                                  metric_name='loss')

            ##############
            # life cycle #
            ##############
            # validation stage without train stage should take no effect
            cb.on_validation_begin(data2)
            cb.on_validation_batch_end(data2)
            cb.on_validation_end(data2)
            cb.on_train_end(data)

            # start the train stage
            cb.on_stage_begin(data)
            self.assertIs(cb.stage, stage)

            # test end the train stage without any metric
            class _FakeLogger(object):
                logs = []

                def warning(self, msg, *args):
                    self.logs.append(('warning', msg % args))

                def get(self, name):
                    self.logs.append(('get', name))
                    return self

            with mock.patch('mltk.callbacks.getLogger', _FakeLogger().get):
                cb.on_train_end(data)

            self.assertEqual(_FakeLogger.logs, [
                ('get', 'mltk.callbacks'),
                ('warning', 'No checkpoint has been saved for early-stopping.  '
                            'Did you forget to update the validation metric '
                            '\'val_acc\'?'),
            ])

            # test state properties
            pfx = '__mltk.callbacks.EarlyStopping.'
            for keys, default in [
                    (('best_metric_value',), None),
                    (('no_improvement_epochs', 'no_improvement_batches'), 0)]:
                for key in keys:
                    self.assertEqual(getattr(cb, key), default)
                    stage.memo[pfx + key] = 123
                    self.assertEqual(getattr(cb, key), 123)
                    setattr(cb, key, 456)
                    self.assertEqual(stage.memo[pfx + key], 456)
                    delattr(cb, key)
                    self.assertEqual(getattr(cb, key), default)

            # test termination condition
            self.assertFalse(cb._need_termination())

            cb.no_improvement_epochs = 11
            self.assertFalse(cb._need_termination())
            cb.no_improvement_epochs = 12
            self.assertTrue(cb._need_termination())
            cb.no_improvement_epochs = 13
            self.assertTrue(cb._need_termination())
            del cb.no_improvement_epochs

            cb.no_improvement_batches = 22
            self.assertFalse(cb._need_termination())
            cb.no_improvement_batches = 23
            self.assertTrue(cb._need_termination())
            cb.no_improvement_batches = 24
            self.assertTrue(cb._need_termination())
            del cb.no_improvement_batches

            # test epoch begin and batch begin
            cb.on_train_epoch_begin(data)
            self.assertEqual(cb.no_improvement_epochs, 1)
            self.assertEqual(cb.no_improvement_batches, 0)

            cb.on_train_batch_begin(data)
            cb.on_train_batch_begin(data)
            self.assertEqual(cb.no_improvement_epochs, 1)
            self.assertEqual(cb.no_improvement_batches, 2)

            cb.on_train_epoch_begin(data2)
            cb.on_train_batch_begin(data2)
            self.assertEqual(cb.no_improvement_epochs, 1)
            self.assertEqual(cb.no_improvement_batches, 2)

            # empty validation loop should take no effect
            cb.on_validation_begin(data)
            cb.on_validation_end(data)
            self.assertEqual(cb._metric_stats.counter, 0)
            self.assertEqual(cb.no_improvement_epochs, 1)
            self.assertEqual(cb.no_improvement_batches, 2)

            # test _update_valid_metric() with new value
            stage.epoch.index = 3
            stage.batch.index = 4
            a.flag = 123456

            self.assertEqual(stage.state_proxy().get_state_dict(), {
                'epoch': 3,
                'batch': 4,
                'memo': {
                    pfx + 'no_improvement_epochs': 1,
                    pfx + 'no_improvement_batches': 2,
                }
            })
            self.assertEqual(a.flag, 123456)
            self.assertEqual(checkpoint.logs, [])
            self.assertIsNone(cb.best_metric_value)
            self.assertFalse(stage.best_validation_mark)

            cb._update_valid_metric(0.875)  # save by earl-stopping
            self.assertAlmostEqual(cb.best_metric_value, 0.875)
            self.assertEqual(latest_saved_flag(), 123456)
            self.assertEqual(checkpoint.logs, [
                ('save', os.path.join(root_dir, 'epoch-3-batch-4/ckpt'))
            ])
            checkpoint.logs.clear()

            # test _update_valid_metric() in with validation batch metrics
            a.flag = 654321
            cb._metric_stats.update(0.125)
            cb.on_validation_begin(data)
            self.assertEqual(cb._metric_stats.counter, 0)
            batch_data = new_callback_data(
                stage=stage, size=3, metrics={'val_acc': 0.995})
            cb.on_validation_batch_end(batch_data)
            batch_data.size = 2
            batch_data.metrics = {'val_acc': 0.9825}
            cb.on_validation_batch_end(batch_data)
            self.assertEqual(cb._metric_stats.counter, 2)
            self.assertAlmostEqual(cb._metric_stats.mean, 0.99)
            cb.on_validation_end(data)
            self.assertAlmostEqual(cb.best_metric_value, 0.99)
            self.assertEqual(checkpoint.logs, checkpoint.logs, [
                ('save', os.path.join(root_dir, 'epoch-3-batch-4_1/ckpt'))
            ])
            self.assertEqual(latest_saved_flag(), 654321)
            checkpoint.logs.clear()

            # test _update_valid_metric() with validation end metrics
            a.flag = 12321
            cb.on_validation_begin(data)
            cb._metric_stats.update(0.125)
            stage_data = new_callback_data(
                stage=stage, metrics={'val_acc': 0.9995})
            cb.on_validation_end(stage_data)
            self.assertAlmostEqual(cb.best_metric_value, 0.9995)
            self.assertEqual(checkpoint.logs, checkpoint.logs, [
                ('save', os.path.join(root_dir, 'epoch-3-batch-4_2/ckpt'))
            ])
            self.assertEqual(latest_saved_flag(), 12321)
            checkpoint.logs.clear()

            # test _update_valid_metric() with inferior value
            cb._update_valid_metric(0.75)
            self.assertAlmostEqual(cb.best_metric_value, 0.9995)
            self.assertEqual(checkpoint.logs, [])

            # test request_termination()
            self.assertFalse(stage.termination_requested)

            stage.termination_requested = False
            cb.no_improvement_batches = 0
            cb.no_improvement_epochs = 12
            cb._update_valid_metric(0.5)
            self.assertTrue(stage.termination_requested)

            stage.termination_requested = False
            cb.no_improvement_epochs = 0
            cb.no_improvement_batches = 23
            cb._update_valid_metric(0.5)
            self.assertTrue(stage.termination_requested)

            # test `train_end()`
            restore_path = os.path.join(
                cb.checkpoint_manager.latest_checkpoint(), 'ckpt')

            a.flag = 111111
            cb.on_train_end(data)
            self.assertEqual(a.flag, 12321)
            self.assertEqual(checkpoint.logs, [('restore', restore_path)])
            checkpoint.logs.clear()

            for err_class in (UserTermination, KeyboardInterrupt, SystemExit):
                cb.restore_on_error = False
                a.flag = 111111
                try:
                    raise err_class()
                except:
                    cb.on_train_end(data)
                self.assertEqual(a.flag, 12321)
                self.assertEqual(
                    checkpoint.logs, [('restore', restore_path)])
                checkpoint.logs.clear()

            for restore_on_error in (False, True):
                cb.restore_on_error = restore_on_error
                a.flag = 111111
                try:
                    raise ValueError()
                except:
                    cb.on_train_end(data)
                if restore_on_error:
                    self.assertEqual(a.flag, 12321)
                    self.assertEqual(
                        checkpoint.logs, [('restore', restore_path)])
                    checkpoint.logs.clear()
                else:
                    self.assertEqual(a.flag, 111111)
                    self.assertEqual(checkpoint.logs, [])
