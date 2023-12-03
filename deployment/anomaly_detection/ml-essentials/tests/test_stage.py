import copy
import types
import unittest
from functools import partial

import mock
import pytest

from mltk import *
from mltk.callbacks import CallbackList, Callback, CallbackData
from mltk.utils import deep_copy, NOT_SET


class _TestHelperCallback(Callback):

    @staticmethod
    def _log_event(name: str,
                   instance: '_TestHelperCallback',
                   data: CallbackData):
        log_dict = {
            'name': name
        }
        log_dict.update({k: getattr(data, k) for k in data.__slots__})
        log_dict['metrics'] = deep_copy(log_dict['metrics'])
        log_dict = {k: v for k, v in log_dict.items()
                    if v is not None}
        instance.logs.append(log_dict)

    def __init__(self):
        self.logs = []
        attrs = dir(self)
        for attr in attrs:
            if attr.startswith('on_'):
                setattr(
                    self,
                    attr,
                    types.MethodType(partial(self._log_event, attr), self),
                )


class CycleCounterTestCase(unittest.TestCase):

    def test_construct(self):
        # default arguments
        c = CycleCounter()
        self.assertEqual(c.index, 0)
        self.assertIsNone(c.total)
        self.assertIsNone(c.original_total)
        self.assertIsNone(c.avg_total)
        self.assertFalse(c.is_active)
        self.assertTrue(c.is_first_loop)
        self.assertEqual(str(c), '0')

        # non-default arguments
        c = CycleCounter(index=1, total=23)
        self.assertEqual(c.index, 1)
        self.assertEqual(c.total, 23)
        self.assertEqual(c.original_total, 23)
        self.assertIsNone(c.avg_total)
        self.assertFalse(c.is_active)
        self.assertTrue(c.is_first_loop)
        self.assertEqual(str(c), '1/23')

    def test_life_cycle(self):
        c = CycleCounter(index=3)
        self.assertEqual(c.index, 3)
        self.assertTrue(c.is_first_loop)

        # test enter loop
        c.enter_loop()
        self.assertEqual(c.index, 3)
        self.assertTrue(c.is_first_loop)

        # test enter without new index
        c.enter()
        self.assertEqual(c.index, 4)
        self.assertTrue(c.is_active)
        self.assertIsNone(c.total)
        c.exit()
        self.assertEqual(c.index, 4)
        self.assertFalse(c.is_active)
        self.assertTrue(c.is_first_loop)
        self.assertIsNone(c.total)

        # test enter with new index
        c.total = 102
        c.enter(100)
        self.assertEqual(c.index, 100)
        self.assertTrue(c.is_active)
        self.assertEqual(c.total, 102)
        c.exit()
        self.assertEqual(c.index, 100)
        self.assertFalse(c.is_active)
        self.assertTrue(c.is_first_loop)
        self.assertEqual(c._avg_total_n_estimates, 0)
        self.assertEqual(c.total, 102)

        # test exit loop
        c.exit_loop()
        self.assertEqual(c.index, 100)
        self.assertFalse(c.is_first_loop)
        self.assertEqual(c._avg_total_n_estimates, 1)
        self.assertEqual(c.avg_total, 100)
        self.assertEqual(c.total, None)  # restored to original total

        # test 2nd next_loop
        c.enter_loop()
        self.assertEqual(c.index, 0)
        c.enter(100)
        c.exit()
        c.exit_loop()
        self.assertEqual(c.index, 100)
        self.assertFalse(c.is_first_loop)
        self.assertEqual(c._avg_total_n_estimates, 2)
        self.assertEqual(c.avg_total, 100)

        # test 3rd next_loop, with different total
        c.enter_loop()
        self.assertEqual(c.index, 0)
        c.enter(102)
        c.exit()
        self.assertEqual(c.index, 102)
        c.exit_loop()
        self.assertEqual(c.index, 102)
        self.assertFalse(c.is_first_loop)
        self.assertEqual(c._avg_total_n_estimates, 3)
        self.assertAlmostEqual(c.avg_total, 100.66666666666667)

    def test_estimated_cycles_ahead(self):
        ########################
        # without manual total #
        ########################
        c = CycleCounter()

        # no total, thus no estimation
        c.enter_loop()
        c.enter(100)
        c.exit()
        self.assertIsNone(c.estimated_cycles_ahead())
        c.exit_loop()
        self.assertEqual(c.avg_total, 100)

        # test count_this_cycle is True
        c.index = 50
        self.assertEqual(c.estimated_cycles_ahead(), 50)  # defaults to ``self.is_active``
        self.assertEqual(c.estimated_cycles_ahead(count_this_cycle=False), 50)
        self.assertEqual(c.estimated_cycles_ahead(count_this_cycle=True), 51)

        c.enter()
        self.assertEqual(c.estimated_cycles_ahead(), 50)  # defaults to ``self.is_active``
        self.assertEqual(c.estimated_cycles_ahead(count_this_cycle=False), 49)
        self.assertEqual(c.estimated_cycles_ahead(count_this_cycle=True), 50)
        c.exit()

        #####################
        # with manual total #
        #####################
        c = CycleCounter(total=101)
        c.enter_loop()
        c.enter(100)
        c.exit()
        self.assertEqual(c.estimated_cycles_ahead(), 1)
        c.exit_loop()
        self.assertEqual(c.avg_total, 100)

        c.enter(100)
        c.exit()
        self.assertEqual(c.estimated_cycles_ahead(), 0)  # estimation improved by avg_total

    def test_timed_counter(self):
        c = TimedCycleCounter(total=10)

        self.assertIsNone(c.start_timestamp)
        self.assertIsNone(c.end_timestamp)
        self.assertIsNone(c.avg_cycle_time)
        self.assertIsNone(c.last_cycle_time)
        self.assertEqual(c._avg_cycle_time_n_estimates, 0)
        self.assertIsNone(c.estimated_time_ahead())

        # enter
        with mock.patch('time.time', return_value=123.):
            c.enter()
        self.assertEqual(c.index, 1)
        self.assertEqual(c.start_timestamp, 123.)
        self.assertIsNone(c.estimated_time_ahead())

        # pre_exit
        with mock.patch('time.time', return_value=200.):
            c.pre_exit()
        self.assertEqual(c.end_timestamp, 200.)
        self.assertEqual(c.last_cycle_time, 77.)
        self.assertEqual(c._avg_cycle_time_n_estimates, 1)
        self.assertEqual(c.avg_cycle_time, 77.)
        self.assertEqual(c.estimated_time_ahead(), 77. * 10)

        c.exit()
        self.assertEqual(c.estimated_time_ahead(), 77. * 9)

        # enter (6)
        with mock.patch('time.time', return_value=211.):
            c.enter(6)
        self.assertEqual(c.index, 6)
        self.assertEqual(c.start_timestamp, 211.)

        # pre_exit
        with mock.patch('time.time', return_value=291.):
            c.pre_exit()
        self.assertEqual(c.end_timestamp, 291.)
        self.assertEqual(c.last_cycle_time, 80.)
        self.assertEqual(c._avg_cycle_time_n_estimates, 2)
        self.assertEqual(c.avg_cycle_time, 78.5)
        self.assertEqual(c.estimated_time_ahead(), 78.5 * 5)

        c.exit()
        self.assertEqual(c.estimated_time_ahead(), 78.5 * 4)


class StageTestCase(unittest.TestCase):

    def test_StageType(self):
        metric_prefixes = {
            'TRAIN': ('', 'train_'),
            'VALIDATION': ('val_', 'valid_'),
            'TEST': ('test_',),
            'PREDICT': ('pred_', 'predict_'),
        }
        metric_keys = [
            'loss', 'train_loss',
            'val_loss', 'valid_loss',
            'test_loss',
            'pred_loss', 'predict_loss',
        ]

        for s in ('TRAIN', 'VALIDATION', 'TEST', 'PREDICT'):
            t: StageType = getattr(StageType, s)
            self.assertEqual(t.name, s)
            self.assertEqual(t.value, s.lower())
            self.assertEqual(t.metric_prefixes, metric_prefixes[s])
            self.assertEqual(t.metric_prefix, metric_prefixes[s][0])

            for key in metric_keys:
                if not any(pfx == '' or key.startswith(pfx)
                           for pfx in metric_prefixes[s]):
                    expected_key = metric_prefixes[s][0] + key
                else:
                    expected_key = key
                self.assertEqual(t.add_metric_prefix(key), expected_key)

    def test_construct(self):
        # test default arguments
        stage = Stage(StageType.TEST)
        self.assertEqual(stage.name, 'test')
        self.assertEqual(stage.type, StageType.TEST)
        self.assertIsNone(stage.epoch)
        self.assertEqual((stage.batch.index, stage.batch.total), (0, None))
        self.assertIsNone(stage.batch_size)
        self.assertIsNone(stage.data_length)
        self.assertIsNone(stage.global_step)
        self.assertEqual(stage.memo, {})
        self.assertIsInstance(stage.callbacks, CallbackList)
        self.assertEqual(list(stage.callbacks), [])
        self.assertIsNone(stage.known_metrics)

        self.assertEqual(stage.best_validation_mark, False)
        self.assertIsNone(stage._current_epoch_size)
        self.assertIsNone(stage._current_epoch_size)
        self.assertFalse(stage.is_active)
        self.assertFalse(stage.termination_requested)
        self.assertIsNone(stage.start_timestamp)
        self.assertIsNone(stage.end_timestamp)

        # test non-default arguments
        callback = Callback()
        stage = Stage(
            StageType.TRAIN,
            epoch=2,
            max_epoch=10,
            batch=4,
            max_batch=24,
            batch_size=17,
            data_length=255,
            global_step=7,
            callbacks=[callback],
            known_metrics=['loss']
        )
        self.assertEqual(stage.name, 'train')
        self.assertEqual(stage.type, StageType.TRAIN)
        self.assertEqual((stage.epoch.index, stage.epoch.total), (2, 10))
        self.assertEqual((stage.batch.index, stage.batch.total), (4, 24))
        self.assertEqual(stage.batch_size, 17)
        self.assertEqual(stage.data_length, 255)
        self.assertEqual(stage.global_step, 7)
        self.assertEqual(stage.callbacks, CallbackList([callback]))
        self.assertEqual(stage.known_metrics, ('loss',))

        # test stage should add prefix to metric names
        stage = Stage(StageType.TEST, known_metrics=['loss'])
        self.assertEqual(stage.known_metrics, ('test_loss',))

    def test_train_cycle(self):
        def assert_logs(names, data):
            buf = []
            for name in names:
                name_data = copy.copy(data)
                name_data['stage'] = stage
                name_data['name'] = name
                buf.append(name_data)
            self.assertEqual(cb.logs, buf)

        for add_callback_mode in (0, 1):
            next_timer_state = [0.]

            def next_timer():
                next_timer_state[0] += 1
                return next_timer_state[0]

            with mock.patch('time.time', next_timer):
                cb = _TestHelperCallback()
                cb2 = _TestHelperCallback()

                ####################
                # test train stage #
                ####################
                initial_callbacks = [cb, cb2] if add_callback_mode == 0 else []
                stage = Stage(
                    StageType.TRAIN,
                    epoch=2,
                    max_epoch=10,
                    batch=3,
                    max_batch=11,
                    global_step=123,
                    callbacks=initial_callbacks,
                )
                if add_callback_mode == 1:
                    stage.add_callback(cb)
                    stage.add_callback(cb2)
                stage.remove_callback(cb2)

                # enter
                cb.logs.clear()
                stage.best_validation_mark = stage.is_active = \
                    stage.termination_requested = stage.start_timestamp = \
                    stage.end_timestamp = NOT_SET
                stage.enter()
                self.assertEqual(stage.epoch.index, 2)
                self.assertEqual(stage.epoch.is_active, False)
                self.assertEqual(stage.batch.index, 3)
                self.assertEqual(stage.batch.is_active, False)
                self.assertEqual(stage.global_step, 123)
                self.assertEqual(stage.best_validation_mark, False)
                self.assertEqual(stage.is_active, True)
                self.assertEqual(stage.termination_requested, False)
                self.assertEqual(stage.start_timestamp, 1.)
                self.assertEqual(stage.end_timestamp, None)
                assert_logs(
                    ['on_stage_begin', 'on_train_begin'],
                    {'start_timestamp': 1.},
                )
                self.assertIsNone(stage.get_eta())

                with pytest.raises(RuntimeError,
                                   match='`Stage` is neither re-entrant, nor '
                                         'reusable'):
                    stage.enter()

                # enter epoch
                cb.logs.clear()
                stage._current_epoch_size = NOT_SET
                stage.best_validation_mark = NOT_SET

                stage.enter_epoch(epoch_size=987)
                self.assertEqual(stage.epoch.index, 3)
                self.assertEqual(stage.epoch.is_active, True)
                self.assertEqual(stage._current_epoch_size, 987)
                self.assertEqual(stage.best_validation_mark, False)
                assert_logs(
                    ['on_epoch_begin', 'on_train_epoch_begin'],
                    {'index': 3, 'size': 987, 'start_timestamp': 2.},
                )

                with pytest.raises(RuntimeError,
                                   match='Stage .* does not have an epoch counter'):
                    Stage(StageType.TEST).enter_epoch()

                # enter batch
                cb.logs.clear()
                stage._current_batch_size = NOT_SET
                stage.best_validation_mark = NOT_SET

                stage.enter_batch(batch_size=123)
                self.assertEqual(stage.batch.index, 4)
                self.assertEqual(stage.global_step, 124)
                self.assertEqual(stage.batch.is_active, True)
                self.assertEqual(stage._current_batch_size, 123)
                self.assertEqual(stage.best_validation_mark, False)
                assert_logs(
                    ['on_batch_begin', 'on_train_batch_begin'],
                    {'index': 4, 'size': 123, 'start_timestamp': 3.},
                )

                # exit batch
                stage.best_validation_mark = True
                cb.logs.clear()
                stage.exit_batch({'loss': 0.25})
                self.assertEqual(stage.batch.index, 4)
                self.assertEqual(stage.global_step, 124)
                self.assertEqual(stage.batch.is_active, False)
                self.assertIsNone(stage._current_batch_size)
                self.assertEqual(stage.best_validation_mark, True)
                assert_logs(
                    ['on_metrics', 'on_train_batch_end', 'on_batch_end'],
                    {
                        'index': 4, 'size': 123, 'start_timestamp': 3.,
                        'end_timestamp': 4., 'exc_time': 1.,
                        'metrics': {'loss': 0.25},
                    },
                )

                # exit epoch
                stage.best_validation_mark = True
                cb.logs.clear()
                stage.exit_epoch({'acc': 0.875})
                self.assertEqual(stage.epoch.index, 3)
                self.assertEqual(stage.epoch.is_active, False)
                self.assertEqual(stage.batch.index, 4)
                self.assertEqual(stage.batch.is_active, False)
                self.assertEqual(stage._current_batch_size, None)
                self.assertEqual(stage.best_validation_mark, True)
                assert_logs(
                    ['on_metrics', 'on_train_epoch_end', 'on_epoch_end'],
                    {
                        'index': 3, 'size': 987, 'start_timestamp': 2.,
                        'end_timestamp': 5., 'exc_time': 3.,
                        'metrics': {'acc': 0.875},
                    },
                )

                with pytest.raises(RuntimeError,
                                   match='Stage .* does not have an epoch counter'):
                    Stage(StageType.TEST).exit_epoch()

                # exit
                stage.end_timestamp = NOT_SET
                stage.best_validation_mark = True
                cb.logs.clear()
                stage.exit({'ll': 0.125})
                self.assertEqual(stage.end_timestamp, 6.)
                self.assertEqual(stage.is_active, False)
                self.assertEqual(stage.epoch.index, 3)
                self.assertEqual(stage.epoch.is_active, False)
                self.assertEqual(stage.batch.index, 4)
                self.assertEqual(stage.batch.is_active, False)
                self.assertEqual(stage.best_validation_mark, True)
                assert_logs(
                    ['on_metrics', 'on_train_end', 'on_stage_end'],
                    {
                        'start_timestamp': 1., 'end_timestamp': 6.,
                        'exc_time': 5., 'metrics': {'ll': 0.125},
                    },
                )

                # cb2 should not be ever called
                self.assertEqual(cb2.logs, [])

    def test_state_proxy(self):
        #########
        # train #
        #########
        stage = Stage(StageType.TRAIN)

        # empty state
        state = stage.state_proxy()
        self.assertEqual(state.get_state_dict(), {
            'epoch': 0,
            'batch': 0,
            'memo': {},
        })

        # set epoch, batch and global_step, and add memo
        stage.epoch.index = 1
        stage.batch.index = 2
        stage.global_step = 99
        stage.memo['acc'] = 0.25
        self.assertEqual(state.get_state_dict(), {
            'epoch': 1,
            'batch': 2,
            'global_step': 99,
            'memo': {'acc': 0.25},
        })

        # let epoch and batch to be active
        stage.epoch.is_active = True
        stage.batch.is_active = True
        self.assertEqual(state.get_state_dict(), {
            'epoch': 0,
            'batch': 1,
            'global_step': 98,
            'memo': {'acc': 0.25},
        })

        # use set_state_dict to update a stage
        stage = Stage(StageType.TRAIN)

        stage.state_proxy().set_state_dict({
            'epoch': 1,
            'batch': 2,
            'global_step': 87,
            'memo': {
                'loss': 0.875,
            }
        })
        self.assertEqual(stage.epoch.index, 1)
        self.assertEqual(stage.batch.index, 2)
        self.assertEqual(stage.global_step, 87)
        self.assertEqual(stage.memo, {'loss': 0.875})

        stage.state_proxy().set_state_dict({
            'epoch': 3,
            'batch': 4,
        })
        self.assertEqual(stage.epoch.index, 3)
        self.assertEqual(stage.batch.index, 4)
        self.assertEqual(stage.global_step, None)
        self.assertEqual(stage.memo, {})

        ########
        # test #
        ########
        stage = Stage(StageType.TEST, batch=123, global_step=33)
        self.assertEqual(stage.state_proxy().get_state_dict(), {
            'batch': 123,
            'global_step': 33,
            'memo': {},
        })

        stage.state_proxy().set_state_dict({
            'batch': 333,
            'memo': {'acc': 0.875}
        })
        self.assertEqual(stage.batch.index, 333)
        self.assertEqual(stage.global_step, None)
        self.assertEqual(stage.memo, {'acc': 0.875})

    def test_request_termination(self):
        stage = Stage(StageType.TRAIN)
        stage.enter()
        self.assertFalse(stage.termination_requested)
        stage.request_termination()
        self.assertTrue(stage.termination_requested)
        stage.enter_epoch()
        self.assertTrue(stage.termination_requested)
        stage.enter_batch()
        self.assertTrue(stage.termination_requested)
        stage.exit_batch()
        self.assertTrue(stage.termination_requested)
        stage.exit_epoch()
        self.assertTrue(stage.termination_requested)
        stage.exit()
        self.assertTrue(stage.termination_requested)

    def test_get_eta(self):
        #########
        # train #
        #########
        stage = Stage(StageType.TRAIN)
        self.assertIsNone(stage.get_eta())

        # use `epoch.total`, `batch.avg_cycle_time` and `batch.total`
        stage.epoch.total = 3
        stage.epoch.index = 2
        stage.epoch.is_active = True
        stage.batch.total = 5
        stage.batch.avg_cycle_time = 1.5
        stage.batch.index = 3
        stage.batch.is_active = True
        self.assertAlmostEqual(stage.get_eta(), (3 + 1 * 5) * 1.5)

        stage.batch.is_active = False
        self.assertAlmostEqual(stage.get_eta(), (2 + 1 * 5) * 1.5)

        stage.epoch.total = None
        self.assertEqual(stage.get_eta(), None)

        # use `epoch.avg_total`, `batch.avg_cycle_time` and `batch.avg_total`
        stage.epoch.avg_total = 7.
        stage.batch.avg_total = 9.
        stage.batch.is_active = True
        self.assertAlmostEqual(stage.get_eta(), (7 + 5 * 9) * 1.5)

        # use `epoch.avg_total` and `epoch.avg_cycle_time`
        stage.epoch.avg_cycle_time = 21
        self.assertAlmostEqual(stage.get_eta(), (7 / 9. + 5) * 21)

        # disable `batch.total`
        stage.batch.total = stage.batch.avg_total = None
        stage.epoch.is_active = True
        self.assertAlmostEqual(stage.get_eta(), 6 * 21)
        stage.epoch.is_active = False
        self.assertAlmostEqual(stage.get_eta(), 5 * 21)

        ########
        # test #
        ########
        stage = Stage(StageType.TEST)
        self.assertIsNone(stage.get_eta())

        stage.batch.total = 8
        stage.batch.index = 3
        stage.batch.avg_cycle_time = 1.5
        stage.batch.is_active = True
        self.assertAlmostEqual(stage.get_eta(), 6 * 1.5)
        stage.batch.is_active = False
        self.assertAlmostEqual(stage.get_eta(), 5 * 1.5)
