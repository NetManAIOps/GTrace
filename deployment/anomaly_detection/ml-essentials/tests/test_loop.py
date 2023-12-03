import codecs
import copy
import json
import os
import random
import unittest
from tempfile import TemporaryDirectory

import mock
import numpy as np
import pytest
from mock import Mock

from mltk import *
from mltk.callbacks import Callback, LoggerCallback, CallbackData
from mltk.loop import BaseLoop, _BatchOnlyLoop


class BaseLoopTestCase(unittest.TestCase):

    def test_construct(self):
        # default args
        stage = Stage(StageType.TEST, batch=2, max_batch=102)
        loop = BaseLoop(stage=stage)
        self.assertEqual(loop._stage, stage)
        self.assertEqual(loop._remote_doc, None)
        self.assertEqual(loop._callbacks[:-1], [])
        self.assertIsInstance(loop.logger, LoggerCallback)
        self.assertIs(loop._callbacks[-1], loop.logger)
        self.assertEqual(loop.batch, 2)
        self.assertEqual(loop.max_batch, 102)

        # default args with remote doc from experiment context, or explicitly None
        with TemporaryDirectory() as temp_dir:
            with Experiment(Config(), output_dir=temp_dir, args=[]) as exp:
                loop = BaseLoop(Stage(StageType.TEST))
                self.assertIs(loop._remote_doc, exp.doc)
                self.assertIs(loop.logger.remote_doc, exp.doc)

                loop = BaseLoop(Stage(StageType.TEST), remote_doc=None)
                self.assertIsNone(loop._remote_doc)
                self.assertIsNone(loop.logger.remote_doc)

        # with remote doc
        stage = Stage(StageType.TEST)
        remote_doc = ExperimentDoc()
        loop = BaseLoop(stage=stage, remote_doc=remote_doc)
        self.assertEqual(loop._remote_doc, remote_doc)

        # with callbacks
        def make_callback(priority):
            ret = Callback()
            ret.priority = priority
            return ret

        stage = Stage(StageType.TEST)
        expected_callbacks = [make_callback(i) for i in range(5)]
        callbacks = copy.copy(expected_callbacks)
        random.shuffle(callbacks)
        loop = BaseLoop(stage=stage, callbacks=callbacks)
        self.assertEqual(loop._callbacks[:-1], expected_callbacks)
        self.assertEqual(list(loop._stage.callbacks[1:-1]), expected_callbacks)

        stage = Stage(StageType.TEST)
        expected_callbacks.append(LoggerCallback())
        callbacks = copy.copy(expected_callbacks[:-1])
        random.shuffle(callbacks)
        stage.add_callback(expected_callbacks[-1])
        loop = BaseLoop(stage=stage)
        for cb in callbacks:
            loop.add_callback(cb)
        self.assertEqual(list(loop._callbacks), expected_callbacks)
        self.assertEqual(list(loop._stage.callbacks[1:]), expected_callbacks)
        self.assertIs(loop.logger, expected_callbacks[-1])

        # remove callback
        cb = expected_callbacks[0]
        loop.remove_callback(cb)
        self.assertEqual(list(loop._callbacks), expected_callbacks[1:])
        self.assertEqual(list(loop._stage.callbacks[1:]), expected_callbacks[1:])

    def test_parent_child(self):
        # the parent loop
        stage = Stage(StageType.TRAIN)
        loop = BaseLoop(stage=stage)

        # the child loop
        stage2 = Stage(StageType.TEST)
        loop2 = BaseLoop(stage=stage2, parent=loop)

        # the parent loop context
        with loop:
            loop.add_metrics({'a': 1.0})
            self.assertEqual(loop._stage_metrics, {'a': 1.0})

            # the child loop context
            with loop2:
                self.assertEqual(loop._child_stack, [loop2])

                loop2.add_metrics({'b': 2.0})
                self.assertEqual(loop2._stage_metrics, {'test_b': 2.0})

                loop.add_metrics({'a': 3.0})
                self.assertEqual(loop._stage_metrics, {'a': 1.0})
                self.assertEqual(loop2._stage_metrics,
                                 {'test_a': 3.0, 'test_b': 2.0})

                loop.add_metrics({'a': 4.0}, add_to_child_=False)
                self.assertEqual(loop._stage_metrics, {'a': 4.0})
                self.assertEqual(loop2._stage_metrics,
                                 {'test_a': 3.0, 'test_b': 2.0})

            loop.add_metrics({'b': 5.0})
            self.assertEqual(loop._stage_metrics, {'a': 4.0, 'b': 5.0})

    def test_add_metrics(self):
        # test add to stage metrics
        stage = Stage(StageType.TRAIN)
        loop = BaseLoop(stage=stage)

        loop.add_metrics({'acc': 0.75, 'loss': 0.125}, loss=0.5)
        self.assertEqual(loop._stage_metrics, {'acc': 0.75, 'loss': 0.5})
        self.assertEqual(loop._epoch_metrics, {})
        self.assertEqual(loop._batch_metrics, {})

        # test add to epoch metrics
        stage.epoch.is_active = True
        loop.add_metrics({'acc': 1.75, 'loss': 1.125}, loss=1.5)
        self.assertEqual(loop._stage_metrics, {'acc': 0.75, 'loss': 0.5})
        self.assertEqual(loop._epoch_metrics, {'acc': 1.75, 'loss': 1.5})
        self.assertEqual(loop._batch_metrics, {})

        # test add to batch metrics
        stage.batch.is_active = True
        loop.add_metrics({'acc': 2.75, 'loss': 2.125}, loss=2.5)
        self.assertEqual(loop._stage_metrics, {'acc': 0.75, 'loss': 0.5})
        self.assertEqual(loop._epoch_metrics, {'acc': 1.75, 'loss': 1.5})
        self.assertEqual(loop._batch_metrics, {'acc': 2.75, 'loss': 2.5})

        # test the metric prefix
        stage = Stage(StageType.TEST)
        loop = BaseLoop(stage=stage)
        loop.add_metrics({'acc': 0.75, 'loss': 0.125}, loss=0.5)
        self.assertEqual(loop._stage_metrics,
                         {'test_acc': 0.75, 'test_loss': 0.5})

    def test_timeit(self):
        stage = Stage(StageType.TRAIN)
        loop = BaseLoop(stage=stage)
        loop.add_metrics = Mock(wraps=loop.add_metrics)

        with pytest.raises(ValueError,
                           match='The metric name for a timer should end '
                                 'with suffix "_time" or "_timer": got '
                                 'metric name \'xyz\''):
            with loop.timeit('xyz'):
                pass

        def fake_timer(v=[1.]):
            v[0] *= 2
            return v[0]

        with mock.patch('time.time', fake_timer):
            with loop.timeit('the_time'):
                pass
            with loop.timeit('another_timer'):
                pass

        self.assertEqual(
            loop._stage_metrics,
            {'the_time': 2.0, 'another_timer': 8.0})

    def test_iter_batches(self):
        # test error calls
        stage = Stage(StageType.TRAIN)
        loop = BaseLoop(stage=stage)
        with pytest.raises(RuntimeError,
                           match='The loop context must be entered before '
                                 'calling `iter_batches()'):
            _ = loop.iter_batches()

        with loop:
            with pytest.raises(ValueError,
                               match='`count` and `limit` cannot be both '
                                     'specified.'):
                _ = loop.iter_batches(count=123, limit=456)

            with pytest.raises(ValueError,
                               match='Any one of `data_generator`, `limit` or '
                                     '`count` is required to be specified when '
                                     '`max_batch` is not configured for the '
                                     'loop.'):
                _ = loop.iter_batches()

            stage.batch.is_active = True
            with pytest.raises(RuntimeError,
                               match=r'`iter_batches\(\)` cannot be called '
                                     r'when a batch is currently running'):
                for _ in loop.iter_batches(count=2):
                    pass

        # test `limit`
        stage = Stage(StageType.TRAIN)
        with BaseLoop(stage=stage) as loop:
            self.assertEqual(list(loop.iter_batches(limit=3)), [1, 2, 3])
            self.assertEqual(list(loop.iter_batches(limit=3)), [])
            self.assertEqual(list(loop.iter_batches(limit=5)), [4, 5])
            stage.batch.index = 1
            self.assertEqual(list(loop.iter_batches(limit=3)), [2, 3])

        # test `count`
        stage = Stage(StageType.TRAIN)
        with BaseLoop(stage=stage) as loop:
            self.assertEqual(list(loop.iter_batches(count=3)), [1, 2, 3])
            self.assertEqual(list(loop.iter_batches(count=3)), [4, 5, 6])

        # test `max_batch`
        stage = Stage(StageType.TRAIN, max_batch=3)
        with BaseLoop(stage=stage) as loop:
            self.assertEqual(list(loop.iter_batches()), [1, 2, 3])
            self.assertEqual(list(loop.iter_batches()), [])
            stage.batch.index = 1
            self.assertEqual(list(loop.iter_batches(limit=3)), [2, 3])

        # test `max_batch` and `limit`
        stage = Stage(StageType.TRAIN, max_batch=5)
        with BaseLoop(stage=stage) as loop:
            self.assertEqual(list(loop.iter_batches(limit=3)), [1, 2, 3])
            self.assertEqual(list(loop.iter_batches(limit=3)), [])
            self.assertEqual(list(loop.iter_batches(limit=7)), [4, 5])

        # test `max_batch` and `count`
        stage = Stage(StageType.TRAIN, max_batch=5)
        with BaseLoop(stage=stage) as loop:
            self.assertEqual(list(loop.iter_batches(count=3)), [1, 2, 3])
            self.assertEqual(list(loop.iter_batches(count=3)), [4, 5])

        # test `data_generator` being a DataStream
        x = np.arange(8)
        g = DataStream.arrays([x], batch_size=3)
        stage = Stage(StageType.TRAIN)
        with BaseLoop(stage=stage) as loop:
            self.assertIsNone(loop.max_batch)
            for i, [batch_x] in loop.iter_batches(g):
                np.testing.assert_equal(batch_x, x[(i - 1) * 3: i * 3])
                self.assertEqual(loop.batch, i)
                self.assertEqual(loop.max_batch, 3)
                self.assertIsNotNone(g._active_iterator)
            self.assertEqual(i, 3)
            self.assertIsNone(g._active_iterator)

        # test `data_generator` being an ordinary iterator
        def x_iterator():
            yield [np.array([0, 1, 2])]
            yield [np.array([3, 4, 5])]
            yield [np.array([6, 7])]

        stage = Stage(StageType.TRAIN)
        with BaseLoop(stage=stage) as loop:
            self.assertIsNone(loop.max_batch)
            for i, [batch_x] in loop.iter_batches(x_iterator()):
                np.testing.assert_equal(batch_x, x[(i - 1) * 3: i * 3])
                self.assertEqual(loop.batch, i)
                self.assertIsNone(loop.max_batch)  # this should be None
            self.assertEqual(i, 3)

        # test `data_generator` and `max_batch`
        x = np.arange(8)
        g = DataStream.arrays([x], batch_size=3)
        stage = Stage(StageType.TRAIN, max_batch=2)
        with BaseLoop(stage=stage) as loop:
            self.assertEqual(loop.max_batch, 2)
            for i, [batch_x] in loop.iter_batches(g):
                np.testing.assert_equal(batch_x, x[(i - 1) * 3: i * 3])
                self.assertEqual(loop.batch, i)
                self.assertEqual(loop.max_batch, 2)
                self.assertIsNotNone(g._active_iterator)
            self.assertEqual(i, 2)
            self.assertIsNone(g._active_iterator)

        # test invalid batch data
        def invalid_iterator():
            yield np.array([1, 2, 3])

        stage = Stage(StageType.TRAIN)
        with BaseLoop(stage=stage) as loop:
            with pytest.raises(ValueError,
                               match='`data_generator` did not yield a '
                                     'non-empty tuple or list of arrays'):
                _ = list(loop.iter_batches(invalid_iterator()))

    def test_complete_metrics_and_outputs_arg(self):
        loop = BaseLoop(Stage(StageType.TEST))
        self.assertEqual(
            loop._complete_metrics_and_outputs_arg((), ()),
            ((), ())
        )

        self.assertEqual(
            loop._complete_metrics_and_outputs_arg(ALL, ()),
            (ALL, ())
        )
        self.assertEqual(
            loop._complete_metrics_and_outputs_arg((), ALL),
            ((), ALL)
        )

        self.assertEqual(
            loop._complete_metrics_and_outputs_arg(ALL, NOT_SET),
            (ALL, ())
        )
        self.assertEqual(
            loop._complete_metrics_and_outputs_arg(NOT_SET, ALL),
            ((), ALL)
        )

        loop.RUN_BATCHES_DEFAULT_METRICS = ALL
        loop.RUN_BATCHES_DEFAULT_OUTPUTS = ()
        self.assertEqual(
            loop._complete_metrics_and_outputs_arg(NOT_SET, NOT_SET),
            (ALL, ())
        )
        self.assertEqual(
            loop._complete_metrics_and_outputs_arg(NOT_SET, ['a']),
            (ALL, ['a'])
        )
        self.assertEqual(
            loop._complete_metrics_and_outputs_arg(['a'], NOT_SET),
            (['a'], ())
        )

        loop.RUN_BATCHES_DEFAULT_METRICS = ()
        loop.RUN_BATCHES_DEFAULT_OUTPUTS = ALL
        self.assertEqual(
            loop._complete_metrics_and_outputs_arg(NOT_SET, NOT_SET),
            ((), ALL)
        )
        self.assertEqual(
            loop._complete_metrics_and_outputs_arg(NOT_SET, ['a']),
            ((), ['a'])
        )
        self.assertEqual(
            loop._complete_metrics_and_outputs_arg(['a'], NOT_SET),
            (['a'], ALL)
        )

    def test_run_batches(self):
        logs = []

        # test no return value
        def batch_fn(batch_x, batch_y):
            i = loop.batch
            np.testing.assert_allclose(batch_x, x[(i - 1) * 3: i * 3])
            np.testing.assert_allclose(batch_y, y[(i - 1) * 3: i * 3])
            logs.append(i)

        x = np.arange(8)
        y = np.arange(8, 16)
        g = DataStream.arrays([x, y], batch_size=3)

        stage = Stage(StageType.TRAIN)
        with BaseLoop(stage=stage) as loop:
            self.assertIsNone(loop.run_batches(batch_fn, g))
        self.assertEqual(logs, [1, 2, 3])
        logs.clear()

        # test return value default args
        def batch_fn(batch_x, batch_y):
            i = loop.batch
            logs.append(i)
            return {'i': i,
                    'x+y': batch_x + batch_y,
                    'sum(x,y)': np.sum(batch_x + batch_y),
                    'avg(x,y)': 0.5 * (batch_x + batch_y)}

        stage = Stage(StageType.TEST)
        with BaseLoop(stage=stage) as loop:
            loop.add_metrics = Mock(loop.add_metrics)
            out = loop.run_batches(batch_fn, g)

            # check the outputs
            expected_out = {
                'avg(x,y)': 7.5,
                'i': 1.875,
                'sum(x,y)': 39.75,
                'x+y': 15.0,
            }
            self.assertEqual(set(out), set(expected_out))
            for key in expected_out:
                np.testing.assert_allclose(out[key], expected_out[key])

            # check the metrics of each batch
            self.assertEqual(
                loop.add_metrics.call_args_list,
                [
                    (({'i': 1.0, 'avg(x,y)': 5.0, 'sum(x,y)': 30.0, 'x+y': 10.0},), {}),
                    (({'i': 2.0, 'avg(x,y)': 8.0, 'sum(x,y)': 48.0, 'x+y': 16.0},), {}),
                    (({'i': 3.0, 'avg(x,y)': 10.5, 'sum(x,y)': 42.0, 'x+y': 21.0},), {}),
                ]
            )

        self.assertEqual(logs, [1, 2, 3])
        logs.clear()

        # test return value with specified metrics, outputs and aggregators
        def batch_fn(batch_x, batch_y):
            i = loop.batch
            logs.append(i)
            return {'i': i,
                    'x+y': batch_x + batch_y,
                    'sum(x,y)': np.sum(batch_x + batch_y),
                    'avg(x,y)': 0.5 * (batch_x + batch_y)}

        stage = Stage(StageType.TEST)
        with BaseLoop(stage=stage) as loop:
            loop.add_metrics = Mock(loop.add_metrics)
            out = loop.run_batches(
                batch_fn,
                g,
                metrics=('avg(x,y)', 'i'),
                outputs=['x+y'],
                aggregators={
                    'sum(x,y)': BatchAggregator('SUM')
                }
            )

            # check the outputs
            expected_out = {
                'avg(x,y)': 7.5,
                'i': 1.875,
                'sum(x,y)': 120,
                'x+y': np.array([8, 10, 12, 14, 16, 18, 20, 22])
            }
            self.assertEqual(set(out), set(expected_out))
            for key in expected_out:
                np.testing.assert_allclose(out[key], expected_out[key])

            # check the metrics of each batch
            self.assertEqual(
                loop.add_metrics.call_args_list,
                [
                    (({'i': 1.0, 'avg(x,y)': 5.0},), {}),
                    (({'i': 2.0, 'avg(x,y)': 8.0},), {}),
                    (({'i': 3.0, 'avg(x,y)': 10.5},), {}),
                ]
            )

        self.assertEqual(logs, [1, 2, 3])
        logs.clear()

        # test error
        def batch_fn(batch_x, batch_y):
            return batch_x + batch_y

        stage = Stage(StageType.TRAIN)
        with BaseLoop(stage=stage) as loop:
            with pytest.raises(TypeError,
                               match='The output of `fn` is expected to be a '
                                     'dict, but got .*'):
                _ = loop.run_batches(batch_fn, g)

    def test_life_cycle(self):
        def summarize_data(data: CallbackData):
            return {
                k: getattr(data, k) for k in data.__slots__
                if k != 'stage' and getattr(data, k) is not None
            }

        class _MyCallback(Callback):
            def on_metrics(self, data: CallbackData):
                logs.append(f'cb:on_metrics:{summarize_data(data)}')

            def on_stage_begin(self, data: CallbackData):
                logs.append(f'cb:stage_begin:{summarize_data(data)}')

            def on_stage_end(self, data: CallbackData):
                logs.append(f'cb:stage_end:{summarize_data(data)}')

            def on_batch_begin(self, data: CallbackData):
                logs.append(f'cb:batch_begin:{summarize_data(data)}')

            def on_batch_end(self, data: CallbackData):
                logs.append(f'cb:batch_end:{summarize_data(data)}')

        def fake_timer():
            return 0.

        logs = []
        x = np.arange(8)
        g = DataStream.arrays([x], batch_size=3)

        stage = Stage(StageType.TRAIN)
        loop = BaseLoop(stage, callbacks=[_MyCallback()])
        loop.on_begin.do(lambda: logs.append('begin'))
        loop.on_end.do(lambda: logs.append('end'))
        loop.on_batch_begin.do(lambda: logs.append('batch_begin'))
        loop.on_batch_end.do(lambda: logs.append('batch_end'))

        with mock.patch('time.time', fake_timer):
            loop.add_metrics({'pre': 1.0})
            with loop:
                loop.add_metrics({'pre': 2.0})
                for i, [batch_x] in loop.iter_batches(g):
                    np.testing.assert_equal(batch_x, x[(i - 1) * 3: i * 3])
                    loop.add_metrics({'i': i, 'avg(x)': np.mean(batch_x)})
                loop.add_metrics({'post': 3.0})

                # test no re-entrant
                with pytest.raises(RuntimeError,
                                   match='BaseLoop is not re-entrant.'):
                    with loop:
                        pass
            loop.add_metrics({'post': 4.0})

        self.assertListEqual(logs, [
            'cb:on_metrics:{\'metrics\': {\'pre\': 1.0}}',
            'begin',
            "cb:stage_begin:{'start_timestamp': 0.0}",
            "cb:on_metrics:{'start_timestamp': 0.0, 'metrics': {'pre': 2.0}}",
            'batch_begin',
            "cb:batch_begin:{'index': 1, 'size': 3, 'start_timestamp': 0.0}",
            'batch_end',
            "cb:on_metrics:{'index': 1, 'size': 3, 'start_timestamp': 0.0, "
            "'end_timestamp': 0.0, 'exc_time': 0.0, 'metrics': {'i': 1, 'avg(x)': 1.0}}",
            "cb:batch_end:{'index': 1, 'size': 3, 'start_timestamp': 0.0, "
            "'end_timestamp': 0.0, 'exc_time': 0.0, 'metrics': {'i': 1, 'avg(x)': 1.0}}",
            'batch_begin',
            "cb:batch_begin:{'index': 2, 'size': 3, 'start_timestamp': 0.0}",
            'batch_end',
            "cb:on_metrics:{'index': 2, 'size': 3, 'start_timestamp': 0.0, "
            "'end_timestamp': 0.0, 'exc_time': 0.0, 'metrics': {'i': 2, 'avg(x)': 4.0}}",
            "cb:batch_end:{'index': 2, 'size': 3, 'start_timestamp': 0.0, "
            "'end_timestamp': 0.0, 'exc_time': 0.0, 'metrics': {'i': 2, 'avg(x)': 4.0}}",
            'batch_begin',
            "cb:batch_begin:{'index': 3, 'size': 2, 'start_timestamp': 0.0}",
            'batch_end',
            "cb:on_metrics:{'index': 3, 'size': 2, 'start_timestamp': 0.0, "
            "'end_timestamp': 0.0, 'exc_time': 0.0, 'metrics': {'i': 3, 'avg(x)': 6.5}}",
            "cb:batch_end:{'index': 3, 'size': 2, 'start_timestamp': 0.0, "
            "'end_timestamp': 0.0, 'exc_time': 0.0, 'metrics': {'i': 3, 'avg(x)': 6.5}}",
            "cb:on_metrics:{'start_timestamp': 0.0, 'metrics': {'pre': 2.0, 'post': 3.0}}",
            'end',
            "cb:on_metrics:{'start_timestamp': 0.0, 'end_timestamp': 0.0, 'exc_time': 0.0, "
            "'metrics': {'pre': 2.0, 'post': 3.0}}",
            "cb:stage_end:{'start_timestamp': 0.0, 'end_timestamp': 0.0, 'exc_time': 0.0, "
            "'metrics': {'pre': 2.0, 'post': 3.0}}",
            'cb:on_metrics:{\'start_timestamp\': 0.0, \'end_timestamp\': 0.0, '
            '\'exc_time\': 0.0, \'metrics\': {\'pre\': 2.0, \'post\': 4.0}}',
        ])


class TrainLoopTestCase(unittest.TestCase):

    def test_construct(self):
        # default args
        loop = TrainLoop()
        self.assertEqual(loop.batch, 0)
        self.assertEqual(loop.epoch, 0)
        self.assertIsNone(loop.max_epoch, None)
        self.assertIsNone(loop.max_batch, None)

        # custom args
        cb = Callback()
        remote_doc = ExperimentDoc()
        loop = TrainLoop(
            max_epoch=12,
            max_batch=23,
            remote_doc=remote_doc,
            callbacks=[cb],
        )
        self.assertEqual(loop.max_epoch, 12)
        self.assertEqual(loop.max_batch, 23)
        self.assertEqual(loop._callbacks[:-1], [cb])
        self.assertEqual(loop._callbacks[-1], loop.logger)
        self.assertIs(loop._remote_doc, remote_doc)
        self.assertIs(loop.logger.remote_doc, remote_doc)

        # error args
        with pytest.raises(ValueError,
                           match='`epochs` must not be specified when '
                                 '`only_batch` is set to True.'):
            _ = TrainLoop(max_epoch=12, only_batch=True)

    def test_iter_batches(self):
        x = np.arange(8)
        g = DataStream.arrays([x], batch_size=3)

        def check_iter_batches(loop):
            batches = list(loop.iter_batches(g))
            self.assertEqual(len(batches), 3)
            for i in range(3):
                np.testing.assert_equal(batches[i][0], i + 1)
                np.testing.assert_equal(batches[i][1][0],
                                        x[i * 3: (i + 1) * 3])

        # ordinary loop
        loop = TrainLoop()
        with loop:
            with pytest.raises(RuntimeError,
                               match=r'The batch loop can only be open inside '
                                     r'an epoch loop.  Did you forget to call '
                                     r'`iter_epochs\(\)`\?'):
                _ = list(loop.iter_batches(g))

            for _ in loop.iter_epochs(1):
                check_iter_batches(loop)

        # only_batch loop
        loop = TrainLoop(only_batch=True)
        with loop:
            self.assertEqual(loop.epoch, 1)
            check_iter_batches(loop)

    def test_iter_epochs(self):
        # test errors
        with TrainLoop(only_batch=True) as loop:
            with pytest.raises(RuntimeError,
                               match=r'The loop is configured with `only_batch'
                                     r' = True`, thus `iter_epochs\(\)` is '
                                     r'prohibited.'):
                _ = loop.iter_epochs()

        loop = TrainLoop()
        with pytest.raises(RuntimeError,
                           match=r'The loop context must be entered before '
                                 r'calling `iter_epochs\(\)`.'):
            _ = loop.iter_epochs()

        with loop:
            for _ in loop.iter_epochs(1):
                with pytest.raises(RuntimeError, match=r'`iter_epochs\(\)` '
                                                       r'is not re-entrant'):
                    _ = loop.iter_epochs()

            with pytest.raises(ValueError,
                               match='`count` and `limit` cannot be both '
                                     'specified.'):
                _ = loop.iter_epochs(1, 2)

            with pytest.raises(ValueError,
                               match='Either `limit` or `count` is required to '
                                     'be specified when `max_epoch` is not '
                                     'configured for the loop.'):
                _ = loop.iter_epochs()

        # test `limit`
        with TrainLoop() as loop:
            self.assertEqual(list(loop.iter_epochs(limit=3)), [1, 2, 3])
            self.assertEqual(list(loop.iter_epochs(limit=3)), [])
            self.assertEqual(list(loop.iter_epochs(limit=5)), [4, 5])
            loop._stage.epoch.index = 1
            self.assertEqual(list(loop.iter_epochs(limit=3)), [2, 3])

        # test `count`
        with TrainLoop() as loop:
            self.assertEqual(list(loop.iter_epochs(count=3)), [1, 2, 3])
            self.assertEqual(list(loop.iter_epochs(count=3)), [4, 5, 6])

        # test `max_batch`
        with TrainLoop(max_epoch=3) as loop:
            self.assertEqual(list(loop.iter_epochs()), [1, 2, 3])
            self.assertEqual(list(loop.iter_epochs()), [])
            loop._stage.epoch.index = 1
            self.assertEqual(list(loop.iter_epochs(limit=3)), [2, 3])

        # test `max_batch` and `limit`
        with TrainLoop(max_epoch=5) as loop:
            self.assertEqual(list(loop.iter_epochs(limit=3)), [1, 2, 3])
            self.assertEqual(list(loop.iter_epochs(limit=3)), [])
            self.assertEqual(list(loop.iter_epochs(limit=7)), [4, 5])

        # test `max_batch` and `count`
        with TrainLoop(max_epoch=5) as loop:
            self.assertEqual(list(loop.iter_epochs(count=3)), [1, 2, 3])
            self.assertEqual(list(loop.iter_epochs(count=3)), [4, 5])

    def test_run_epochs(self):
        loop = TrainLoop()
        loop.iter_epochs = Mock(wraps=loop.iter_epochs)
        loop.run_batches = Mock(wraps=loop.run_batches)
        loop.add_metrics = Mock(wraps=loop.add_metrics)
        g = DataStream.arrays([np.arange(8)], batch_size=3)

        def batch_fn(batch_x):
            return {'avg(x)': np.mean(batch_x)}

        with loop:
            loop.run_epochs(batch_fn, g, limit=2)

        self.assertEqual(
            loop.iter_epochs.call_args_list,
            [(
                (),
                {'limit': 2, 'count': None}
            )]
        )
        self.assertEqual(
            loop.run_batches.call_args_list,
            [(
                (batch_fn, g),
                {'excludes': (), 'metrics': NOT_SET},
            )] * 2
        )
        self.assertEqual(
            loop.add_metrics.call_args_list,
            [
                (({'avg(x)': 1.0},), {}),
                (({'avg(x)': 4.0},), {}),
                (({'avg(x)': 6.5},), {}),
            ] * 2
        )

    def test_run(self):
        g = DataStream.arrays([np.arange(8)], batch_size=3)

        def batch_fn(batch_x):
            return {'avg(x)': np.mean(batch_x)}

        def check_call_args(loop, run_fn, n_epochs=2):
            self.assertEqual(
                run_fn.call_args_list,
                [(
                    (batch_fn, g),
                    {'excludes': (), 'metrics': NOT_SET},
                )]
            )
            self.assertEqual(
                loop.add_metrics.call_args_list,
                [
                    (({'avg(x)': 1.0},), {}),
                    (({'avg(x)': 4.0},), {}),
                    (({'avg(x)': 6.5},), {}),
                ] * n_epochs
            )

        # context, only_batch = False
        loop = TrainLoop(max_epoch=2)
        loop.run_epochs = Mock(wraps=loop.run_epochs)
        loop.add_metrics = Mock(wraps=loop.add_metrics)
        with loop:
            self.assertIsNone(loop.run(batch_fn, g))
        check_call_args(loop, loop.run_epochs)

        # no context, only_batch = False
        loop = TrainLoop(max_epoch=2)
        loop.run_epochs = Mock(wraps=loop.run_epochs)
        loop.add_metrics = Mock(wraps=loop.add_metrics)
        self.assertIsNone(loop.run(batch_fn, g))
        check_call_args(loop, loop.run_epochs)

        # context, only_batch = True
        loop = TrainLoop(only_batch=True)
        loop.run_batches = Mock(wraps=loop.run_batches)
        loop.add_metrics = Mock(wraps=loop.add_metrics)
        with loop:
            self.assertEqual(loop.run(batch_fn, g), {'avg(x)': 3.5})
        check_call_args(loop, loop.run_batches, 1)

        # no context, only_batch = True
        loop = TrainLoop(only_batch=True)
        loop.run_batches = Mock(wraps=loop.run_batches)
        loop.add_metrics = Mock(wraps=loop.add_metrics)
        self.assertEqual(loop.run(batch_fn, g), {'avg(x)': 3.5})
        check_call_args(loop, loop.run_batches, 1)

    def test_life_cycle(self):
        def summarize_data(data: CallbackData):
            return {
                k: getattr(data, k) for k in data.__slots__
                if k != 'stage' and getattr(data, k) is not None
            }

        class _MyCallback(Callback):
            def on_stage_begin(self, data: CallbackData):
                logs.append(f'cb:stage_begin:{summarize_data(data)}')

            def on_stage_end(self, data: CallbackData):
                logs.append(f'cb:stage_end:{summarize_data(data)}')

            def on_batch_begin(self, data: CallbackData):
                logs.append(f'cb:batch_begin:{summarize_data(data)}')

            def on_batch_end(self, data: CallbackData):
                logs.append(f'cb:batch_end:{summarize_data(data)}')

            def on_epoch_begin(self, data: CallbackData):
                logs.append(f'cb:epoch_begin:{summarize_data(data)}')

            def on_epoch_end(self, data: CallbackData):
                logs.append(f'cb:epoch_end:{summarize_data(data)}')

        def fake_timer():
            return 0.

        logs = []
        x = np.arange(8)
        g = DataStream.arrays([x], batch_size=3)

        loop = TrainLoop(max_epoch=2, callbacks=[_MyCallback()])
        loop.on_begin.do(lambda: logs.append('begin'))
        loop.on_end.do(lambda: logs.append('end'))
        loop.on_batch_begin.do(lambda: logs.append('batch_begin'))
        loop.on_batch_end.do(lambda: logs.append('batch_end'))
        loop.on_epoch_begin.do(lambda: logs.append('epoch_begin'))
        loop.on_epoch_end.do(lambda: logs.append('epoch_end'))

        for freq in [1, 2]:
            loop.run_after_every(
                lambda freq=freq: logs.append(f'after_{freq}_epochs'),
                epochs=freq,
            )
            loop.run_after_every(
                lambda freq=freq: logs.append(f'after_{freq}_batches'),
                batches=freq,
            )

        cb = loop.run_after_every(lambda: logs.append('XXX'), epochs=1)
        loop.remove_after_every(cb)
        cb = loop.run_after_every(lambda: logs.append('YYY'), batches=1)
        loop.remove_after_every(cb)

        self.assertIsNone(loop.run_after_every(lambda: logs.append('ZZZ')))
        loop.remove_after_every(None)

        with pytest.raises(ValueError,
                           match='`epochs` and `batches` cannot be both '
                                 'specified'):
            _ = loop.run_after_every(lambda: logs.append('WWW'), epochs=1, batches=1)

        with pytest.raises(ValueError,
                           match='`epochs` must be a positive integer'):
            _ = loop.run_after_every(lambda: logs.append('WWW'), epochs=0)

        with pytest.raises(ValueError,
                           match='`batches` must be a positive integer'):
            _ = loop.run_after_every(lambda: logs.append('WWW'), batches=0)

        with mock.patch('time.time', fake_timer):
            with loop:
                loop.add_metrics({'pre_stage': 2.0})
                for j in loop.iter_epochs():
                    loop.add_metrics({'pre_epoch': 3.0, 'j': j})
                    for i, [batch_x] in loop.iter_batches(g):
                        np.testing.assert_equal(batch_x, x[(i - 1) * 3: i * 3])
                        loop.add_metrics({'i': i, 'avg(x)': np.mean(batch_x)})
                    loop.add_metrics({'post_epoch': 4.0})
                loop.add_metrics({'post_stage': 5.0})

        self.assertEqual(logs, [
            'begin',
            "cb:stage_begin:{'start_timestamp': 0.0}",
            'epoch_begin',
            "cb:epoch_begin:{'index': 1, 'start_timestamp': 0.0}",
            'batch_begin',
            "cb:batch_begin:{'index': 1, 'size': 3, 'start_timestamp': 0.0}",
            'batch_end',
            'after_1_batches',
            "cb:batch_end:{'index': 1, 'size': 3, 'start_timestamp': 0.0, "
            "'end_timestamp': 0.0, 'exc_time': 0.0, 'metrics': {'i': 1, 'avg(x)': 1.0}}",
            'batch_begin',
            "cb:batch_begin:{'index': 2, 'size': 3, 'start_timestamp': 0.0}",
            'batch_end',
            'after_1_batches',
            'after_2_batches',
            "cb:batch_end:{'index': 2, 'size': 3, 'start_timestamp': 0.0, "
            "'end_timestamp': 0.0, 'exc_time': 0.0, 'metrics': {'i': 2, 'avg(x)': 4.0}}",
            'batch_begin',
            "cb:batch_begin:{'index': 3, 'size': 2, 'start_timestamp': 0.0}",
            'batch_end',
            'after_1_batches',
            "cb:batch_end:{'index': 3, 'size': 2, 'start_timestamp': 0.0, "
            "'end_timestamp': 0.0, 'exc_time': 0.0, 'metrics': {'i': 3, 'avg(x)': 6.5}}",
            'epoch_end',
            'after_1_epochs',
            "cb:epoch_end:{'index': 1, 'start_timestamp': 0.0, 'end_timestamp': 0.0, "
            "'exc_time': 0.0, 'metrics': {'pre_epoch': 3.0, 'j': 1, 'post_epoch': 4.0}}",
            'epoch_begin',
            "cb:epoch_begin:{'index': 2, 'start_timestamp': 0.0}",
            'batch_begin',
            "cb:batch_begin:{'index': 1, 'size': 3, 'start_timestamp': 0.0}",
            'batch_end',
            'after_1_batches',
            "cb:batch_end:{'index': 1, 'size': 3, 'start_timestamp': 0.0, "
            "'end_timestamp': 0.0, 'exc_time': 0.0, 'metrics': {'i': 1, 'avg(x)': 1.0}}",
            'batch_begin',
            "cb:batch_begin:{'index': 2, 'size': 3, 'start_timestamp': 0.0}",
            'batch_end',
            'after_1_batches',
            'after_2_batches',
            "cb:batch_end:{'index': 2, 'size': 3, 'start_timestamp': 0.0, "
            "'end_timestamp': 0.0, 'exc_time': 0.0, 'metrics': {'i': 2, 'avg(x)': 4.0}}",
            'batch_begin',
            "cb:batch_begin:{'index': 3, 'size': 2, 'start_timestamp': 0.0}",
            'batch_end',
            'after_1_batches',
            "cb:batch_end:{'index': 3, 'size': 2, 'start_timestamp': 0.0, "
            "'end_timestamp': 0.0, 'exc_time': 0.0, 'metrics': {'i': 3, 'avg(x)': 6.5}}",
            'epoch_end',
            'after_1_epochs',
            'after_2_epochs',
            "cb:epoch_end:{'index': 2, 'start_timestamp': 0.0, 'end_timestamp': 0.0, "
            "'exc_time': 0.0, 'metrics': {'pre_epoch': 3.0, 'j': 2, 'post_epoch': 4.0}}",
            'end',
            "cb:stage_end:{'start_timestamp': 0.0, 'end_timestamp': 0.0, 'exc_time': 0.0, "
            "'metrics': {'pre_stage': 2.0, 'post_stage': 5.0}}"
        ])

    def test_run_on_error(self):
        # test not run on error
        logs = []
        loop = TrainLoop(max_epoch=2)
        loop.run_after_every((lambda: logs.append(loop.epoch)), epochs=1)
        with pytest.raises(RuntimeError, match='break the loop'):
            with loop:
                for j in loop.iter_epochs():
                    if j == 2:
                        raise RuntimeError(f'break the loop')
                    else:
                        pass
        self.assertListEqual(logs, [1])

        # test run on error
        logs = []
        loop = TrainLoop(max_epoch=2)
        loop.run_after_every((lambda: logs.append(loop.epoch)), epochs=1, on_error=True)
        with pytest.raises(RuntimeError, match='break the loop'):
            with loop:
                for j in loop.iter_epochs():
                    if j == 2:
                        raise RuntimeError(f'break the loop')
                    else:
                        pass
        self.assertListEqual(logs, [1, 2])

    def test_sub_loop(self):
        cb = Callback()
        remote_doc = ExperimentDoc()
        loop = TrainLoop(remote_doc=remote_doc, callbacks=[cb])

        for type_ in ('validation', 'test', 'predict'):
            factory = getattr(loop, type_)
            sub_loop: BaseLoop = factory()
            self.assertEqual(sub_loop._stage.type,
                             getattr(StageType, type_.upper()))
            self.assertIs(sub_loop.logger, loop.logger)
            self.assertIs(sub_loop._remote_doc, loop._remote_doc)
            self.assertEqual(sub_loop._callbacks, loop._callbacks)
            self.assertIs(sub_loop.parent, loop)

    def test_final_result(self):
        with TemporaryDirectory() as temp_dir:
            with Experiment(Config, output_dir=temp_dir, args=[]) as exp:
                train_stream = DataStream.arrays(
                    [np.arange(5, dtype=np.float32)], batch_size=2)
                test_stream = DataStream.arrays(
                    [np.arange(10, 13, dtype=np.float32)], batch_size=2)

                loop = TrainLoop(max_epoch=1)
                loop.run(
                    (lambda x: {'loss': np.mean(x ** 2), 'test_acc': np.mean(x)}),
                    train_stream
                )
                loop.test().run((lambda x: {'test_acc': np.mean(x)}), test_stream)

            with codecs.open(os.path.join(temp_dir, 'result.json'), 'rb', 'utf-8') as f:
                cnt = json.loads(f.read())

            self.assertAlmostEqual(cnt['test_acc']['mean'], 11.0)
            self.assertAlmostEqual(cnt['loss']['mean'], 6.0)


class BatchOnlyLoopTestCase(unittest.TestCase):

    def test_run(self):
        g = DataStream.arrays([np.arange(8)], batch_size=3)

        def batch_fn(batch_x):
            return {'avg(x)': np.mean(batch_x)}

        def check_call_args(loop):
            self.assertEqual(
                loop.run_batches.call_args_list,
                [(
                    (batch_fn, g),
                    {'metrics': NOT_SET, 'outputs': NOT_SET,
                     'aggregators': None, 'excludes': ()},
                )]
            )
            self.assertEqual(
                loop.add_metrics.call_args_list,
                [
                    (({'avg(x)': 1.0},), {}),
                    (({'avg(x)': 4.0},), {}),
                    (({'avg(x)': 6.5},), {}),
                ]
            )

        # context, only_batch = True
        loop = _BatchOnlyLoop(stage=Stage(StageType.TEST))
        loop.run_batches = Mock(wraps=loop.run_batches)
        loop.add_metrics = Mock(wraps=loop.add_metrics)
        with loop:
            self.assertEqual(loop.run(batch_fn, g), {'avg(x)': 3.5})
        check_call_args(loop)

        # no context, only_batch = True
        loop = _BatchOnlyLoop(stage=Stage(StageType.TEST))
        loop.run_batches = Mock(wraps=loop.run_batches)
        loop.add_metrics = Mock(wraps=loop.add_metrics)
        self.assertEqual(loop.run(batch_fn, g), {'avg(x)': 3.5})
        check_call_args(loop)

    def test_validation_loop(self):
        loop = ValidationLoop()
        self.assertEqual(loop._stage.type, StageType.VALIDATION)
        self.assertEqual(loop.RUN_BATCHES_DEFAULT_METRICS, ALL)
        self.assertEqual(loop.RUN_BATCHES_DEFAULT_OUTPUTS, ())

    def test_test_loop(self):
        loop = TestLoop()
        self.assertEqual(loop._stage.type, StageType.TEST)
        self.assertEqual(loop.RUN_BATCHES_DEFAULT_METRICS, ALL)
        self.assertEqual(loop.RUN_BATCHES_DEFAULT_OUTPUTS, ())

    def test_predict_loop(self):
        loop = PredictLoop()
        self.assertEqual(loop._stage.type, StageType.PREDICT)
        self.assertEqual(loop.RUN_BATCHES_DEFAULT_METRICS, ())
        self.assertEqual(loop.RUN_BATCHES_DEFAULT_OUTPUTS, ALL)
