import time
import unittest
import warnings

import numpy as np
import pytest

from mltk.data import *

# Do not delete the following line!
# It checks whether DataStream is exposed to the root package.
from mltk import DataStream
from tests.helpers import slow_test


class DataStreamTestCase(unittest.TestCase):

    def test_construct(self):
        # test properties
        args = {
            'batch_size': 12,
            'array_count': 2,
            'data_shapes': ((), (3, 5)),
            'data_length': 25,
            'random_state': np.random.RandomState(),
        }

        s1 = DataStream()
        s2 = DataStream(**args)

        for attr in args:
            self.assertIsNone(getattr(s1, attr))
            self.assertEqual(getattr(s2, attr), args[attr])

        # test argument validation
        s = DataStream(array_count=3)
        self.assertEqual(s.array_count, 3)
        self.assertIsNone(s.data_shapes)

        s = DataStream(data_shapes=args['data_shapes'])
        self.assertEqual(s.data_shapes, args['data_shapes'])
        self.assertEqual(s.array_count, 2)

        with pytest.raises(TypeError,
                           match='`random_state` is not np.random.RandomState'
                                 ': .*'):
            _ = DataStream(random_state=object())

    def test_to_arrays_stream(self):
        x = np.random.normal(size=[5, 4])
        y = np.random.normal(size=[5, 2, 3])
        stream = DataStream.arrays([x, y], batch_size=3)
        self.assertIsNone(stream.random_state)

        stream2 = stream.to_arrays_stream()
        self.assertIsNone(stream2.random_state)

        rs = np.random.RandomState()
        stream3 = stream.to_arrays_stream(random_state=rs)
        self.assertIs(stream3.random_state, rs)

    def test_select(self):
        # additional tests when source stream does not report array_count
        # and data_shapes

        def mapper(x, y, z):
            return x + y, y - z

        x = np.random.normal(size=[5, 3])
        y = np.random.normal(size=[5, 1])
        z = np.random.normal(size=[5, 3])

        source = DataStream.arrays([x, y, z], batch_size=3).map(mapper)
        self.assertIsNone(source.data_shapes)
        self.assertIsNone(source.array_count)

        stream = source.select([-1, 0, 1])
        self.assertEqual(stream.array_count, 3)
        self.assertIsNone(stream.data_shapes)

        a, b, c = stream.get_arrays()
        np.testing.assert_allclose(a, y - z)
        np.testing.assert_allclose(b, x + y)
        np.testing.assert_allclose(c, y - z)

        a, b, c = stream.get_arrays(1)
        np.testing.assert_allclose(a, (y - z)[:3])
        np.testing.assert_allclose(b, (x + y)[:3])
        np.testing.assert_allclose(c, (y - z)[:3])

        a, b, c = stream.get_arrays(0)
        np.testing.assert_allclose(a, (y - z)[:0])
        np.testing.assert_allclose(b, (x + y)[:0])
        np.testing.assert_allclose(c, (y - z)[:0])

        # index out of range error
        stream = source.select([0, 1, 2])
        with pytest.raises(IndexError, match='.* index out of range'):
            for _ in stream:
                pass


class ArraysDataStreamTestCase(unittest.TestCase):

    def test_stream(self):
        # test argument validation
        with pytest.raises(ValueError, match='`arrays` must not be empty.'):
            _ = ArraysDataStream([], batch_size=3, shuffle=False,
                                 skip_incomplete=False)

        with pytest.raises(ValueError, match='`arrays` must be arrays.'):
            _ = ArraysDataStream(
                [np.arange(5), object()],
                batch_size=3, shuffle=False, skip_incomplete=False
            )

        with pytest.raises(ValueError,
                           match='`arrays` must be at least 1-d arrays.'):
            _ = ArraysDataStream([np.asarray(0.)], batch_size=3, shuffle=False,
                                 skip_incomplete=False)

        with pytest.raises(ValueError,
                           match='`arrays` must have the same length.'):
            _ = ArraysDataStream(
                [np.arange(5), np.arange(6)],
                batch_size=3, shuffle=False, skip_incomplete=False,
            )

        # test non-shuffle stream
        x = np.random.normal(size=[5, 4])
        y = np.random.normal(size=[5, 2, 3])
        stream = ArraysDataStream([x, y], batch_size=3, shuffle=False,
                                  skip_incomplete=False)
        self.assertEqual(stream.data_shapes, ((4,), (2, 3)))
        self.assertEqual(stream.array_count, 2)
        self.assertEqual(stream.data_length, 5)
        self.assertIsNone(stream.random_state)
        self.assertEqual(stream.batch_size, 3)
        self.assertEqual(stream.batch_count, 2)
        self.assertFalse(stream.shuffle)
        self.assertFalse(stream.skip_incomplete)

        # test shuffle stream
        rs = np.random.RandomState(1234)
        stream = ArraysDataStream([x, y], batch_size=3, shuffle=True,
                                  skip_incomplete=True, random_state=rs)
        self.assertEqual(stream.data_shapes, ((4,), (2, 3)))
        self.assertEqual(stream.array_count, 2)
        self.assertEqual(stream.data_length, 3)
        self.assertIs(stream.random_state, rs)
        self.assertEqual(stream.batch_size, 3)
        self.assertEqual(stream.batch_count, 1)
        self.assertTrue(stream.shuffle)
        self.assertTrue(stream.skip_incomplete)

        rs0 = np.random.RandomState(1234)
        indices = np.arange(5, dtype=np.int32)
        rs0.shuffle(indices)
        arrays = stream.get_arrays()
        np.testing.assert_equal(arrays[0], x[indices[:3]])
        np.testing.assert_equal(arrays[1], y[indices[:3]])

        rs0.shuffle(indices)
        arrays = stream.get_arrays()
        np.testing.assert_equal(arrays[0], x[indices[:3]])
        np.testing.assert_equal(arrays[1], y[indices[:3]])

    def test_copy(self):
        x = np.random.normal(size=[5, 4])
        y = np.random.normal(size=[5, 2, 3])
        rs = np.random.RandomState(1234)
        stream = ArraysDataStream([x, y], batch_size=3, shuffle=False,
                                  skip_incomplete=False)

        stream2 = stream.copy(batch_size=4, shuffle=True, skip_incomplete=True,
                              random_state=rs)
        self.assertIsInstance(stream2, ArraysDataStream)
        self.assertEqual(stream2.batch_size, 4)
        self.assertTrue(stream2.shuffle)
        self.assertTrue(stream2.skip_incomplete)
        self.assertIs(stream2.random_state, rs)


class IntSeqDataStreamTestCase(unittest.TestCase):

    def test_stream(self):
        # test argument validation
        with pytest.raises(ValueError, match='`batch_size` is required.'):
            _ = IntSeqDataStream(5)

        # test non-shuffle stream
        stream = IntSeqDataStream(5, batch_size=3)
        self.assertEqual(stream.start, 0)
        self.assertEqual(stream.stop, 5)
        self.assertEqual(stream.step, 1)
        self.assertEqual(stream.dtype, np.int32)
        self.assertEqual(stream.batch_size, 3)
        self.assertEqual(stream.batch_count, 2)
        self.assertEqual(stream.array_count, 1)
        self.assertEqual(stream.data_shapes, ((),))
        self.assertEqual(stream.data_length, 5)
        self.assertFalse(stream.shuffle)
        self.assertFalse(stream.skip_incomplete)
        np.testing.assert_equal(stream.get_arrays()[0], np.arange(5))

        # test shuffle stream
        rs = np.random.RandomState(1234)
        stream = IntSeqDataStream(
            1, 10, 2, dtype=np.int64, batch_size=3, shuffle=True,
            skip_incomplete=True, random_state=rs
        )
        self.assertEqual(stream.start, 1)
        self.assertEqual(stream.stop, 10)
        self.assertEqual(stream.step, 2)
        self.assertEqual(stream.dtype, np.int64)
        self.assertEqual(stream.batch_size, 3)
        self.assertEqual(stream.batch_count, 1)
        self.assertEqual(stream.array_count, 1)
        self.assertEqual(stream.data_shapes, ((),))
        self.assertEqual(stream.data_length, 3)
        self.assertTrue(stream.shuffle)
        self.assertTrue(stream.skip_incomplete)

        rs0 = np.random.RandomState(1234)
        ans = np.arange(1, 10, 2, dtype=np.int64)
        rs0.shuffle(ans)
        ans = ans[:3]
        np.testing.assert_equal(stream.get_arrays()[0], ans)

    def test_copy(self):
        rs = np.random.RandomState(1234)
        stream = IntSeqDataStream(5, dtype=np.int32, batch_size=3)
        stream2 = stream.copy(dtype=np.int64, batch_size=4, shuffle=True,
                              skip_incomplete=True, random_state=rs)
        self.assertIsInstance(stream2, IntSeqDataStream)
        self.assertEqual(stream2.dtype, np.int64)
        self.assertEqual(stream2.batch_size, 4)
        self.assertTrue(stream2.shuffle)
        self.assertTrue(stream2.skip_incomplete)
        self.assertIs(stream2.random_state, rs)


class UserGeneratorDataStreamTestCase(unittest.TestCase):

    def test_validate_batch(self):
        # no constraint
        stream = UserGeneratorDataStream()
        x = np.arange(5)
        batch = stream._validate_batch(x)
        self.assertIsInstance(batch, tuple)
        self.assertEqual(len(batch), 1)
        self.assertIs(batch[0], x)

        # with constraint
        stream = UserGeneratorDataStream(
            batch_size=5,
            array_count=2,
            data_shapes=((), (3,))
        )

        with pytest.raises(ValueError,
                           match='batch size of the mapper output is not '
                                 'valid: expected <= 5, got 6'):
            stream._validate_batch([np.arange(6), x])

        with pytest.raises(ValueError,
                           match='batch size of the 1-th mapper output != '
                                 'the first output'):
            stream._validate_batch([x, np.arange(6)])

        with pytest.raises(ValueError,
                           match='user generator returned invalid number of '
                                 'arrays: expected 2, got 1'):
            stream._validate_batch(x)

        with pytest.raises(ValueError,
                           match=r'data shape of the 1-th mapper output is not '
                                 r'valid: expected \(3,\), got \(\)'):
            stream._validate_batch([x, np.arange(5)])

        y = np.random.normal(size=[5, 3])
        batch = stream._validate_batch([x, y])
        self.assertIsInstance(batch, tuple)
        self.assertIs(batch[0], x)
        self.assertIs(batch[1], y)


class GeneratorFactoryDataStreamTestCase(unittest.TestCase):

    def test_stream_and_copy(self):
        def g():
            for i in range(3):
                yield np.arange(i * 3, (i + 1) * 3, dtype=np.int32)

        stream = GeneratorFactoryDataStream(g)
        self.assertIs(stream.factory, g)

        for attr in ('batch_size', 'array_count', 'data_length', 'data_shapes',
                     'random_state'):
            self.assertIsNone(getattr(stream, attr))

        stream2 = stream.copy()
        self.assertIsNot(stream2, stream)
        self.assertIsInstance(stream2, GeneratorFactoryDataStream)


class GatherDataStreamTestCase(unittest.TestCase):

    def test_stream(self):
        # argument validation
        with pytest.raises(ValueError,
                           match='At least one data stream should be '
                                 'specified'):
            _ = GatherDataStream([])

        with pytest.raises(TypeError,
                           match='The 1-th element of `streams` is not '
                                 'an instance of DataStream: <object.*>'):
            _ = GatherDataStream([DataStream.int_seq(5, batch_size=3),
                                  object()])

        def my_generator():
            if False:
                yield

        with pytest.raises(ValueError,
                           match='Inconsistent batch size among the specified '
                                 'streams: encountered 4 at the 3-th stream, '
                                 'but has already encountered 3 before.'):
            _ = GatherDataStream([
                DataStream.generator(my_generator),
                DataStream.int_seq(5, batch_size=3),
                DataStream.generator(my_generator),
                DataStream.int_seq(5, batch_size=4),
            ])

        with pytest.raises(ValueError,
                           match='Inconsistent data length among the specified '
                                 'streams: encountered 6 at the 3-th stream, '
                                 'but has already encountered 5 before.'):
            _ = GatherDataStream([
                DataStream.generator(my_generator),
                DataStream.int_seq(5, batch_size=3),
                DataStream.generator(my_generator),
                DataStream.int_seq(6, batch_size=3),
            ])

        # test property inheritance
        rs0 = np.random.RandomState(1234)
        x = rs0.normal(size=[5, 1])
        y = rs0.normal(size=[5, 2])
        z = rs0.normal(size=[5, 3])

        rs = np.random.RandomState(1234)
        stream_x = DataStream.arrays([x], batch_size=3, random_state=rs)
        stream_yz = DataStream.arrays([y, z], batch_size=3)

        stream = GatherDataStream([stream_x, stream_yz])
        self.assertTupleEqual(stream.streams, (stream_x, stream_yz))
        self.assertEqual(stream.batch_size, 3)
        self.assertEqual(stream.data_shapes, ((1,), (2,), (3,)))
        self.assertEqual(stream.data_length, 5)
        self.assertEqual(stream.array_count, 3)
        self.assertEqual(stream.batch_count, 2)
        self.assertIs(stream.random_state, rs)

        arrays = stream.get_arrays()
        self.assertEqual(len(arrays), 3)
        np.testing.assert_equal(arrays[0], x)
        np.testing.assert_equal(arrays[1], y)
        np.testing.assert_equal(arrays[2], z)

        # test no property to inherit
        stream_1 = DataStream.generator(my_generator)
        stream_2 = DataStream.generator(my_generator)

        stream = GatherDataStream([stream_1, stream_2])
        for attr in ('batch_size', 'array_count', 'data_length', 'data_shapes',
                     'random_state'):
            self.assertIsNone(getattr(stream, attr))

        stream = GatherDataStream([stream_1, stream_2], random_state=rs)
        self.assertIs(stream.random_state, rs)

    def test_override_random_state(self):
        rs0 = np.random.RandomState(1234)
        x = rs0.normal(size=[5, 1])
        y = rs0.normal(size=[5, 2])
        z = rs0.normal(size=[5, 3])

        rs = np.random.RandomState(1234)
        stream_x = DataStream.arrays([x], batch_size=3, random_state=rs)
        stream_yz = DataStream.arrays([y, z], batch_size=3)

        # test overriding random state
        rs2 = np.random.RandomState(1234)
        stream = GatherDataStream([stream_x, stream_yz], random_state=rs2)
        self.assertIs(stream.random_state, rs2)

        # test copy with overrided random state
        rs3 = np.random.RandomState(1234)
        stream2 = stream.copy(random_state=rs3)
        self.assertIsInstance(stream2, GatherDataStream)
        self.assertIs(stream2.random_state, rs3)


class MapperDataStreamTestCase(unittest.TestCase):

    def test_stream(self):
        def identity(*args):
            return args

        # test argument validation
        with pytest.raises(TypeError,
                           match='`source` is not a DataStream: <object.*>'):
            _ = MapperDataStream(object(), identity)

        # test property inheritance
        np.random.seed(1234)
        rs = np.random.RandomState()
        x = np.random.normal(size=[5, 1])
        y = np.random.normal(size=[5, 2])
        source = DataStream.arrays([x, y], batch_size=3, random_state=rs)

        mapped = MapperDataStream(source, identity, preserve_shapes=False)
        self.assertIs(mapped.source, source)
        self.assertEqual(mapped.data_length, 5)
        self.assertEqual(mapped.batch_size, 3)
        self.assertEqual(mapped.batch_count, 2)
        self.assertIsNone(mapped.array_count)
        self.assertIsNone(mapped.data_shapes)
        self.assertIs(mapped.random_state, rs)

        mapped = MapperDataStream(source, identity, preserve_shapes=True)
        self.assertEqual(mapped.data_length, 5)
        self.assertEqual(mapped.batch_size, 3)
        self.assertEqual(mapped.batch_count, 2)
        self.assertEqual(mapped.array_count, 2)
        self.assertEqual(mapped.data_shapes, ((1,), (2,)))
        self.assertIs(mapped.random_state, rs)

        # test override
        rs2 = np.random.RandomState()
        mapped = MapperDataStream(
            source,
            identity,
            # in fact, these overrides are incorrect
            batch_size=7,
            array_count=1,
            data_shapes=((3,),),
            data_length=11,
            random_state=rs2,
        )
        self.assertEqual(mapped.data_length, 11)
        self.assertEqual(mapped.batch_size, 7)
        self.assertEqual(mapped.batch_count, 2)
        self.assertEqual(mapped.array_count, 1)
        self.assertEqual(mapped.data_shapes, ((3,),))
        self.assertIs(mapped.random_state, rs2)

    def test_copy(self):
        rs = np.random.RandomState()
        source = DataStream.int_seq(5, batch_size=3)

        mapped = MapperDataStream(source, lambda *args: args)
        mapped2 = mapped.copy(
            batch_size=7,
            array_count=1,
            data_shapes=((3,),),
            data_length=11,
            random_state=rs
        )
        self.assertEqual(mapped2.data_length, 11)
        self.assertEqual(mapped2.batch_size, 7)
        self.assertEqual(mapped2.array_count, 1)
        self.assertEqual(mapped2.data_shapes, ((3,),))
        self.assertIs(mapped2.random_state, rs)


class ThreadingDataStreamTestCase(unittest.TestCase):

    def test_stream(self):
        np.random.seed(1234)
        rs = np.random.RandomState()
        x = np.random.normal(size=[5, 1])
        y = np.random.normal(size=[5, 2])
        source = DataStream.arrays([x, y], batch_size=3, random_state=rs)

        # test argument validation
        with pytest.raises(TypeError,
                           match='`source` is not a DataStream: <object.*>'):
            _ = ThreadingDataStream(object(), prefetch=2)

        with pytest.raises(ValueError,
                           match='`prefetch` must be at least 1'):
            _ = ThreadingDataStream(source, prefetch=0)

        # test threaded with context
        stream = ThreadingDataStream(source, prefetch=3)
        self.assertIs(stream.source, source)
        self.assertEqual(stream.prefetch, 3)
        self.assertEqual(stream.batch_size, 3)
        self.assertEqual(stream.batch_count, 2)
        self.assertEqual(stream.array_count, 2)
        self.assertIs(stream.random_state, rs)
        self.assertEqual(stream.data_length, 5)
        self.assertEqual(stream.data_shapes, ((1,), (2,)))

        self.assertFalse(stream._worker_alive)
        self.assertFalse(stream._initialized)
        with stream:
            arrays = stream.get_arrays()
            self.assertEqual(len(arrays), 2)
            np.testing.assert_equal(arrays[0], x)
            np.testing.assert_equal(arrays[1], y)
        self.assertFalse(stream._worker_alive)
        self.assertFalse(stream._initialized)

        # test threaded without context
        stream = ThreadingDataStream(source, prefetch=3)
        self.assertFalse(stream._worker_alive)
        self.assertFalse(stream._initialized)
        arrays = stream.get_arrays()
        self.assertTrue(stream._worker_alive)
        self.assertTrue(stream._initialized)
        stream.close()
        self.assertFalse(stream._worker_alive)
        self.assertFalse(stream._initialized)
        stream.close()  # double close should not cause an error

        self.assertEqual(len(arrays), 2)
        np.testing.assert_equal(arrays[0], x)
        np.testing.assert_equal(arrays[1], y)

    @slow_test
    def test_iterator(self):
        class _MyError(Exception):
            pass

        epoch_counter = [0]
        external_counter = [1]

        seq_flow = DataStream.int_seq(0, 10, batch_size=2)
        map_flow = seq_flow.map(
            lambda x: (x + epoch_counter[0] * 10 + external_counter[0] * 100,))

        def make_iterator():
            epoch_counter[0] += 1
            return map_flow

        it_flow = DataStream.generator(make_iterator)
        with it_flow.threaded(prefetch=2) as flow:
            # the first epoch, expect 0 .. 10
            np.testing.assert_array_equal(
                [[110, 111], [112, 113], [114, 115], [116, 117], [118, 119]],
                [a[0] for a in flow]
            )
            time.sleep(.1)
            external_counter[0] += 1

            # the second epoch, the epoch counter should affect more than
            # the external counter
            np.testing.assert_array_equal(
                # having `prefetch = 2` should affect 3 items, because
                # while the queue size is 2, there are 1 additional prefetched
                # item waiting to be enqueued
                [[120, 121], [122, 123], [124, 125], [226, 227], [228, 229]],
                [a[0] for a in flow]
            )
            time.sleep(.1)
            external_counter[0] += 1

            # the third epoch, we shall carry out an incomplete epoch by break
            for a in flow:
                np.testing.assert_array_equal([230, 231], a[0])
                break
            time.sleep(.1)
            external_counter[0] += 1

            # verify that the epoch counter increases after break
            for i, (a,) in enumerate(flow):
                # because the interruption is not well-predictable under
                # multi-threading context, we shall have a weaker verification
                # than the above
                self.assertTrue((340 + i * 2 == a[0]) or (440 + i * 2 == a[0]))
                self.assertTrue((341 + i * 2 == a[1]) or (441 + i * 2 == a[1]))
            time.sleep(.1)
            external_counter[0] += 1

            # carry out the fourth, incomplete epoch by error
            try:
                for a in flow:
                    np.testing.assert_array_equal([450, 451], a[0])
                    raise _MyError()
            except _MyError:
                pass
            time.sleep(.1)
            external_counter[0] += 1

            # verify that the epoch counter increases after error
            for i, (a,) in enumerate(flow):
                self.assertTrue((560 + i * 2 == a[0]) or (660 + i * 2 == a[0]))
                self.assertTrue((561 + i * 2 == a[1]) or (661 + i * 2 == a[1]))

    def test_iter_reentrant_warn(self):
        stream = DataStream.int_seq(5, batch_size=3)

        # test open and close, no warning
        iterator = iter(stream)
        np.testing.assert_equal(next(iterator)[0], [0, 1, 2])
        iterator.close()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            batches = list(stream)
            self.assertEqual(len(batches), 2)
            np.testing.assert_equal(batches[0][0], [0, 1, 2])
            np.testing.assert_equal(batches[1][0], [3, 4])

        self.assertEqual(len(w), 0)

        # test open without close, cause warning
        iterator = iter(stream)
        np.testing.assert_equal(next(iterator)[0], [0, 1, 2])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            batches = list(stream)
            self.assertEqual(len(batches), 2)
            np.testing.assert_equal(batches[0][0], [0, 1, 2])
            np.testing.assert_equal(batches[1][0], [3, 4])

        self.assertEqual(len(w), 1)
        self.assertTrue(issubclass(w[-1].category, UserWarning))
        self.assertRegex(
            str(w[-1].message),
            r'Another iterator of the DataStream .* is still active, '
            r'will close it automatically.'
        )

        with pytest.raises(StopIteration):
            _ = next(iterator)  # this iterator should have been closed

        # test no warning the second time
        iterator = iter(stream)
        np.testing.assert_equal(next(iterator)[0], [0, 1, 2])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            batches = list(stream)
            self.assertEqual(len(batches), 2)
            np.testing.assert_equal(batches[0][0], [0, 1, 2])
            np.testing.assert_equal(batches[1][0], [3, 4])

        self.assertEqual(len(w), 0)

    def test_auto_init(self):
        epoch_counter = [0]

        seq_flow = DataStream.int_seq(0, 10, batch_size=2)
        map_flow = seq_flow.map(
            lambda x: (x + epoch_counter[0] * 10,))

        def make_iterator():
            epoch_counter[0] += 1
            return map_flow

        it_flow = DataStream.generator(make_iterator)
        flow = it_flow.threaded(3)

        batches = [b[0] for b in flow]
        np.testing.assert_array_equal(
            [[10, 11], [12, 13], [14, 15], [16, 17], [18, 19]], batches)

        batches = [b[0] for b in flow]
        np.testing.assert_array_equal(
            [[20, 21], [22, 23], [24, 25], [26, 27], [28, 29]], batches)

        flow.close()
        batches = [b[0] for b in flow]
        np.testing.assert_array_equal(
            [[40, 41], [42, 43], [44, 45], [46, 47], [48, 49]], batches)

        flow.close()

    def test_source_error(self):
        class _MyError(Exception):
            pass

        def g():
            for i in range(6):
                if i >= 1:
                    raise _MyError('err msg')
                yield np.arange(i * 3, (i + 1) * 3)

        source = DataStream.generator(g)
        threaded = source.threaded()
        self.assertEqual(threaded.prefetch, 5)

        def f():
            # get the first batch should be okay
            it = iter(threaded)
            batch = next(it)
            self.assertIsInstance(batch, tuple)
            self.assertEqual(len(batch), 1)
            np.testing.assert_equal(batch[0], np.arange(3))

            # get the second batch should raise an error
            with pytest.raises(_MyError, match='err msg'):
                _ = next(it)

        # enter the first loop
        f()

        # should be able to recover from the error in previous loop
        f()

    def test_copy(self):
        source = DataStream.int_seq(5, batch_size=3)
        stream = source.threaded(3)
        self.assertIs(stream.source, source)
        self.assertEqual(stream.prefetch, 3)

        stream2 = stream.copy(prefetch=1)
        self.assertIsInstance(stream2, ThreadingDataStream)
        self.assertIs(stream2.source, source)
        self.assertEqual(stream2.prefetch, 1)
