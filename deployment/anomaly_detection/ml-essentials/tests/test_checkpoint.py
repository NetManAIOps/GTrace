import os
import time
import unittest
from datetime import datetime
from tempfile import TemporaryDirectory

import mock
import pytest

from mltk import BaseCheckpoint, SimpleStatefulObject, CheckpointManager, StatefulObjectGroup


def file_get_contents(path) -> bytes:
    with open(path, 'rb') as f:
        return f.read()


class _MyCheckpoint(BaseCheckpoint):

    ckpt_is_dir: bool = True
    save_raise_error: bool = False
    restore_raise_error: bool = False

    def __init__(self):
        self.logs = []

    def _save(self, checkpoint_path: str) -> None:
        if os.path.exists(checkpoint_path):
            raise FileExistsError(checkpoint_path)
        if self.ckpt_is_dir:
            os.makedirs(checkpoint_path, exist_ok=False)
            with open(os.path.join(checkpoint_path, 'data.txt'), 'wb') as f:
                f.write(b'directory')
        else:
            with open(checkpoint_path, 'wb') as f:
                f.write(b'file')
        self.logs.append(('save', checkpoint_path))
        if self.save_raise_error:
            raise RuntimeError('save_raise_error')

    def _restore(self, checkpoint_path: str) -> None:
        if self.ckpt_is_dir:
            if not os.path.isdir(checkpoint_path):
                raise NotADirectoryError(checkpoint_path)
        else:
            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(checkpoint_path)
        self.logs.append(('restore', checkpoint_path))
        if self.restore_raise_error:
            raise RuntimeError('restore_raise_error')


class BaseCheckpointTestCase(unittest.TestCase):

    def test_no_state_object(self):
        ckpt = _MyCheckpoint()

        with TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, 'a')
            sub_path = os.path.join(path, 'ckpt')

            # cannot recover from non-exist checkpoint
            with pytest.raises(IOError,
                               match=f'Checkpoint does not exist: {sub_path}'):
                ckpt.restore(path)

            # refuse to overwrite a file
            with open(path, 'wb') as f:
                f.write(b'')
            with pytest.raises(IOError,
                               match='`checkpoint_dir` already exists'):
                ckpt.save(path)
            with pytest.raises(IOError,
                               match='Checkpoint does not exist'):
                ckpt.restore(path)

            # save & restore without state objects
            ckpt.save(path, overwrite=True)
            ckpt.restore(path)
            self.assertEqual(ckpt.logs, [
                ('save', sub_path),
                ('restore', sub_path),
            ])
            self.assertEqual(
                file_get_contents(os.path.join(sub_path, 'data.txt')),
                b'directory'
            )
            ckpt.logs.clear()

            # refuse to overwrite a directory
            with pytest.raises(IOError,
                               match='`checkpoint_dir` already exists'):
                ckpt.save(path)

            # save & restore without state object, as a file
            ckpt.ckpt_is_dir = False
            ckpt.save(path, overwrite=True)
            ckpt.restore(path)
            self.assertEqual(ckpt.logs, [
                ('save', sub_path),
                ('restore', sub_path),
            ])
            self.assertEqual(file_get_contents(sub_path), b'file')
            ckpt.logs.clear()

    def test_with_state_object(self):

        with TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, 'a')
            state_path = os.path.join(path, 'state.npz')

            # save
            a = SimpleStatefulObject()
            b = SimpleStatefulObject()
            state_objects = {'a': a, 'b': b}
            ckpt = _MyCheckpoint()

            a.key1 = 123
            a.key2 = '456'
            b.key3 = 789.0

            ckpt.save(path, state_objects)
            self.assertTrue(os.path.isfile(state_path))

            # restore
            a = SimpleStatefulObject()
            b = SimpleStatefulObject()
            state_objects = {'a': a, 'b': b}

            ckpt.restore(path, StatefulObjectGroup(state_objects))
            self.assertEqual(a.key1, 123)
            self.assertEqual(a.key2, '456')
            self.assertEqual(b.key3, 789.0)

            # test error during restoration
            ckpt.restore_raise_error = True
            a.key1 = 1230
            a.key2 = '4560'
            b.key3 = 7890.0

            with pytest.raises(RuntimeError, match='restore_raise_error'):
                ckpt.restore(path, state_objects)

            self.assertEqual(a.key1, 1230)
            self.assertEqual(a.key2, '4560')
            self.assertEqual(b.key3, 7890.0)

            # test state file not exist
            ckpt.restore_raise_error = False
            os.remove(state_path)
            with pytest.raises(IOError,
                               match=f'State file does not exist: {state_path}'):
                ckpt.restore(path, state_objects)


class CheckpointManagerTestCase(unittest.TestCase):

    def test_construct(self):
        with TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, 'ckpt')
            state_objects = SimpleStatefulObject()
            ckpt = _MyCheckpoint()

            # most default
            mgr = CheckpointManager(
                checkpoint=ckpt,
                root_dir=path,
            )
            self.assertEqual(mgr.checkpoint, ckpt)
            self.assertEqual(mgr.state_objects, None)
            self.assertEqual(mgr.root_dir, path)
            self.assertEqual(mgr.checkpoint_index_file, 'checkpoint.json')
            self.assertEqual(mgr.max_to_keep, None)

            # all specified
            mgr = CheckpointManager(
                checkpoint=ckpt,
                root_dir=path,
                state_objects=state_objects,
                checkpoint_index_file='ckpt.json',
                max_to_keep=5
            )
            self.assertEqual(mgr.checkpoint, ckpt)
            self.assertEqual(mgr.state_objects, state_objects)
            self.assertEqual(mgr.root_dir, path)
            self.assertEqual(mgr.checkpoint_index_file, 'ckpt.json')
            self.assertEqual(mgr.max_to_keep, 5)

            # error argument
            with pytest.raises(ValueError,
                               match='`max_to_keep` must >= 1: got 0'):
                _ = CheckpointManager(ckpt, path, max_to_keep=0)

    def test_save_restore(self):
        a = SimpleStatefulObject()
        b = SimpleStatefulObject()
        state_objects = {'a': a, 'b': b}
        ckpt = _MyCheckpoint()

        with TemporaryDirectory() as temp_dir:
            root_dir = os.path.join(temp_dir, 'ckpt')
            checkpoints = []

            #######################################
            # the first checkpoint manager object #
            #######################################
            mgr = CheckpointManager(
                checkpoint=ckpt,
                root_dir=root_dir,
                state_objects=state_objects,
            )
            self.assertIsNone(mgr.latest_checkpoint())
            mgr.restore_latest()  # no error should be raised

            with pytest.raises(IOError, match='No checkpoint can be restored'):
                mgr.restore_latest(True)

            # save new checkpoints, with auto name
            base_time = int(time.time()) + 0.01
            base_name = datetime.utcfromtimestamp(base_time). \
                strftime('%Y-%m-%d %H-%M-%S')

            class FakeDateTime(object):

                _counter = 0

                @classmethod
                def now(cls):
                    cls._counter += 1
                    return datetime.utcfromtimestamp(
                        base_time + 0.01 * int(cls._counter // 4))

            for i in range(3):
                with mock.patch('mltk.checkpoint.datetime', FakeDateTime):
                    a.value = 100 + i
                    b.value = 200 + i
                    checkpoints.append(mgr.save())
                    desired_path = os.path.join(root_dir, base_name)
                    desired_path += f'.0{i + 1}0000'
                    self.assertEqual(checkpoints[-1], desired_path)
                    self.assertEqual(mgr.latest_checkpoint(), checkpoints[-1])

            # save new checkpoints, with specified name
            base_name = 'my_checkpoint'

            a.value = 103
            b.value = 203
            checkpoints.append(mgr.save(base_name + '_xxx'))
            desired_path = os.path.join(root_dir, base_name + '_xxx')
            self.assertEqual(checkpoints[-1], desired_path)
            self.assertEqual(mgr.latest_checkpoint(), checkpoints[-1])

            for i in range(3):
                a.value = 104 + i
                b.value = 204 + i
                checkpoints.append(mgr.save(base_name))
                desired_path = os.path.join(root_dir, base_name)
                if i > 0:
                    desired_path += f'_{i}'
                self.assertEqual(checkpoints[-1], desired_path)
                self.assertEqual(mgr.latest_checkpoint(), checkpoints[-1])

            self.assertEqual(mgr.checkpoint_list(), checkpoints)
            self.assertEqual(ckpt.logs, [
                ('save', os.path.join(p, 'ckpt'))
                for p in checkpoints
            ])
            ckpt.logs.clear()

            ########################################
            # the second checkpoint manager object #
            ########################################
            mgr = CheckpointManager(
                checkpoint=ckpt,
                root_dir=root_dir,
                state_objects=state_objects,
            )
            self.assertEqual(mgr.latest_checkpoint(), checkpoints[-1])
            self.assertEqual(mgr.checkpoint_list(), checkpoints)

            # restore latest
            a.value = b.value = None
            mgr.restore_latest()
            self.assertEqual(a.value, 106)
            self.assertEqual(b.value, 206)
            self.assertEqual(ckpt.logs, [
                ('restore', os.path.join(checkpoints[-1], 'ckpt'))
            ])
            ckpt.logs.clear()

            # restore from each checkpoint
            for i, ckpt_path in enumerate(checkpoints):
                mgr.restore(ckpt_path)
            self.assertEqual(a.value, 100 + i)
            self.assertEqual(b.value, 200 + i)
            self.assertEqual(ckpt.logs, [
                ('restore', os.path.join(p, 'ckpt'))
                for p in checkpoints
            ])
            ckpt.logs.clear()

    def test_max_to_keep(self):
        a = SimpleStatefulObject()
        b = SimpleStatefulObject()
        state_objects = {'a': a, 'b': b}
        ckpt = _MyCheckpoint()

        with TemporaryDirectory() as temp_dir:
            root_dir = os.path.join(temp_dir, 'ckpt')
            checkpoints = []

            #######################################
            # the first checkpoint manager object #
            #######################################
            mgr = CheckpointManager(
                checkpoint=ckpt,
                root_dir=root_dir,
                state_objects=state_objects,
                max_to_keep=3,
            )
            base_name = 'my_checkpoint'
            for i in range(7):
                a.value = 100 + i
                b.value = 200 + i
                checkpoints.append(mgr.save(base_name))
                desired_path = os.path.join(root_dir, base_name)
                if i > 0:
                    desired_path += f'_{i}'
                self.assertEqual(checkpoints[-1], desired_path)
                self.assertEqual(mgr.latest_checkpoint(), checkpoints[-1])
                self.assertTrue(os.path.isdir(desired_path))

            self.assertEqual(mgr.checkpoint_list(), checkpoints[-3:])
            self.assertEqual(ckpt.logs, [
                ('save', os.path.join(p, 'ckpt'))
                for p in checkpoints
            ])
            ckpt.logs.clear()

            for p in checkpoints[:-3]:
                self.assertFalse(os.path.isdir(p))
            for p in checkpoints[-3:]:
                self.assertTrue(os.path.isdir(p))

            ########################################
            # the second checkpoint manager object #
            ########################################
            mgr = CheckpointManager(
                checkpoint=ckpt,
                root_dir=root_dir,
                state_objects=state_objects,
                max_to_keep=2,
            )
            self.assertEqual(mgr.checkpoint_list(), checkpoints[-3:])
            self.assertEqual(mgr.latest_checkpoint(), checkpoints[-1])

            # restore latest
            a.value = b.value = None
            mgr.restore_latest()
            self.assertEqual(a.value, 106)
            self.assertEqual(b.value, 206)
            self.assertEqual(ckpt.logs, [
                ('restore', os.path.join(checkpoints[-1], 'ckpt'))
            ])
            ckpt.logs.clear()

            # save new, this time it should only keep 2 checkpoints
            a.value = 107
            b.value = 207
            checkpoints.append(mgr.save(base_name))
            desired_path = os.path.join(root_dir, base_name + '_7')
            self.assertEqual(checkpoints[-1], desired_path)
            self.assertEqual(ckpt.logs, [
                ('save', os.path.join(checkpoints[-1], 'ckpt'))
            ])
            self.assertEqual(mgr.latest_checkpoint(), checkpoints[-1])
            self.assertEqual(mgr.checkpoint_list(), checkpoints[-2:])
            self.assertFalse(os.path.isdir(checkpoints[-3]))
            ckpt.logs.clear()

            # now save new but raise error
            ckpt.save_raise_error = True
            a.value = 108
            b.value = 208
            with pytest.raises(RuntimeError, match='save_raise_error'):
                mgr.save(base_name)
            desired_path = os.path.join(root_dir, base_name + '_8')
            self.assertEqual(ckpt.logs, [
                ('save', os.path.join(desired_path, 'ckpt'))
            ])
            self.assertEqual(mgr.latest_checkpoint(), checkpoints[-1])
            self.assertEqual(mgr.checkpoint_list(), checkpoints[-2:])
            self.assertFalse(os.path.isdir(desired_path))
            self.assertTrue(os.path.isdir(checkpoints[-1]))
            ckpt.logs.clear()
