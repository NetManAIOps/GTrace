import codecs
import json
import os
import shutil
import time
from datetime import datetime
from logging import getLogger
from typing import *

from .stateful import StatefulObject, StatefulObjectGroup, StateSaver

__all__ = ['Checkpointable', 'BaseCheckpoint', 'CheckpointManager']

Checkpointable = Union[StatefulObject, Dict[str, StatefulObject]]
"""
Type of objects that can be saved and restored as `state_objects` via
:meth:`save()` and :meth:`restore()` of :class:`BaseCheckpoint`. 
"""


class BaseCheckpoint(object):
    """
    Base interface of a checkpoint object.

    Any attribute attached to a checkpoint object should be saved via
    :meth:`save()`, and restored via :meth:`restore()`.

    The true checkpoint classes for specific backends should be implemented
    in the modules under the package ``mltk.integration``.
    """

    def _save(self, checkpoint_path: str) -> None:
        raise NotImplementedError()

    def save(self,
             checkpoint_dir: str,
             state_objects: Optional[Checkpointable] = None,
             overwrite: bool = False) -> None:
        """
        Save checkpoint to `checkpoint_dir`.

        Args:
            checkpoint_dir: The directory where to save the checkpoint.
            state_objects: Additional stateful object(s) to be saved,
                alongside the backend checkpoint file.
            overwrite: Whether or not to overwrite exist checkpoint?
        """
        checkpoint_dir = os.path.abspath(checkpoint_dir)
        if state_objects is not None and \
                not isinstance(state_objects, StatefulObject):
            state_objects = StatefulObjectGroup(state_objects)

        # check whether or not we shall overwrite existing file/directory
        if os.path.exists(checkpoint_dir):
            if not overwrite:
                raise IOError(f'`checkpoint_dir` already exists: '
                              f'{checkpoint_dir}')
            if os.path.isdir(checkpoint_dir):
                shutil.rmtree(checkpoint_dir)
            else:
                os.remove(checkpoint_dir)

        # now save the checkpoint and state objects
        os.makedirs(checkpoint_dir, exist_ok=True)
        state_path = os.path.join(checkpoint_dir, 'state.npz')
        ckpt_path = os.path.join(checkpoint_dir, 'ckpt')

        if state_objects is not None:
            StateSaver(state_objects).save(state_path)
        self._save(ckpt_path)

    def _restore(self, checkpoint_path: str) -> None:
        raise NotImplementedError()

    def restore(self,
                checkpoint_dir: str,
                state_objects: Optional[Checkpointable] = None) -> None:
        """
        Restore checkpoint from `checkpoint_dir`.

        Args:
            checkpoint_dir: The directory where the checkpoint was saved.
            state_objects: Additional stateful objects to be restored,
                alongside the backend checkpoint file.
        """
        # backup the original object state
        checkpoint_dir = os.path.abspath(checkpoint_dir)
        if state_objects is not None and \
                not isinstance(state_objects, StatefulObject):
            state_objects = StatefulObjectGroup(state_objects)

        # check whether the checkpoint exists
        state_path = os.path.join(checkpoint_dir, 'state.npz')
        ckpt_path = os.path.join(checkpoint_dir, 'ckpt')

        if not os.path.exists(ckpt_path):
            raise IOError(f'Checkpoint does not exist: {ckpt_path}')
        if state_objects is not None and not os.path.isfile(state_path):
            raise IOError(f'State file does not exist: {state_path}')

        # load the state object and checkpoint
        original_state = state_objects.get_state_dict() \
            if state_objects is not None else None

        try:
            if state_objects is not None:
                StateSaver(state_objects).load(state_path)
            self._restore(ckpt_path)
        except:
            if state_objects is not None:
                state_objects.set_state_dict(original_state)
            raise


class CheckpointList(Sequence[str]):
    """
    A sequence of checkpoint paths, with O(1) time cost to verify whether
    or not an element is in the list.

    >>> ckpt_list = CheckpointList(['a', 'b', 'a'])
    >>> list(ckpt_list)
    ['a', 'b', 'a']
    >>> 'a' in ckpt_list
    True
    >>> 'b' in ckpt_list
    True
    >>> 'c' in ckpt_list
    False

    >>> ckpt_list.pop_front()
    'a'
    >>> list(ckpt_list)
    ['b', 'a']
    >>> 'a' in ckpt_list
    True
    >>> 'b' in ckpt_list
    True

    >>> ckpt_list.pop_front()
    'b'
    >>> list(ckpt_list)
    ['a']
    >>> 'a' in ckpt_list
    True
    >>> 'b' in ckpt_list
    False

    >>> ckpt_list.pop_front()
    'a'
    >>> list(ckpt_list)
    []
    >>> 'a' in ckpt_list
    False
    """

    _list: List[str]
    _count: Dict[str, int]

    def __init__(self, checkpoints: Sequence[str] = ()):
        self._list = []
        self._count = {}
        for item in checkpoints:
            self.push_back(item)

    def push_back(self, item) -> None:
        self._list.append(item)
        if item not in self._count:
            self._count[item] = 1
        else:
            self._count[item] += 1

    def pop_front(self) -> str:
        item = self._list.pop(0)
        self._count[item] -= 1
        if self._count[item] <= 0:
            self._count.pop(item)
        return item

    def __len__(self) -> int:
        return len(self._list)

    def __bool__(self) -> bool:
        return bool(self._list)

    def __iter__(self) -> Iterator[str]:
        return iter(self._list)

    def __contains__(self, item) -> bool:
        return item in self._count

    def __getitem__(self, item) -> str:
        return self._list[item]


class CheckpointManager(object):
    """
    Save and restore checkpoint and state objects, with version control.
    """

    checkpoint: BaseCheckpoint
    """The checkpoint object to be saved and restored."""

    state_objects: Optional[Checkpointable]
    """The state objects to be saved and restored."""

    root_dir: str
    """The root directory of the checkpoints."""

    checkpoint_index_file: str
    """The index file of the checkpoints."""

    max_to_keep: Optional[int]
    """The maximum number of checkpoints to keep."""

    _checkpoint_list: CheckpointList
    """The list of directory names, of the checkpoints."""

    def __init__(self,
                 checkpoint: BaseCheckpoint,
                 root_dir: str,
                 state_objects: Optional[Checkpointable] = None,
                 checkpoint_index_file: str = 'checkpoint.json',
                 max_to_keep: Optional[int] = None):
        """
        Construct a new :class:`CheckpointManager`.

        Args:
            checkpoint: The checkpoint object.
            root_dir: The root directory, where to store the checkpoints.
            state_objects: The stateful objects.
            checkpoint_index_file: The checkpoint index file, will be
                at ``os.path.join(root_dir, checkpoint_index_file)``.
            max_to_keep: Maximum checkpoints to keep.  Old checkpoints
                will be removed automatically.
        """
        if max_to_keep is not None and max_to_keep < 1:
            raise ValueError(f'`max_to_keep` must >= 1: got {max_to_keep!r}')

        root_dir = os.path.abspath(root_dir)
        self.checkpoint = checkpoint
        self.state_objects = state_objects
        self.root_dir = root_dir
        self.checkpoint_index_file = checkpoint_index_file
        self.max_to_keep = max_to_keep

        # load the checkpoint index file
        index_path = os.path.join(root_dir, checkpoint_index_file)
        if os.path.isfile(index_path):
            with codecs.open(index_path, 'rb', 'utf-8') as f:
                cnt = f.read()
            index_content = json.loads(cnt)
        else:
            index_content = {}
        self._checkpoint_list = CheckpointList(
            index_content.get('checkpoint_list', []))

    def _save_index_file(self):
        index_path = os.path.join(self.root_dir, self.checkpoint_index_file)
        cnt = json.dumps({
            'checkpoint_list': list(self._checkpoint_list),
        })
        with codecs.open(index_path, 'wb', 'utf-8') as f:
            f.write(cnt)

    def checkpoint_list(self) -> List[str]:
        """
        Get the list of checkpoint paths.

        Returns:
            The list of checkpoint paths.
        """
        return [os.path.join(self.root_dir, p) for p in self._checkpoint_list]

    def latest_checkpoint(self) -> Optional[str]:
        """
        Get the path of the latest checkpoint.

        Returns:
            Checkpoint path, or :obj:`None` if no checkpoint has been saved.
        """
        if self._checkpoint_list:
            return os.path.join(self.root_dir, self._checkpoint_list[-1])

    def save(self, name: Optional[str] = None) -> str:
        """
        Save a new checkpoint.

        Args:
            name: Base name of the checkpoint.  Will be deduplicated.
                If not specified, use the current date time as name.

        Returns:
            The checkpoint path, which can be restored via :meth:`restore()`.
        """
        # get a unique checkpoint name
        if name is None:
            name = datetime.now().strftime('%Y-%m-%d %H-%M-%S.%f')
            while name in self._checkpoint_list:
                time.sleep(0.01)
                name = datetime.now().strftime('%Y-%m-%d %H-%M-%S.%f')
        else:
            max_idx = -1
            pfx = f'{name}_'
            for ckpt_name in self._checkpoint_list:
                if ckpt_name == name:
                    max_idx = max(max_idx, 0)
                elif ckpt_name.startswith(pfx):
                    ckpt_idx = ckpt_name[len(pfx):]
                    try:
                        max_idx = max(max_idx, int(ckpt_idx))
                    except ValueError:
                        pass
            if max_idx > -1:
                name = f'{name}_{max_idx + 1}'

        # now save the checkpoint and index file
        path = os.path.join(self.root_dir, name)
        names_to_purge = []

        try:
            # save checkpoint and update index file
            self.checkpoint.save(path, self.state_objects, overwrite=True)
            self._latest_checkpoint = name
            self._checkpoint_list.push_back(name)

            # purge old checkpoint if `max_to_keep` is configured
            if self.max_to_keep is not None:
                while len(self._checkpoint_list) > self.max_to_keep:
                    names_to_purge.append(self._checkpoint_list.pop_front())

            # save the new index file
            self._save_index_file()
        except:
            shutil.rmtree(path)
            raise

        # checkpoint saved, purge old checkpoint
        if names_to_purge is not None:
            for old_name in names_to_purge:
                if old_name in self._checkpoint_list:  # pragma: no cover
                    continue  # should generally not happen
                old_path = os.path.join(self.root_dir, old_name)
                try:
                    if os.path.exists(old_path):
                        shutil.rmtree(old_path)
                except Exception:  # pragma: no cover
                    getLogger(__name__).warning(
                        'Failed to purge old checkpoint: %s', old_path)

        return path

    def restore(self, path: str) -> None:
        """
        Restore from a checkpoint.

        Args:
            path: Path of the checkpoint.
        """
        self.checkpoint.restore(path, self.state_objects)

    def restore_latest(self, raise_not_exist: bool = False) -> None:
        """
        Restore from the latest checkpoint.

        Args:
            raise_not_exist: Whether to raise an :class:`IOError` if
                the latest checkpoint does not exist?  Defaults to False.
        """
        latest_checkpoint = self.latest_checkpoint()
        if latest_checkpoint is not None:
            self.restore(latest_checkpoint)
        elif raise_not_exist:
            raise IOError('No checkpoint can be restored.')
