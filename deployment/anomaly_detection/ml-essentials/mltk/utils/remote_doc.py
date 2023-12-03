import copy
import time
from enum import Enum
from logging import getLogger
from threading import Thread, Condition, Semaphore
from typing import *

from ..typing_ import *
from .misc import deep_copy

__all__ = [
    'RemoteUpdateMode', 'RemoteDoc',
]


def merge_updates(target: DocumentType,
                  *sources: DocumentType,
                  keys_to_expand: Tuple[str, ...] = (),
                  copy_target: bool = False) -> DocumentType:
    """
    Merge update fields from `source` dicts into `target`.

    Args:
        target: The target update dict.
        \\*sources: The source update dicts.
        keys_to_expand: A list of field names, which are expected to be nested
            dict and should be expanded.
        copy_target: Whether or not to get a copy of the target before merging
            the values from the sources?  Defaults to :obj:`False`.

    Returns:
        The merged updates.
    """
    if copy_target:
        target = copy.copy(target)
    for source in sources:
        if source:
            for key, val in source.items():
                if key in keys_to_expand and isinstance(val, dict):
                    target.pop(key, None)
                    for val_key, val_val in val.items():
                        target[f'{key}.{val_key}'] = val_val
                else:
                    # TODO: do we need to raise error if `key in keys_to_expand`
                    #       but val is not a dict?
                    target[key] = val
    return target


def merge_updates_into_doc(doc: DocumentType,
                           *updates: DocumentType,
                           keys_to_expand: Tuple[str, ...] = (),
                           copy_doc: bool = False) -> DocumentType:
    """
    Merge update fields from `updates` into the document `doc`.

    Args:
        doc: The document dict.
        *updates: The source update dict.
        keys_to_expand: A list of field names, which are expected to be nested
            dict and should be expanded.
        copy_doc: Whether or not to get a copy of `doc` before merging
            the values from the sources?  Defaults to :obj:`False`.

    Returns:
        The merged document.
    """
    if copy_doc:
        doc = copy.copy(doc)
    expand_prefixes = tuple(f'{k}.' for k in keys_to_expand)

    for update in updates:
        if update:
            for key, val in update.items():
                if key in keys_to_expand and isinstance(val, dict):
                    if key not in doc:
                        doc[key] = {}
                    doc[key].update(val)
                else:
                    for pfx in expand_prefixes:
                        if key.startswith(pfx):
                            left_key = key[:len(pfx) - 1]
                            right_key = key[len(pfx):]
                            if left_key not in doc:
                                doc[left_key] = {}
                            doc[left_key][right_key] = val
                            break
                    else:
                        doc[key] = val
    return doc


class RemoteUpdateMode(int, Enum):
    """
    Update mode.  Larger number indicates more frequent attempts to push
    updates to the remote.
    """

    STOPPED = 9999  # background thread exited
    NONE = 0  # no pending update
    RETRY = 1  # retry previous failed update
    RELAXED = 2  # update later
    IMMEDIATELY = 3  # update immediately


class RemoteDoc(Mapping[str, Any]):
    """
    Class that pushes update of a document to remote via a background thread.
    """

    def __init__(self,
                 retry_interval: float,
                 relaxed_interval: float,
                 heartbeat_interval: Optional[float] = None,
                 keys_to_expand: Sequence[str] = (),
                 local_value: Optional[DocumentType] = None):
        from ..events import EventHost, Event

        # parameters
        self.retry_interval: float = retry_interval
        self.relaxed_interval: float = relaxed_interval
        self.heartbeat_interval: Optional[float] = heartbeat_interval
        self.keys_to_expand: Tuple[str, ...] = tuple(keys_to_expand)

        # events
        self.events: EventHost = EventHost()
        # (updates: Dict[str, Any]) -> None, triggered before calling `push_to_remote`
        self.on_before_push: Event = self.events['on_before_push']
        # (updates: Dict[str, Any], remote_doc: Dict[str, Any]) -> None, trigger after calling `push_to_remote`
        self.on_after_push: Event = self.events['on_after_push']

        # the local value is maintained according to :meth:`update()` and the
        # response of `_push_to_remote()`.
        self.local_value: Dict[str, Any] = deep_copy(local_value or {})

        # the pending updates
        self._updates: Optional[DocumentType] = {}
        self._update_mode: RemoteUpdateMode = RemoteUpdateMode.NONE
        self._last_push_time: float = 0.

        # state of the background worker
        self._thread: Optional[Thread] = None
        self._cond = Condition()
        self._start_sem = Semaphore(0)
        self._ref_count = 0  # number of times `start_worker` has been called

    def __iter__(self):
        return iter(self.local_value)

    def __contains__(self, item):
        return item in self.local_value

    def __getitem__(self, item):
        return self.local_value[item]

    def __len__(self) -> int:
        return len(self.local_value)

    @property
    def heartbeat_enabled(self) -> bool:
        """
        Whether or not heartbeat is enabled?

        Heartbeat is enabled if and only if `heartbeat_interval` is not None.
        """
        return self.heartbeat_interval is not None

    def _push_to_remote(self, updates: DocumentType) -> Optional[DocumentType]:
        """
        Push pending updates to the remote, and return the updated document
        at the remote.  Subclasses should implement this method.

        Args:
            updates: The updates, flatten according to `keys_to_expand`.

        Returns:
            The updated whole document.
        """
        raise NotImplementedError()

    def push_to_remote(self, updates: DocumentType) -> Optional[DocumentType]:
        """
        Push pending updates to the remote, and return the updated document
        at the remote.

        Args:
            updates: The updates to be pushed to the remote.

        Returns:
            The updated whole document.
        """
        self.on_before_push.fire(updates)
        remote_doc = self._push_to_remote(updates)
        self.on_after_push.fire(updates, remote_doc)
        return remote_doc

    def update(self, fields: DocumentType, immediately: bool = False):
        """
        Queue updates of the remote document into the background worker.

        Args:
            fields: The document updates.
            immediately: Whether or not to let the background worker
                push updates to the remote immediately?  Defaults to
                :obj:`False`, i.e., the updates will be pushed to remote
                later at a proper time.
        """
        with self._cond:
            # set pending updates
            self._updates = merge_updates(
                self._updates, fields,
                keys_to_expand=self.keys_to_expand)
            self.local_value = merge_updates_into_doc(
                self.local_value, self._updates,
                keys_to_expand=self.keys_to_expand)

            # set the update mode
            if immediately:
                self._update_mode = RemoteUpdateMode.IMMEDIATELY
            else:
                self._update_mode = RemoteUpdateMode.RELAXED

            # notify the background thread about the new updates
            self._cond.notify_all()

    def flush(self):
        """Push pending updates to the remote in the foreground thread."""
        with self._cond:
            updates = copy.copy(self._updates)
            self._updates.clear()
            if self._update_mode != RemoteUpdateMode.STOPPED:
                self._update_mode = RemoteUpdateMode.NONE
            if updates:
                try:
                    remote_updated_doc = self.push_to_remote(updates)
                except:
                    self._merge_back(updates, RemoteUpdateMode.RETRY)
                    raise
                else:
                    # here no need to merge `updates` into `remote_updated_doc`,
                    # since we've obtained the lock, thus the updates must
                    # never be written during this period.
                    self.local_value = remote_updated_doc

    def _merge_back(self, updates: Optional[DocumentType],
                    mode: RemoteUpdateMode):
        """
        Merge back unsuccessful updates.

        The caller must obtain ``self._cond`` lock before calling this method.

        Args:
            updates: The updates that was not pushed to remote successfully.
            mode: The minimum remote update mode after merged.
        """
        if updates:
            for k, v in updates.items():
                if k not in self._updates:
                    self._updates[k] = v
        if self._update_mode < mode:
            self._update_mode = mode

    def _thread_func(self):
        self._last_push_time = 0.
        self._update_mode = RemoteUpdateMode.NONE

        # notify the main thread that this worker has started
        self._start_sem.release()

        # the main worker loop
        while True:
            # check the update mode and pending updates
            with self._cond:
                mode = self._update_mode
                last_push_time = self._last_push_time
                if mode == RemoteUpdateMode.STOPPED:
                    break

                now_time = time.time()
                elapsed = now_time - last_push_time
                target_itv = {
                    RemoteUpdateMode.IMMEDIATELY: 0,
                    RemoteUpdateMode.RELAXED: self.relaxed_interval,
                    RemoteUpdateMode.RETRY: self.retry_interval,
                    RemoteUpdateMode.NONE: self.heartbeat_interval
                }[mode]

                # if target sleep interval has been reached, we plan to
                # push the updates.  otherwise we plan to do nothing.
                if target_itv is not None:
                    if elapsed >= target_itv:
                        # move the pending updates from the shared status to
                        # the private zone of this thread worker
                        updates = copy.copy(self._updates)
                        self._updates.clear()
                        self._update_mode = RemoteUpdateMode.NONE
                    else:
                        # no plan to execute now, sleep for a bit more
                        self._cond.wait(target_itv - elapsed)
                        continue  # go into next loop to check status again
                else:
                    # no plan to execute, and no sleep interval can be inferred,
                    # wait for an indefinite amount of time
                    self._cond.wait()
                    continue  # go into next loop to check status again

            # now, since the plan has been set, we are about ot execute the plan
            merge_back = False
            remote_updated_doc = None

            if updates is not None:
                try:
                    remote_updated_doc = self.push_to_remote(updates)
                except Exception:
                    getLogger(__name__).warning(
                        'Failed to push updates to remote.', exc_info=True)
                    merge_back = True
                finally:
                    last_push_time = time.time()

            # write back to the shared status
            with self._cond:
                # merge back updates
                if merge_back:
                    self._merge_back(updates, RemoteUpdateMode.RETRY)
                else:
                    self._merge_back(None, RemoteUpdateMode.NONE)

                # merge remote doc to local value
                if remote_updated_doc:
                    self.local_value = merge_updates_into_doc(
                        # note we use `remote_updated_doc` to replace any old
                        # `local_value`, thus here we let `doc = remote_updated_doc`
                        remote_updated_doc,
                        self._updates,  # also merge the updates into the doc
                        keys_to_expand=self.keys_to_expand)

                # set the last push time
                self._last_push_time = last_push_time

        # finally, de-reference the thread object
        self._thread = None

    @property
    def running(self) -> bool:
        """Check whether or not the background worker is running."""
        return self._thread is not None

    def start_worker(self):
        """Start the background worker."""
        if self._thread is None:
            self._thread = Thread(target=self._thread_func, daemon=True)
            self._thread.start()
            self._start_sem.acquire()
        self._ref_count += 1

    def stop_worker(self):
        """
        Stop the background worker.

        This method will try to call ``flush()``, but if it fails, this method
        will not raise error.  Instead, the updates that were failed to push
        will be kept in this :class:`RemoteDoc` instance, and its update mode
        will be set to ``RemoteUpdateMode.RETRY``.
        """
        if self._thread is not None:
            self._ref_count -= 1
            if self._ref_count <= 0:
                thread = self._thread
                with self._cond:
                    self._update_mode = RemoteUpdateMode.STOPPED
                    self._cond.notify_all()
                thread.join()
                self._ref_count = 0

                try:
                    self.flush()
                except Exception:
                    getLogger(__name__).warning(
                        'Failed to push remaining update.',
                        exc_info=True
                    )

    def __enter__(self):
        self.start_worker()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_worker()
