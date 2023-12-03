import os
import re
import time
from datetime import datetime
from logging import getLogger
from typing import *

import pytz
import requests
from bson import ObjectId
from cachetools import LRUCache
from urllib.parse import quote as urlquote

from .typing_ import *
from .utils import json_dumps, json_loads, RemoteDoc

__all__ = ['MLStorageClient', 'ExperimentDoc']

_PATH_SEP_SPLITTER = re.compile(r'[/\\]')
_INVALID_PATH_CHARS = re.compile(r'[<>:"|?*]')


def normalize_relpath(path: str) -> str:
    """
    Normalize the `path`, enforcing `path` to be relative, translating "\\"
    into "/", reducing contiguous "/", eliminating "." and "..", and checking
    whether the `path` contains invalid characters.

    >>> normalize_relpath(r'/a/.\\b/c/../d')
    'a/b/d'
    >>> normalize_relpath('c:\\\\windows')
    Traceback (most recent call last):
        ...
    ValueError: Path contains invalid character(s): 'c:\\\\windows'
    >>> normalize_relpath('../')
    Traceback (most recent call last):
        ...
    ValueError: Path jump out of root: '../'

    Args:
        path: The relative path to be normalized.

    Returns:
        The normalized relative path.

    Raises:
        ValueError: If any ".." would jump out of root, or the path contains
            invalid characters.
    """
    if _INVALID_PATH_CHARS.search(path):
        raise ValueError(f'Path contains invalid character(s): {path!r}')
    segments = _PATH_SEP_SPLITTER.split(path)
    ret = []
    for segment in segments:
        if segment == '..':
            try:
                ret.pop()
            except IndexError:
                raise ValueError(f'Path jump out of root: {path!r}')
        elif segment not in ('', '.'):
            ret.append(segment)
    return '/'.join(ret)


class MLStorageClient(object):
    """
    Client binding for MLStorage Server API v1.
    """

    def __init__(self, uri: str):
        """
        Construct a new :class:`ClientV1`.

        Args:
            uri: Base URI of the MLStorage server, e.g., "http://example.com".
        """
        uri = uri.rstrip('/')
        self._uri = uri
        self._storage_dir_cache = LRUCache(128)

    def _update_storage_dir_cache(self, doc):
        self._storage_dir_cache[doc['_id']] = doc['storage_dir']

    @property
    def uri(self) -> str:
        """Get the base URI of the MLStorage server."""
        return self._uri

    def do_request(self, method: str, endpoint: str, decode_json: bool = True,
                   **kwargs) -> Union[requests.Response, Any]:
        """
        Send request of HTTP `method` to given `endpoint`.

        Args:
            method: The HTTP request method.
            endpoint: The endpoint of the API, should start with a slash "/".
                For example, "/_query".
            decode_json: Whether or not to decode the response body as JSON?
            \\**kwargs: Arguments to be passed to :func:`requests.request`.

        Returns:
            The response object if ``decode_json = False``, or the decoded
            JSON object.
        """
        uri = f'{self.uri}/v1{endpoint}'
        if 'json' in kwargs:
            json_obj = kwargs.pop('json')
            json_str = json_dumps(json_obj, no_dollar_field=True)
            kwargs['data'] = json_str
            kwargs.setdefault('headers', {})
            kwargs['headers']['Content-Type'] = 'application/json'

        resp = requests.request(method, uri, **kwargs)
        try:
            resp.raise_for_status()
            cnt = resp.content
            if decode_json:
                content_type = resp.headers.get('content-type') or ''
                content_type = content_type.split(';', 1)[0]
                if content_type != 'application/json':
                    raise IOError(f'The response from {uri} is not JSON: '
                                  f'HTTP code is {resp.status_code}')
                cnt = json_loads(cnt)
        finally:
            resp.close()

        return cnt

    def query(self,
              filter: Optional[FilterType] = None,
              sort: Optional[str] = None,
              skip: int = 0,
              limit: Optional[int] = None) -> List[DocumentType]:
        """
        Query experiment documents according to the `filter`.

        Args:
            filter: The filter dict.
            sort: Sort by which field, a string matching the pattern
                ``[+/-]<field>``.  "+" means ASC order, while "-" means
                DESC order.  For example, "start_time", "+start_time" and
                "-stop_time".
            skip: The number of records to skip.
            limit: The maximum number of records to retrieve.

        Returns:
            The documents of the matched experiments.
        """
        uri = f'/_query?skip={skip}'
        if sort is not None:
            uri += f'&sort={urlquote(sort)}'
        if limit is not None:
            uri += f'&limit={limit}'
        ret = self.do_request('POST', uri, json=filter or {})
        for doc in ret:
            self._update_storage_dir_cache(doc)
        return ret

    def get(self, id: ExperimentId) -> DocumentType:
        """
        Get the document of an experiment by its `id`.

        Args:
            id: The id of the experiment.

        Returns:
            The document of the retrieved experiment.
        """
        ret = self.do_request('GET', f'/_get/{id}')
        self._update_storage_dir_cache(ret)
        return ret

    def heartbeat(self, id: ExperimentId) -> None:
        """
        Send heartbeat packet for the experiment `id`.

        Args:
            id: The id of the experiment.
        """
        self.do_request('POST', f'/_heartbeat/{id}', data=b'')

    def create(self, doc_fields: DocumentType) -> DocumentType:
        """
        Create an experiment.

        Args:
            doc_fields: The document fields of the new experiment.

        Returns:
            The document of the created experiment.
        """
        doc_fields = dict(doc_fields)
        ret = self.do_request('POST', '/_create', json=doc_fields)
        self._update_storage_dir_cache(ret)
        return ret

    def update(self, id: ExperimentId, doc_fields: DocumentType) -> DocumentType:
        """
        Update the document of an experiment.

        Args:
            id: ID of the experiment.
            doc_fields: The fields to be updated.

        Returns:
            The document of the updated experiment.
        """
        ret = self.do_request('POST', f'/_update/{id}', json=doc_fields)
        self._update_storage_dir_cache(ret)
        return ret

    def add_tags(self, id: ExperimentId, tags: Iterable[str]) -> DocumentType:
        """
        Add tags to an experiment document.

        Args:
            id: ID of the experiment.
            tags: New tags to be added.

        Returns:
            The document of the updated experiment.
        """
        old_doc = self.get(id)
        new_tags = old_doc.get('tags', [])
        for tag in tags:
            if tag not in new_tags:
                new_tags.append(tag)
        return self.update(id, {'tags': new_tags})

    def delete(self, id: ExperimentId) -> List[ExperimentId]:
        """
        Delete an experiment.

        Args:
            id: ID of the experiment.

        Returns:
            List of deleted experiment IDs.
        """
        ret = self.do_request('POST', f'/_delete/{id}', data=b'')
        for i in ret:
            self._storage_dir_cache.pop(i, None)
        return ret

    def set_finished(self,
                     id: ExperimentId,
                     status: str,
                     doc_fields: Optional[DocumentType] = None
                     ) -> DocumentType:
        """
        Set the status of an experiment.

        Args:
            id: ID of the experiment.
            status: The new status, one of {"RUNNING", "COMPLETED", "FAILED"}.
            doc_fields: Optional new document fields to be set.

        Returns:
            The document of the updated experiment.
        """
        doc_fields = dict(doc_fields or ())
        doc_fields['status'] = status
        ret = self.do_request('POST', f'/_set_finished/{id}', json=doc_fields)
        self._update_storage_dir_cache(ret)
        return ret

    def get_storage_dir(self, id: ExperimentId) -> str:
        """
        Get the storage directory of an experiment.

        Args:
            id: ID of the experiment.

        Returns:
            The storage directory of the experiment.
        """
        id = str(id)
        storage_dir = self._storage_dir_cache.get(id, None)
        if storage_dir is None:
            doc = self.get(id)
            storage_dir = doc['storage_dir']
        return storage_dir

    def get_file(self, id: ExperimentId, path: str) -> bytes:
        """
        Get the content of a file in the storage directory of an experiment.

        Args:
            id: ID of the experiment.
            path: Relative path of the file.

        Returns:
            The file content.
        """
        id = str(id)
        path = normalize_relpath(path)
        return self.do_request(
            'GET', f'/_getfile/{id}/{path}', decode_json=False)


class ExperimentDoc(RemoteDoc):
    """:class:`RemoteDoc` class for experiment document."""

    client: Optional[MLStorageClient]
    id: Optional[ExperimentId]
    has_set_finished: bool = False

    def __init__(self,
                 client: Optional[MLStorageClient] = None,
                 id: Optional[ExperimentId] = None,
                 enable_heartbeat: bool = True):
        """
        Construct a new :class:`ExperimentDoc`.

        Args:
            client: The client of MLStorage server.
            id: ID of the experiment.
            enable_heartbeat: Whether or not to enable heartbeat to the
                remote server?
        """
        super().__init__(
            retry_interval=30,
            relaxed_interval=5,
            heartbeat_interval=120 if enable_heartbeat else None,
            keys_to_expand=('result', 'progress', 'webui', 'exc_info'),
        )

        self.client = client
        self.id = id

    @classmethod
    def from_env(cls, enable_heartbeat: bool = False
                 ) -> Optional['ExperimentDoc']:
        """
        Construct a :class:`ExperimentDoc` according to context.

        Args:
            enable_heartbeat: Whether or not to update heartbeat time?
        """
        # check the environment variables to determine the remote server
        server_uri = os.environ.get('MLSTORAGE_SERVER_URI', None)
        experiment_id = os.environ.get('MLSTORAGE_EXPERIMENT_ID', None)
        if not server_uri or not experiment_id:
            server_uri = experiment_id = None

        # construct the client object
        if server_uri and experiment_id:
            client = MLStorageClient(server_uri)
            experiment_id = ObjectId(experiment_id)
        else:
            client = None

        # construct the object if either the remote server or local result
        # config file is configured.
        if client is not None:
            remote_doc = ExperimentDoc(
                client=client,
                id=experiment_id,
                enable_heartbeat=enable_heartbeat,
            )
            return remote_doc

    @classmethod
    def default_doc(cls) -> Optional['ExperimentDoc']:
        """
        Get the default :class:`ExperimentDoc` object.

        If there is an active :class:`Experiment` object at the context stack,
        then returns its ``remote_doc`` attribute.  Otherwise attempt to create
        a new instance of :class:`ExperimentDoc` via :meth:`from_env()`.

        Returns:
            The default :class:`ExperimentDoc` object, or :obj:`None` if no
            :class:`ExperimentDoc` can be constructed according to the context.
        """
        from .experiment import get_active_experiment
        exp = get_active_experiment()
        if exp is not None:
            return exp.doc
        return ExperimentDoc.from_env()

    def now_time_literal(self) -> str:
        """
        Get the current time literal.

        Using this method, we can use `client.update` as a drop-in replacement
        for the `heartbeat` and `set_finished` API call.
        """
        return datetime.utcnow().replace(tzinfo=pytz.UTC).isoformat()

    RESULT_KEY_PREFIX = 'result.'

    def _push_to_remote(self, updates: DocumentType) -> Optional[DocumentType]:
        if self.client is not None and self.id is not None:
            if self.heartbeat_enabled and 'heartbeat' not in updates:
                updates['heartbeat'] = self.now_time_literal()
            return self.client.update(self.id, updates)

    def set_finished(self,
                     status: str,
                     updates: Optional[DocumentType] = None,
                     retry_intervals: Sequence[float] = (10, 20, 30, 50, 80,
                                                         130, 210)):
        """
        Set the experiment to be finished.

        Args:
            status: The finish status, one of ``{"COMPLETED", "FAILED"}``.
            updates: The other fields to be updated.
            retry_intervals: The intervals to sleep between two attempts
                to save the finish status.
        """
        if self._thread is not None:
            raise RuntimeError('`set_finished` must only be called when '
                               'the background worker is not running.')

        # compose the updates dict
        if updates is None:
            updates: DocumentType = {}
        now_time = self.now_time_literal()
        updates.setdefault('status', status)
        updates.setdefault('heartbeat', now_time)
        updates.setdefault('stop_time', now_time)

        # feed into this remote doc
        self.update(updates)

        # now call flush to actually push the updates
        # try to save the final status
        last_ex: Optional[Exception] = None
        for itv in (0,) + tuple(retry_intervals):
            if itv > 0:
                time.sleep(itv)
            try:
                self.flush()
            except Exception as ex:
                last_ex = ex
                getLogger(__name__).warning(
                    'Failed to store the final status of the experiment %s',
                    self.id, exc_info=True
                )
            else:
                last_ex = None
                self.has_set_finished = True
                break

        # if still failed, raise error
        if last_ex is not None:
            raise last_ex
