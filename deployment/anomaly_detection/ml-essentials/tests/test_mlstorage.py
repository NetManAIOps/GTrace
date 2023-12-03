import os
import time
import unittest
import uuid
from datetime import datetime
from functools import partial

import httpretty
import pytest
import pytz
import requests
from bson import ObjectId
from mock import Mock

from mltk import MLStorageClient, ExperimentDoc
from mltk.utils import json_dumps, json_loads
from tests.helpers import *


class MLStorageClientTestCase(unittest.TestCase):

    def setUp(self):
        self.client = MLStorageClient('http://127.0.0.1')

    @httpretty.activate
    def test_interface(self):
        c = MLStorageClient('http://127.0.0.1')
        self.assertEqual(c.uri, 'http://127.0.0.1')

        c = MLStorageClient('http://127.0.0.1/')
        self.assertEqual(c.uri, 'http://127.0.0.1')

        # test invalid response should trigger error
        httpretty_register_uri(
            httpretty.POST, 'http://127.0.0.1/v1/_query', body='hello')
        with pytest.raises(IOError,
                           match=r'The response from http://127.0.0.1/v1/'
                                 r'_query\?skip=0 is not JSON: HTTP code is '
                                 r'200'):
            _ = self.client.query()

    @httpretty.activate
    def test_bare_query(self):
        def callback(request, uri, response_headers,
                     expected_body, expected_skip, expected_limit,
                     expected_sort, response_body):
            content_type = request.headers.get('Content-Type').split(';', 1)[0]
            self.assertEqual(content_type, 'application/json')
            self.assertEqual(request.querystring['skip'][0], str(expected_skip))
            if expected_limit is not None:
                self.assertEqual(
                    request.querystring['limit'][0], str(expected_limit))
            if expected_sort is not None:
                self.assertEqual(
                    request.querystring['sort'][0], str(expected_sort))
            if expected_body is not None:
                self.assertEqual(request.body, expected_body)
            response_headers['content-type'] = 'application/json; charset=utf-8'
            return [200, response_headers, response_body]

        object_ids = [str(ObjectId()) for _ in range(5)]
        docs = [
            {'_id': object_ids[i], 'storage_dir': f'/{object_ids[i]}',
             'uuid': str(uuid.uuid4())}
            for i in range(len(object_ids))
        ]

        # test bare query
        for doc in docs:
            httpretty_register_uri(
                httpretty.GET,
                f'http://127.0.0.1/v1/_get/{doc["_id"]}',
                body=partial(
                    callback, expected_body=None, expected_skip=None,
                    expected_limit=None, expected_sort=None,
                    response_body=json_dumps(doc)
                )
            )
        httpretty_register_uri(
            httpretty.POST,
            'http://127.0.0.1/v1/_query',
            body=partial(
                callback, expected_body=b'{}', expected_skip=0,
                expected_limit=None, expected_sort=None,
                response_body=json_dumps(docs[:2])
            )
        )
        ret = self.client.query()
        self.assertListEqual(ret, docs[:2])

        for obj_id in object_ids[:2]:  # test the storage dir cache
            self.assertEqual(self.client.get_storage_dir(obj_id), f'/{obj_id}')

        with pytest.raises(requests.exceptions.ConnectionError):
            _ = self.client.get_storage_dir(object_ids[2])

    @httpretty.activate
    def test_query(self):
        def callback(request, uri, response_headers,
                     expected_body, expected_skip, expected_limit,
                     expected_sort, response_body):
            content_type = request.headers.get('Content-Type').split(';', 1)[0]
            self.assertEqual(content_type, 'application/json')
            self.assertEqual(request.querystring['skip'][0], str(expected_skip))
            if expected_limit is not None:
                self.assertEqual(
                    request.querystring['limit'][0], str(expected_limit))
            if expected_sort is not None:
                self.assertEqual(
                    request.querystring['sort'][0], str(expected_sort))
            self.assertEqual(request.body, expected_body)
            response_headers['content-type'] = 'application/json; charset=utf-8'
            return [200, response_headers, response_body]

        object_ids = [str(ObjectId()) for _ in range(5)]
        docs = [
            {'_id': object_ids[i], 'storage_dir': f'/{object_ids[i]}',
             'uuid': str(uuid.uuid4())}
            for i in range(len(object_ids))
        ]

        # test query
        httpretty_register_uri(
            httpretty.POST,
            'http://127.0.0.1/v1/_query',
            body=partial(
                callback, expected_body=b'{"name":"hint"}', expected_skip=1,
                expected_limit=99, expected_sort='-start_time',
                response_body=json_dumps(docs[2:4])
            )
        )
        ret = self.client.query(filter={'name': 'hint'}, sort='-start_time',
                                skip=1, limit=99)
        self.assertListEqual(ret, docs[2:4])

    @httpretty.activate
    def test_get(self):
        object_id = str(ObjectId())
        doc = {
            '_id': object_id,
            'storage_dir': f'/{object_id}',
            'uuid': str(uuid.uuid4()),
        }

        def callback(request, uri, response_headers):
            response_headers['content-type'] = 'application/json; charset=utf-8'
            return [200, response_headers, json_dumps(doc)]

        httpretty_register_uri(
            httpretty.GET, f'http://127.0.0.1/v1/_get/{object_id}',
            body=callback
        )
        ret = self.client.get(object_id)

        self.assertDictEqual(ret, doc)
        self.assertEqual(self.client.get_storage_dir(object_id),
                         doc['storage_dir'])

    @httpretty.activate
    def test_heartbeat(self):
        def callback(request, uri, response_headers):
            self.assertEqual(request.body, b'')
            response_headers['content-type'] = 'application/json; charset=utf-8'
            heartbeat_received[0] = True
            return [200, response_headers, b'{}']

        heartbeat_received = [False]
        object_id = str(ObjectId())
        httpretty_register_uri(
            httpretty.POST, f'http://127.0.0.1/v1/_heartbeat/{object_id}',
            body=callback
        )
        self.client.heartbeat(object_id)
        self.assertTrue(heartbeat_received[0])

    @httpretty.activate
    def test_add_tags(self):
        object_id = str(ObjectId())
        doc_fields = {
            'uuid': str(uuid.uuid4()),
            'name': 'hello',
            'storage_dir': f'/{object_id}',
        }

        def callback(request, uri, response_headers):
            response_headers[
                'content-type'] = 'application/json; charset=utf-8'
            o = {'_id': object_id}
            o.update(doc_fields)
            return [200, response_headers, json_dumps(o)]

        httpretty_register_uri(
            httpretty.GET, f'http://127.0.0.1/v1/_get/{object_id}',
            body=callback
        )

        # test add_tags
        doc_fields['tags'] = ['abc', '123']

        def callback(request, uri, response_headers):
            content_type = request.headers.get('Content-Type').split(';', 1)[0]
            self.assertEqual(content_type, 'application/json')
            fields = json_loads(request.body)
            self.assertDictEqual(
                fields, {'tags': ['abc', '123', 'hello, world!']})
            o = {'_id': object_id}
            o.update(doc_fields)
            response_headers['content-type'] = 'application/json; charset=utf-8'
            return [200, response_headers, json_dumps(o)]

        httpretty_register_uri(
            httpretty.POST, f'http://127.0.0.1/v1/_update/{object_id}',
            body=callback
        )
        ret = self.client.add_tags(object_id, ['hello, world!', '123'])
        expected = {'_id': object_id}
        expected.update(doc_fields)
        self.assertDictEqual(ret, expected)

    @httpretty.activate
    def test_create_update_delete(self):
        object_id = str(ObjectId())
        doc_fields = {
            'uuid': str(uuid.uuid4()),
            'name': 'hello',
            'storage_dir': f'/{object_id}',
        }

        # test create
        def callback(request, uri, response_headers):
            content_type = request.headers.get('Content-Type').split(';', 1)[0]
            self.assertEqual(content_type, 'application/json')
            fields = json_loads(request.body)
            self.assertDictEqual(fields, doc_fields)
            o = {'_id': object_id}
            o.update(doc_fields)
            response_headers['content-type'] = 'application/json; charset=utf-8'
            return [200, response_headers, json_dumps(o)]

        httpretty_register_uri(
            httpretty.POST, f'http://127.0.0.1/v1/_create',
            body=callback
        )
        ret = self.client.create(doc_fields)

        expected = {'_id': object_id}
        expected.update(doc_fields)
        self.assertDictEqual(ret, expected)
        self.assertEqual(self.client.get_storage_dir(object_id),
                         doc_fields['storage_dir'])

        # test update
        doc_fields['storage_dir'] = f'/new/{object_id}'

        def callback(request, uri, response_headers):
            content_type = request.headers.get('Content-Type').split(';', 1)[0]
            self.assertEqual(content_type, 'application/json')
            fields = json_loads(request.body)
            self.assertDictEqual(
                fields, {'storage_dir': doc_fields['storage_dir']})
            o = {'_id': object_id}
            o.update(doc_fields)
            response_headers['content-type'] = 'application/json; charset=utf-8'
            return [200, response_headers, json_dumps(o)]

        httpretty_register_uri(
            httpretty.POST, f'http://127.0.0.1/v1/_update/{object_id}',
            body=callback
        )
        ret = self.client.update(
            object_id, {'storage_dir': doc_fields['storage_dir']})

        expected = {'_id': object_id}
        expected.update(doc_fields)
        self.assertDictEqual(ret, expected)
        self.assertEqual(self.client.get_storage_dir(object_id),
                         doc_fields['storage_dir'])

        # test delete
        def callback(request, uri, response_headers):
            self.assertEqual(request.body, b'')
            response_headers['content-type'] = 'application/json; charset=utf-8'
            return [200, response_headers, json_dumps([object_id])]

        httpretty_register_uri(
            httpretty.POST, f'http://127.0.0.1/v1/_delete/{object_id}',
            body=callback
        )
        self.assertListEqual(self.client.delete(object_id), [object_id])

        httpretty_register_uri(
            httpretty.GET,
            f'http://127.0.0.1/v1/_get/{object_id}',
            body=lambda: [404, {}, b'{}']
        )

        with pytest.raises(requests.exceptions.ConnectionError):
            _ = self.client.get_storage_dir(object_id)

    @httpretty.activate
    def test_set_finished(self):
        object_id = str(ObjectId())
        doc_fields = {
            '_id': object_id,
            'uuid': str(uuid.uuid4()),
            'name': 'hello',
            'status': 'COMPLETED',
            'storage_dir': f'/{object_id}',
        }

        # test set status only
        def callback(request, uri, response_headers):
            content_type = request.headers.get('Content-Type').split(';', 1)[0]
            self.assertEqual(content_type, 'application/json')
            fields = json_loads(request.body)
            self.assertDictEqual(fields, {'status': 'COMPLETED'})
            response_headers['content-type'] = 'application/json; charset=utf-8'
            return [200, response_headers, json_dumps(doc_fields)]

        httpretty_register_uri(
            httpretty.POST, f'http://127.0.0.1/v1/_set_finished/{object_id}',
            body=callback
        )
        ret = self.client.set_finished(object_id, 'COMPLETED')
        self.assertDictEqual(ret, doc_fields)

        # test set status with new fields
        def callback(request, uri, response_headers):
            content_type = request.headers.get('Content-Type').split(';', 1)[0]
            self.assertEqual(content_type, 'application/json')
            fields = json_loads(request.body)
            self.assertDictEqual(
                fields, {'name': 'hello', 'status': 'COMPLETED'})
            response_headers['content-type'] = 'application/json; charset=utf-8'
            return [200, response_headers, json_dumps(doc_fields)]

        httpretty_register_uri(
            httpretty.POST, f'http://127.0.0.1/v1/_set_finished/{object_id}',
            body=callback
        )
        ret = self.client.set_finished(
            object_id, 'COMPLETED', {'name': 'hello'})
        self.assertDictEqual(ret, doc_fields)
        self.assertEqual(
            self.client.get_storage_dir(object_id), f'/{object_id}')

    @httpretty.activate
    def test_get_storage_dir(self):
        object_id = str(ObjectId())
        doc_fields = {'_id': object_id, 'storage_dir': f'/{object_id}'}

        def callback(request, uri, response_headers):
            response_headers['content-type'] = 'application/json; charset=utf-8'
            counter[0] += 1
            return [200, response_headers, json_dumps(doc_fields)]

        counter = [0]
        httpretty_register_uri(
            httpretty.GET, f'http://127.0.0.1/v1/_get/{object_id}',
            body=callback
        )

        self.assertEqual(
            self.client.get_storage_dir(object_id), f'/{object_id}')
        self.assertEqual(counter[0], 1)
        self.assertEqual(
            self.client.get_storage_dir(object_id), f'/{object_id}')
        self.assertEqual(counter[0], 1)

    @httpretty.activate
    def test_get_file(self):
        object_id = str(ObjectId())

        httpretty_register_uri(
            httpretty.GET,
            f'http://127.0.0.1/v1/_getfile/{object_id}/hello.txt',
            body=b'hello, world'
        )
        self.assertEqual(self.client.get_file(object_id, '/./hello.txt'),
                         b'hello, world')


class ExperimentDocTestCase(unittest.TestCase):

    def test_construct(self):
        client = Mock()
        id = ObjectId()

        doc = ExperimentDoc(client, id)
        self.assertEqual(doc.retry_interval, 30)
        self.assertEqual(doc.relaxed_interval, 5)
        self.assertEqual(doc.heartbeat_interval, 120)
        self.assertEqual(doc.keys_to_expand,
                         ('result', 'progress', 'webui', 'exc_info'))
        self.assertIs(doc.client, client)
        self.assertEqual(doc.id, id)
        self.assertEqual(doc.heartbeat_enabled, True)
        self.assertEqual(doc.has_set_finished, False)
        self.assertIsInstance(doc.now_time_literal(), str)

        doc = ExperimentDoc(client, id, enable_heartbeat=False)
        self.assertEqual(doc.heartbeat_enabled, False)
        self.assertEqual(doc.heartbeat_interval, None)

    def test_from_env(self):
        os.environ.pop('MLSTORAGE_SERVER_URI', None)
        os.environ.pop('MLSTORAGE_EXPERIMENT_ID', None)
        self.assertIsNone(ExperimentDoc.from_env())

        server_uri = 'http://127.0.0.1:8080'
        experiment_id = ObjectId()
        os.environ['MLSTORAGE_SERVER_URI'] = server_uri
        os.environ['MLSTORAGE_EXPERIMENT_ID'] = str(experiment_id)
        doc = ExperimentDoc.from_env()
        self.assertEqual(doc.heartbeat_enabled, False)
        self.assertEqual(doc.id, experiment_id)
        self.assertEqual(doc.client.uri, server_uri)

        doc = ExperimentDoc.from_env(enable_heartbeat=True)
        self.assertEqual(doc.heartbeat_enabled, True)

        os.environ.pop('MLSTORAGE_SERVER_URI', None)
        os.environ.pop('MLSTORAGE_EXPERIMENT_ID', None)

    def test_push_to_remote(self):
        client = MLStorageClient('http://127.0.0.1:8080')
        id = ObjectId()
        return_value = {'id': id, 'flag': 123}
        now_time = datetime.utcnow().replace(tzinfo=pytz.UTC).isoformat()
        client.update = Mock(return_value=return_value)

        # test push_to_remote without heartbeat field
        push_updates = {'id': id, 'flag': 456}
        doc = ExperimentDoc(client, id)
        doc.now_time_literal = Mock(return_value=now_time)
        self.assertEqual(doc.push_to_remote(push_updates), return_value)

        self.assertEqual(client.update.call_args[0][0], id)
        updates_arg = client.update.call_args[0][1]
        self.assertIn('heartbeat', updates_arg)
        updates_arg.pop('heartbeat')
        self.assertEqual(updates_arg, push_updates)

        # test with heartbeat field
        now_time2 = datetime.utcnow().replace(tzinfo=pytz.UTC).isoformat()
        push_updates = {'id': id, 'flag': 456, 'heartbeat': now_time2}
        doc = ExperimentDoc(client, id)
        doc.now_time_literal = Mock(return_value=now_time)
        self.assertEqual(doc.push_to_remote(push_updates), return_value)
        self.assertEqual(client.update.call_args[0][0], id)
        self.assertEqual(client.update.call_args[0][1], push_updates)

    def test_set_finished(self):
        client = MLStorageClient('http://127.0.0.1:8080')
        id = ObjectId()
        doc = ExperimentDoc(client, id)
        now_time = datetime.utcnow().replace(tzinfo=pytz.UTC).isoformat()
        doc.now_time_literal = Mock(return_value=now_time)

        # set_finished should only be called when thread is None
        client.update = Mock(return_value={})
        with doc:
            with pytest.raises(RuntimeError,
                               match='`set_finished` must only be called '
                                     'when the background worker is not '
                                     'running'):
                _ = doc.set_finished('FAILED', retry_intervals=(0.1, 0.2))

        # test error retry
        start_time = time.time()
        retry_times = []
        expected_updates = {
            'status': 'COMPLETED',
            'heartbeat': now_time,
            'stop_time': now_time,
            'abc': 123,
        }

        def f(v_id, v_updates):
            self.assertEqual(v_id, id)
            self.assertEqual(v_updates, expected_updates)
            retry_times.append(time.time())
            raise RuntimeError(f'retry count: {len(retry_times)}')

        client.update = Mock(wraps=f)
        with pytest.raises(RuntimeError, match='retry count: 3'):
            doc.set_finished('COMPLETED', {'abc': 123},
                             retry_intervals=(0.1, 0.2))

        self.assertEqual(len(retry_times), 3)
        self.assertLess(abs(retry_times[0] - start_time), 0.02)
        self.assertLess(abs(retry_times[1] - retry_times[0] - 0.1), 0.02)
        self.assertLess(abs(retry_times[2] - retry_times[1] - 0.2), 0.02)
        self.assertEqual(doc.has_set_finished, False)

        # test success
        return_value = {'id': id, 'flags': 456}
        client.update = Mock(return_value=return_value)
        doc.set_finished('COMPLETED', {'abc': 123}, retry_intervals=(0.1, 0.2))
        self.assertEqual(client.update.call_count, 1)
        self.assertEqual(client.update.call_args[0][0], id)
        self.assertEqual(client.update.call_args[0][1], expected_updates)
        self.assertEqual(doc.has_set_finished, True)
