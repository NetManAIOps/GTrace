# -*- coding: utf-8 -*-

import os
import re
import socket
import subprocess
import sys
import time
import unittest
from contextlib import contextmanager
from tempfile import TemporaryDirectory

import httpretty
import mock
import pytest
import requests
from bson import ObjectId
from click.testing import CliRunner
from mock import Mock

import mltk
from mltk import ConfigValidationError, MLStorageClient, validate_config, ProgramOutputReceiver, config_params
from mltk.mlrunner import (ProgramHost, MLRunnerConfig, SourceCopier,
                           MLRunnerConfigLoader, TemporaryFileCleaner,
                           JsonFileWatcher, MLRunner, mlrun, ControlServer)
from mltk.utils import json_dumps, json_loads
from tests.helpers import *


class MLRunnerConfigTestCase(unittest.TestCase):

    def test_validate(self):
        # test all empty
        config = MLRunnerConfig()
        config.source.includes = []
        config.source.excludes = []

        config.args = 'xyz'
        with pytest.raises(ConfigValidationError,
                           match='\'server\' is required.'):
            _ = validate_config(config)

        del config.args
        config.server = 'http://127.0.0.1:8080'
        with pytest.raises(ConfigValidationError,
                           match='\'args\' is required.'):
            _ = validate_config(config)

        config.args = ''
        with pytest.raises(ConfigValidationError,
                           match='\'args\' must not be empty'):
            _ = validate_config(config)

        config.args = []
        with pytest.raises(ConfigValidationError,
                           match='\'args\' must not be empty'):
            _ = validate_config(config)

        config.args = ['sh', '-c', 'echo hello']
        config = validate_config(config)
        for key in ('name', 'description', 'tags', 'env', 'gpu',
                    'work_dir', 'daemon'):
            self.assertIsNone(config[key])
        self.assertEqual(config.source.includes, [])
        self.assertEqual(config.source.excludes, [])

        # test .args
        config = MLRunnerConfig(args=['sh', 123],
                                server='http://127.0.0.1:8080')
        self.assertEqual(validate_config(config).args, ['sh', '123'])
        config = MLRunnerConfig(args='exit 0',
                                server='http://127.0.0.1:8080')
        self.assertEqual(validate_config(config).args, 'exit 0')

        # test .tags
        config.tags = ['hello', 123]
        self.assertListEqual(validate_config(config).tags, ['hello', '123'])

        # test .env
        config.env = {'value': 123}
        self.assertEqual(validate_config(config).env, {'value': '123'})

        # test .gpu
        config.gpu = [1, 2]
        self.assertListEqual(validate_config(config).gpu, [1, 2])

        # test .daemon
        config.daemon = 'exit 0'
        with pytest.raises(ConfigValidationError,
                           match='at daemon: value is not a sequence'):
            _ = validate_config(config)

        config.daemon = ['exit 0', ['sh', '-c', 'exit 1']]
        self.assertListEqual(validate_config(config).daemon, [
            'exit 0', ['sh', '-c', 'exit 1']
        ])

        # test .source.includes & .source.excludes using literals
        includes = r'.*\.py$'
        excludes = re.compile(r'.*/\.svn$')

        config.source.includes = includes
        config.source.excludes = excludes

        c = validate_config(config)
        self.assertIsInstance(c.source.includes, list)
        self.assertEqual(len(c.source.includes), 1)
        self.assertEqual(c.source.includes[0].pattern, includes)

        self.assertIsInstance(c.source.excludes, list)
        self.assertEqual(len(c.source.excludes), 1)
        self.assertIs(c.source.excludes[0], excludes)

        # test .source.includes & .source.excludes using lists
        includes = [r'.*\.py$', re.compile(r'.*\.exe$')]
        excludes = [r'.*/\.git$', re.compile(r'.*/\.svn$')]

        config.source.includes = includes
        config.source.excludes = excludes

        c = validate_config(config)
        self.assertIsInstance(c.source.includes, list)
        self.assertEqual(len(c.source.includes), 2)
        self.assertEqual(c.source.includes[0].pattern, includes[0])
        self.assertIs(c.source.includes[1], includes[1])

        self.assertIsInstance(c.source.excludes, list)
        self.assertEqual(len(c.source.excludes), 2)
        self.assertEqual(c.source.excludes[0].pattern, excludes[0])
        self.assertIs(c.source.excludes[1], excludes[1])


class MLRunnerConfigLoaderTestCase(unittest.TestCase):

    maxDiff = None

    def test_loader(self):
        with TemporaryDirectory() as temp_dir:
            # prepare for the test dir
            prepare_dir(temp_dir, {
                'sys1': {
                    '.mlrun.yaml': b'clone_from: sys1\n'
                                   b'args: sys1/.mlrun.yaml args\n'
                },
                'sys2': {
                    '.mlrun.yml': b'args: sys2/.mlrun.yml args\n'
                                  b'name: sys2/.mlrun.yml',
                },
                'work': {
                    '.mlrun.yml': b'name: work/.mlrun.yml\n'
                                  b'server: http://127.0.0.1:8080',
                    '.mlrun.yaml': b'server: http://127.0.0.1:8081\n'
                                   b'tags: [1, 2, 3]',
                    '.mlrun.json': b'{"tags": [4, 5, 6],'
                                   b'"description": "work/.mlrun.json"}',
                    'nested': {
                        '.mlrun.yml': b'description: work/nested/.mlrun.yml\n'
                                      b'resume_from: xyz'
                    }
                },
                'config1.yml': b'resume_from: zyx\n'
                               b'source.src_dir: config1',
                'config2.yml': b'source.src_dir: config2\n'
                               b'logging.log_file: config2.log',
            })

            # test loader
            config = MLRunnerConfig(env={'a': '1'}, clone_from='code')
            loader = MLRunnerConfigLoader(
                config=config,
                config_files=[
                    os.path.join(temp_dir, 'config1.yml'),
                    os.path.join(temp_dir, 'config2.yml')
                ],
                work_dir=os.path.join(temp_dir, 'work/nested'),
                system_paths=[
                    os.path.join(temp_dir, 'sys1'),
                    os.path.join(temp_dir, 'sys2')
                ],
            )
            expected_config_files = [
                os.path.join(temp_dir, 'sys1/.mlrun.yaml'),
                os.path.join(temp_dir, 'sys2/.mlrun.yml'),
                os.path.join(temp_dir, 'work/.mlrun.yml'),
                os.path.join(temp_dir, 'work/.mlrun.yaml'),
                os.path.join(temp_dir, 'work/.mlrun.json'),
                os.path.join(temp_dir, 'work/nested/.mlrun.yml'),
                os.path.join(temp_dir, 'config1.yml'),
                os.path.join(temp_dir, 'config2.yml'),
            ]
            self.assertListEqual(
                loader.list_config_files(), expected_config_files)
            load_order = []
            loader.load_config_files(on_load=load_order.append)
            self.assertListEqual(load_order, expected_config_files)

            config = loader.get()
            self.assertEqual(config.logging.log_file, 'config2.log')
            self.assertEqual(config.source.src_dir, 'config2')
            self.assertEqual(config.resume_from, 'zyx')
            self.assertEqual(config.description, 'work/nested/.mlrun.yml')
            self.assertListEqual(config.tags, ['4', '5', '6'])
            self.assertEqual(config.server, 'http://127.0.0.1:8081')
            self.assertEqual(config.name, 'work/.mlrun.yml')
            self.assertEqual(config.args, 'sys2/.mlrun.yml args')
            self.assertEqual(config.clone_from, 'sys1')
            self.assertEqual(config.env, {'a': '1'})

            # test bare loader
            loader = MLRunnerConfigLoader(system_paths=[])
            self.assertListEqual(loader.list_config_files(), [])
            loader.load_config_files()

            # test just one config file
            cfg_file = os.path.join(temp_dir, 'config.json')
            write_file_content(cfg_file, b'{"args": "exit 0",'
                                         b'"server":"http://127.0.0.1:8080"}')
            loader = MLRunnerConfigLoader(config_files=[cfg_file])
            loader.load_config_files()
            self.assertEqual(loader.get(), MLRunnerConfig(
                server='http://127.0.0.1:8080',
                args='exit 0'
            ))

            # test error on non-exist user specified config file
            loader = MLRunnerConfigLoader(
                config_files=[
                    os.path.join(temp_dir, 'not-exist.yml')
                ]
            )
            with pytest.raises(IOError, match='User specified config file '
                                              '.* does not exist'):
                loader.load_config_files()


class MockMLServer(object):

    def __init__(self, root_dir, uri='http://127.0.0.1:8080'):
        self._root_dir = root_dir
        self._uri = uri
        self._db = {}

    @property
    def root_dir(self):
        return self._root_dir

    @property
    def uri(self):
        return self._uri

    @property
    def db(self):
        return self._db

    def register_uri(self):
        httpretty_register_uri(
            'GET',
            re.compile(self.uri + '/v1/_get/([A-Za-z0-9]+)'),
            self.handle_get
        )
        httpretty_register_uri(
            'POST',
            re.compile(self.uri + '/v1/_heartbeat/([A-Za-z0-9]+)'),
            self.handle_heartbeat
        )
        httpretty_register_uri(
            'POST',
            re.compile(self.uri + '/v1/_create'),
            self.handle_create
        )
        httpretty_register_uri(
            'POST',
            re.compile(self.uri + '/v1/_update/([A-Za-z0-9]+)'),
            self.handle_update
        )
        httpretty_register_uri(
            'POST',
            re.compile(self.uri + '/v1/_set_finished/([A-Za-z0-9]+)'),
            self.handle_set_finished
        )

    def json_response(self, response_headers, cnt, status=200):
        if not isinstance(cnt, (str, bytes)):
            cnt = json_dumps(cnt)
        if not isinstance(cnt, bytes):
            cnt = cnt.encode('utf-8')
        response_headers['content-type'] = 'application/json; charset=utf-8'
        return [status, response_headers, cnt]

    def handle_heartbeat(self, request, uri, response_headers):
        id = re.search(r'/v1/_heartbeat/([A-Za-z0-9]+)$', uri).group(1)
        assert(id in self.db)
        return self.json_response(response_headers, {})

    def handle_get(self, request, uri, response_headers):
        id = re.search(r'/v1/_get/([A-Za-z0-9]+)$', uri).group(1)
        assert(id in self.db)
        return self.json_response(response_headers, self.db[id])

    def handle_create(self, request, uri, response_headers):
        assert(request.headers.get('Content-Type', '').split(';', 1)[0] ==
               'application/json')
        body = json_loads(request.body.decode('utf-8'))
        body['_id'] = str(ObjectId())
        body['storage_dir'] = os.path.join(self.root_dir, body['_id'])
        self.db[body['_id']] = body
        return self.json_response(response_headers, self.db[body['_id']])

    def handle_update(self, request, uri, response_headers):
        id = re.search(r'/v1/_update/([A-Za-z0-9]+)$', uri).group(1)
        assert(id in self.db)
        assert (request.headers.get('Content-Type', '').split(';', 1)[0] ==
                'application/json')
        body = json_loads(request.body.decode('utf-8'))
        if body:
            for key, val in body.items():
                parts = key.split('.', 1)
                if len(parts) == 2:
                    self.db[id].setdefault(parts[0], {})
                    self.db[id][parts[0]][parts[1]] = val
                else:
                    self.db[id][key] = val
        return self.json_response(response_headers, self.db[id])

    def handle_set_finished(self, request, uri, response_headers):
        id = re.search(r'/v1/_set_finished/([A-Za-z0-9]+)$', uri).group(1)
        assert (id in self.db)
        assert (request.headers.get('Content-Type', '').split(';', 1)[0] ==
                'application/json')
        body = json_loads(request.body.decode('utf-8'))
        if body:
            for key, val in body.items():
                parts = key.split('.', 1)
                if len(parts) == 2:
                    self.db[id].setdefault(parts[0], {})
                    self.db[id][parts[0]][parts[1]] = val
                else:
                    self.db[id][key] = val
        return self.json_response(response_headers, self.db[id])

    def __enter__(self):
        self.register_uri()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class MLRunnerTestCase(unittest.TestCase):

    maxDiff = None

    @slow_test
    @httpretty.activate
    def test_run(self):
        with TemporaryDirectory() as temp_dir:
            # prepare for the source dir
            source_root = os.path.join(temp_dir, 'source')
            source_fiels = {
                'a.py': b'print("a.py")\n',
                'a.txt': b'a.txt content\n',
                'nested': {
                    'b.sh': b'echo "b.sh"\n'
                }
            }
            prepare_dir(source_root, source_fiels)

            with chdir_context(source_root):
                # prepare for the test server
                output_root = os.path.join(temp_dir, 'output')
                server = MockMLServer(output_root)

                # prepare for the test experiment runner
                config = MLRunnerConfig(
                    server=server.uri,
                    name='test',
                    args=[
                        sys.executable,
                        '-c',
                        'print("hello")\n'
                        'print("[Epoch 2/5, Batch 3/6, Step 4, ETA 5s] epoch_time: 1s; loss: 0.25 (±0.1); span: 5")\n'
                        'open("config.json", "wb").write(b"{\\"max_epoch\\": 123}\\n")\n'
                        'open("config.defaults.json", "wb").write(b"{\\"max_epoch\\": 100}\\n")\n'
                        'open("webui.json", "wb").write(b"{\\"TB\\": \\"http://tb:7890\\"}\\n")\n'
                        'import time\n'
                        'time.sleep(1)\n'
                        'open("result.json", "wb").write(b"{\\"acc\\": 0.99}\\n")\n'
                    ],
                    daemon=[
                        ['echo', 'daemon 1'],
                        'echo "Serving HTTP on 0.0.0.0 port 12367 '
                        '(http://0.0.0.0:12367/)"',
                        'env',
                    ],
                    env={'MY_ENV': 'abc'},
                    gpu=[1, 2]
                )
                config.source.copy_to_dst = True
                config.source.cleanup = False
                config.integration.watch_json_files = True
                config.integration.parse_stdout = True
                runner = MLRunner(config, retry_intervals=(0.1, 0.2, 0.3))

                # run the test experiment
                with server:
                    code = runner.run()
                    self.assertEqual(code, 0)

                # check the result
                self.assertEqual(len(server.db), 1)
                doc = list(server.db.values())[0]

                output_dir = os.path.join(output_root, doc['_id'])
                size, inode = compute_fs_size_and_inode(output_dir)

                exc_info = doc.pop('exc_info')
                self.assertIsInstance(exc_info, dict)
                self.assertEqual(exc_info['hostname'], socket.gethostname())
                self.assertIsInstance(exc_info['pid'], int)
                self.assertIsInstance(exc_info['env'], dict)
                self.assertEqual(exc_info['env']['PYTHONUNBUFFERED'], '1')
                self.assertEqual(exc_info['env']['MLSTORAGE_SERVER_URI'],
                                 'http://127.0.0.1:8080')
                self.assertEqual(exc_info['env']['MLSTORAGE_EXPERIMENT_ID'],
                                 doc['_id'])
                self.assertEqual(exc_info['env']['MLSTORAGE_OUTPUT_DIR'],
                                 output_dir)
                self.assertEqual(exc_info['env']['PWD'], output_dir)
                self.assertEqual(exc_info['env']['CUDA_VISIBLE_DEVICES'], '1,2')
                self.assertIn(output_dir, exc_info['env']['PYTHONPATH'])
                self.assertIn(source_root, exc_info['env']['PYTHONPATH'])

                control_port = doc.pop('control_port')
                self.assertIsInstance(control_port, dict)
                self.assertTrue(control_port['kill'].startswith('http://'))

                hostname = socket.gethostname()
                stop_time = doc.pop('stop_time')
                self.assertEqual(doc.pop('heartbeat'), stop_time)
                self.assertDictEqual(doc, {
                    'name': config.name,
                    '_id': doc["_id"],
                    'storage_dir': os.path.join(
                        temp_dir, f'output/{doc["_id"]}'),
                    'args': config.args,
                    'progress': {'epoch': '2/5', 'batch': '3/6', 'step': 4, 'eta': '5s'},
                    'result': {'acc': 0.99, 'span': '5', 'loss': '0.25 (±0.1)'},
                    'exit_code': code,
                    'storage_size': size,
                    'storage_inode': inode,
                    'status': 'COMPLETED',
                    'config': {'max_epoch': 123},
                    'default_config': {'max_epoch': 100},
                    'webui': {'SimpleHTTP': f'http://{hostname}:12367/',
                              'TB': 'http://tb:7890'},
                })

                output_snapshot = dir_snapshot(output_dir)
                self.assertEqual(
                    set(output_snapshot),
                    {'config.defaults.json', 'config.json', 'console.log',
                     'daemon.0.log', 'daemon.1.log', 'daemon.2.log',
                     'mlrun.log', 'result.json', 'webui.json', 'source.zip',
                     'a.py', 'nested'}
                )

                self.assertEqual(output_snapshot['config.defaults.json'],
                                 b'{"max_epoch": 100}\n')
                self.assertEqual(output_snapshot['config.json'],
                                 b'{"max_epoch": 123}\n')
                self.assertEqual(output_snapshot['console.log'],
                                 b'hello\n'
                                 b'[Epoch 2/5, Batch 3/6, Step 4, ETA 5s] '
                                 b'epoch_time: 1s; loss: 0.25 (\xc2\xb10.1); '
                                 b'span: 5\n')
                self.assertEqual(output_snapshot['result.json'],
                                 b'{"acc": 0.99}\n')
                self.assertEqual(output_snapshot['webui.json'],
                                 b'{"TB": "http://tb:7890"}\n')

                self.assertEqual(output_snapshot['daemon.0.log'],
                                 b'daemon 1\n')
                self.assertTrue(output_snapshot['daemon.1.log'].startswith(
                    b'Serving HTTP on 0.0.0.0 port 12367 '
                    b'(http://0.0.0.0:12367/)'
                ))

                self.assertIn(b'MY_ENV=abc\n', output_snapshot['daemon.2.log'])
                self.assertIn(b'PYTHONUNBUFFERED=1\n',
                              output_snapshot['daemon.2.log'])

                self.assertEqual(output_snapshot['a.py'], b'print("a.py")\n')
                self.assertDictEqual(output_snapshot['nested'], {
                    'b.sh': b'echo "b.sh"\n',
                })

                self.assertDictEqual(
                    zip_snapshot(os.path.join(output_dir, 'source.zip')),
                    {
                        'a.py': b'print("a.py")\n',
                        'nested': {
                            'b.sh': b'echo "b.sh"\n'
                        }
                    }
                )

    @slow_test
    @httpretty.activate
    def test_copy_arg_files_resume_and_clone(self):
        parent_id = str(ObjectId())

        with TemporaryDirectory() as temp_dir:
            # prepare for the source dir
            source_root = os.path.join(temp_dir, 'source')
            source_fiels = {
                'a.py': b'print("a.py")\n',
                'a.txt': b'a.txt content\n',
                'nested': {
                    'b.sh': b'echo "b.sh"\nexit 1\n'
                }
            }
            prepare_dir(source_root, source_fiels)

            with chdir_context(source_root):
                # prepare for the test server
                output_root = os.path.join(temp_dir, 'output')
                server = MockMLServer(output_root)

                # test copy arg files
                config = MLRunnerConfig(
                    server=server.uri,
                    args='sh nested/b.sh a.txt'.split(' '),
                    parent_id=parent_id,
                )
                config.source.make_archive = False  # test no source archive
                runner = MLRunner(config, retry_intervals=(0.1, 0.2, 0.3))

                with server:
                    code = runner.run()
                    self.assertEqual(code, 1)

                self.assertEqual(len(server.db), 1)
                doc = list(server.db.values())[0]
                self.assertEqual(doc['exit_code'], 1)
                self.assertEqual(doc['status'], 'COMPLETED')
                self.assertEqual(doc['parent_id'], parent_id)
                self.assertEqual(doc['name'], ' '.join(config.args))

                output_snapshot = dir_snapshot(doc['storage_dir'])
                self.assertEqual(output_snapshot['console.log'], b'b.sh\n')
                self.assertEqual(output_snapshot['nested'], {
                    'b.sh': b'echo "b.sh"\nexit 1\n',
                })
                self.assertNotIn('source.zip', output_snapshot)
                self.assertNotIn('a.txt', output_snapshot)
                self.assertNotIn('a.py', output_snapshot)

                # test resume from
                config = MLRunnerConfig(
                    server=server.uri,
                    args='echo hello',
                    resume_from=doc['_id'],
                )
                config.source.make_archive = False
                runner = MLRunner(config, retry_intervals=(0.1, 0.2, 0.3))

                with server:
                    code = runner.run()
                    self.assertEqual(code, 0)

                self.assertEqual(len(server.db), 1)
                doc2 = list(server.db.values())[0]
                self.assertEqual(doc2, doc)

                output_snapshot = dir_snapshot(doc['storage_dir'])
                self.assertEqual(output_snapshot['console.log'],
                                 b'b.sh\nhello\n')

                # test clone from
                config = MLRunnerConfig(
                    server=server.uri,
                    args='echo hi',
                    clone_from=doc['_id'],
                )
                config.source.make_archive = False
                runner = MLRunner(config, retry_intervals=(0.1, 0.2, 0.3))

                with server:
                    code = runner.run()
                    self.assertEqual(code, 0)

                self.assertEqual(len(server.db), 2)
                doc3 = list(server.db.values())[1]
                self.assertNotEqual(doc3['_id'], doc['_id'])
                self.assertNotEqual(doc3['storage_dir'], doc['storage_dir'])

                output_snapshot = dir_snapshot(doc3['storage_dir'])
                self.assertEqual(output_snapshot['console.log'],
                                 b'b.sh\nhello\nhi\n')
                self.assertEqual(output_snapshot['nested'], {
                    'b.sh': b'echo "b.sh"\nexit 1\n',
                })

    @slow_test
    @httpretty.activate
    def test_work_dir(self):
        parent_id = str(ObjectId())

        with TemporaryDirectory() as temp_dir, \
                set_environ_context(MLSTORAGE_EXPERIMENT_ID=parent_id):
            # prepare for the source dir
            source_root = os.path.join(temp_dir, 'source')
            source_fiels = {
                'a.py': b'print("a.py")\n',
                'a.txt': b'a.txt content\n',
                'nested': {
                    'b.sh': b'echo "b.sh"\nexit 1\n'
                }
            }
            prepare_dir(source_root, source_fiels)

            # prepare for the work dir
            work_dir = os.path.join(temp_dir, 'work')
            os.makedirs(work_dir)

            with chdir_context(source_root):
                # prepare for the test server
                output_root = os.path.join(temp_dir, 'output')
                server = MockMLServer(output_root)

                # test copy arg files
                config = MLRunnerConfig(
                    server=server.uri,
                    args='echo nested/b.sh a.txt; echo hello > temp.txt',
                    daemon=['pwd'],
                    work_dir=work_dir,
                )
                config.source.make_archive = False  # test no source archive
                runner = MLRunner(config, retry_intervals=(0.1, 0.2, 0.3))

                with server:
                    code = runner.run()
                    self.assertEqual(code, 0)

                self.assertEqual(len(server.db), 1)
                doc = list(server.db.values())[0]
                self.assertEqual(doc['exit_code'], 0)
                self.assertEqual(doc['status'], 'COMPLETED')
                self.assertEqual(doc['parent_id'], parent_id)
                self.assertEqual(doc['name'], config.args)
                self.assertNotEqual(doc['storage_dir'], work_dir)

                output_snapshot = dir_snapshot(doc['storage_dir'])
                self.assertEqual(output_snapshot['console.log'],
                                 b'nested/b.sh a.txt\n')
                self.assertEqual(output_snapshot['daemon.0.log'],
                                 f'{work_dir}\n'.encode('utf-8'))
                self.assertEqual(output_snapshot['nested'], {
                    'b.sh': b'echo "b.sh"\nexit 1\n',
                })
                self.assertNotIn('source.zip', output_snapshot)
                self.assertNotIn('a.txt', output_snapshot)
                self.assertNotIn('a.py', output_snapshot)

                work_dir_snapshot = dir_snapshot(work_dir)
                self.assertDictEqual(work_dir_snapshot, {
                    'temp.txt': b'hello\n'
                })

    @slow_test
    @httpretty.activate
    def test_inner_error(self):
        def mock_gethostname():
            raise RuntimeError('cannot get hostname')

        with TemporaryDirectory() as temp_dir, \
                mock.patch('socket.gethostname', mock_gethostname):
            # prepare for the test server
            output_root = os.path.join(temp_dir, 'output')
            server = MockMLServer(output_root)

            # test copy arg files
            config = MLRunnerConfig(
                server=server.uri,
                args='echo hello',
            )
            config.source.make_archive = False  # test no source archive
            runner = MLRunner(config, retry_intervals=(0.1, 0.2, 0.3))

            with pytest.raises(RuntimeError, match='cannot get hostname'):
                with server:
                    _ = runner.run()

            self.assertEqual(len(server.db), 1)
            doc = list(server.db.values())[0]

            self.assertEqual(doc['status'], 'FAILED')
            self.assertEqual(doc['error']['message'], 'cannot get hostname')
            self.assertIn('RuntimeError: cannot get hostname',
                          doc['error']['traceback'])

            log_cnt = get_file_content(
                os.path.join(doc['storage_dir'], 'mlrun.log'))
            self.assertIn(b'RuntimeError: cannot get hostname', log_cnt)

    @slow_test
    @httpretty.activate
    def test_outer_error(self):
        @contextmanager
        def mock_configure_logger(*args, **kwargs):
            raise RuntimeError('cannot configure logger')
            yield

        with TemporaryDirectory() as temp_dir, \
                mock.patch('mltk.mlrunner.configure_logger',
                           mock_configure_logger):
            # prepare for the test server
            output_root = os.path.join(temp_dir, 'output')
            server = MockMLServer(output_root)

            # test copy arg files
            config = MLRunnerConfig(
                server=server.uri,
                args='echo hello',
            )
            config.source.make_archive = False  # test no source archive
            runner = MLRunner(config, retry_intervals=(0.1, 0.2, 0.3))

            with pytest.raises(RuntimeError, match='cannot configure logger'):
                with server:
                    _ = runner.run()

            self.assertEqual(len(server.db), 1)
            doc = list(server.db.values())[0]

            self.assertEqual(doc['status'], 'FAILED')
            self.assertEqual(doc['error']['message'], 'cannot configure logger')
            self.assertIn('RuntimeError: cannot configure logger',
                          doc['error']['traceback'])


def make_PatchedMLRunner():
    class PatchedMLRunner(MLRunner):

        exit_code: int = 0
        last_instance: 'PatchedMLRunner' = None

        def __init__(self, config):
            super().__init__(config)
            self.__class__.last_instance = self

        def run(self):
            return self.exit_code

    return PatchedMLRunner


def make_PatchedMLRunnerConfigLoader():
    class PatchedMLRunnerConfigLoader(MLRunnerConfigLoader):

        last_instance: 'PatchedMLRunnerConfigLoader' = None

        def __init__(self, config_files, **kwargs):
            super().__init__(config_files=config_files, **kwargs)
            self.__class__.last_instance = self
            self.loaded = False

        def load_config_files(self, on_load=None):
            super().load_config_files(on_load)
            self.loaded = True

    return PatchedMLRunnerConfigLoader


class MLRunTestCase(unittest.TestCase):

    def test_mlrun(self):
        @config_params(undefined_fields=True)
        class MyMLRunnerConfig(MLRunnerConfig):
            pass

        with TemporaryDirectory() as temp_dir:
            config1 = os.path.join(temp_dir, 'config1.yml')
            config2 = os.path.join(temp_dir, 'config2.yml')
            write_file_content(config1, b'key1: 123')
            write_file_content(config2, b'key2: 456')

            runner = CliRunner()

            # test default arguments
            result = runner.invoke(
                mlrun,
                ['-s', 'http://127.0.0.1:8080', '--print-config',
                 '--', 'echo', 'hello']
            )
            self.assertEqual(result.exit_code, 0)
            self.assertEqual(
                result.output.rstrip(),
                mltk.format_config(
                    MLRunnerConfig(
                        server='http://127.0.0.1:8080',
                        args=['echo', 'hello'],
                        source=MLRunnerConfig.source(),
                        integration=MLRunnerConfig.integration()
                    ),
                    sort_keys=True
                )
            )

            # test various arguments
            with set_environ_context(MLSTORAGE_SERVER_URI='http://127.0.0.1:8080'):
                result = runner.invoke(mlrun, [
                    '-C', config1,
                    '--config-file=' + config2,
                    '-c', 'echo hello',
                    '-e', 'MY_ENV=abc',
                    '--env=MY_ENV2=def',
                    '-g', '1,2',
                    '--gpu=3',
                    '-n', 'test',
                    '--description=testing',
                    '-t', 'first',
                    '--tags=second',
                    '--daemon=echo hello',
                    '-D', 'echo hi',
                    '--tensorboard',
                    '--no-source-archive',
                    '--parse-stdout',
                    '--watch-files',
                    '--copy-source',
                    '--resume-from=xyzz',
                    '--clone-from=zyxx',
                    '--print-config',
                ])
                self.assertEqual(result.exit_code, 0)
                self.assertEqual(
                    result.output.rstrip(),
                    mltk.format_config(
                        MyMLRunnerConfig(
                            server='http://127.0.0.1:8080',
                            args='echo hello',
                            env={'MY_ENV': 'abc', 'MY_ENV2': 'def'},
                            gpu=[1, 2, 3],
                            name='test',
                            description='testing',
                            tags=['first', 'second'],
                            daemon=[
                                'echo hello',
                                'echo hi',
                                'tensorboard --logdir=. --port=0 --host=0.0.0.0',
                            ],
                            source=MLRunnerConfig.source(
                                copy_to_dst=True,
                                make_archive=False,
                            ),
                            integration=MLRunnerConfig.integration(
                                parse_stdout=True,
                                watch_json_files=True,
                            ),
                            resume_from='xyzz',
                            clone_from='zyxx',
                            key1=123,
                            key2=456,
                        ),
                        sort_keys=True
                    )
                )

    def test_legacy_args(self):
        with TemporaryDirectory() as temp_dir:
            runner = CliRunner()

            # test default arguments
            result = runner.invoke(mlrun, [
                '-s', 'http://127.0.0.1:8080',
                '--legacy',
                '--print-config',
                '--',
                'echo', 'hello',
            ])
            self.assertEqual(result.exit_code, 0)
            self.assertEqual(
                result.output.rstrip(),
                mltk.format_config(
                    MLRunnerConfig(
                        source=MLRunnerConfig.source(),
                        integration=MLRunnerConfig.integration(
                            parse_stdout=True,
                            watch_json_files=True,
                        ),
                        server='http://127.0.0.1:8080',
                        args=['echo', 'hello'],
                    ),
                    sort_keys=True,
                )
            )


class ProgramHostTestCase(unittest.TestCase):

    @slow_test
    def test_run(self):
        def run_and_get_output(*args, **kwargs):
            with TemporaryDirectory() as temp_dir:
                log_file = os.path.join(temp_dir, 'log.txt')
                kwargs.setdefault('log_to_stdout', False)
                kwargs.setdefault('log_file', log_file)
                host = ProgramHost(*args, **kwargs)
                code = host.run()
                if os.path.isfile(log_file):
                    output = get_file_content(log_file)
                else:
                    output = None
                return code, output

        # test exit code
        host = ProgramHost('exit 123', log_to_stdout=False)
        self.assertEqual(host.run(), 123)

        host = ProgramHost(['sh', '-c', 'exit 123'],
                           log_to_stdout=False)
        self.assertEqual(host.run(), 123)

        # test environment dict
        code, output = run_and_get_output(
            'env',
            env={
                'MY_ENV_VAR': 'hello',
                b'MY_ENV_VAR_2': b'hi',
            },
        )
        self.assertEqual(code, 0)
        self.assertIn(b'MY_ENV_VAR=hello\n', output)
        self.assertIn(b'MY_ENV_VAR_2=hi\n', output)

        # test work dir
        with TemporaryDirectory() as temp_dir:
            temp_dir = os.path.realpath(temp_dir)
            code, output = run_and_get_output('pwd', work_dir=temp_dir)
            self.assertEqual(code, 0)
            self.assertEqual(output, temp_dir.encode('utf-8') + b'\n')

        # test stdout
        with TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, 'log.txt')
            fd = os.open(log_file, os.O_CREAT | os.O_WRONLY | os.O_TRUNC)
            stdout_fd = sys.stdout.fileno()
            stdout_fd2 = None

            try:
                sys.stdout.flush()
                stdout_fd2 = os.dup(stdout_fd)
                os.dup2(fd, stdout_fd)

                # run the program
                code, output = run_and_get_output(
                    'echo "hello"', log_to_stdout=True)
                self.assertEqual(code, 0)
                self.assertEqual(output, b'hello\n')
                self.assertEqual(get_file_content(log_file), output)
            finally:
                if stdout_fd2 is not None:
                    os.dup2(stdout_fd2, stdout_fd)
                    os.close(stdout_fd2)

        # test log parser
        class MyParser(ProgramOutputReceiver):
            def __init__(self):
                super().__init__([])

            def start(self):
                super().start()
                logs.append('start')

            def put_output(self, data: bytes):
                logs.append(data)

            def stop(self):
                super().stop()
                logs.append('flush')

        logs = []
        code, output = run_and_get_output(
            [
                sys.executable, '-c',
                r'import sys, time; '
                r'sys.stdout.write("hello\n"); '
                r'sys.stdout.flush(); '
                r'time.sleep(0.1); '
                r'sys.stdout.write("world\n")'
            ],
            log_receiver=MyParser()
        )
        self.assertEqual(code, 0)
        self.assertEqual(output, b'hello\nworld\n')
        self.assertListEqual(logs, ['start', b'hello\n', b'world\n', 'flush'])

        # test log parser with error
        class MyParser(ProgramOutputReceiver):
            def __init__(self):
                super().__init__([])

            def start(self):
                super().start()
                logs.append('start')

            def put_output(self, content: bytes):
                logs.append(content)
                raise RuntimeError('some error occurred')

            def stop(self):
                super().stop()
                logs.append('flush')
                raise RuntimeError('some error occurred')

        logs = []
        code, output = run_and_get_output(
            [
                sys.executable, '-c',
                r'import sys, time; '
                r'sys.stdout.write("hello\n"); '
                r'sys.stdout.flush(); '
                r'time.sleep(0.1); '
                r'sys.stdout.write("world\n")'
            ],
            log_receiver=MyParser()
        )
        self.assertEqual(code, 0)
        self.assertEqual(output, b'hello\nworld\n')
        self.assertListEqual(logs, ['start', b'hello\n', b'world\n', 'flush'])

        # test log file
        with TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, 'log.txt')

            # test append
            code, output = run_and_get_output('echo hello', log_file=log_file)
            self.assertEqual(code, 0)
            code, output = run_and_get_output('echo hi', log_file=log_file)
            self.assertEqual(code, 0)
            self.assertEqual(get_file_content(log_file), b'hello\nhi\n')

            # test no append
            code, output = run_and_get_output(
                'echo hey', log_file=log_file, append_to_file=False)
            self.assertEqual(code, 0)
            self.assertEqual(get_file_content(log_file), b'hey\n')

            # test fileno
            log_fileno = os.open(
                log_file, os.O_CREAT | os.O_TRUNC | os.O_WRONLY)
            try:
                code, output = run_and_get_output(
                    'echo goodbye', log_file=log_fileno)
                self.assertEqual(code, 0)
            finally:
                os.close(log_fileno)
            self.assertEqual(get_file_content(log_file), b'goodbye\n')

    @slow_test
    def test_kill(self):
        with TemporaryDirectory() as temp_dir:
            # test kill by SIGINT
            host = ProgramHost(
                [
                    sys.executable,
                    '-u', '-c',
                    'import time, sys\n'
                    'for i in range(100):\n'
                    '  sys.stdout.write(str(i) + "\\n")\n'
                    '  sys.stdout.flush()\n'
                    '  time.sleep(0.1)\n'
                ],
                log_to_stdout=False,
            )

            with host.exec_proc() as proc:
                time.sleep(0.5)
                start_time = time.time()
                host.kill(ctrl_c_timeout=3)
                stop_time = time.time()

            host.kill()
            _ = proc.wait()
            # self.assertNotEqual(code, 0)
            self.assertLess(abs(stop_time - start_time), 0.5)  # that is, 20% difference of time

            # test kill by SIGKILL
            log_file = os.path.join(temp_dir, 'console.log')
            host = ProgramHost(
                [
                    sys.executable,
                    '-u', '-c',
                    'import sys, time\n'
                    'while True:\n'
                    '  try:\n'
                    '    time.sleep(0.1)\n'
                    '  except KeyboardInterrupt:\n'
                    '    sys.stdout.write("kbd interrupt\\n")\n'
                    '    sys.stdout.flush()\n'
                ],
                log_file=log_file
            )

            with host.exec_proc() as proc:
                time.sleep(0.5)
                start_time = time.time()
                host.kill(ctrl_c_timeout=0.5)
                stop_time = time.time()

            host.kill()
            _ = proc.wait()
            self.assertEqual(get_file_content(log_file), b'kbd interrupt\n')
            self.assertLess(abs(stop_time - start_time - 0.5), 0.1)


class SourceCopierTestCase(unittest.TestCase):

    def test_copier(self):
        includes = MLRunnerConfig.source.includes
        excludes = MLRunnerConfig.source.excludes

        with TemporaryDirectory() as temp_dir:
            # prepare for the source dir
            source_dir = os.path.join(temp_dir, 'src')
            prepare_dir(source_dir, {
                'a.py': b'a.py',
                'b.txt': b'b.txt',
                '.git': {
                    'c.py': b'c.py',
                },
                'dir': {
                    'd.sh': b'd.sh',
                },
                'dir2': {
                    'nested': {
                        'e.sh': b'e.sh'
                    },
                    'f.sh': b'f.sh',
                },
                'override.py': b'source'
            })

            # test copy source
            dest_dir = os.path.join(temp_dir, 'dst')
            prepare_dir(dest_dir, {'override.py': b'dest'})

            copier = SourceCopier(source_dir, dest_dir, includes, excludes)
            copier.clone_dir()
            self.assertEqual(copier.file_count, 4)
            dest_content = dir_snapshot(dest_dir)
            dest_expected = {
                'a.py': b'a.py',
                'dir': {
                    'd.sh': b'd.sh',
                },
                'dir2': {
                    'nested': {
                        'e.sh': b'e.sh'
                    },
                    'f.sh': b'f.sh',
                },
                'override.py': b'dest'
            }
            self.assertDictEqual(dest_content, dest_expected)

            # test pack zip
            zip_file = os.path.join(temp_dir, 'source.zip')
            copier.pack_zip(zip_file)
            zip_content = zip_snapshot(zip_file)
            dest_expected['override.py'] = b'source'
            self.assertDictEqual(zip_content, dest_expected)

            # test cleanup
            write_file_content(
                os.path.join(dest_dir, 'dir/more.txt'),
                b'more.txt')  # more file
            os.remove(os.path.join(dest_dir, 'dir2/f.sh'))  # fewer file
            copier.cleanup_dir()
            dest_content = dir_snapshot(dest_dir)
            self.assertDictEqual(dest_content, {
                'dir': {
                    'more.txt': b'more.txt'
                },
                'override.py': b'dest'
            })

    def test_copy_args_files(self):
        with TemporaryDirectory() as temp_dir:
            src = os.path.join(temp_dir, 'src')
            dst1 = os.path.join(temp_dir, 'dst1')
            dst2 = os.path.join(temp_dir, 'dst2')
            prepare_dir(temp_dir, {'e.sh': b'e.sh'})

            # prepare for the source dir
            prepare_dir(src, {
                'a.py': b'a.py',
                'a.sh': b'a.sh',
                'a.txt': b'a.txt',
                'nested': {
                    'b.sh': b'b.sh',
                    'b.py': b'b.py',
                },
                '.git': {
                    'c.py': b'c.py'
                },
            })

            # test copy according to command line
            copier = SourceCopier(src, dst1, MLRunnerConfig.source.includes,
                                  MLRunnerConfig.source.excludes)
            copier.copy_args_files(
                'python a.py a.txt nested/b.py ./nested/../nested/b.sh '
                '.git/c.py d.sh ../e.sh'
            )
            self.assertDictEqual(dir_snapshot(dst1), {
                'a.py': b'a.py',
                'nested': {
                    'b.sh': b'b.sh',
                    'b.py': b'b.py',
                }
            })

            # test copy according to args
            copier = SourceCopier(src, dst2, MLRunnerConfig.source.includes,
                                  MLRunnerConfig.source.excludes)
            copier.copy_args_files(
                'python a.py a.txt nested/b.py ./nested/../nested/b.sh '
                '.git/c.py d.sh ../e.sh'.split(' ')
            )
            self.assertDictEqual(dir_snapshot(dst2), {
                'a.py': b'a.py',
                'nested': {
                    'b.sh': b'b.sh',
                    'b.py': b'b.py',
                }
            })


class TemporaryFileCleanerTestCase(unittest.TestCase):

    maxDiff = None

    def test_cleanup(self):
        with TemporaryDirectory() as temp_dir:
            prepare_dir(temp_dir, {
                '.git': {
                    'a.pyc': b'a.pyc'
                },
                '__pycache__': {
                    'b.pyc': b'b.pyc',
                    'g.txt': b'g.txt',
                },
                'nested': {
                    '__pycache__': {
                        'c.pyc': b'c.pyc',
                        'd.pyc': b'd.pyc',
                        'Thumbs.db': b'Thumbs.db',
                        '.DS_Store': b'.DS_Store',
                    },
                    'e.pyc': b'e.pyc',
                },
                'h.DS_Store': b'h.DS_Store'
            })

            cleaner = TemporaryFileCleaner(temp_dir)
            cleaner.cleanup()

            self.assertDictEqual(dir_snapshot(temp_dir), {
                '.git': {
                    'a.pyc': b'a.pyc',
                },
                '__pycache__': {
                    'g.txt': b'g.txt',
                },
                'nested': {},
                'h.DS_Store': b'h.DS_Store'
            })

            # test cleanup non-exist directory
            cleaner = TemporaryFileCleaner(os.path.join(temp_dir, 'non-exist'))
            cleaner.cleanup()


class JsonFileWatcherTestCase(unittest.TestCase):

    @slow_test
    def test_watcher(self):
        with TemporaryDirectory() as temp_dir:
            logs = []
            path_a = os.path.join(temp_dir, 'a.json')
            path_b = os.path.join(temp_dir, 'b.json')
            path_c = os.path.join(temp_dir, 'c.json')
            os.makedirs(os.path.join(temp_dir, 'd.json'))

            watcher = JsonFileWatcher(
                root_dir=temp_dir,
                file_names=['a.json', 'b.json', 'd.json'],
                interval=0.1,
            )

            def on_json_updated(*args):
                logs.append(args)

            def raise_error(*args):
                raise RuntimeError('raised error')

            watcher.on_json_updated.do(on_json_updated)
            watcher.on_json_updated.do(raise_error)

            with watcher:
                write_file_content(path_a, b'{"a": 1}')
                write_file_content(path_c, b'{"c": 3}')
                time.sleep(0.12)
                write_file_content(path_b, b'{"b": 2}')
                time.sleep(0.12)
                self.assertListEqual(logs, [
                    ('a.json', {'a': 1}), ('b.json', {'b': 2})
                ])

                write_file_content(path_a, b'{"a": 4}')
                time.sleep(0.12)
                self.assertListEqual(logs, [
                    ('a.json', {'a': 1}), ('b.json', {'b': 2}),
                    ('a.json', {'a': 4})
                ])

            self.assertListEqual(logs, [
                ('a.json', {'a': 1}), ('b.json', {'b': 2}),
                ('a.json', {'a': 4}),
                # the forced, final check
                ('a.json', {'a': 4}), ('b.json', {'b': 2})
            ])


class ControlPortServerTestCase(unittest.TestCase):

    @slow_test
    def test_server(self):
        server = ControlServer('127.0.0.1', 12379)
        self.assertEqual(server.uri, 'http://127.0.0.1:12379')

        server = ControlServer('127.0.0.1')
        logs = []

        with server.run_in_background():
            time.sleep(0.5)

            # test not found
            r = requests.get(server.uri)
            self.assertEqual(r.status_code, 404)

            r = requests.post(server.uri, data=b'')
            self.assertEqual(r.status_code, 404)

            # test kill
            server.on_kill.do(lambda: logs.append('on kill'))
            self.assertListEqual(logs, [])
            r = requests.post(server.uri + '/kill', json={})
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.content, b'{}')
            self.assertEqual(r.headers['Content-Type'].split(';', 1)[0],
                             'application/json')
            self.assertListEqual(logs, ['on kill'])

            # test kill but error
            def on_kill_error():
                raise RuntimeError('error on kill')

            server.on_kill.do(on_kill_error)
            r = requests.post(server.uri + '/kill', json={})
            self.assertEqual(r.status_code, 500)
