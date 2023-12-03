import os
import stat
import unittest
import zipfile
from contextlib import contextmanager
from typing import *

from httpretty import httpretty

from mltk.callbacks import CallbackData
from mltk.utils import NOT_SET

__all__ = [
    'slow_test', 'remote_test', 'pytorch_test',
    'get_file_content', 'write_file_content', 'dir_snapshot',
    'prepare_dir', 'zip_snapshot', 'chdir_context', 'set_environ_context',
    'compute_fs_size_and_inode', 'new_callback_data', 'httpretty_register_uri',
]

FAST = os.environ.get('FAST_TEST', '0') == '1'
LOCAL = os.environ.get('LOCAL_TEST', '0') == '1'
REMOTE = os.environ.get('LOCAL_TEST', '0') == '1'

try:
    import torch
except ImportError:
    HAS_PYTORCH = False
else:
    HAS_PYTORCH = True


def slow_test(method):
    return unittest.skipIf(
        FAST, 'slow tests are skipped in fast test mode'
    )(method)


def remote_test(method):
    return unittest.skipUnless(
        REMOTE,
        'remote tests are skipped unless REMOTE=1'
    )(method)


def pytorch_test(method):
    return unittest.skipUnless(
        HAS_PYTORCH,
        'PyTorch is installed'
    )(method)


def get_file_content(path):
    with open(path, 'rb') as f:
        return f.read()


def write_file_content(path, content):
    with open(path, 'wb') as f:
        f.write(content)


def dir_snapshot(path):
    ret = {}
    for name in os.listdir(path):
        f_path = os.path.join(path, name)
        if os.path.isdir(f_path):
            ret[name] = dir_snapshot(f_path)
        else:
            ret[name] = get_file_content(f_path)
    return ret


def prepare_dir(path, snapshot):
    os.makedirs(path, exist_ok=True)

    for name, value in snapshot.items():
        f_path = os.path.join(path, name)
        if isinstance(value, dict):
            prepare_dir(f_path, value)
        else:
            write_file_content(f_path, value)


def zip_snapshot(path):
    ret = {}

    def put_entry(arcname, cnt):
        t = ret
        segments = arcname.strip('/').split('/')
        for n in segments[:-1]:
            if n not in t:
                t[n] = {}
            t = t[n]

        assert(segments[-1] not in t)
        t[segments[-1]] = cnt

    with zipfile.ZipFile(path, 'r') as zip_file:
        for e in zip_file.infolist():
            if e.filename.endswith('/'):
                put_entry(e.filename, {})
            else:
                put_entry(e.filename, zip_file.read(e.filename))

    return ret


@contextmanager
def chdir_context(path):
    pwd = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(pwd)


@contextmanager
def set_environ_context(key_values: Dict[str, str] = None, **kwargs):
    key_values = dict(key_values or ())
    key_values.update(kwargs)

    old_values = {}

    try:
        for key, value in key_values.items():
            old_values[key] = os.environ.pop(key, NOT_SET)
            if value is not NOT_SET:
                os.environ[key] = value
        yield

    finally:
        for key, value in old_values.items():
            if value is NOT_SET:
                os.environ.pop(key)
            else:
                os.environ[key] = value


def compute_fs_size_and_inode(path):
    try:
        st = os.stat(path)
        if stat.S_ISDIR(st.st_mode):
            size, inode = st.st_size, 1
            for name in os.listdir(path):
                f_path = os.path.join(path, name)
                f_size, f_inode = compute_fs_size_and_inode(f_path)
                size += f_size
                inode += f_inode
            return size, inode
        else:
            return st.st_size, 1
    except IOError:
        return 0, 0


def new_callback_data(**kwargs):
    for field in CallbackData.__slots__:
        kwargs.setdefault(field, None)
    return CallbackData(**kwargs)


def httpretty_register_uri(*args, **kwargs):
    httpretty.register_uri(*args, **kwargs)
