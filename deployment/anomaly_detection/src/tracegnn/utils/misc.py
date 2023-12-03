import re
import sys
from contextlib import contextmanager
from tempfile import TemporaryDirectory
from typing import *
from urllib.request import urlretrieve

import os

__all__ = [
    'abspath_relative_to_file',
    'fake_tqdm',
    'ensure_parent_exists',
    'as_local_file',
    'trace_id_to_str',
    'str_to_trace_id'
]


def trace_id_to_str(trace_id: Tuple[int, int]) -> str:
    trace_id_high = trace_id[0] % (1 << 64)
    trace_id_low = trace_id[1] % (1 << 64)

    return int.to_bytes((trace_id_high << 64) + trace_id_low, length=16, byteorder='little').hex()


def str_to_trace_id(trace_id_str: str) -> Tuple[int, int]:
    trace_id = int.from_bytes(bytes.fromhex(trace_id_str), byteorder='little')

    return trace_id // (1 << 64), trace_id % (1 << 64)


def abspath_relative_to_file(path, file_path):
    return os.path.join(
        os.path.split(os.path.abspath(file_path))[0],
        path
    )


def fake_tqdm(data, *args, **kwargs):
    yield from data


def ensure_parent_exists(path):
    if path is not None:
        path = os.path.abspath(path)
        parent_dir = os.path.split(path)[0]
        if not os.path.isdir(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)
        return path


@contextmanager
def as_local_file(uri: str) -> ContextManager[str]:
    if re.match(r'^https?://', uri):
        m = re.match(r'^(https?://[^/]+)/([a-z0-9]{24})/(.*)?$', uri)
        if m:
            uri = f'{m.group(1)}/v1/_getfile/{m.group(2)}'
            if m.group(3):
                uri += f'/{m.group(3)}'
        with TemporaryDirectory() as temp_dir:
            filename = os.path.join(temp_dir, uri.rstrip('/').rsplit('/', 1)[-1])
            print(f'Download: {uri}', file=sys.stderr)
            urlretrieve(uri, filename=filename)
            yield filename
    elif uri.startswith('file://'):
        yield uri[7:]
    else:
        yield uri
