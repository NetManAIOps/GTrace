"""Wraps a BytesDB into TraceGraphDB."""
import os
import pickle as pkl
import re
from contextlib import contextmanager
from typing import *

import numpy as np

from .bytes_db import *
from .trace_graph import *

__all__ = ['TraceGraphDB', 'open_trace_graph_db']


class TraceGraphDB(object):

    bytes_db: BytesDB
    protocol: int

    def __init__(self, bytes_db: BytesDB, protocol: Optional[int] = None):
        if protocol is None:
            protocol = pkl.DEFAULT_PROTOCOL
        self.bytes_db = bytes_db
        self.protocol = protocol

    def __enter__(self):
        self.bytes_db.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.bytes_db.__exit__(exc_type, exc_val, exc_tb)

    def __len__(self) -> int:
        return self.data_count()

    def __getitem__(self, item: int):
        return self.get(item)

    def __iter__(self):
        for i in range(self.data_count()):
            yield self.get(i)

    def __repr__(self):
        desc = repr(self.bytes_db)
        desc = desc[desc.find('(') + 1: -1]
        return f'TraceGraphDB({desc})'

    def sample_n(self,
                 n: int,
                 with_id: bool = False
                 ) -> List[Union[TraceGraph, Tuple[int, TraceGraph]]]:
        ret = []
        indices = np.random.randint(self.data_count(), size=n)
        for i in indices:
            g = self.get(i)
            if with_id:
                ret.append((int(i), g))
            else:
                ret.append(g)
        return ret

    def data_count(self) -> int:
        return self.bytes_db.data_count()

    def get(self, item: int) -> TraceGraph:
        return TraceGraph.from_bytes(self.bytes_db.get(item))

    def set(self, item: int, g: TraceGraph):
        self.bytes_db.set(item, g.to_bytes())

    def add(self, g: TraceGraph) -> int:
        return self.bytes_db.add(g.to_bytes(protocol=self.protocol))

    @contextmanager
    def write_batch(self):
        with self.bytes_db.write_batch():
            yield self

    def commit(self):
        self.bytes_db.commit()

    def optimize(self):
        self.bytes_db.optimize()

    def close(self):
        self.bytes_db.close()


def open_trace_graph_db(input_dir: str,
                        root_dir: Optional[str] = None,
                        names: Optional[Sequence[str]] = (),
                        protocol: Optional[int] = None,
                        ) -> Tuple[TraceGraphDB, TraceGraphIDManager]:
    """
    Open a single ByteDB or a ByteMultiDB as TraceGraphDB, according to the location of `service_id.yml`.

    :param input_dir: The TraceGraphDB directory, or the parent directory of TraceGraphDB.
    :param root_dir: The directory where there is `service_id.yml`.
    :param names: If `input_dir` is the parent of TraceGraphDB directories, specify this can filter the DBs.
    :param protocol: If `protocol` is not None, then use "_bytes_{protocol}.db" as the db file name.
    """
    file_name = f'_bytes_{protocol}.db' if protocol else '_bytes.db'

    if names:
        names = sum([n.split(',') for n in names], [])

    if os.path.exists(os.path.join(input_dir, 'service_id.yml')):
        root_dir = input_dir
        if names:
            input_dirs = [os.path.join(input_dir, name) for name in names]
            for n, p in zip(names, input_dirs):
                if not os.path.exists(p):
                    raise IOError(f'DB {n!r} not exist: {p}')
        else:
            input_dirs = [
                os.path.join(input_dir, name)
                for name in os.listdir(input_dir)
                if re.match(r'^(\d{4}-\d{2}-\d{2}|train|val|drop.*|l?test.*)?$', name)
            ]
    else:
        input_dirs = [input_dir]

    if root_dir is None:
        root_dir = os.path.split(input_dir)[0]
        if not os.path.exists(os.path.join(root_dir, 'service_id.yml')):
            raise ValueError(f'`{root_dir}` is not a TraceGraphDB directory.')

    id_manager = TraceGraphIDManager(root_dir)
    if len(input_dirs) == 1:
        db = TraceGraphDB(
            BytesSqliteDB(input_dirs[0], file_name=file_name),
            protocol=protocol,
        )
    elif input_dirs:
        db = TraceGraphDB(
            BytesMultiDB(*[
                BytesSqliteDB(path, file_name=file_name)
                for path in input_dirs
            ]),
            protocol=protocol,
        )
    else:
        raise ValueError(f'No TraceGraphDB under `{input_dir}`')

    return db, id_manager
