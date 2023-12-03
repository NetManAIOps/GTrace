from typing import *

import mltk
import torch
from dgl.data import DGLDataset

from tracegnn.data import *
from .config import ExpConfig
from loguru import logger
from tqdm import tqdm
import numpy as np

import os


def init_config(config: ExpConfig):
    processed_dir = os.path.join(config.dataset_root_dir, config.dataset, 'processed')
    id_manager = TraceGraphIDManager(processed_dir)

    # Set DatasetParams
    config.DatasetParams.operation_cnt = id_manager.num_operations
    config.DatasetParams.service_cnt = id_manager.num_services
    config.DatasetParams.status_cnt = id_manager.num_status

    # Set runtime info
    if config.RuntimeInfo.latency_range is None:
        tmp_latency_dict: Dict[int,List[float]] = {}

        config.RuntimeInfo.latency_range = torch.zeros([config.DatasetParams.operation_cnt + 1, 2], dtype=torch.float)
        config.RuntimeInfo.latency_p98 = torch.zeros([config.DatasetParams.operation_cnt + 1], dtype=torch.float)
        
        # TODO: Set default value
        config.RuntimeInfo.latency_range[:, :] = 50.0

        with TraceGraphDB(BytesSqliteDB(os.path.join(processed_dir, 'train'))) as db:
            logger.info('Get latency range...')
            t = db if not config.enable_tqdm else tqdm(db)
            for g in t:
                for _, nd in g.iter_bfs():
                    tmp_latency_dict.setdefault(nd.operation_id, [])
                    tmp_latency_dict[nd.operation_id].append(nd.features.avg_latency)
        for op, vals in tmp_latency_dict.items():
            vals_p99 = np.percentile(vals, 99)
            vals = np.array(vals)
            if np.any(vals < vals_p99):
                vals = vals[vals < vals_p99]

            # Set a minimum value for vals to avoid nan
            
            # TODO: set this
            vals_mean, vals_std = np.mean(vals), max(np.std(vals), 10.0)
            # vals_mean, vals_std = np.mean(vals), np.std(vals)

            config.RuntimeInfo.latency_range[op] = torch.tensor([vals_mean, vals_std])
            config.RuntimeInfo.latency_p98[op] = np.percentile(vals, 98)
        
        config.RuntimeInfo.latency_range = config.RuntimeInfo.latency_range.to(config.device)
        config.RuntimeInfo.latency_p98 = config.RuntimeInfo.latency_p98.to(config.device)


class TrainDataset(DGLDataset):
    def __init__(self, config: ExpConfig, valid=False):
        self.config = config

        # Load id_manager and basic_info
        processed_dir = os.path.join(config.dataset_root_dir, config.dataset, 'processed')
        self.id_manager = TraceGraphIDManager(processed_dir)

        if not valid:
            self.train_db = TraceGraphDB(BytesSqliteDB(os.path.join(processed_dir, 'train')))
        else:
            self.train_db = TraceGraphDB(BytesSqliteDB(os.path.join(processed_dir, 'val')))

        # Set config
        init_config(config)

        # Show info
        logger.info(f'{len(self.train_db)} in {"train" if not valid else "val"} dataset.')

    def process(self):
        pass

    def __len__(self):
        return len(self.train_db)

    def __getitem__(self, index):
        graph: TraceGraph = self.train_db.get(index)
        dgl_graph = graph_to_dgl(graph)

        return dgl_graph


class TestDataset(DGLDataset):
    def __init__(self, config: ExpConfig, test_path: str=None):
        self.config = config

        # Load id_manager and basic_info
        processed_dir = os.path.join(config.dataset_root_dir, config.dataset, 'processed')
        self.id_manager = TraceGraphIDManager(processed_dir)
        test_path = test_path if test_path is not None else os.path.join(processed_dir, config.test_dataset)
        self.test_db = TraceGraphDB(BytesSqliteDB(test_path))

        # Show info
        logger.info(f'{len(self.test_db)} in test dataset.')

    def process(self):
        pass

    def __len__(self):
        return len(self.test_db)

    def __getitem__(self, index):
        graph1: TraceGraph
        graph2: TraceGraph

        graph1, graph2 = self.test_db.get(index)
        dgl_graph1, dgl_graph2 = graph_to_dgl(graph1), graph_to_dgl(graph2)

        # Set label of graph (one-hot label for 0 - drop, 1 - latency)
        graph_label = torch.zeros([2], dtype=torch.bool)
        if graph1.anomaly == 1:
            graph_label[0] = True
        if graph2.anomaly == 2:
            graph_label[1] = True

        return dgl_graph1, dgl_graph2, graph_label


class DetectionDataset(DGLDataset):
    def __init__(self, config: ExpConfig, test_path: str=None):
        self.config = config

        # Load id_manager and basic_info
        processed_dir = os.path.join(config.dataset_root_dir, config.dataset, 'processed')
        self.id_manager = TraceGraphIDManager(processed_dir)
        test_path = test_path if test_path is not None else os.path.join(processed_dir, config.test_dataset)
        self.test_db = TraceGraphDB(BytesSqliteDB(test_path))

        # Get valid list
        self.valid_list: List[int] = []

        g: TraceGraph
        for i, g in enumerate(tqdm(self.test_db, desc='Get Valid List')):
            for _, nd in g.iter_bfs():
                if nd.operation_id >= config.DatasetParams.operation_cnt or \
                   nd.service_id >= config.DatasetParams.service_cnt or \
                   nd.status_id >= config.DatasetParams.status_cnt:
                   break
            else:
                self.valid_list.append(i)

    def process(self):
        pass

    def __len__(self):
        return len(self.valid_list)

    def __getitem__(self, index):
        graph: TraceGraph

        index = self.valid_list[index]
        graph = self.test_db.get(index)
        dgl_graph = graph_to_dgl(graph)

        return dgl_graph, torch.tensor(graph.trace_id[0], dtype=torch.int64), torch.tensor(graph.trace_id[1], dtype=torch.int64)
