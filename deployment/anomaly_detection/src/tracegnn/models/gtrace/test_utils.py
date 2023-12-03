from urllib.error import HTTPError
from typing import *

import mltk
import torch
import json
from tracegnn.models.gtrace.dataset import init_config

from tracegnn.models.gtrace.models.level_model import LevelModel
from tracegnn.models.gtrace.config import ExpConfig
from tracegnn.utils import *
from tracegnn.data import *
from loguru import logger
import torch


__all__ = [
    'load_model_local',
    'load_model'
]

def load_model_local(prefix: str, device: str='cpu') -> Tuple[LevelModel, ExpConfig, TraceGraphIDManager]:
    logger.info('Loading model and configs from file...')
    # try:
    # Load id_manager
    id_manager = TraceGraphIDManager(f'{prefix}/id_manager')

    # Load config
    config_loader = mltk.ConfigLoader(ExpConfig)
    config_loader.load_file(f'{prefix}/model/config.json')
    config = config_loader.get(discard_undefined=mltk.type_check.DiscardMode.WARN)

    # Patch config
    config.RuntimeInfo.latency_range = torch.load(f'{prefix}/model/latency_range.pth', map_location=device)
    config.RuntimeInfo.latency_p98 = torch.load(f'{prefix}/model/latency_p98.pth', map_location=device)

    # Get nll_p99
    nll_p99 = json.load(open(f'{prefix}/model/nll_p99.json', 'rt'))
    config.NllInfo.node_nll_p99 = nll_p99['node_nll_p99']
    config.NllInfo.graph_latency_nll_p99 = nll_p99['graph_latency_nll_p99']
    config.NllInfo.graph_structure_nll_p99 = nll_p99['graph_structure_nll_p99']

    # Configure DatasetParams
    config.DatasetParams.operation_cnt = id_manager.num_operations
    config.DatasetParams.service_cnt = id_manager.num_services
    config.DatasetParams.status_cnt = id_manager.num_status

    # Load model
    model = LevelModel(config)
    model.load_state_dict(torch.load(
        f'{prefix}/model/model.pth',
        map_location=device
    ))

    config.device = device
    logger.info('Model loaded successfully!')

    return model, config, id_manager


def load_model(device: str='cpu') -> Tuple[LevelModel, ExpConfig, TraceGraphIDManager]:
    logger.info('Loading model and configs from file...')
    # try:
    # Load id_manager
    id_manager = TraceGraphIDManager('/tmp/data/id_manager')

    # Load config
    config_loader = mltk.ConfigLoader(ExpConfig)
    config_loader.load_file('/tmp/data/model/config.json')
    config = config_loader.get(discard_undefined=mltk.type_check.DiscardMode.WARN)

    # Patch config
    config.RuntimeInfo.latency_range = torch.load('/tmp/data/model/latency_range.pth', map_location=device)
    config.RuntimeInfo.latency_p98 = torch.load('/tmp/data/model/latency_p98.pth', map_location=device)

    # Get nll_p99
    nll_p99 = json.load(open('/tmp/data/model/nll_p99.json', 'rt'))
    config.NllInfo.node_nll_p99 = nll_p99['node_nll_p99']
    config.NllInfo.graph_latency_nll_p99 = nll_p99['graph_latency_nll_p99']
    config.NllInfo.graph_structure_nll_p99 = nll_p99['graph_structure_nll_p99']

    # Configure DatasetParams
    config.DatasetParams.operation_cnt = id_manager.num_operations
    config.DatasetParams.service_cnt = id_manager.num_services
    config.DatasetParams.status_cnt = id_manager.num_status

    # Load model
    model = LevelModel(config)
    model.load_state_dict(torch.load(
        '/tmp/data/model/model.pth',
        map_location=device
    ))
    config.device = device
    logger.info('Model loaded successfully!')

    return model, config, id_manager
