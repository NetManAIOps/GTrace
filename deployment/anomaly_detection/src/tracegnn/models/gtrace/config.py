import mltk
import torch
from typing import *


# Presets
model_presets = {
    'large': {
        'embedding_size': 32,
        'graph_embedding_size': 64,
        'decoder_feature_size': 32,
        'latency_feature_size': 16,
        'latency_gcn_layers': 5
    },
    'small': {
        'embedding_size': 4,
        'graph_embedding_size': 4,
        'decoder_feature_size': 4,
        'latency_feature_size': 4,
        'latency_gcn_layers': 5
    }
}


# Define ExpConfig for configuration and runtime information
class ExpConfig(mltk.Config):
    # Base Train Info
    device: str = 'cuda:0'
    dataset: str = 'dataset_b'
    test_dataset: str = 'test'
    seed: int = 1234

    batch_size: int = 128
    test_batch_size: int = 32
    max_epochs: int = 50

    enable_tqdm: bool = True

    # Dataset Info
    dataset_root_dir: str = '/srv/data/'

    # Model info
    class Latency(mltk.Config):
        embedding_type: str = 'normal'
        latency_feature_length: int = 1
        latency_embedding: int = 10
        latency_max_value: float = 50.0

    class Model(mltk.Config):
        vae: bool = True
        anneal: bool = False
        kl_weight: float = 1e-2
        n_z: int = 5

        latency_model: str = 'tree' # tree / gat
        structure_model: str = 'tree' # tree / gcn
        model_type: str = 'large'
        embedding_size: int = model_presets[model_type]['embedding_size']
        graph_embedding_size: int = model_presets[model_type]['graph_embedding_size']
        decoder_feature_size: int = model_presets[model_type]['decoder_feature_size']
        latency_feature_size: int = model_presets[model_type]['latency_feature_size']
        latency_gcn_layers: int = model_presets[model_type]['latency_gcn_layers']


    decoder_max_nodes: int = 100

    # Dataset Params
    class RuntimeInfo:
        # Operation latency range (op -> (mean, std))
        latency_range: torch.Tensor = None
        latency_p98: torch.Tensor = None

    class DatasetParams:
        # Basic Dataset Info
        operation_cnt: int = None
        service_cnt: int = None
        status_cnt: int = None

    class NllInfo:
        node_nll_p99: float = None
        graph_latency_nll_p99: float = None
        graph_structure_nll_p99: float = None
