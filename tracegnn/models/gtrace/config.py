import mltk
import torch
from typing import *


# Define ExpConfig for configuration and runtime information
class ExpConfig(mltk.Config):
    # Base Train Info
    device: str = 'cpu'
    dataset: str = 'dataset_b'
    test_dataset: str = 'test'
    seed: int = 1234

    batch_size: int = 128
    test_batch_size: int = 64
    max_epochs: int = 80
    enable_tqdm: bool = True

    # Dataset Info
    dataset_root_dir: str = 'dataset'

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
        latency_input: bool = False
        embedding_size: int = 4
        graph_embedding_size: int = 4
        decoder_feature_size: int = 4
        latency_feature_size: int = 4
        latency_gcn_layers: int = 5


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
