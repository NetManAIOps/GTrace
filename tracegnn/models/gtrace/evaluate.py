from typing import *

from tracegnn.models.gtrace.models.level_model import LevelModel
from .config import ExpConfig

import mltk
import dgl
from loguru import logger
import dgl.dataloading
import torch
import torch.backends.cudnn
import torch.nn.functional as F
from tqdm import tqdm
import pickle
import numpy as np
import multiprocessing as mp
import os

from tracegnn.data import *
from tracegnn.utils.analyze_nll import analyze_anomaly_nll
from .utils import dgl_graph_key
from .models.level_model import calculate_nll



def evaluate(config: ExpConfig,
          dataloader: dgl.dataloading.GraphDataLoader, 
          model: LevelModel):
    device = config.device
    n_z = config.Model.n_z

    # Train model
    logger.info('Start Evaluation with nll...')
    model.eval()

    nll_list = []
    label_list = []

    latency_nll_list = []
    structure_nll_list = []
    graph_label_list = []

    # sample_label for debugging
    node_sample_label_list = []
    graph_sample_label_list = []

    nll_with_nodes = {}

    with torch.no_grad():
        t = tqdm(dataloader) if config.enable_tqdm else dataloader
        test_graphs: dgl.DGLGraph
        label_graphs: dgl.DGLGraph
        graph_labels: torch.Tensor


        for label_graphs, test_graphs, graph_labels in t:
            # Empty cache first
            if 'cuda' in config.device:
                torch.cuda.empty_cache()

            test_graphs = test_graphs.to(device)
            label_graphs = label_graphs.to(device)
            graph_labels = graph_labels.to(device)

            pred = model(test_graphs, n_z=n_z)
            
            # Calculate nll
            nll_structure, nll_latency = calculate_nll(config, pred, test_graphs)
            test_graphs.ndata['nll_latency'] = nll_latency

            # Unbatch graph batches for evaluation
            test_graph_list: List[dgl.DGLGraph] = dgl.unbatch(test_graphs)

            # Iterate to get latency anomaly degree
            for i in range(test_graphs.batch_size):
                graph_key = dgl_graph_key(test_graph_list[i])

                # Set node level evaluation data  
                nll_list.extend(test_graph_list[i].ndata['nll_latency'].tolist())
                label_list.extend(test_graph_list[i].ndata['anomaly'].tolist())
                node_sample_label_list.extend([graph_key] * test_graph_list[i].num_nodes())
                graph_sample_label_list.append(graph_key)

                # Set graph level evaluation data
                if graph_labels[i, 0]:
                    graph_label_list.append(1)
                elif graph_labels[i, 1]:
                    graph_label_list.append(2)
                else:
                    graph_label_list.append(0)
                    nll_with_nodes.setdefault(test_graph_list[i].num_nodes(), [])
                    nll_with_nodes[test_graph_list[i].num_nodes()].append(nll_structure[i].item())

                # The latency nll of the whole graph is the mean of the node degrees
                # graph_latency_nll = test_graph_list[i].ndata['nll_latency'].mean().item()
                graph_latency_nll = test_graph_list[i].ndata['nll_latency'].max().item()

                latency_nll_list.append(graph_latency_nll)
                structure_nll_list.append(nll_structure[i].item())

        # Set evaluation output
        logger.info('--------------------Node Level-----------------------')
        # Get node level result
        node_result = analyze_anomaly_nll(
            nll_list=np.array(nll_list, dtype=np.float32),
            label_list=np.array(label_list, dtype=np.int64),
            threshold=np.percentile(nll_list, 98),
            sample_label_list=node_sample_label_list
        )
        logger.info(node_result)

        logger.info('-------------------Graph Level Latency-----------------------')
        # Get graph level result
        latency_mask = (np.array(graph_label_list, dtype=np.int64) != 1)
        latency_result = analyze_anomaly_nll(
            nll_list=np.array(latency_nll_list, dtype=np.float32)[latency_mask],
            label_list=np.array(graph_label_list, dtype=np.int64)[latency_mask],
            threshold=np.percentile(latency_nll_list, 98)
        )
        logger.info(latency_result)

        logger.info('-------------------Graph Level Structure-----------------------')
        # Get graph level result
        structure_mask = (np.array(graph_label_list, dtype=np.int64) != 2)
        structure_result = analyze_anomaly_nll(
            nll_list=np.array(structure_nll_list, dtype=np.float32)[structure_mask],
            label_list=np.array(graph_label_list, dtype=np.int64)[structure_mask],
            threshold=np.percentile(structure_nll_list, 98)
        )
        logger.info(structure_result)

    model.train()
