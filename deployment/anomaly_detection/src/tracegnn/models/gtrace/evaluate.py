from typing import *
from tracegnn.models.gtrace.models.level_model import LevelModel, calculate_nll, log_exp_mean, normal_loss, calculate_node_nll
from tracegnn.models.gtrace.config import ExpConfig
from tracegnn.models.gtrace.models.loss_func_np import normal_loss_np, log_exp_mean_np

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
from lru import LRU
import time
import os

from tracegnn.data import *
from tracegnn.utils.analyze_nll import analyze_anomaly_nll
from tracegnn.models.gtrace.utils import dgl_graph_key
from tracegnn.models.gtrace.models.latency_embedding import latency_to_feature, feature_to_latency

class Evaluator:
    def __init__(self, config: ExpConfig, model: LevelModel, n_z: int=5, cache_items=(1 << 18) + 1000, device='cpu'):
        self.cache_items = cache_items
        self.config = config
        self.model = model.to(device)
        self.device = device

        self.structure_sum_cache: torch.Tensor = torch.zeros([cache_items, 1, model.encoder.tree_lstm.x_size * 3], dtype=torch.float, device=self.device)
        self.structure_c_cache: torch.Tensor = torch.zeros([cache_items, 1, model.encoder.tree_lstm.x_size], dtype=torch.float, device=self.device)

        self.latency_sum_cache = torch.zeros([cache_items, n_z, model.latency_decoder.tree_lstm.x_size * 3], dtype=torch.float, device=self.device)
        self.latency_c_cache = torch.zeros([cache_items, n_z, model.latency_decoder.tree_lstm.x_size], dtype=torch.float, device=self.device)
        self.log_p_q_cache = torch.zeros([cache_items, n_z], device=self.device)
        self.latency_mu_cache = torch.zeros([cache_items, n_z, 1], device=self.device)
        self.latency_logvar_cache = torch.zeros([cache_items, n_z, 1], device=self.device)

        self.z_graph_mu_cache = torch.zeros([cache_items, config.Model.graph_embedding_size], device=self.device)
        self.z_graph_logvar_cache = torch.zeros([cache_items, config.Model.graph_embedding_size], device=self.device)

        self.graph_cache = LRU(cache_items)

    def get_batch_nll(self,
                      results: Tuple[dgl.DGLGraph, torch.Tensor, set, torch.Tensor, torch.Tensor, dict]):
        config = self.config
        # Node-level cache
        all_keys, created_keys, node_cnt, node_hit = self.__node_model(results, self.model)

        # Get NLL with cache
        graph_cnt, graph_hit = self.__cache_batch_nll(results, all_keys, created_keys, self.model)
        return node_cnt, node_hit, graph_cnt, graph_hit

        
    def __cache_batch_nll(self,
                          results: Tuple[dgl.DGLGraph, torch.Tensor, set, torch.Tensor, torch.Tensor, dict],
                          all_keys: torch.Tensor,
                          created_keys: Set[int],
                          model: LevelModel):
        config = self.config
        n_z = config.Model.n_z

        batch_graphs = results[-1]
        
        # Get graph labels
        latency_label: torch.Tensor = latency_to_feature(config, batch_graphs['latency'], batch_graphs['operation_id']).unsqueeze(1).expand([-1,config.Model.n_z,-1])
        
        # Iterate all graphs
        node_latency_nll: List[List[float]] = []
        graph_latency_nll: List[float] = []
        graph_structure_nll: List[float] = []

        num_graphs = batch_graphs['batch_size']
        batch_num_nodes = batch_graphs['batch_num_nodes']
        batch_num_nodes_np = batch_graphs['batch_num_nodes'].cpu().numpy()

        # Latency NLL (With node cache)
        latency_mu = self.latency_mu_cache[all_keys]
        latency_logvar = self.latency_logvar_cache[all_keys]
        log_p_q = self.log_p_q_cache[all_keys]
        log_p_x2 = normal_loss(latency_label, latency_mu, latency_logvar)
        cur_latency_nll = -torch.mean((log_p_q + log_p_x2))


        # Graph NLL (With graph cache)
        node_idx: int = 0
        graph_hit_cnt: int = 0
        nodes_prefix_sum = torch.zeros_like(batch_num_nodes)
        nodes_prefix_sum[1:] = torch.cumsum(batch_num_nodes[:-1], dim=0)
        graph_keys = batch_graphs['node_hash'][nodes_prefix_sum].detach().cpu().numpy()

        # Data for no-cache inference
        infer_graph_key = []
        infer_i = []
        infer_z_graph_mu = []
        infer_z_graph_logvar = []
        infer_operation_id = []
        infer_status_id = []

        for i in range(num_graphs):
            num_nodes = batch_num_nodes_np[i]
            cur_idx = all_keys[node_idx:node_idx + num_nodes]
            cur_graph_key = graph_keys[i]

            # Structure
            # Check if in cache
            if cur_graph_key in self.graph_cache:
                graph_hit_cnt += 1
            else:
                infer_graph_key.append(cur_graph_key)
                infer_i.append(i)

                # Get encoded
                z_graph_mu = self.z_graph_mu_cache[cur_idx]
                z_graph_logvar = self.z_graph_logvar_cache[cur_idx]
                
                z_graph_mu = torch.mean(z_graph_mu, dim=0)
                z_graph_logvar = torch.tanh(torch.mean(z_graph_logvar, dim=0))
                infer_z_graph_mu.append(z_graph_mu)
                infer_z_graph_logvar.append(z_graph_logvar)

                infer_operation_id.append(batch_graphs['operation_id'][node_idx:node_idx+batch_num_nodes[i]])
                infer_status_id.append(batch_graphs['status_id'][node_idx:node_idx+batch_num_nodes[i]])

            node_idx += batch_num_nodes[i]

        # No-cache inference
        batch_size = 32
        for step in range(len(infer_i) // batch_size + 1):
            s, e = step * batch_size, min((step + 1) * batch_size, len(infer_i))
            if s == e:
                break
            # Process data
            tensor_z_graph_mu = torch.stack(infer_z_graph_mu[s:e], dim=0)
            tensor_z_graph_logvar = torch.stack(infer_z_graph_logvar[s:e], dim=0)
            tensor_operation_id = torch.cat(infer_operation_id[s:e], dim=0)
            tensor_status_id = torch.cat(infer_status_id[s:e], dim=0)
            tensor_i = torch.tensor(infer_i[s:e], device=self.config.device)

            # Run model
            with torch.no_grad():
                pred = model.graph_infer(tensor_z_graph_mu, tensor_z_graph_logvar, n_z)
                node_batchs = batch_num_nodes[tensor_i]
                nll = calculate_nll(config, pred, tensor_operation_id, tensor_status_id, node_batchs, tensor_z_graph_mu, tensor_z_graph_logvar).detach()

            # Store into cache
            nll = nll.detach().cpu().numpy()

            for i in range(len(infer_graph_key[s:e])):
                self.graph_cache[infer_graph_key[s:e][i]] = {}
                self.graph_cache[infer_graph_key[s:e][i]]['nll'] = nll[i]

        print(f"---> graph hit: {graph_hit_cnt} / {num_graphs}")
        return num_graphs, graph_hit_cnt


    def __node_model(self,
                    results: Tuple[dgl.DGLGraph, torch.Tensor, set, torch.Tensor, torch.Tensor, dict],
                    model: LevelModel):
        config = self.config
        # Load cal graph
        cal_graph, all_keys, created_keys, cal_graph_keys, data_idx, batch_graphs = results

        num_nodes = cal_graph.num_nodes()
        if batch_graphs['node_size'] == 0:
            return [], [], len(all_keys), len(all_keys) - num_nodes

        if num_nodes == 0:
            return all_keys, created_keys, len(all_keys), len(all_keys) - num_nodes

        n_z = config.Model.n_z

        structure_sum = torch.zeros([num_nodes, 1, model.encoder.tree_lstm.x_size * 3], dtype=torch.float, device=self.device)
        structure_c = torch.zeros([num_nodes, 1, model.encoder.tree_lstm.x_size], dtype=torch.float, device=self.device)

        latency_sum = torch.zeros([num_nodes, n_z, model.latency_decoder.tree_lstm.x_size * 3], dtype=torch.float, device=self.device)
        latency_c = torch.zeros([num_nodes, n_z, model.latency_decoder.tree_lstm.x_size], dtype=torch.float, device=self.device)

        if len(data_idx) > 0:
            structure_sum[data_idx] = self.structure_sum_cache[cal_graph_keys[data_idx]]
            structure_c[data_idx] = self.structure_c_cache[cal_graph_keys[data_idx]]
            latency_sum[data_idx] = self.latency_sum_cache[cal_graph_keys[data_idx]]
            latency_c[data_idx] = self.latency_c_cache[cal_graph_keys[data_idx]]

        # Store into graph
        cal_graph.ndata['structure_sum'] = structure_sum
        cal_graph.ndata['structure_c'] = structure_c
        cal_graph.ndata['latency_sum'] = latency_sum
        cal_graph.ndata['latency_c'] = latency_c
        
        # Run model inference and get result
        model.eval()

        with torch.no_grad():
            pred = model.node_infer(cal_graph, n_z)
            log_p_q = calculate_node_nll(config, pred, cal_graph)

        print(f"---> Cal graph: {num_nodes} / {len(all_keys)} nodes")

        # Store into cache for created keys
        cal_graph_keys = cal_graph_keys.cpu().numpy()
        for i, node_key in enumerate(cal_graph_keys):
            if node_key not in created_keys: continue
            self.structure_sum_cache[node_key] = cal_graph.ndata['structure_sum'][i].detach()
            self.structure_c_cache[node_key] = cal_graph.ndata['structure_c'][i].detach()
            self.latency_sum_cache[node_key] = cal_graph.ndata['latency_sum'][i].detach()
            self.latency_c_cache[node_key] = cal_graph.ndata['latency_c'][i].detach()
            self.log_p_q_cache[node_key] = log_p_q[i].detach()
            self.latency_mu_cache[node_key] = pred['latency_mu'][i].detach()
            self.latency_logvar_cache[node_key] = pred['latency_logvar'][i].detach()
            self.z_graph_mu_cache[node_key] = pred['z_graph_mu'][i].detach()
            self.z_graph_logvar_cache[node_key] = pred['z_graph_logvar'][i].detach()

        return all_keys, created_keys, len(all_keys), len(all_keys) - num_nodes
