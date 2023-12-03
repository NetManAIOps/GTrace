import torch
import torch.jit
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch.conv import GraphConv, GATConv
from ..config import ExpConfig
from .latency_embedding import latency_to_feature
from .tree_lstm import TreeLSTM
from typing import *


class LevelEmbedding(nn.Module):
    def __init__(self, config: ExpConfig):
        super(LevelEmbedding, self).__init__()
        self.config = config

        # Embeded node type
        self.operation_embed = nn.Embedding(config.DatasetParams.operation_cnt, config.Model.embedding_size)
        self.service_embed = nn.Embedding(config.DatasetParams.service_cnt, config.Model.embedding_size)
        self.status_embed = nn.Embedding(config.DatasetParams.status_cnt, config.Model.embedding_size)

    def forward(self, g: dgl.DGLGraph):
        operation = self.operation_embed(g.ndata['operation_id'])
        service = self.service_embed(g.ndata['service_id'])
        status = self.status_embed(g.ndata['status_id'])

        if self.config.Model.latency_input:
            latency = latency_to_feature(self.config, g.ndata['latency'], g.ndata['operation_id'], clip=False)
            features = torch.cat([operation, service, status, latency], dim=-1)
        else:
            features = torch.cat([operation, service, status], dim=-1)
        return features


class LevelEncoder(nn.Module):
    def __init__(self, config: ExpConfig):
        super(LevelEncoder, self).__init__()
        self.config = config

        # DGL GCN Encoder
        self.embedding = LevelEmbedding(config)
        self.gcn1 = GraphConv(config.Model.embedding_size * 3, config.Model.graph_embedding_size)
        self.gcn2 = GraphConv(config.Model.graph_embedding_size, config.Model.graph_embedding_size * 2)

        self.depth_weight = nn.Embedding(
            num_embeddings=50,
            embedding_dim=1
        )

    def forward(self, g: dgl.DGLGraph, embed: torch.Tensor):
        g_re = dgl.to_bidirected(g.clone().cpu(), copy_ndata=False).to(self.config.device)
        # g_re = g
        # Encode
        out = torch.relu(self.gcn1(g_re, embed))
        g.ndata['node_encode'] = self.gcn2(g_re, out)

        # Readout
        g.ndata['depth_weight'] = self.depth_weight(g.ndata['node_depth'])
        graph_encode = dgl.readout_nodes(g, 'node_encode')#, weight='depth_weight')
        graph_mu, graph_logvar = torch.split(graph_encode, graph_encode.size(-1) // 2, dim=-1)
        graph_logvar = torch.tanh(graph_logvar)

        return graph_mu, graph_logvar


class TreeLSTMLevelEncoder(nn.Module):
    def __init__(self, config: ExpConfig):
        super(TreeLSTMLevelEncoder, self).__init__()
        self.config = config

        # DGL GCN Encoder
        if config.Model.latency_input:
            self.tree_lstm = TreeLSTM(config.Model.embedding_size * 3 + config.Latency.latency_feature_length, config.Model.graph_embedding_size * 2)
        else:
            self.tree_lstm = TreeLSTM(config.Model.embedding_size * 3, config.Model.graph_embedding_size * 2)

    def forward(self, g: dgl.DGLGraph, embed: torch.Tensor):
        g.ndata['node_encode'] = self.tree_lstm(g, embed.unsqueeze(1)).squeeze(1)
        graph_encode = dgl.readout_nodes(g, 'node_encode')
        # batch_num_nodes = g.batch_num_nodes()
        # graph_encode = g.ndata['node_encode'][torch.cumsum(batch_num_nodes, 0)-batch_num_nodes[0]]
        graph_mu, graph_logvar = torch.split(graph_encode, graph_encode.size(-1) // 2, dim=-1)
        graph_logvar = torch.tanh(graph_logvar)

        return graph_mu, graph_logvar


class LevelDecoder(nn.Module):
    def __init__(self, config: ExpConfig):
        super(LevelDecoder, self).__init__()
        self.config = config

        # A single decoder
        self.mlp = nn.Sequential(
            nn.Linear(config.Model.graph_embedding_size, config.Model.graph_embedding_size),
            nn.ReLU(),
            nn.Linear(config.Model.graph_embedding_size, config.Model.graph_embedding_size),
            nn.ReLU(),
            nn.Linear(config.Model.graph_embedding_size, config.decoder_max_nodes * config.Model.decoder_feature_size * 2)
        )

        # Decoded -> Op, Svc, Status, Exist, Leaf
        self.op_decoder = nn.Linear(config.Model.decoder_feature_size, config.DatasetParams.operation_cnt + 1)
        # self.svc_decoder = nn.Linear(config.Model.decoder_feature_size, config.DatasetParams.service_cnt)
        self.status_decoder = nn.Linear(config.Model.decoder_feature_size, config.DatasetParams.status_cnt)

    def forward(self, graph_encode: torch.Tensor):
        decoded = self.mlp(graph_encode).reshape(
            graph_encode.size(0), graph_encode.size(1), self.config.decoder_max_nodes, -1)
        
        # node_type = operation_id & not_exist (the last one)
        node_type = self.op_decoder(decoded[..., :self.config.Model.decoder_feature_size])
        status = self.status_decoder(decoded[..., self.config.Model.decoder_feature_size:self.config.Model.decoder_feature_size * 2])

        return node_type, status


class LevelLatencyEncoder(nn.Module):
    def __init__(self, config: ExpConfig):
        super(LevelLatencyEncoder, self).__init__()
        self.config = config

        self.gcn_input = GATConv(config.Model.latency_feature_size, config.Model.latency_feature_size * 2, 1)
        # self.gcn_input = GraphConv(config.Model.latency_feature_size, config.Model.latency_feature_size * 2, norm='none')
        self.linear_input = nn.Linear(config.Model.embedding_size * 3 # + config.Model.graph_embedding_size
            ,config.Model.latency_feature_size)

    def forward(self, g: dgl.DGLGraph, embed: torch.Tensor):
        # graph_embed_broadcast = dgl.broadcast_nodes(g, graph_embed)
        # x = F.relu(self.linear_input(torch.cat([embed, graph_embed_broadcast], dim=-1)))

        x = F.relu(self.linear_input(embed))

        # g_re = g
        g_re = dgl.to_bidirected(g.clone().cpu(), copy_ndata=False).to(self.config.device)
        # g_re = dgl.reverse(g, copy_ndata=False, copy_edata=False)
        y = self.gcn_input(g_re, x).reshape(x.size(0), -1)
        # y = y / (torch.norm(y, dim=-1, keepdim=True) + 1e-7)

        z_latency_mu, z_latency_logvar = torch.split(y, y.size(-1) // 2, dim=-1)
        z_latency_logvar = torch.tanh(z_latency_logvar)

        return z_latency_mu, z_latency_logvar

    
class LevelLatencyDecoder(nn.Module):
    def __init__(self, config: ExpConfig):
        super(LevelLatencyDecoder, self).__init__()
        self.config = config

        # Aggregate information
        self.gcn_list = nn.ModuleList()
        for i in range(config.Model.latency_gcn_layers):
            self.gcn_list.append(GATConv(config.Model.latency_feature_size, config.Model.latency_feature_size, 1))
            # self.gcn_list.append(GraphConv(config.Model.latency_feature_size, config.Model.latency_feature_size, norm='none'))

        self.linear_output_mu = nn.Linear(config.Model.latency_feature_size, config.Latency.latency_feature_length)
        self.linear_output_logvar = nn.Linear(config.Model.latency_feature_size, config.Latency.latency_feature_length)

        # Add op-wise output layer
        self.op_wise_output = nn.Embedding(
            num_embeddings=config.DatasetParams.operation_cnt + 1,
            embedding_dim=2 * config.Latency.latency_feature_length ** 2 # * config.Latency.latency_feature_length
        )

        # self.dropout = nn.Dropout(0.2)

    def forward(self, g: dgl.DGLGraph, z_latency: torch.Tensor):
        # g_re = g
        g_re = dgl.to_bidirected(g.clone().cpu(), copy_ndata=False).to(self.config.device)
        # g_re = dgl.reverse(g, copy_ndata=False, copy_edata=False)

        y = z_latency
        for layer in self.gcn_list:
            # [num_nodes x n_z x feat]
            y = F.relu(layer(g_re, y).reshape(z_latency.size(0), z_latency.size(1), -1))
            # y = y / (torch.norm(y, dim=-1, keepdim=True) + 1e-7)

        mu = self.linear_output_mu(y)
        logvar = self.linear_output_logvar(y)

        w_mu, w_logvar = torch.split(
            self.op_wise_output(g.ndata['operation_id']).reshape(-1, 1, self.config.Latency.latency_feature_length,
                self.config.Latency.latency_feature_length * 2),
            split_size_or_sections=self.config.Latency.latency_feature_length,
            dim=-1
        )
        # print(f'{w_mu.shape=},{mu.shape=}')
        mu = (mu.unsqueeze(2) @ (w_mu)).squeeze(2)
        logvar = (logvar.unsqueeze(2) @ (w_logvar)).squeeze(2)

        return mu, logvar


class TreeLSTMLatencyEncoder(nn.Module):
    def __init__(self, config: ExpConfig):
        super(TreeLSTMLatencyEncoder, self).__init__()
        self.config = config

        if config.Model.latency_input:
            self.linear_input = nn.Sequential(
                nn.Linear(config.Model.embedding_size * 3 + config.Latency.latency_feature_length, config.Model.latency_feature_size),
                nn.ReLU(),
                nn.Linear(config.Model.latency_feature_size, config.Model.latency_feature_size * 2)
            )
        else:
            self.linear_input = nn.Sequential(
                nn.Linear(config.Model.embedding_size * 3, config.Model.latency_feature_size),
                nn.ReLU(),
                nn.Linear(config.Model.latency_feature_size, config.Model.latency_feature_size * 2)
            )

    def forward(self, g: dgl.DGLGraph, embed: torch.Tensor):
        y = self.linear_input(embed)

        z_latency_mu, z_latency_logvar = torch.split(y, y.size(-1) // 2, dim=-1)
        z_latency_logvar = torch.tanh(z_latency_logvar)

        return z_latency_mu, z_latency_logvar


class TreeLSTMLatencyDecoder(nn.Module):
    def __init__(self, config: ExpConfig):
        super(TreeLSTMLatencyDecoder, self).__init__()
        self.config = config

        # Tree LSTM
        self.tree_lstm = TreeLSTM(config.Model.latency_feature_size, config.Model.latency_feature_size)

        # Mu & Sigma
        self.linear_output_mu = nn.Linear(config.Model.latency_feature_size, config.Latency.latency_feature_length)
        self.linear_output_logvar = nn.Linear(config.Model.latency_feature_size, config.Latency.latency_feature_length)

        # Add op-wise output layer
        self.op_wise_output = nn.Embedding(
            num_embeddings=config.DatasetParams.operation_cnt + 1,
            embedding_dim=2 * config.Latency.latency_feature_length ** 2
        )

    def forward(self, g: dgl.DGLGraph, z_latency: torch.Tensor):
        y = self.tree_lstm(g, z_latency)

        mu = self.linear_output_mu(y)
        logvar = self.linear_output_logvar(y)

        w_mu, w_logvar = torch.split(
            self.op_wise_output(g.ndata['operation_id']).reshape(-1, 1, self.config.Latency.latency_feature_length,
                self.config.Latency.latency_feature_length * 2),
            split_size_or_sections=self.config.Latency.latency_feature_length,
            dim=-1
        )

        mu = (mu.unsqueeze(2) @ (w_mu)).squeeze(2)
        logvar = (logvar.unsqueeze(2) @ (w_logvar)).squeeze(2)

        return mu, logvar


class LevelModel(nn.Module):
    def __init__(self, config: ExpConfig):
        super(LevelModel, self).__init__()
        self.config = config

        # Structure Model
        self.embedding = LevelEmbedding(config)
        
        if self.config.Model.structure_model == 'gcn':
            self.encoder = LevelEncoder(config)
        elif self.config.Model.structure_model == 'tree':
            self.encoder = TreeLSTMLevelEncoder(config)
        else:
            raise NotImplementedError()

        self.decoder = LevelDecoder(config)

        # Latency Model
        if self.config.Model.latency_model == 'gat':
            self.latency_encoder = LevelLatencyEncoder(config)
            self.latency_decoder = LevelLatencyDecoder(config)
        elif self.config.Model.latency_model == 'tree':
            self.latency_encoder = TreeLSTMLatencyEncoder(config)
            self.latency_decoder = TreeLSTMLatencyDecoder(config)
        else:
            raise NotImplementedError()

    def sample_z(self, z_mu: torch.Tensor, z_logvar: torch.Tensor) -> torch.Tensor:
        # Sample eps from N (0, I)
        eps = torch.randn_like(z_mu)
        sigma = torch.exp(z_logvar * 0.5)

        result = z_mu + sigma * eps
        return result

    def forward(self, g: dgl.DGLGraph, n_z: int=None):
        # Structure
        embed = self.embedding(g)
        z_graph_mu, z_graph_logvar = self.encoder(g, embed)

        # Expand to n_z
        assert n_z is None or n_z > 1
        n_z = 1 if n_z is None else n_z

        z_graph_mu = z_graph_mu.unsqueeze(1).expand([-1, n_z, -1])
        z_graph_logvar = z_graph_logvar.unsqueeze(1).expand([-1, n_z, -1])

        # Sample z_graph
        if self.config.Model.vae:
            z_graph_sample = self.sample_z(z_graph_mu, z_graph_logvar)
        else:
            z_graph_sample = z_graph_mu

        # Decode graph structure
        node_type, status = self.decoder(z_graph_sample)

        # Latency
        z_latency_mu, z_latency_logvar = self.latency_encoder(g, embed)

        z_latency_mu = z_latency_mu.unsqueeze(1).expand([-1, n_z, -1])
        z_latency_logvar = z_latency_logvar.unsqueeze(1).expand([-1, n_z, -1])

        # Sample z_latency
        if self.config.Model.vae:
            z_latency_sample = self.sample_z(z_latency_mu, z_latency_logvar)
        else:
            z_latency_sample = z_latency_mu

        # Decode latency
        latency_mu, latency_logvar = self.latency_decoder(g, z_latency_sample)

        # Squeeze result
        if n_z == 1:
            node_type, status = node_type.squeeze(1), status.squeeze(1)
            latency_mu, latency_logvar = latency_mu.squeeze(1), latency_logvar.squeeze(1)
            z_graph_sample = z_graph_sample.squeeze(1)
            z_latency_mu, z_latency_logvar = z_latency_mu.squeeze(1), z_latency_logvar.squeeze(1)
            z_graph_mu, z_graph_logvar = z_graph_mu.squeeze(1), z_graph_logvar.squeeze(1)

        return {
            'node_type': node_type,
            'status': status,
            'latency_mu': latency_mu,
            'latency_logvar': latency_logvar,
            'z_graph_mu': z_graph_mu,
            'z_graph_logvar': z_graph_logvar,
            'z_latency_mu': z_latency_mu,
            'z_latency_logvar': z_latency_logvar,
            'z_latency_sample': z_latency_sample,
            'encoded': z_graph_sample
        }


def normal_loss(label: torch.Tensor,
                mu: torch.Tensor,
                logvar: torch.Tensor,
                reduction: str = 'mean',
                eps: float = 1e-7,
                positive_only: bool = False) -> torch.Tensor:
    """
        Calculate the loss with normal distribution.
    """

    if positive_only:
        loss_mask = (label > mu)
        loss = (mu - label) ** 2 / (2 * torch.exp(logvar) + eps) * loss_mask + 0.5 * logvar
    else:
        loss = (mu - label) ** 2 / (2 * torch.exp(logvar) + eps) + 0.5 * logvar

    
    if reduction == 'mean':
        return torch.mean(loss)
    else:
        return loss


def kl_loss(mu: torch.Tensor,
            logvar: torch.Tensor) -> torch.Tensor:
    result = -0.5 * torch.mean(logvar + 1. - torch.exp(logvar) - mu ** 2)

    return result


def get_tensor_label(config: ExpConfig,
                     graphs: dgl.DGLGraph,
                     clip: bool=True,
                     mask_extend: int=0):
    # Unbatch the graphs first
    graph_list: List[dgl.DGLGraph] = dgl.unbatch(graphs)
    device = graphs.device

    # Set the node_type
    # Expand the true node_type
    b = len(graph_list)

    node_type_label = torch.zeros([b, config.decoder_max_nodes], dtype=torch.long, device=device)
    status_label = torch.zeros([b, config.decoder_max_nodes], dtype=torch.long, device=device)
    latency_label = torch.zeros(
        [b, config.decoder_max_nodes, config.Latency.latency_feature_length], dtype=torch.float, device=device)
    mask = torch.zeros([b, config.decoder_max_nodes], dtype=torch.bool, device=device)

    for i in range(b):
        node_cnt = graph_list[i].num_nodes()

        node_type_label[i, :node_cnt] = graph_list[i].ndata['operation_id']
        node_type_label[i, node_cnt:] = config.DatasetParams.operation_cnt

        status_label[i, :node_cnt] = graph_list[i].ndata['status_id']
        mask[i, :node_cnt + mask_extend] = True

    latency_label = latency_to_feature(config, graphs.ndata['latency'], graphs.ndata['operation_id'], clip=clip)

    return node_type_label, status_label, latency_label, mask


def log_exp_mean(x: torch.Tensor) -> torch.Tensor:
    x_min = x.min(dim=1, keepdim=True)[0]

    x = torch.minimum(torch.ones_like(x) * 20.0, x - x_min)

    result = torch.log(torch.mean(torch.exp(x), dim=1) + 1e-7)
    return result + x_min[:, 0]


def calculate_nll(config: ExpConfig, pred: dict, graphs: dgl.DGLGraph):
    n_z = config.Model.n_z
    # q(z1|x)  [num_graphs x n_z]
    log_q_z1 = -normal_loss(pred['encoded'], pred['z_graph_mu'],
        pred['z_graph_logvar'], reduction='none')
    log_q_z1 = torch.mean(log_q_z1, dim=-1)

    # p(z1) [num_graphs x n_z]
    log_p_z1 = -normal_loss(pred['encoded'], torch.zeros_like(pred['encoded']),
        torch.ones_like(pred['encoded']), reduction='none')
    log_p_z1 = torch.mean(log_p_z1, dim=-1)

    # p(node_type,status|z1) [num_graphs x n_z]
    node_type_label, status_label, latency_label, mask = get_tensor_label(config, graphs, clip=False, mask_extend=5)
    log_p_x1 = F.cross_entropy(
        pred['node_type'].transpose(1, -1), node_type_label.unsqueeze(-1).expand([-1,-1,n_z]), reduction='none') * mask.unsqueeze(-1)
    status_mask = (status_label != 0).unsqueeze(-1)
    log_p_x1 += F.cross_entropy(
        pred['status'].transpose(1, -1), status_label.unsqueeze(-1).expand([-1,-1,n_z]), reduction='none') * status_mask
    log_p_x1 = -torch.max(log_p_x1.transpose(1, 2), dim=-1)[0]

    # TODO: q(z2|x) [num_nodes x n_z]
    log_q_z2 = -normal_loss(pred['z_latency_sample'], pred['z_latency_mu'],
        pred['z_latency_logvar'], reduction='none')
    log_q_z2 = torch.mean(log_q_z2, dim=-1)

    # p(z2) [num_nodes x n_z]
    log_p_z2 = -normal_loss(pred['z_latency_sample'], torch.zeros_like(pred['z_latency_sample']),
        torch.ones_like(pred['z_latency_sample']), reduction='none')
    log_p_z2 = torch.mean(log_p_z2, dim=-1)

    # p(latency|z2) [num_nodes x n_z]
    log_p_x2 = -normal_loss(latency_label.unsqueeze(1).expand([-1,n_z,-1]),
        pred['latency_mu'], pred['latency_logvar'], reduction='none')
    log_p_x2 = torch.mean(log_p_x2, dim=-1)

    # nll1
    nll_1 = -log_exp_mean(log_p_x1 + log_p_z1 - log_q_z1)
    nll_2 = -log_exp_mean(log_p_x2 + log_p_z2 - log_q_z2)

    return nll_1, nll_2


def level_model_loss(config: ExpConfig,
                     pred: dict,
                     graphs: dgl.DGLGraph,
                     return_detail: bool=False) -> torch.Tensor:
    # Get labels
    node_type_label, status_label, latency_label, mask = get_tensor_label(config, graphs)

    # Calculate loss
    node_type_loss = F.cross_entropy(pred['node_type'].transpose(1, 2), node_type_label)

    status_loss = F.cross_entropy(pred['status'].transpose(1, 2), status_label, reduction='none')
    status_loss = torch.sum(status_loss * mask) / torch.sum(mask)

    operation_loss_dict = torch.zeros([config.DatasetParams.operation_cnt], dtype=torch.float, device=config.device)
    operation_cnt_dict = torch.zeros([config.DatasetParams.operation_cnt], dtype=torch.long, device=config.device)

    if config.Latency.embedding_type != 'class':
        latency_loss = normal_loss(latency_label, pred['latency_mu'], pred['latency_logvar'], reduction='none')
        operation_id = graphs.ndata['operation_id']
        latency_loss = torch.mean(latency_loss, dim=1)
        operation_cnt_dict = torch.index_add(operation_cnt_dict, 0, operation_id, torch.ones_like(operation_id))
        operation_loss_dict = torch.index_add(operation_loss_dict, 0, operation_id, latency_loss)
        latency_loss = latency_loss.mean()
    else:
        latency_loss = F.cross_entropy(pred['latency_mu'], latency_label)

    # Set whether to add vae to model loss
    if config.Model.vae:
        # Calculate kl loss
        kl_structure = kl_loss(pred['z_graph_mu'], pred['z_graph_logvar'])

        # TODO: prior distribution
        kl_latency = kl_loss(pred['z_latency_mu'], pred['z_latency_logvar'])
    else:
        kl_structure, kl_latency = torch.tensor(0.0), torch.tensor(0.0)

    if return_detail:
        return node_type_loss, status_loss, latency_loss, kl_structure, kl_latency, operation_loss_dict, operation_cnt_dict
    return node_type_loss, status_loss, latency_loss, kl_structure, kl_latency
