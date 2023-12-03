import dgl
import torch
from typing import *


def dgl_graph_key(graph: dgl.DGLGraph) -> str:
    return edges_to_key(graph.ndata['operation_id'], *graph.edges())

@torch.jit.script
def edges_to_key(operation_id: torch.Tensor, u_list: torch.Tensor, v_list: torch.Tensor) -> str:
    mask = u_list != v_list
    u_id: List[int] = operation_id[u_list][mask].tolist()
    v_id: List[int] = operation_id[v_list][mask].tolist()

    graph_key = f'0,{operation_id[0].item()};' + ';'.join(sorted([f'{u},{v}' for (u, v) in zip(u_id, v_id)]))

    return graph_key
