import cffi
import numpy as np
import dgl
import torch
from typing import *


ffi = cffi.FFI()
ffi.cdef(
    """
    void initialize_data_fetch(
        const char* test_file,
        const char* id_manager_file
    );

    void start_workers();

    void get_trace_graph_batch(
        // Input params
        unsigned long long batch_size,
        // Output params (Graph)
        unsigned long long **batch_num_nodes,
        unsigned long long **operation_id,
        unsigned long long **service_id,
        unsigned long long **status_id,
        unsigned long long **node_id,
        unsigned long long **duration,
        unsigned long long **start_time,
        unsigned long long **node_hash,
        unsigned long long **batch_num_edges,
        unsigned long long **u,
        unsigned long long **v
    );

    void get_tree_cache_batch(
        // Input params
        unsigned long long batch_size,
        // Output params (Graph)
        unsigned long long **batch_num_nodes,
        unsigned long long **operation_id,
        unsigned long long **service_id,
        unsigned long long **status_id,
        unsigned long long **node_id,
        unsigned long long **duration,
        unsigned long long **start_time,
        unsigned long long **node_hash,
        unsigned long long **batch_num_edges,
        unsigned long long **u,
        unsigned long long **v,
        // Output params (Cache)
        unsigned long long **all_keys_arr,
        unsigned long long **created_keys_arr,
        unsigned long long *created_keys_size,
        unsigned long long **cal_graph_keys_arr,
        unsigned long long **cal_graph_operation_id,
        unsigned long long **cal_graph_service_id,
        unsigned long long **cal_graph_status_id,
        unsigned long long **cal_graph_edge_u,
        unsigned long long **cal_graph_edge_v,
        unsigned long long **cal_graph_edge_cnt,
        unsigned long long *cal_graph_node_size,
        unsigned long long *cal_graph_edge_size
    );
    """
)
DL = ffi.dlopen("tracegnn/models/gtrace/cache/build/libdata-fetch-lib.so")


def initialize_data_fetch(
    test_file: str='dataset_b/raw/2022-04-13.csv',
    id_manager_file: str='dataset_b/processed'):
    # Set input cffi data
    test_file = ffi.new("char[]", bytes(test_file, encoding='ascii'))
    id_manager_file = ffi.new("char[]", bytes(id_manager_file, encoding='ascii'))

    DL.initialize_data_fetch(test_file, id_manager_file)

def start_workers():
    DL.start_workers()

def c_arr_to_graphs(
    batch_size,
    batch_num_nodes,
    operation_id,
    service_id,
    status_id,
    node_id,
    duration,
    start_time,
    node_hash,
    batch_num_edges,
    u,
    v,
    device: str='cpu'):
    # Set batch size
    batch_size = int(batch_size)
    batch_num_nodes = torch.frombuffer(ffi.buffer(batch_num_nodes[0], 8 * batch_size), dtype=torch.long).to(device)
    batch_num_edges = torch.frombuffer(ffi.buffer(batch_num_edges[0], 8 * batch_size), dtype=torch.long).to(device)

    node_size = batch_num_nodes.sum().item()
    operation_id = torch.frombuffer(ffi.buffer(operation_id[0], 8 * node_size), dtype=torch.long).to(device)
    service_id = torch.frombuffer(ffi.buffer(service_id[0], 8 * node_size), dtype=torch.long).to(device)
    status_id = torch.frombuffer(ffi.buffer(status_id[0], 8 * node_size), dtype=torch.long).to(device)
    node_id = torch.frombuffer(ffi.buffer(node_id[0], 8 * node_size), dtype=torch.long).to(device)
    duration = torch.frombuffer(ffi.buffer(duration[0], 8 * node_size), dtype=torch.long).to(device)
    start_time = torch.frombuffer(ffi.buffer(start_time[0], 8 * node_size), dtype=torch.long).to(device)
    node_hash = torch.frombuffer(ffi.buffer(node_hash[0], 8 * node_size), dtype=torch.long).to(device)

    edge_size = batch_num_edges.sum().item()
    u = torch.frombuffer(ffi.buffer(u[0], 8 * edge_size), dtype=torch.long).to(device)
    v = torch.frombuffer(ffi.buffer(v[0], 8 * edge_size), dtype=torch.long).to(device)

    return {
        'batch_size': batch_size,
        'batch_num_nodes': batch_num_nodes,
        'batch_num_edges': batch_num_edges,
        'node_size': node_size,
        'operation_id': operation_id,
        'service_id': service_id,
        'status_id': status_id,
        'node_id': node_id,
        'latency': duration,
        'start_time': start_time,
        'node_hash': node_hash,
        'edge_size': edge_size,
        'u': u,
        'v': v
    }


def get_trace_graph_batch(batch_size: int=256, device: str='cpu'):
    # Set input cffi data
    batch_size = ffi.cast("unsigned long long", batch_size)

    # Set output cffi data
    batch_num_nodes = ffi.new("unsigned long long *[1]")
    operation_id = ffi.new("unsigned long long *[1]")
    service_id = ffi.new("unsigned long long *[1]")
    status_id = ffi.new("unsigned long long *[1]")
    node_id = ffi.new("unsigned long long *[1]")
    duration = ffi.new("unsigned long long *[1]")
    start_time = ffi.new("unsigned long long *[1]")
    node_hash = ffi.new("unsigned long long *[1]")
    batch_num_edges = ffi.new("unsigned long long *[1]")
    u = ffi.new("unsigned long long *[1]")
    v = ffi.new("unsigned long long *[1]")

    # Execute C code
    DL.get_trace_graph_batch(
        batch_size,
        batch_num_nodes,
        operation_id,
        service_id,
        status_id,
        node_id,
        duration,
        start_time,
        node_hash,
        batch_num_edges,
        u,
        v)

    # Construct graphs
    trace_graphs = c_arr_to_graphs(batch_size, batch_num_nodes, operation_id, service_id,
        status_id, node_id, duration, start_time, node_hash, batch_num_edges, u, v, device)

    return trace_graphs


def graph_batch_to_dgl(graph_batch: dict):
    result: List[dgl.DGLGraph] = []

    node_idx: int = 0
    edge_idx: int = 0

    batch_num_nodes = graph_batch['batch_num_nodes'].cpu().numpy()
    batch_num_edges = graph_batch['batch_num_edges'].cpu().numpy()

    for i in range(graph_batch['batch_size']):
        node_size = batch_num_nodes[i]
        edge_size = batch_num_edges[i]

        graph = dgl.graph((
            graph_batch['u'][edge_idx:edge_idx+edge_size],
            graph_batch['v'][edge_idx:edge_idx+edge_size]
        ))
        graph.ndata['operation_id'] = graph_batch['operation_id'][node_idx:node_idx+node_size]
        graph.ndata['service_id'] = graph_batch['service_id'][node_idx:node_idx+node_size]
        graph.ndata['status_id'] = graph_batch['status_id'][node_idx:node_idx+node_size]
        graph.ndata['latency'] = graph_batch['latency'][node_idx:node_idx+node_size]
        graph.ndata['start_time'] = graph_batch['start_time'][node_idx:node_idx+node_size]
        graph.ndata['node_hash'] = graph_batch['node_hash'][node_idx:node_idx+node_size]

        result.append(graph)

        node_idx += node_size
        edge_idx += edge_size

    return result


def get_tree_cache_batch(batch_size: int=256, device: str='cpu'):
    # Set input cffi data
    batch_size = ffi.cast("unsigned long long", batch_size)

    # Set output cffi data
    batch_num_nodes = ffi.new("unsigned long long *[1]")
    operation_id = ffi.new("unsigned long long *[1]")
    service_id = ffi.new("unsigned long long *[1]")
    status_id = ffi.new("unsigned long long *[1]")
    node_id = ffi.new("unsigned long long *[1]")
    duration = ffi.new("unsigned long long *[1]")
    start_time = ffi.new("unsigned long long *[1]")
    node_hash = ffi.new("unsigned long long *[1]")
    batch_num_edges = ffi.new("unsigned long long *[1]")
    u = ffi.new("unsigned long long *[1]")
    v = ffi.new("unsigned long long *[1]")

    all_keys_arr = ffi.new("unsigned long long *[1]")
    created_keys_arr = ffi.new("unsigned long long *[1]")
    created_keys_size = ffi.new("unsigned long long[1]")
    cal_graph_keys_arr = ffi.new("unsigned long long *[1]")
    cal_graph_operation_id = ffi.new("unsigned long long *[1]")
    cal_graph_service_id = ffi.new("unsigned long long *[1]")
    cal_graph_status_id = ffi.new("unsigned long long *[1]")
    cal_graph_edge_u = ffi.new("unsigned long long *[1]")
    cal_graph_edge_v = ffi.new("unsigned long long *[1]")
    cal_graph_edge_cnt = ffi.new("unsigned long long *[1]")
    cal_graph_node_size = ffi.new("unsigned long long[1]")
    cal_graph_edge_size = ffi.new("unsigned long long[1]")

    # Execute C code
    DL.get_tree_cache_batch(
        batch_size,
        batch_num_nodes,
        operation_id,
        service_id,
        status_id,
        node_id,
        duration,
        start_time,
        node_hash,
        batch_num_edges,
        u,
        v,
        all_keys_arr,
        created_keys_arr,
        created_keys_size,
        cal_graph_keys_arr,
        cal_graph_operation_id,
        cal_graph_service_id,
        cal_graph_status_id,
        cal_graph_edge_u,
        cal_graph_edge_v,
        cal_graph_edge_cnt,
        cal_graph_node_size,
        cal_graph_edge_size
    )

    # Construct graphs
    trace_graphs = c_arr_to_graphs(batch_size, batch_num_nodes, operation_id, service_id,
        status_id, node_id, duration, start_time, node_hash, batch_num_edges, u, v, device)

    # Fetch data
    node_size = trace_graphs['node_size']

    # Build dgl graph
    cal_graph_node_size = cal_graph_node_size[0]
    cal_graph_edge_size = cal_graph_edge_size[0]
    all_keys = torch.frombuffer(ffi.buffer(all_keys_arr[0], 8 * node_size), dtype=torch.long).to(device)
    created_keys = set(np.frombuffer(ffi.buffer(created_keys_arr[0], 8 * created_keys_size[0]), dtype=np.int64))

    # Check if the size of cal_graph is zero
    if cal_graph_node_size != 0:
        cal_graph_keys = torch.frombuffer(ffi.buffer(cal_graph_keys_arr[0], 8 * cal_graph_node_size), dtype=torch.long).to(device)
        u=torch.frombuffer(ffi.buffer(cal_graph_edge_u[0], 8 * cal_graph_edge_size), dtype=torch.long).to(device)
        v=torch.frombuffer(ffi.buffer(cal_graph_edge_v[0], 8 * cal_graph_edge_size), dtype=torch.long).to(device)
        data_idx = torch.tensor([i for (i, v) in enumerate(cal_graph_keys.cpu().numpy()) if v not in created_keys]).to(device)

        g = dgl.graph((u, v), num_nodes=cal_graph_node_size, device=device)

        g.ndata['operation_id'] = torch.frombuffer(ffi.buffer(cal_graph_operation_id[0], 8 * cal_graph_node_size), dtype=torch.long).to(device)
        g.ndata['service_id'] = torch.frombuffer(ffi.buffer(cal_graph_service_id[0], 8 * cal_graph_node_size), dtype=torch.long).to(device)
        g.ndata['status_id'] = torch.frombuffer(ffi.buffer(cal_graph_status_id[0], 8 * cal_graph_node_size), dtype=torch.long).to(device)
        g.edata['cnt'] = torch.frombuffer(ffi.buffer(cal_graph_edge_cnt[0], 8 * cal_graph_edge_size), dtype=torch.long).to(device).unsqueeze(-1).unsqueeze(-1)
    else:
        cal_graph_keys = torch.empty(0, device=device)
        data_idx = torch.empty(0, device=device)
        g = dgl.graph(([], []), num_nodes=0, device=device)
    
    return g, all_keys, created_keys, cal_graph_keys, data_idx, trace_graphs


if __name__ == "__main__":
    initialize_data_fetch()
    start_workers()
    cnt = 0
    while True:
        print(cnt)
        cnt += 1
        get_tree_cache_batch(256, device='cuda:0')
