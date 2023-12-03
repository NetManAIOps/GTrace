#include "models.h"
#include "utils.h"
#include "data_fetch.h"
#include "controller.h"
#include <vector>
#include <cstring>


// Define global variables
Controller controller;


// Export function
void initialize_data_fetch(
        const char* test_file,
        const char* id_manager_file) {
    controller.initialize(test_file, id_manager_file);
}


void start_workers() {
    controller.start_workers();
}


void trace_graph_batch_to_c_arr(
        // Input params
        const std::vector<TraceGraph>& trace_graph_batch,
        // Output params
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
) {
    /// Convert trace graph batch to c_arr, without reindex nodes and edges
    /// !!! The node_id and edge_id is very different from dgl.batch!
    auto batch_size {trace_graph_batch.size()};
    size_t total_nodes {0}, total_edges {0};

    for (const auto& trace_graph: trace_graph_batch) {
        total_nodes += trace_graph.node_id.size();
        total_edges += trace_graph.u.size();
    }

    // Initialize
    *batch_num_nodes = new unsigned long long[batch_size];
    *batch_num_edges = new unsigned long long[batch_size];

    *operation_id = new unsigned long long[total_nodes];
    *service_id = new unsigned long long[total_nodes];
    *status_id = new unsigned long long[total_nodes];
    *node_id = new unsigned long long[total_nodes];
    *duration = new unsigned long long[total_nodes];
    *start_time = new unsigned long long[total_nodes];
    *node_hash = new unsigned long long[total_nodes];

    *u = new unsigned long long[total_edges];
    *v = new unsigned long long[total_edges];

    // Copy data
    for (size_t batch_idx {}, node_idx {}, edge_idx {}; const auto& trace_graph: trace_graph_batch) {
        (*batch_num_nodes)[batch_idx] = trace_graph.node_id.size();
        (*batch_num_edges)[batch_idx] = trace_graph.u.size();
        batch_idx++;

        memcpy((*operation_id) + node_idx,
               trace_graph.operation_id.data(), trace_graph.operation_id.size() * sizeof(unsigned long long));
        memcpy((*service_id) + node_idx,
               trace_graph.service_id.data(), trace_graph.service_id.size() * sizeof(unsigned long long));
        memcpy((*status_id) + node_idx,
               trace_graph.status_id.data(), trace_graph.status_id.size() * sizeof(unsigned long long));
        memcpy((*node_id) + node_idx,
               trace_graph.node_id.data(), trace_graph.node_id.size() * sizeof(unsigned long long));
        memcpy((*duration) + node_idx,
               trace_graph.duration.data(), trace_graph.duration.size() * sizeof(unsigned long long));
        memcpy((*start_time) + node_idx,
               trace_graph.start_time.data(), trace_graph.start_time.size() * sizeof(unsigned long long));
        memcpy((*node_hash) + node_idx,
               trace_graph.node_hash.data(), trace_graph.node_hash.size() * sizeof(unsigned long long));
        node_idx += trace_graph.node_id.size();

        memcpy((*u) + edge_idx,
               trace_graph.u.data(), trace_graph.u.size() * sizeof(unsigned long long));
        memcpy((*v) + edge_idx,
               trace_graph.v.data(), trace_graph.v.size() * sizeof(unsigned long long));
        edge_idx += trace_graph.u.size();
    }
}

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
) {
    // Process
    auto &&trace_graph_batch {controller.consume_trace_graph_batch(batch_size)};

    trace_graph_batch_to_c_arr(
            trace_graph_batch,
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
            v);
}

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
) {
	// Process
    auto &&[trace_graph_batch, cache_result] {controller.consume_tree_cache_batch(batch_size)};
	auto &&[all_keys, created_keys, cal_graph_item_ids, cal_graph_nodes, cal_graph_edges] = cache_result;

    trace_graph_batch_to_c_arr(
            trace_graph_batch,
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
            v);

	vector_to_c_arr(all_keys, all_keys_arr);
	vector_to_c_arr(std::vector(created_keys.begin(), created_keys.end()), created_keys_arr);
	*created_keys_size = created_keys.size();

	std::vector<unsigned long long> cal_graph_op_vec, cal_graph_svc_vec, cal_graph_sta_vec, cal_graph_u_vec, cal_graph_v_vec, cal_graph_cnt_vec;

	for (const auto& nd: cal_graph_nodes) {
		cal_graph_op_vec.emplace_back(nd.operation_id);
		cal_graph_svc_vec.emplace_back(nd.service_id);
		cal_graph_sta_vec.emplace_back(nd.status_id);
	}

	for (unsigned long long cal_u {0}; cal_u < cal_graph_edges.size(); ++cal_u) {
		for (const auto &[cal_v, cnt]: cal_graph_edges[cal_u]) {
			cal_graph_u_vec.emplace_back(cal_u);
			cal_graph_v_vec.emplace_back(cal_v);
			cal_graph_cnt_vec.emplace_back(cnt);
		}
	}

    vector_to_c_arr(cal_graph_item_ids, cal_graph_keys_arr);
	vector_to_c_arr(cal_graph_op_vec, cal_graph_operation_id);
	vector_to_c_arr(cal_graph_svc_vec, cal_graph_service_id);
	vector_to_c_arr(cal_graph_sta_vec, cal_graph_status_id);
	vector_to_c_arr(cal_graph_u_vec, cal_graph_edge_u);
	vector_to_c_arr(cal_graph_v_vec, cal_graph_edge_v);
	vector_to_c_arr(cal_graph_cnt_vec, cal_graph_edge_cnt);
	*cal_graph_node_size = cal_graph_op_vec.size();
	*cal_graph_edge_size = cal_graph_u_vec.size();
}
