#pragma once

extern "C" {
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
}
