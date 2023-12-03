#ifndef DATA_FETCH_LOCAL_CONTROLLER_H
#define DATA_FETCH_LOCAL_CONTROLLER_H

#include <filesystem>
#include <chrono>
#include "utils.h"
#include "models.h"
#include "graph_builder.h"
#include "fetch_local.h"
#include "tree_cache.h"
#include <boost/lockfree/queue.hpp>


// Hyper params
constexpr size_t num_workers = 8;
constexpr size_t worker_queue_size = (1 << 24);
constexpr time_t fetch_interval = 60;
constexpr time_t time_offset = -10 * 3600;

using namespace std::chrono_literals;


class Controller {
    std::vector<SpanQueue> worker_queue;
    TraceIdManager id_manager;
    TraceQueue trace_queue;
    std::vector<std::thread> threads;
    TreeCache tree_cache;

    void fetch_data(const std::string& test_file) {
        time_t nxt_fetch_ts = current_ts();
        // Fetch from local file
        fetch_from_file(
            nxt_fetch_ts+time_offset,
            nxt_fetch_ts+time_offset+fetch_interval,
            worker_queue,
            id_manager,
            0,
            test_file);
    }

public:
    Controller(size_t cache_size = (1ull << 18)): trace_queue(worker_queue_size), tree_cache(cache_size) {}

    void initialize(
            const std::string& test_file = "../dataset/dataset_b/raw/2022-04-13.csv",
            const std::string& id_manager_file = "../dataset/dataset_b/processed/"
            ) {
        // Initialize the controller with the given dataset
        // The controller with simulate the db fetch process and store the raw spans into memory
        // Print status info
        spdlog::info("[data_fetch] test_file: {}", test_file);
        spdlog::info("[data_fetch] id_manager_file: {}", id_manager_file);
        spdlog::info("[data_fetch] Num workers: " + std::to_string(num_workers));
        spdlog::info("[data_fetch] Fetch interval:" + std::to_string(fetch_interval));

        // Load id_manager
        id_manager = load_id_manager(id_manager_file);

        // Fetch data
        for (size_t i{}; i < num_workers; ++i) {
            worker_queue.emplace_back();
        }
        fetch_data(test_file);
        spdlog::info("[data_fetch] Data fetch local initialized successfully.");
    }

    void start_workers() {
        // Start workers
        for (size_t i{}; i < num_workers; ++i) {
            threads.emplace_back(graph_builder_start, std::ref(worker_queue[i]), std::ref(trace_queue));
        }
        spdlog::info("[data_fetch] Started {} workers for processing.", num_workers);
    }

    auto consume_trace_graph() {
        while(true) {
            if (!trace_queue.empty()) {
                std::shared_ptr<TraceGraph> result;
                trace_queue.pop(result);
                return result;
            }
            else {
                std::this_thread::sleep_for(1us);
            }
        }
    }

    auto consume_trace_graph_batch(size_t batch_size = 256) {
        std::vector<TraceGraph> result(batch_size);
        for (auto &v: result) {
            v = *consume_trace_graph();
        }
        return result;
    }

    auto consume_tree_cache_batch(size_t batch_size = 256) {
        auto trace_graph_batch {consume_trace_graph_batch(batch_size)};
        auto cache_result {tree_cache.insert_batch_trees(trace_graph_batch)};
        return std::make_tuple(trace_graph_batch, cache_result);
    }
};


#endif //DATA_FETCH_LOCAL_CONTROLLER_H
