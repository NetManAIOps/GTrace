#ifndef THREAD_TEST_GRAPH_BUILDER_H
#define THREAD_TEST_GRAPH_BUILDER_H

#include <vector>
#include <queue>
#include <map>
#include <cstdint>
#include <atomic>
#include <functional>
#include <thread>
#include <chrono>
#include <random>
#include <algorithm>
#include <queue>
#include <boost/lockfree/queue.hpp>
#include "models.h"
#include "safe_queue.h"


using namespace std::chrono_literals;
using SpanQueue = std::queue<RawSpan>;
using TraceQueue = boost::lockfree::queue<TraceGraph*>;

class GraphBuilder {
public:
    // Hyperparameters
    // These parameters are initialized with default values, which can be changed manually
    time_t time_range = 30;
    size_t min_node_count = 2, max_node_count = 100;

    // FIFO queue of trace_id for window
    std::queue<std::pair<std::pair<uint64_t, uint64_t>, time_t>> fifo_queue;

    // Map: trace_id -> dict of spans
    std::map<std::pair<uint64_t, uint64_t>, std::unordered_map<uint64_t, RawSpan>> trace_dict;


    GraphBuilder() = default;

    // Build graph with evicted traces
    void build_graph(const std::unordered_map<uint64_t,
                       RawSpan>& trace_map,
                       TraceQueue& trace_queue) const {
        // Fast filter too small graphs
        if (trace_map.size() < min_node_count) {
            return;
        }

        // SpanId tree
        std::vector<uint64_t> root_span_id;
        std::unordered_map<uint64_t, std::unordered_set<uint64_t>> child_dict;

        // Build SpanId Tree
        for (const auto& [span_id, raw_span]: trace_map) {
            if (trace_map.contains(raw_span.parentSpanId)) {
                if (!child_dict.contains(raw_span.parentSpanId)) {
                    child_dict[raw_span.parentSpanId] = std::unordered_set<uint64_t>{};
                }
                if (!child_dict.contains(span_id)) {
                    child_dict[span_id] = std::unordered_set<uint64_t>{};
                }
                child_dict[raw_span.parentSpanId].insert(span_id);
            }
            else {
                // It is a root node!
                root_span_id.emplace_back(span_id);
            }
        }

        // Build
        size_t success_cnt = 0;

        // Regenerate trace_id to avoid duplication
        auto [trace_id_high, trace_id_low] = trace_map.begin() -> second.traceId;

        // Iterate over all traces with the same trace_id
        for (const auto &root: root_span_id) {
            // Trace Graph
            auto trace_graph_pointer = new TraceGraph();
            TraceGraph &trace_graph = *trace_graph_pointer;
            trace_graph.trace_id = {trace_id_high, trace_id_low};
            trace_id_high++;

            // DFS to build tree
            std::function<std::pair<uint64_t, uint64_t>(uint64_t)> dfs =
                    [&](uint64_t cur_id) {
                const auto &cur_span = trace_map.at(cur_id);

                // Append features
                uint64_t node_id {trace_graph.node_id.size()};
                trace_graph.service_id.emplace_back(cur_span.serviceId);
                trace_graph.operation_id.emplace_back(cur_span.operationId);
                trace_graph.status_id.emplace_back(cur_span.statusId);
                trace_graph.node_id.emplace_back(node_id);

                trace_graph.duration.emplace_back(cur_span.duration);
                trace_graph.start_time.emplace_back(
                        static_cast<uint64_t>(cur_span.startTime) * 1000ull +
                        static_cast<uint64_t>(cur_span.nanosecond / 1000000ull));
                trace_graph.node_hash.emplace_back(0ull);
                trace_graph.children_dict.emplace_back();

                // Get children hash
                std::vector<uint64_t> child_hash_list;
                for (const auto &child: child_dict[cur_id]) {
                    auto [child_id, child_hash] =dfs(child);
                    child_hash_list.emplace_back(child_hash);
                    trace_graph.u.emplace_back(node_id);
                    trace_graph.v.emplace_back(child_id);
                    trace_graph.children_dict[node_id].emplace_back(child_id);
                }

                // Calculate hash
                uint64_t hash_value {cur_span.operationId};

                std::sort(child_hash_list.begin(), child_hash_list.end());
                for (const auto& v: child_hash_list) {
                    hash_value = hash_value * 998244353ull + v;
                }
                trace_graph.node_hash[node_id] = hash_value;
                return std::make_pair(node_id, hash_value);
            };

            dfs(root);

            // Filter graphs
            if (trace_graph.node_id.size() > max_node_count || trace_graph.node_id.size() < min_node_count) {
                delete trace_graph_pointer;
                continue;
            }

            // Put into trace queue
            trace_queue.push(trace_graph_pointer);
        }
    }

    // Put a new span into graph builder
    void put(const RawSpan& raw_span, TraceQueue& trace_queue) {
        // Evict old traces (Before a certain time range of current trace)
        auto cur_time = raw_span.startTime;
        while (!fifo_queue.empty() && cur_time > fifo_queue.front().second + time_range) {
            // Evict one trace
            auto cur_trace_id = fifo_queue.front().first;
            fifo_queue.pop();

            // Build graph
            auto cur_trace = trace_dict[cur_trace_id];
            build_graph(cur_trace, trace_queue);

            // Remove from trace_dict to reduce memory usage
            trace_dict.erase(trace_dict.find(cur_trace_id));
        }

        if (!trace_dict.contains(raw_span.traceId)) {
            // Create a new trace
            trace_dict[raw_span.traceId] = std::unordered_map<uint64_t, RawSpan>{};
            fifo_queue.emplace(raw_span.traceId, raw_span.startTime);
        }
        trace_dict[raw_span.traceId][raw_span.spanId] = raw_span;
    }
};


// thread enter function
[[noreturn]] void graph_builder_start(SpanQueue &span_queue,
                                      TraceQueue &trace_queue) {
    GraphBuilder graph_builder;

    while (true) {
        if (!span_queue.empty()) {
            auto raw_span = span_queue.front();
            span_queue.pop();
            graph_builder.put(raw_span, trace_queue);
        }
    }
}

#endif //THREAD_TEST_GRAPH_BUILDER_H
