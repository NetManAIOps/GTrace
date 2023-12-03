#ifndef THREAD_TEST_MODELS_H
#define THREAD_TEST_MODELS_H

#include <string>
#include <cstdint>
#include <chrono>
#include <unordered_map>
#include <unordered_set>
#include <type_traits>
#include <spdlog/spdlog.h>

struct RawSpan {
    std::pair<std::uint64_t, std::uint64_t> traceId;
    int64_t spanId;
    int64_t parentSpanId;

    uint64_t serviceId;
    uint64_t operationId;
    uint64_t statusId;

    time_t startTime;
    uint64_t duration;
    uint64_t nanosecond;
};


struct TraceGraph {
    // Overall features
    std::pair<std::uint64_t, std::uint64_t> trace_id;

    // Node features
    std::vector<uint64_t> service_id, operation_id, status_id, node_id;
    std::vector<uint64_t> duration;
    std::vector<uint64_t> node_hash;
    std::vector<uint64_t> start_time;

    // Edges
    std::vector<uint64_t> u, v;
    std::vector<std::vector<uint64_t>> children_dict;
};

struct TraceIdManager {
    std::string version;

    std::unordered_map<std::string, size_t> operation_id;
    std::unordered_map<std::string, size_t> service_id;
    std::unordered_map<std::string, size_t> status_id;
};

#endif //THREAD_TEST_MODELS_H
