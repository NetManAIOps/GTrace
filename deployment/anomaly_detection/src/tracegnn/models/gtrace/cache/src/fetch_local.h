#ifndef THREAD_TEST_CLICKHOUSE_FETCH_H
#define THREAD_TEST_CLICKHOUSE_FETCH_H

#include <iostream>
#include <vector>
#include <string>
#include <cstdint>
#include <queue>
#include "models.h"
#include "time_format.h"
#include <csv.h>
#include <spdlog/spdlog.h>
#include <boost/lockfree/spsc_queue.hpp>
#include <boost/lockfree/queue.hpp>


using SpanQueue = std::queue<RawSpan>;

void fetch_from_file(
        time_t start_time,
        time_t end_time,
        std::vector<SpanQueue>& worker_queue,
        TraceIdManager& id_manager,
        time_t start_time_delta,
        const std::string& test_file
) {
    using namespace std::chrono;

    static time_t time_delta = system_clock::to_time_t(system_clock::now());

    // Fetch from file
    auto left_date = ts2date(start_time), right_date = ts2date(end_time);
    spdlog::info("[data_fetch] Initializing... Loading data to memory.");

    // Start select
    try {
        io::CSVReader<11>
                      // io::trim_chars<' ', '\t'>,
                      // io::double_quote_escape<',', '"'>>
                      csv_in {test_file};
        csv_in.read_header(io::ignore_extra_column,
            "traceIdHigh",
            "traceIdLow",
            "spanId",
            "parentSpanId",
            "serviceName",
            "operationName",
            "status",
            "startTime",
            "duration",
            "nanosecond",
            "DBhash");

        // Collect result
        auto num_workers {worker_queue.size()};
        size_t total_count = 0;

        // Read from file
        int64_t traceIdHigh, traceIdLow;
        int64_t spanId, parentSpanId;
        std::string serviceName, operationName, status;
        std::string startTime;
        uint64_t duration, nanosecond;
        uint64_t dbhash;
        bool id_manager_update = false;

        while (csv_in.read_row(
                traceIdHigh,
                traceIdLow,
                spanId,
                parentSpanId,
                serviceName,
                operationName,
                status,
                startTime,
                duration,
                nanosecond,
                dbhash)) {
            // Filter dbhash
            if (dbhash != 0) {
                continue;
            }

            // Assign the RawSpan to the corresponding worker according to its trace_id
            // Modify traceIdLow
            traceIdLow += static_cast<int64_t>(start_time);

            size_t cur_index = (size_t) traceIdLow % num_workers;

            // Convert name -> id
            if (!id_manager.service_id.contains(serviceName)) {
                id_manager.service_id[serviceName] = id_manager.service_id.size();
                id_manager_update = true;
                spdlog::info("New serviceName: {} -> {}", serviceName, id_manager.service_id[serviceName]);
                continue;
            }

            operationName = serviceName + "/" + operationName;
            if (!id_manager.operation_id.contains(operationName)) {
                id_manager.operation_id[operationName] = id_manager.operation_id.size();
                id_manager_update = true;
                spdlog::info("New operationName: {} -> {}", operationName, id_manager.operation_id[operationName]);
                continue;
            }

            if (!id_manager.status_id.contains(status)) {
                id_manager.status_id[status] = id_manager.status_id.size();
                id_manager_update = true;
                spdlog::info("New status: {} -> {}", status, id_manager.status_id[status]);
                continue;
            }

            auto cur_time_delta {system_clock::to_time_t(system_clock::now()) - time_delta};

            worker_queue[cur_index].push(RawSpan{
                    .traceId = {traceIdHigh, traceIdLow},
                    .spanId = spanId,
                    .parentSpanId = parentSpanId,
                    .serviceId = id_manager.service_id[serviceName],
                    .operationId = id_manager.operation_id[operationName],
                    .statusId = id_manager.status_id[status],
                    .startTime = date2ts(startTime) + start_time_delta,
                    .duration = duration,
                    .nanosecond = nanosecond
            });

            total_count += 1;
        }

        // Update id_manager
        if (id_manager_update) {
            spdlog::info("Dump new id_manager successfully.");
        }

        spdlog::info(std::to_string(total_count) + " spans processed.");
    }
    catch (const std::exception& err) {
        spdlog::error("Error occur: {}", err.what());
    }
    catch (...) {
        spdlog::error("Unknown error when fetching spans.");
    }
}

#endif //THREAD_TEST_CLICKHOUSE_FETCH_H
