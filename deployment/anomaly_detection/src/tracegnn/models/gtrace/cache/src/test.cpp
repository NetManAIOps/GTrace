#include <iostream>
#include <string>
#include <thread>
#include <spdlog/spdlog.h>
#include "controller.h"
using namespace std::chrono_literals;

Controller controller;

int main() {
    spdlog::info("[data_fetch] Start local testing.");

    controller.initialize();

    controller.start_workers();
    // std::this_thread::sleep_for(1000s);

    for (size_t cnt {}; ; cnt++) {
        auto p = controller.consume_tree_cache_batch(256);

        if (cnt % 20ull == 0) {
            spdlog::info("[data_fetch] {} traces processed.", cnt * 256);
        }
    }
}
