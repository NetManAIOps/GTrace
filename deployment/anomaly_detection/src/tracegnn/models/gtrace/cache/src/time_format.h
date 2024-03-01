#pragma once
#include <iostream>
#include <string>
#include <string_view>
#include <iomanip>
#include <ctime>
#include <sstream>

constexpr size_t DATE_LEN = 50;

time_t current_ts() {
    time_t result = std::time(nullptr);
    return result;
}

time_t date2ts(std::string_view s) {
    std::tm t{};
    std::istringstream ss(std::string{s});
    ss >> std::get_time(&t, "%Y-%m-%d %H:%M:%S");
    if(ss.fail()) {
        throw std::runtime_error("Failed to parse date/time");
    }

    return mktime(&t);
}

std::string ts2date(time_t ts) {
    auto tm = gmtime(&ts);
    char s[DATE_LEN] = {0};
    strftime(s, DATE_LEN, "%Y-%m-%d %H:%M:%S", tm);

    return std::string{s};
}
