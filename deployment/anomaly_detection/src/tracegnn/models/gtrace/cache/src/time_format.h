#pragma once
#include <iostream>
#include <string>
#include <string_view>
#include <iomanip>
#include <ctime>

constexpr size_t DATE_LEN = 50;

time_t current_ts() {
    time_t result = std::time(nullptr);
    return result;
}


time_t date2ts(std::string_view s) {
    std::tm t{};
    strptime(s.data(), "%Y-%m-%d %H:%M:%S", &t);

    return timegm(&t);
}


std::string ts2date(time_t ts) {
    auto tm = gmtime(&ts);
    char s[DATE_LEN] = {0};
    strftime(s, DATE_LEN, "%Y-%m-%d %H:%M:%S", tm);

    return std::string{s};
}
