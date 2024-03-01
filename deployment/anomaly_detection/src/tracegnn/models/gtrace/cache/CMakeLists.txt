cmake_minimum_required(VERSION 3.22)

# Configuration
option(DATA_FETCH_TEST "make test" ON)
option(DATA_FETCH_DISABLE_KAFKA "disable kafka connection" OFF)

set(CMAKE_TOOLCHAIN_FILE ${CMAKE_CURRENT_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake
        CACHE STRING "Vcpkg toolchain file")
set(CMAKE_CXX_STANDARD 20)

project(data-fetch-test CXX)
add_subdirectory(
        lrucache11
)
add_executable(data-fetch-test
        src/test.cpp)

target_compile_features(data-fetch-test PRIVATE cxx_std_20)

find_package(spdlog CONFIG REQUIRED)
find_package(Boost REQUIRED)
find_package(yaml-cpp CONFIG REQUIRED)

target_include_directories(data-fetch-test PRIVATE
        src
        "fast-cpp-csv-parser"
        )
target_link_libraries(data-fetch-test
        PRIVATE
        spdlog::spdlog
        Boost::boost
        yaml-cpp::yaml-cpp
        LRUCache11
        )

# Build lib
project(data-fetch-lib)
add_library(data-fetch-lib SHARED
        src/data_fetch.cpp)
target_compile_features(data-fetch-lib PRIVATE cxx_std_20)
target_include_directories(data-fetch-lib PRIVATE
        src
        "fast-cpp-csv-parser"
        include
        )
set_target_properties(data-fetch-lib PROPERTIES SUFFIX ".so")
target_link_libraries(data-fetch-lib
        PRIVATE
        spdlog::spdlog
        Boost::boost
        yaml-cpp::yaml-cpp
        LRUCache11
        )
