#ifndef DATA_FETCH_LOCAL_UTILS_H
#define DATA_FETCH_LOCAL_UTILS_H

#include "models.h"
#include <string>
#include <yaml-cpp/yaml.h>
#include <filesystem>


TraceIdManager load_id_manager(const std::filesystem::path& id_manager_file) {
    // Read files
    TraceIdManager result;

    // Load operation_id
    YAML::Node operation_id_config = YAML::LoadFile(id_manager_file / "operation_id.yml");
    for (auto &&node: operation_id_config) {
        result.operation_id[node.first.as<std::string>()] = node.second.as<size_t>();
    }

    // Load service_id
    YAML::Node service_id_config = YAML::LoadFile(id_manager_file / "service_id.yml");
    for (auto &&node: service_id_config) {
        result.service_id[node.first.as<std::string>()] = node.second.as<size_t>();
    }

    // Load status_id
    YAML::Node status_id_config = YAML::LoadFile(id_manager_file / "status_id.yml");
    for (auto &&node: status_id_config) {
        result.status_id[node.first.as<std::string>()] = node.second.as<size_t>();
    }

    return result;
}

char* string_to_char_array(const std::string& s) {
    char* c_arr {new char[s.length() + 1]};
    strcpy(c_arr, s.c_str());

    return c_arr;
}

std::string string_join(const std::string& d, const std::vector<std::string>& ss) {
    bool flag {false};
    std::string result;

    for (const auto& s: ss) {
        if (flag) {
            result += d;
        }
        else {
            flag = true;
        }

        result += s;
    }

    return result;
}


template <class T_vec, class T_p>
void vector_to_c_arr(std::vector<T_vec> vec, T_p **result) {
    T_p *result_pointer {new T_p[vec.size()]};

    memcpy(result_pointer, vec.data(), vec.size() * sizeof(T_vec));

    *result = result_pointer;
}


void string_vector_to_c_arr(std::vector<std::string> vec, char ***result) {
    char **result_pointer {new char*[vec.size()]};

    for (size_t i {}; i < vec.size(); ++i) {
        result_pointer[i] = string_to_char_array(vec[i]);
    }

    *result = result_pointer;
}

#endif //DATA_FETCH_LOCAL_UTILS_H
