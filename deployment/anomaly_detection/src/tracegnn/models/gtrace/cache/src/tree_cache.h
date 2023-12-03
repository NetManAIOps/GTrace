#ifndef DATA_FETCH_LOCAL_TREE_CACHE_H
#define DATA_FETCH_LOCAL_TREE_CACHE_H

#include <string>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <tuple>
#include <iostream>
#include "utils.h"
#include "models.h"
#include "LRUCache11.hpp"


struct TreeNode {
    uint64_t node_hash;
    uint64_t operation_id;
    uint64_t service_id;
    uint64_t status_id;
};


class TreeCache {
    lru11::Cache<uint64_t, TreeNode> lru_cache;
    std::unordered_map<uint64_t, uint64_t> item_id_map;
    std::unordered_set<uint64_t> available_item_ids;

public:
    explicit TreeCache(size_t lru_max_size = (1ul << 16),
              size_t elasticity = 1000ul):
            lru_cache{lru_max_size, elasticity}, item_id_map {}, available_item_ids {} {
        for (size_t i {}; i < lru_max_size + elasticity; ++i) {
            available_item_ids.insert(i);
        }
    }

    auto insert_batch_trees(const std::vector<TraceGraph>& trees) {
        // Store the list of all node keys
        std::vector<uint64_t> all_keys;
        std::unordered_set<uint64_t> created_hash, created_keys;

        // Store the tmp nodes (node_hash[tree_node])
        std::unordered_map<uint64_t, TreeNode> tmp_lru_cache;

        // Map to store the edges (node_hash[child_node_hash[cnt]])
        std::unordered_map<uint64_t, std::unordered_map<uint64_t, uint64_t>> all_edges;

        // std::cout << "Step 1: build trees." << std::endl;
        // Iterate the trees in a batch to get all keys
        for (const auto& tree: trees) {
            // Check if the key is in the lru_dict
            for (size_t nd {}; nd < tree.node_id.size(); ++nd) {
                auto node_hash {tree.node_hash[nd]};

                if(TreeNode vOut; !lru_cache.tryGet(node_hash, vOut)) {
                    // If in the lru_dict, then alright
                    // If not in the lru_dict, then create new one
                    vOut = TreeNode {
                            .node_hash = node_hash,
                            .operation_id = tree.operation_id[nd],
                            .service_id = tree.service_id[nd],
                            .status_id = tree.status_id[nd]
                    };
                    lru_cache.insert(node_hash, vOut);

                    // Add to created key
                    created_hash.insert(node_hash);
                    tmp_lru_cache[node_hash] = vOut;
                }
                else {
                    tmp_lru_cache[node_hash] = vOut;
                }

                // Store edge data
                if (!all_edges.contains(node_hash)) {
                    for (const auto& c: tree.children_dict.at(nd)) {
                        if (!all_edges[node_hash].contains(tree.node_hash[c])) {
                            all_edges[node_hash][tree.node_hash[c]] = 0;
                        }
                        all_edges[node_hash][tree.node_hash[c]]++;
                    }
                }
            }
        }

        // std::cout << "Step 2: get evicted items." << std::endl;
        // Get evicted keys when inserting keys into lru_cache
        auto &&evicted_keys_vec {lru_cache.getEvictedItems()};
        std::unordered_set<uint64_t> evicted_keys {evicted_keys_vec.begin(), evicted_keys_vec.end()};
        for (const auto& evicted: evicted_keys) {
            available_item_ids.insert(item_id_map[evicted]);
        }

        for (const auto& created: created_hash) {
            auto cur_item_id = *available_item_ids.begin();
            available_item_ids.erase(cur_item_id);
            item_id_map[created] = cur_item_id;
            created_keys.insert(cur_item_id);
        }

        for (const auto& tree: trees) {
            for (const auto& node_hash: tree.node_hash) {
                all_keys.emplace_back(item_id_map[node_hash]);
            }
        }

//        for (auto p: all_keys) {
//            if (available_item_ids.contains(p)) {
//                spdlog::error("Key {} already evicted!", p);
//            }
//        }

        // std::cout << "Step 3: Build calculation graph." << std::endl;
        // Build calculation graph
        // 1. For created_keys, they are in calculation graph.
        // 2. For the children of created_keys, they are also in calculation graph.
        std::vector<TreeNode> cal_graph_nodes;
        std::vector<uint64_t> cal_graph_item_ids;
        std::unordered_map<uint64_t, size_t> node_dict;
        // Edge: {child_id, cnt}
        std::vector<std::unordered_map<size_t, uint64_t>> cal_graph_edges;

        // Iterate the created_keys
        for (const auto& node_hash: created_hash) {
            if (node_dict.contains(node_hash)) {
                continue;
            }

            // Add current node to the cal graph
            node_dict[node_hash] = cal_graph_nodes.size();
            cal_graph_nodes.emplace_back(tmp_lru_cache[node_hash]);
            cal_graph_item_ids.emplace_back(item_id_map[node_hash]);
            cal_graph_edges.emplace_back();

            // Add children to the list
            for (const auto &[child_node_hash, cnt]: all_edges[node_hash]) {
                if (!node_dict.contains(child_node_hash)) {
                    node_dict[child_node_hash] = cal_graph_nodes.size();
                    cal_graph_nodes.emplace_back(tmp_lru_cache[child_node_hash]);
                    cal_graph_item_ids.emplace_back(item_id_map[child_node_hash]);
                    cal_graph_edges.emplace_back();
                }

                // Add new edges to this node
                cal_graph_edges[node_dict[node_hash]][node_dict[child_node_hash]] = cnt;
            }
        }

        // Return the results
        return std::make_tuple(all_keys, created_keys, cal_graph_item_ids, cal_graph_nodes, cal_graph_edges);
    }
};


#endif //DATA_FETCH_LOCAL_TREE_CACHE_H
