#pragma once

#include "../core.h"
#include "solve_strategy.h"
#include "../../third_party/hnswlib/hnswlib.h"

class HnswStrategy : public SolveStrategy {
 public:
  HnswStrategy(std::string source_path,
               std::string query_path,
               std::string codebooks_path,
               std::string index_path)
      : SolveStrategy(source_path, query_path, codebooks_path, index_path) {}

  void solve() {
    // Build HNSW index

#if defined(DIS_L2)
    hnswlib::L2Space space(data_dim_);
#else
    hnswlib::InnerProductSpace space(data_dim_);
#endif
    hnswlib::HierarchicalNSW<float> hnsw(&space, data_num_, M_, ef_construction_);

    std::cout << "begin to solve" << std::endl;

    if (std::filesystem::exists(codebooks_path_)) {
      std::cout << "load index from " << index_path_ << std::endl;
      hnsw.loadIndex(index_path_, &space, 0);
    } else {
      std::cout << "build index to " << index_path_ << std::endl;
      auto s_build = std::chrono::system_clock::now();
#pragma omp parallel for schedule(dynamic)
      for (int i = 0; i < data_num_; ++i) {
        hnsw.addPoint(data_set_[i].data(), i);

        if (i % 10000 == 0) {
          auto cur_time = std::chrono::system_clock::now();
          std::cout << cur_time.time_since_epoch().count() << " build index: " << i << "/" << data_num_
                    << std::endl;
        }
      }
      auto e_build = std::chrono::system_clock::now();
      std::cout << "build cost: " << time_cost(s_build, e_build) << " (ms)\n";

      {
        std::filesystem::path fsPath(codebooks_path_);
        fsPath.remove_filename();
        std::filesystem::create_directories(fsPath);
        std::ofstream out(codebooks_path_, std::ios::binary);
        std::cout << "save index: " + index_path_ << std::endl;
        hnsw.saveIndex(index_path_);
      }
    }

    // Solve query
    auto s_solve = std::chrono::system_clock::now();
    hnsw.setEf(ef_search_);

    // cc debug
    for (int k = 0; k < REPEATED_COUNT; k++) {
#pragma omp parallel for schedule(dynamic) num_threads(NUM_THREADS)
      for (int i = 0; i < query_num_; ++i) {
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result =
            hnsw.searchKnn(query_set_[i].data(), K);
        while (!result.empty() && knn_results_[i].size() < K) {
          knn_results_[i].emplace_back(result.top().second);
          result.pop();
        }
        while (knn_results_[i].size() < K) {
          knn_results_[i].emplace_back(-1);
        }
      }
    }
    auto e_solve = std::chrono::system_clock::now();

    std::cout << "solve cost: " << (time_cost(s_solve, e_solve) / REPEATED_COUNT) << " (ms)\n";
    std::cout << "metric_hops: " << (hnsw.metric_hops / REPEATED_COUNT / query_num_)
              << ", metric_distance_computations: "
              << (hnsw.metric_distance_computations / REPEATED_COUNT / query_num_) << std::endl;
  }
};
