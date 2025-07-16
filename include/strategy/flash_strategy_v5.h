#pragma once

#include <alloca.h>
#include <cstdlib>
// #include <gperftools/malloc_extension.h>

#include "solve_strategy.h"
#include "../../third_party/hnswlib/flash_l2.h"
#include "../../third_party/hnswlib/hnswalg_flash_v5.h"

using Eigen::MatrixXf;
using Eigen::VectorXf;

class FlashStrategy_V5 : public SolveStrategy {
 public:
  FlashStrategy_V5(std::string source_path,
                   std::string query_path,
                   std::string codebooks_path,
                   std::string index_path)
      : SolveStrategy(source_path, query_path, codebooks_path, index_path) {
    data_path_ = source_path;
    subvector_num_ = SUBVECTOR_NUM;

    sample_num_ = std::max((size_t)(data_num_ * 0.1), SAMPLE_NUM);
    // sample_num_ = (size_t)(data_num_ * 0.1);
    ori_dim = data_dim_;
  }

  ~FlashStrategy_V5() {}

  void solve() {
    // With PQ CLUSTER_NUM set to 16, each cluster can be represented using 4 bits.
    // This allows storing two subvectors in a single byte, effectively saving space.

    hnswlib::FlashL2 flash_space(SUBVECTOR_NUM, CLUSTER_NUM, data_dim_);
    hnswlib::HnswFlash<float>* hnsw = nullptr;

    // Malloc
    Eigen::setNbThreads(NUM_THREADS);

    bool need_build_index = true;
    if (std::filesystem::exists(codebooks_path_)) {
      std::cout << "load codebooks from " << codebooks_path_ << std::endl;
      if (std::filesystem::exists(index_path_)) {
        std::cout << "load index from " << index_path_ << std::endl;
        hnsw = new hnswlib::HnswFlash<float>(&flash_space, index_path_);

        need_build_index = false;
      }
    } else {
      std::cout << "generate codebooks to " << codebooks_path_ << std::endl;

      // Generate/Read PQ's codebooks
      auto s_gen = std::chrono::system_clock::now();

      auto e_gen = std::chrono::system_clock::now();
      std::cout << "generate codebooks cost: " << time_cost(s_gen, e_gen) << " (ms)\n";

      {
        std::filesystem::path fsPath(codebooks_path_);
        fsPath.remove_filename();
        std::filesystem::create_directories(fsPath);
        std::ofstream out(codebooks_path_, std::ios::binary);
      }
    }

    // Build index
    if (need_build_index) {
      std::cout << "build index to " << index_path_ << std::endl;

      auto s_build = std::chrono::system_clock::now();
      hnsw = new hnswlib::HnswFlash<float>(&flash_space, data_num_, M_, ef_construction_);

      // train
      {
        std::vector<size_t> subset_index(sample_num_);
        std::random_device rd;
        std::mt19937 g(rd());
        // std::mt19937 g(19260817);
        std::uniform_int_distribution<size_t> dis(0, data_num_ - 1);
        for (size_t i = 0; i < sample_num_; ++i) {
          subset_index[i] = dis(g);
        }

        std::vector<float> train_set;
        for (size_t i = 0; i < sample_num_; ++i) {
          if (i % 10000 == 0) {
            std::cout << "generate train set: " << i << std::endl;
          }
          train_set.insert(train_set.end(), data_set_[subset_index[i]].begin(),
                           data_set_[subset_index[i]].end());
        }
        std::cout << "generate trainset finsih: total size: " << train_set.size()
                  << ", sample num: " << sample_num_ << std::endl;

        std::cout << "begin to train" << std::endl;
        hnsw->train(sample_num_, train_set.data());
      }

      // Encode data with PQ and SQ and add point
#pragma omp parallel for schedule(dynamic) num_threads(NUM_THREADS)
      for (size_t i = 0; i < data_num_; ++i) {
        // for (size_t i = 0; i < 2; ++i) {
        hnsw->addPoint(data_set_[i].data(), i);

        // std::cout << "add data: " << i << std::endl;
        // std::string debug_data = "";
        // for (size_t k = 0; k < data_dim_; ++k) {
        //   debug_data += std::to_string(data_set_[i][k]) + ", ";
        // }

        if (i % 100000 == 0) {
          std::cout << "add point: " << i << std::endl;
        }
      }
      auto e_build = std::chrono::system_clock::now();
      std::cout << "build cost: " << time_cost(s_build, e_build) << " (ms)\n";

      // Save Index
      {
        hnsw->saveIndex(index_path_);
      }
    }

#if defined(DEBUG_LOG)
    constexpr bool need_debug = true;
#else
    constexpr bool need_debug = false;
#endif
    // search
    auto s_solve = std::chrono::system_clock::now();

    hnsw->setEf(EF_SEARCH);

    for (size_t k = 0; k < REPEATED_COUNT; ++k) {
#pragma omp parallel for schedule(dynamic) num_threads(NUM_THREADS)
      for (size_t i = 0; i < query_num_; ++i) {
        // search
        float* query_data = query_set_[i].data();
        auto ret = hnsw->searchKnn(query_data, K);

        knn_results_[i].resize(K);
        size_t size = std::min(K, ret.size());
        for (size_t j = 0; j < size; ++j) {
          auto& top = ret.top();
          knn_results_[i][size - 1 - j] = top.second;
          ret.pop();
        }

        for (size_t j = size; j < K; ++j) {
          knn_results_[i][j] = -1;
        }
      }
    }

    auto e_solve = std::chrono::system_clock::now();
    std::cout << "solve cost: " << (time_cost(s_solve, e_solve) / REPEATED_COUNT) << " (ms)" << std::endl;
    std::cout << "metric_hops: " << (hnsw->metric_hops / REPEATED_COUNT / query_num_)
              << ", metric_distance_computations: "
              << (hnsw->metric_distance_computations / REPEATED_COUNT / query_num_) << std::endl;

    std::cout << "pq encode cost: " << pq_encode_cost / 1000000 << " (ms)" << std::endl;
    std::cout << "\tpq dist cost: " << pq_dist_cost / 1000000 << " (ms)" << std::endl;
    std::cout << "\tpq quant cost: " << pq_quant_cost / 1000000 << " (ms)" << std::endl;
    std::cout << "knn search cost: " << knn_search_cost / 1000000 << " (ms)" << std::endl;
    // std::cout << "\tknn upper layer cost: " << hnsw->knn_upper_layer_cost / 1000000 << " (ms)" <<
    // std::endl; std::cout << "\tknn base layer cost: " << hnsw->knn_base_layer_cost / 1000000 << " (ms)" <<
    // std::endl;
    std::cout << "rerank cost: " << rerank_cost / 1000000 << " (ms)" << std::endl;
  };

 protected:
  std::string data_path_;

  size_t subvector_num_{0};
  size_t sample_num_{0};

  int64_t pq_encode_cost = 0;
  int64_t pq_dist_cost = 0;
  int64_t pq_quant_cost = 0;
  int64_t knn_search_cost = 0;
  int64_t rerank_cost = 0;

  size_t ori_dim{0};  // The original dim of data before PCA
};
