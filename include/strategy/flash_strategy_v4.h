#pragma once

#include <alloca.h>
#include <cstdlib>
// #include <gperftools/malloc_extension.h>

#include "solve_strategy.h"
#include "../space/space_flash.h"
#include "../../third_party/hnswlib/space_l2_v2.h"
#include "../../third_party/hnswlib/hnswalg_flash_v4.h"
#include "../../third_party/hnswlib/flash_l2.h"

using Eigen::MatrixXf;
using Eigen::VectorXf;
using hnswlib::pq_dist_t;

class FlashStrategy_V4 : public SolveStrategy {
 public:
  FlashStrategy_V4(std::string source_path,
                   std::string query_path,
                   std::string codebooks_path,
                   std::string index_path)
      : SolveStrategy(source_path, query_path, codebooks_path, index_path) {
    data_path_ = source_path;
    subvector_num_ = SUBVECTOR_NUM;
    hnswlib::subvec_num_v4_ = subvector_num_;
    // sample_num_ = std::max((size_t)(data_num_ * 0.1), SAMPLE_NUM);
    sample_num_ = (size_t)(data_num_ * 0.1);
    ori_dim = data_dim_;
    pre_length_ = (size_t*)malloc(subvector_num_ * sizeof(size_t));
    subvector_length_ = (size_t*)malloc(subvector_num_ * sizeof(size_t));
  }

  ~FlashStrategy_V4() {}

  void solve() {
    // With PQ CLUSTER_NUM set to 16, each cluster can be represented using 4 bits.
    // This allows storing two subvectors in a single byte, effectively saving space.

    hnswlib::data_dim_ = subvector_num_;
    hnswlib::FlashSpace flash_space(subvector_num_);
    hnswlib::HierarchicalNSWFlash_V4<data_t>* hnsw;

    // Malloc
    Eigen::setNbThreads(NUM_THREADS);
    // To save memory and avoid excessive malloc calls during vector encoding, we allocate space for each
    // thread separately.
    uint8_t** thread_encoded_vector = (uint8_t**)malloc(NUM_THREADS * sizeof(uint8_t*));
    for (int i = 0; i < NUM_THREADS; ++i) {
      thread_encoded_vector[i] = (uint8_t*)aligned_alloc(
          64, SUBVECTOR_NUM * CLUSTER_NUM * sizeof(data_t) + subvector_num_ * sizeof(uint16_t));
    }
    // Save the distance table if SAVE_MEMORY is not enabled.
    // If the distance table is not saved, the SDC will be used to compute the distance between points.

    // Create index
    // If the index is already saved, load it from the file system
    bool need_build_index = true;
    if (std::filesystem::exists(codebooks_path_)) {
      std::cout << "load codebooks from " << codebooks_path_ << std::endl;
      std::ifstream in(codebooks_path_, std::ios::binary);
#if defined(USE_PCA)
      {
        VectorXf tmp1(ori_dim);
        for (int j = 0; j < ori_dim; ++j) {
          in.read(reinterpret_cast<char*>(&tmp1(j)), sizeof(float));
        }
        data_mean_ = tmp1;
      }
      {
        MatrixXf tmp2(ori_dim, PRINCIPAL_DIM);
        for (int i = 0; i < ori_dim; ++i) {
          for (int j = 0; j < PRINCIPAL_DIM; ++j) {
            in.read(reinterpret_cast<char*>(&tmp2(i, j)), sizeof(float));
          }
        }
        principal_components = tmp2;
      }

      data_dim_ = PRINCIPAL_DIM;
#endif
      in.read(reinterpret_cast<char*>(&qmin), sizeof(float));
      std::cout << "load qmin: " << qmin << std::endl;
      in.read(reinterpret_cast<char*>(&qmax), sizeof(float));
      std::cout << "load qmax: " << qmax << std::endl;

      for (int i = 0; i < subvector_num_; ++i) {
        in.read(reinterpret_cast<char*>(&pre_length_[i]), sizeof(size_t));
      }
      for (int i = 0; i < subvector_num_; ++i) {
        in.read(reinterpret_cast<char*>(&subvector_length_[i]), sizeof(size_t));
      }

      auto& codebooks = hnswlib::flash_codebooks_v4_;
      codebooks = (float*)malloc(CLUSTER_NUM * data_dim_ * sizeof(float));

      for (size_t i = 0, index = 0; i < SUBVECTOR_NUM; ++i) {
        for (size_t j = 0; j < CLUSTER_NUM; ++j) {
          for (size_t k = 0; k < subvector_length_[i]; ++k, ++index) {
            in.read(reinterpret_cast<char*>(&codebooks[index]), sizeof(float));
          }
        }
      }

      auto& dist = hnswlib::flash_dist_v4_;
      dist = (data_t*)malloc(SUBVECTOR_NUM * CLUSTER_NUM * CLUSTER_NUM * sizeof(data_t));
      for (int i = 0; i < SUBVECTOR_NUM * CLUSTER_NUM * CLUSTER_NUM; ++i) {
        in.read(reinterpret_cast<char*>(&dist[i]), sizeof(data_t));
      }

      if (std::filesystem::exists(index_path_)) {
        std::cout << "load index from " << index_path_ << std::endl;
        hnsw = new hnswlib::HierarchicalNSWFlash_V4<data_t>(&flash_space, index_path_);

#if defined(RERANK)
        if (data_set_.empty()) {
          uint32_t tmp_dim = 0;
          ReadData(data_path_, data_set_, data_num_, tmp_dim);

#if defined(USE_PCA)
          pcaEncode(data_set_);
#endif
        }
#endif
        need_build_index = false;
      }
    } else {
      std::cout << "generate codebooks to " << codebooks_path_ << std::endl;

#if defined(USE_PCA)
      // Generate the PCA matrix and encode the data to reduce the dimension to PRINCIPAL_DIM
      auto s_encode_data_pca = std::chrono::system_clock::now();
      generate_matrix(data_set_, sample_num_);
      pcaEncode(data_set_);

      data_dim_ = PRINCIPAL_DIM;
      auto e_encode_data_pca = std::chrono::system_clock::now();
      std::cout << "pca encode data cost: " << time_cost(s_encode_data_pca, e_encode_data_pca) << " (ms)\n";
#else
      // If PCA is not used, simply initialize these variables
      size_t length = data_dim_ / subvector_num_;
      for (int i = 0; i < subvector_num_; ++i) {
        subvector_length_[i] = length;
        pre_length_[i] = i * length;
      }
#endif
      // Generate/Read PQ's codebooks
      auto s_gen = std::chrono::system_clock::now();

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

        generate_codebooks(sample_num_, train_set.data());
      }

      auto e_gen = std::chrono::system_clock::now();
      std::cout << "generate codebooks cost: " << time_cost(s_gen, e_gen) << " (ms)\n";

      {
        std::filesystem::path fsPath(codebooks_path_);
        fsPath.remove_filename();
        std::filesystem::create_directories(fsPath);
        std::ofstream out(codebooks_path_, std::ios::binary);

#if defined(USE_PCA)
        // save pca info
        for (int j = 0; j < ori_dim; ++j) {
          out.write(reinterpret_cast<char*>(&data_mean_(j)), sizeof(float));
        }
        for (int i = 0; i < ori_dim; ++i) {
          for (int j = 0; j < PRINCIPAL_DIM; ++j) {
            out.write(reinterpret_cast<char*>(&principal_components(i, j)), sizeof(float));
          }
        }
#endif

        // save pq info
        out.write(reinterpret_cast<char*>(&qmin), sizeof(float));
        out.write(reinterpret_cast<char*>(&qmax), sizeof(float));
        for (int i = 0; i < subvector_num_; ++i) {
          out.write(reinterpret_cast<char*>(&pre_length_[i]), sizeof(size_t));
        }
        for (int i = 0; i < subvector_num_; ++i) {
          out.write(reinterpret_cast<char*>(&subvector_length_[i]), sizeof(size_t));
        }

        auto& codebooks = hnswlib::flash_codebooks_v4_;
        for (size_t i = 0, index = 0; i < SUBVECTOR_NUM; ++i) {
          for (size_t j = 0; j < CLUSTER_NUM; ++j) {
            for (size_t k = 0; k < subvector_length_[i]; ++k, ++index) {
              out.write(reinterpret_cast<char*>(&codebooks[index]), sizeof(float));
            }
          }
        }

        auto& dist = hnswlib::flash_dist_v4_;
        for (int i = 0; i < SUBVECTOR_NUM * CLUSTER_NUM * CLUSTER_NUM; ++i) {
          out.write(reinterpret_cast<char*>(&dist[i]), sizeof(data_t));
        }
      }
    }

    // Build index
    if (need_build_index) {
      std::cout << "build index to " << index_path_ << std::endl;

      auto s_build = std::chrono::system_clock::now();
      hnsw = new hnswlib::HierarchicalNSWFlash_V4<data_t>(&flash_space, data_num_, M_, ef_construction_);
      // Encode data with PQ and SQ and add point
#pragma omp parallel for schedule(dynamic) num_threads(NUM_THREADS)
      for (size_t i = 0; i < data_num_; ++i) {
        uint8_t* encoded_data = thread_encoded_vector[omp_get_thread_num()];
        pqEncode(data_set_[i].data(),
                 (encode_t*)(encoded_data + subvector_num_ * CLUSTER_NUM * sizeof(data_t)),
                 (data_t*)encoded_data, 0);
        hnsw->addPoint(encoded_data, i);

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
#if defined(ADSAMPLING)
    hnswlib::init_ratio();
#endif
#if defined(USE_PCA)
    pcaEncode(query_set_);
#endif
    hnsw->setEf(EF_SEARCH);

    for (size_t k = 0; k < REPEATED_COUNT; ++k) {
#pragma omp parallel for schedule(dynamic) num_threads(NUM_THREADS)
      for (size_t i = 0; i < query_num_; ++i) {
        // Encode query with PQ
        // uint8_t* encoded_query = thread_encoded_vector[omp_get_thread_num()];
        thread_local std::vector<char> encoded_query(SUBVECTOR_NUM * CLUSTER_NUM * sizeof(pq_dist_t) +
                                                     SUBVECTOR_NUM * sizeof(encode_t));

        auto s_pq_encode = std::chrono::steady_clock::now();
        pqEncode(query_set_[i].data(),
                 (encode_t*)(encoded_query.data() + subvector_num_ * CLUSTER_NUM * sizeof(data_t)),
                 (data_t*)encoded_query.data(), true);
        auto e_pq_encode = std::chrono::steady_clock::now();
        pq_encode_cost += time_cost(s_pq_encode, e_pq_encode);

        // search
#if defined(RERANK)
        size_t rerank_topk = K * 1.2f;
        if (K < 10) {
          rerank_topk = K + 10;
        }

        auto s_knn = std::chrono::steady_clock::now();

        std::priority_queue<std::pair<data_t, hnswlib::labeltype>> tmp =
            hnsw->searchKnn(encoded_query.data(), rerank_topk);
        auto e_knn = std::chrono::steady_clock::now();
        knn_search_cost += time_cost(s_knn, e_knn);

        std::priority_queue<std::pair<float, hnswlib::labeltype>,
                            std::vector<std::pair<float, hnswlib::labeltype>>, std::greater<>>
            result;

        if (need_debug && i == 0) {
          std::cout << "search rerank res: " << std::endl;
        }

        auto s_rerank = std::chrono::steady_clock::now();
        while (!tmp.empty()) {
          float res = 0;
          const auto& top_item = tmp.top();
          if (need_debug && i == 0) {
            std::cout << "[" << (data_t)top_item.first << ", " << top_item.second << "]" << "\t";
          }
          size_t data_id = top_item.second;

          // res =
          //     hnswlib::L2SqrSIMD16ExtSSE(data_set_[data_id].data(), query_set_[i].data(), &ori_dim,
          //     nullptr);

          res = hnswlib::FlashL2::RerankWithSSE16(data_set_[data_id].data(), query_set_[i].data(), &ori_dim);

          result.emplace(res, data_id);
          tmp.pop();
        }
        auto e_rerank = std::chrono::steady_clock::now();
        rerank_cost += time_cost(s_rerank, e_rerank);

        if (need_debug && i == 0) {
          std::cout << std::endl;
        }
#else
        std::priority_queue<std::pair<data_t, hnswlib::labeltype>> result = hnsw->searchKnn(encoded_query, K);
#endif
        if (need_debug && i == 0) {
          std::cout << "search topk res: " << std::endl;
        }

        while (!result.empty() && knn_results_[i].size() < K) {
          knn_results_[i].emplace_back(result.top().second);
          if (need_debug && i == 0) {
            std::cout << "[" << result.top().first << ", " << result.top().second << "]" << "\t";
          }

          result.pop();
        }
        if (need_debug && i == 0) {
          std::cout << std::endl;
        }

        while (knn_results_[i].size() < K) {
          knn_results_[i].emplace_back(-1);
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
    std::cout << "\tknn upper layer cost: " << hnsw->knn_upper_layer_cost / 1000000 << " (ms)" << std::endl;
    std::cout << "\tknn base layer cost: " << hnsw->knn_base_layer_cost / 1000000 << " (ms)" << std::endl;
    std::cout << "rerank cost: " << rerank_cost / 1000000 << " (ms)" << std::endl;

    for (int i = 0; i < NUM_THREADS; ++i) {
      free(thread_encoded_vector[i]);
    }
    free(pre_length_);
    free(subvector_length_);
    free(thread_encoded_vector);
    free(hnswlib::flash_codebooks_v4_);
    free(hnswlib::flash_dist_v4_);
    free(hnswlib::flash_data_dist_table_);
  };

 protected:
  void generate_codebooks(int n, const float* x) {
    size_t subspace_len = data_dim_ / SUBVECTOR_NUM;
    size_t pre_subspace_size = 0;

    std::cout << "subspace_len: " << subspace_len << std::endl;

    // generate codebook
    auto* codebooks = hnswlib::flash_codebooks_v4_;
    codebooks = (float*)malloc(CLUSTER_NUM * data_dim_ * sizeof(float));

    for (size_t i = 0; i < SUBVECTOR_NUM; ++i) {
      std::cout << "begin kMeans for subspace: (" << i + 1 << " / " << SUBVECTOR_NUM << ")" << std::endl;
      Eigen::MatrixXf subspace_data(n, subspace_len);
      size_t cur_subspace_prelen = i * subspace_len;
      for (size_t j = 0; j < n; ++j) {
        float* cur_emb = const_cast<float*>(x) + j * data_dim_;
        subspace_data.row(j) = Eigen::Map<Eigen::VectorXf>(cur_emb + cur_subspace_prelen, subspace_len);
      }

      Eigen::MatrixXf centroid_matrix = kMeans(subspace_data, CLUSTER_NUM, MAX_ITERATIONS);
      auto* cur_codebook_ptr = codebooks + pre_subspace_size;
      for (size_t j = 0; j < CLUSTER_NUM; ++j) {
        Eigen::VectorXf row = centroid_matrix.row(j);
        std::copy(row.data(), row.data() + row.size(), cur_codebook_ptr + j * subspace_len);
      }

      pre_subspace_size += CLUSTER_NUM * subspace_len;
    }

    // get quantize param
    pre_subspace_size = 0;
    qmax = 0;
    qmin = std::numeric_limits<pq_dist_t>::max();

    float* tmp_table = (float*)malloc(SUBVECTOR_NUM * CLUSTER_NUM * CLUSTER_NUM * sizeof(float));
    memset(tmp_table, 0, SUBVECTOR_NUM * CLUSTER_NUM * CLUSTER_NUM * sizeof(float));
    float* ptr_tmp_table = tmp_table;
    for (size_t i = 0; i < SUBVECTOR_NUM; ++i) {
      float max_dis = 0;
      auto* cur_codebook_ptr = codebooks + pre_subspace_size;

      for (size_t c1 = 0; c1 < CLUSTER_NUM; ++c1) {
        for (size_t c2 = 0; c2 < CLUSTER_NUM; ++c2) {
          if (c1 == c2) {
            continue;
          }
          Eigen::VectorXf v1 =
              Eigen::Map<Eigen::VectorXf>(cur_codebook_ptr + c1 * subspace_len, subspace_len);
          Eigen::VectorXf v2 =
              Eigen::Map<Eigen::VectorXf>(cur_codebook_ptr + c2 * subspace_len, subspace_len);

          *ptr_tmp_table = (v1 - v2).squaredNorm();
          qmin = std::min(qmin, *ptr_tmp_table);
          max_dis = std::max(max_dis, *ptr_tmp_table);
          ptr_tmp_table += 1;
        }
      }

      qmax += max_dis;
      pre_subspace_size += CLUSTER_NUM * subspace_len;
    }
    qmax -= qmin;

    ptr_tmp_table = tmp_table;

    hnswlib::flash_dist_v4_ =
        (pq_dist_t*)malloc(SUBVECTOR_NUM * CLUSTER_NUM * CLUSTER_NUM * sizeof(pq_dist_t));
    pq_dist_t* ptr_pq_center_dis_table_ = hnswlib::flash_dist_v4_;

    for (size_t i = 0; i < SUBVECTOR_NUM; ++i) {
      for (size_t c1 = 0; c1 < CLUSTER_NUM; ++c1) {
        for (size_t c2 = 0; c2 < CLUSTER_NUM; ++c2) {
          float ratio = (*ptr_tmp_table - qmin) / qmax;
          if (ratio < 0) {
            ratio = 0;
          } else if (ratio > 1) {
            ratio = 1;
          }
          *ptr_pq_center_dis_table_ = ratio * std::numeric_limits<pq_dist_t>::max();

          ++ptr_tmp_table;
          ++ptr_pq_center_dis_table_;
        }
      }
    }

    free(tmp_table);
  }

  void generate_codebooks_org(std::vector<std::vector<float>>& data_set_, size_t sample_num) {
    // Sample sample_num data points from the range [0, data_num_)
    std::vector<size_t> subset_data(sample_num_);
    std::random_device rd;
    std::mt19937 g(rd());
    // std::mt19937 g(19260817);
    std::uniform_int_distribution<size_t> dis(0, data_num_ - 1);
    for (size_t i = 0; i < sample_num; ++i) {
      subset_data[i] = dis(g);
    }

    auto& codebooks = hnswlib::flash_codebooks_v4_;
    codebooks = (float*)malloc(CLUSTER_NUM * data_dim_ * sizeof(float));

    // Iterate through each subvector
    size_t pre_subvector_size = 0;
    for (size_t i = 0; i < subvector_num_; ++i) {
      MatrixXf subvector_data(sample_num, subvector_length_[i]);
      for (size_t j = 0; j < sample_num; ++j) {
        // Map the subvectors in the dataset to the rows of the Eigen matrix
        // The dimension of subvecotr[i] is in range [pre_length_[i], pre_length_[i] + subvector_length_[i])
        subvector_data.row(j) =
            Eigen::Map<VectorXf>(data_set_[subset_data[j]].data() + pre_length_[i], subvector_length_[i]);
      }
      // Perform k-means clustering on the subvector data to obtain the cluster center matrix.
      MatrixXf centroid_matrix = kMeans(subvector_data, CLUSTER_NUM, MAX_ITERATIONS);

      auto* cur_codebook_ptr = codebooks + pre_subvector_size;
      // Store each cluster center from the cluster center matrix into the codebook.
      for (int r = 0; r < centroid_matrix.rows(); ++r) {
        Eigen::VectorXf row = centroid_matrix.row(r);
        std::copy(row.data(), row.data() + row.size(), cur_codebook_ptr + r * subvector_length_[i]);
      }

      pre_subvector_size += CLUSTER_NUM * subvector_length_[i];
    }

    // Calculate the distance table between the clusters of each subvector
    hnswlib::flash_dist_v4_ = (data_t*)malloc(SUBVECTOR_NUM * CLUSTER_NUM * CLUSTER_NUM * sizeof(data_t));
    data_t* dist_ptr = hnswlib::flash_dist_v4_;
    float* fdist = (float*)malloc(SUBVECTOR_NUM * CLUSTER_NUM * CLUSTER_NUM * sizeof(float));
    float* fdist_ptr = fdist;
    qmin = FLT_MAX;
    qmax = 0;
    pre_subvector_size = 0;
    for (size_t i = 0; i < subvector_num_; ++i) {
      float max_dist = 0;
      auto* cur_codebook_ptr = codebooks + pre_subvector_size;

      for (size_t j1 = 0; j1 < CLUSTER_NUM; ++j1) {
        for (size_t j2 = 0; j2 < CLUSTER_NUM; ++j2) {
          VectorXf v1 =
              Eigen::Map<VectorXf>(cur_codebook_ptr + j1 * subvector_length_[i], subvector_length_[i]);
          VectorXf v2 =
              Eigen::Map<VectorXf>(cur_codebook_ptr + j2 * subvector_length_[i], subvector_length_[i]);
          *fdist_ptr = (v1 - v2).squaredNorm();
          qmin = std::min(*fdist_ptr, qmin);
          max_dist = std::max(*fdist_ptr, max_dist);
          fdist_ptr++;
        }
      }

      qmax += max_dist;
      pre_subvector_size += CLUSTER_NUM * subvector_length_[i];
    }

    qmax -= qmin;
    fdist_ptr = fdist;

    // Perform SQ on distance table
    for (int i = 0; i < subvector_num_; ++i) {
      for (int j1 = 0; j1 < CLUSTER_NUM; ++j1) {
        for (int j2 = 0; j2 < CLUSTER_NUM; ++j2) {
          float value = (*fdist_ptr - qmin) / qmax;
          if (value > 1) value = 1.0f;
          *dist_ptr = (double)std::numeric_limits<data_t>::max() * value;
          fdist_ptr++;
          dist_ptr++;
        }
      }
    }

    free(fdist);
  }

  MatrixXf kMeanspp_init(const MatrixXf& data, int k) {
    size_t n_samples = data.rows();
    size_t n_features = data.cols();

    Eigen::MatrixXf centers(k, n_features);
    Eigen::VectorXf min_distance = Eigen::VectorXf::Constant(n_samples, std::numeric_limits<float>::max());

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> index(0, n_samples - 1);

    size_t first_idx = index(gen);
    centers.row(0) = data.row(first_idx);

    for (size_t c = 1; c < k; ++c) {
      for (int i = 0; i < n_samples; ++i) {
        float dist = (data.row(i) - centers.row(c - 1)).squaredNorm();
        min_distance[i] = std::min(min_distance[i], dist);
      }

      float dist_sum = min_distance.sum();
      std::uniform_real_distribution<float> dist_pick(0.0, dist_sum);
      float r = dist_pick(gen);

      float acc = 0;
      int next_index = 0;
      for (; next_index < n_samples; ++next_index) {
        acc += min_distance[next_index];
        if (acc >= r) {
          break;
        }
      }

      centers.row(c) = data.row(next_index);
    }

    return centers;
  }

  Eigen::MatrixXf kMeans(const Eigen::MatrixXf& train_dataset, size_t cluster_num, size_t max_iteration) {
    size_t data_num = train_dataset.rows();
    size_t data_dim = train_dataset.cols();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, data_num - 1);

    Eigen::MatrixXf centroids = kMeanspp_init(train_dataset, cluster_num);

    // kMeans
    std::vector<size_t> labels(data_num);
    for (size_t iter = 0; iter < max_iteration; ++iter) {
#pragma omp parallel for schedule(static) num_threads(omp_get_max_threads())
      for (size_t i = 0; i < data_num; ++i) {
        float min_dist = std::numeric_limits<float>::max();
        size_t best_index = 0;
        for (size_t j = 0; j < cluster_num; ++j) {
          float dist = (train_dataset.row(i) - centroids.row(j)).squaredNorm();
          if (dist < min_dist) {
            min_dist = dist;
            best_index = j;
          }
        }
        labels[i] = best_index;
      }

      // update new centroids
      Eigen::MatrixXf new_centroids = Eigen::MatrixXf::Zero(cluster_num, data_dim);
      std::vector<int> counts(cluster_num, 0);

#pragma omp parallel for schedule(static) num_threads(omp_get_max_threads())
      for (size_t i = 0; i < data_num; ++i) {
        new_centroids.row(labels[i]) += train_dataset.row(i);
        counts[labels[i]]++;
      }

#pragma omp parallel for schedule(static) num_threads(omp_get_max_threads())
      for (size_t i = 0; i < cluster_num; ++i) {
        if (counts[i] > 0) {
          new_centroids.row(i) /= counts[i];
        } else {
          // If a centroid has no points assigned, reinitialize it randomly
          new_centroids.row(i) = train_dataset.row(dis(gen));
        }
      }

      if (new_centroids.isApprox(centroids, 1e-3)) {
        break;
      }
      centroids = new_centroids;
    }

    return centroids;
  }

  MatrixXf kMeans_org(const MatrixXf& data_set, size_t cluster_num, size_t max_iterations) {
    // Initialize centroids randomly
    size_t data_num = data_set.rows();
    size_t data_dim = data_set.cols();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, data_num - 1);

    // MatrixXf centroids(cluster_num, data_dim);
    // for (size_t i = 0; i < cluster_num; ++i) {
    //     centroids.row(i) = data_set.row(dis(gen));
    // }

    MatrixXf centroids = kMeanspp_init(data_set, cluster_num);

    // kMeans
    std::vector<size_t> labels(data_num);
    auto startTime = std::chrono::steady_clock::now();
    for (size_t iter = 0; iter < max_iterations; ++iter) {
      progressBar(iter, max_iterations, startTime);

      // Assign labels to each data point, that is, find the nearest cluster center to it.
#pragma omp parallel for schedule(dynamic) num_threads(NUM_THREADS)
      for (size_t i = 0; i < data_num; ++i) {
        float min_dist = FLT_MAX;
        size_t best_index = 0;
        for (size_t j = 0; j < cluster_num; ++j) {
          float dist = (data_set.row(i) - centroids.row(j)).squaredNorm();
          if (dist < min_dist) {
            min_dist = dist;
            best_index = j;
          }
        }
        labels[i] = best_index;
      }

      // Update the cluster centers, calculating the mean of all data points in each cluster as the new
      // cluster center.
      MatrixXf new_centroids = MatrixXf::Zero(cluster_num, data_dim);
      std::vector<int> counts(cluster_num, 0);
#pragma omp parallel for schedule(dynamic) num_threads(NUM_THREADS)
      for (size_t i = 0; i < data_num; ++i) {
        new_centroids.row(labels[i]) += data_set.row(i);
        counts[labels[i]]++;
      }
      for (size_t j = 0; j < cluster_num; ++j) {
        if (counts[j] != 0) {
          new_centroids.row(j) /= counts[j];
        } else {
          new_centroids.row(j) =
              data_set.row(dis(gen));  // Reinitialize a random centroid if no points are assigned
        }
      }

      if (new_centroids.isApprox(centroids, 1e-3)) {
        // std::cout << "Converged at iteration " << iter << std::endl;
        break;  // Convergence check
      }

      centroids = new_centroids;
    }
    progressBar(max_iterations, max_iterations, startTime);
    return centroids;
  }

  static inline float sum_four(__m128 v) {
    __m128 sum1 = _mm_hadd_ps(v, v);        // [a+b, c+d, a+b, c+d]
    __m128 sum2 = _mm_hadd_ps(sum1, sum1);  // [a+b+c+d, a+b+c+d, ...]
    return _mm_cvtss_f32(sum2);             // 取第一个元素
  }

  inline float sum_first_two(__m128 v) {
    __m128 sum = _mm_add_ss(v, _mm_shuffle_ps(v, v, 0x55));
    return _mm_cvtss_f32(sum);
  }

  /**
   * Perform PQ encoding on the given data and compute the distance table between the encoded vectors and the
   * original data. Then, perform SQ encoding on the distance table with an upper bound of the sum of the
   * maximum distance from each subvector. When encoding base data, the distance table with qmin and qmax
   * remains stable. When encoding query data, the distance table with qmin and qmax needs to be recalculated.
   * @param data Pointer to the data to be encoded
   * @param encoded_vector Pointer to the encoded vector
   * @param dist_table Pointer to the distance table
   * @param is_query Flag indicating whether the data is a query: 1 for query data, 0 for non-query data
   */

  void pqEncode(float* data, encode_t* encode_vector, data_t* dist_table, bool is_query = true) {
    thread_local std::vector<float> raw_dist_table(SUBVECTOR_NUM * CLUSTER_NUM);
    float* codebook_ptr = hnswlib::flash_codebooks_v4_;

    size_t dist_table_index = 0;
    float min_dist = std::numeric_limits<float>::max(), max_dist = 0;
    size_t subspace_len = ori_dim / SUBVECTOR_NUM;

    // 填充 raw_dist_table
    size_t cur_prelen = 0;
    for (size_t i = 0; i < SUBVECTOR_NUM; ++i) {
      float* data_ptr = data + i * subspace_len;
      encode_t best_index = 0;

      float subspace_min_dist = std::numeric_limits<float>::max();
      float subspace_max_dist = 0;
      if (subspace_len == 4) {
        __m128 cal_res;
        cal_res = _mm_set1_ps(0);

        __m128 v1;
        __m128 v2;
        __m128 diff;
        v1 = _mm_loadu_ps(data_ptr);
        // 每次处理 4 个 float, 即 1 个 cluster
        for (size_t j = 0; j < CLUSTER_NUM; ++j) {
          float res = 0;
          v2 = _mm_loadu_ps(codebook_ptr);
          diff = _mm_sub_ps(v1, v2);
          cal_res = _mm_mul_ps(diff, diff);
          res = sum_four(cal_res);

          if (res < subspace_min_dist) {
            subspace_min_dist = res;
            best_index = j;
          } else if (res > subspace_max_dist) {
            subspace_max_dist = res;
          }

          raw_dist_table[dist_table_index++] = res;
          codebook_ptr += 4;
        }
      } else if (subspace_len == 2) {
        __m128 cal_res;
        cal_res = _mm_set1_ps(0);

        __m128 v1;
        __m128 v2;
        __m128 diff;

        __m128 a = _mm_set1_ps(data_ptr[0]);  // [a, a, a, a]
        __m128 b = _mm_set1_ps(data_ptr[1]);  // [b, b, b, b]
        v1 = _mm_unpacklo_ps(a, b);           // [a, b, a, b]
        alignas(16) float tmp_res[4];

        // 每次处理 4 个 float, 即 2 个 cluster
        for (size_t j = 0; j < CLUSTER_NUM; j += 2) {
          v2 = _mm_loadu_ps(codebook_ptr);
          diff = _mm_sub_ps(v1, v2);
          cal_res = _mm_mul_ps(diff, diff);
          cal_res = _mm_hadd_ps(cal_res, cal_res);  // 【a+b, c+d, a+b, c+d】
          _mm_store_ps(tmp_res, cal_res);

          for (size_t k = 0; k < 2; ++k) {
            auto res = tmp_res[k];
            if (res < subspace_min_dist) {
              subspace_min_dist = res;
              best_index = j + k;
            } else if (res > subspace_max_dist) {
              subspace_max_dist = res;
            }
          }

          raw_dist_table[dist_table_index] = tmp_res[0];
          raw_dist_table[dist_table_index + 1] = tmp_res[1];
          dist_table_index += 2;
          codebook_ptr += 4;
        }
      } else if (subspace_len == 1) {
        __m128 cal_res;
        cal_res = _mm_set1_ps(0);

        __m128 v1;
        __m128 v2;
        __m128 diff;

        v1 = _mm_set1_ps(*data_ptr);
        alignas(16) float tmp_res[4];

        // 每次处理 4 个 float, 即 4 个 cluster
        for (size_t j = 0; j < CLUSTER_NUM; j += 4) {
          v2 = _mm_loadu_ps(codebook_ptr);
          diff = _mm_sub_ps(v1, v2);
          cal_res = _mm_mul_ps(diff, diff);

          _mm_store_ps(tmp_res, cal_res);
          for (size_t k = 0; k < 4; ++k) {
            auto cur_res = tmp_res[k];
            if (cur_res < subspace_min_dist) {
              subspace_min_dist = cur_res;
              best_index = j + k;
            } else if (cur_res > subspace_max_dist) {
              subspace_max_dist = cur_res;
            }
          }

          raw_dist_table[dist_table_index] = tmp_res[0];
          raw_dist_table[dist_table_index + 1] = tmp_res[1];
          raw_dist_table[dist_table_index + 2] = tmp_res[2];
          raw_dist_table[dist_table_index + 3] = tmp_res[3];
          dist_table_index += 4;
          codebook_ptr += 4;
        }
      }

      min_dist = std::min(min_dist, subspace_min_dist);
      max_dist += subspace_max_dist;
      encode_vector[i] = best_index;
    }
    max_dist -= min_dist;

    // 量化 raw_dist_table，并将结果填充到 dist_table
    // query 使用独立的 qmin 和 qmax
    // index data 使用码本 qmin 和 qmax
    if (!is_query) {
      max_dist = qmax;
      min_dist = qmin;
    }

    auto* raw_dist_table_ptr = raw_dist_table.data();
    float qscale = 1 / max_dist;
    for (size_t i = 0; i < SUBVECTOR_NUM; ++i) {
      for (size_t j = 0; j < CLUSTER_NUM; ++j) {
        float ratio = (*raw_dist_table_ptr - min_dist) * qscale;
        if (ratio < 0) {
          ratio = 0;
        } else if (ratio > 1) {
          ratio = 1;
        }

        *dist_table = (pq_dist_t)(ratio * std::numeric_limits<pq_dist_t>::max());
        ++dist_table;
        ++raw_dist_table_ptr;
      }
    }
  }

  void pqEncode_org(float* data, encode_t* encoded_vector, data_t* dist_table, int is_query = 1) {
    thread_local std::vector<float> dist(CLUSTER_NUM * SUBVECTOR_NUM);
    // std::unique_ptr<float, decltype(&std::free)> dist_ptr(dist, &std::free);
    // Calculate the distance from each subvector to each cluster center.
    float* codebook_ptr = hnswlib::flash_codebooks_v4_;
    size_t cur_codebook_prelen = 0;

    size_t dist_index = 0;
    auto s_pq_dist = std::chrono::steady_clock::now();

    float min_dist = FLT_MAX, max_dist = 0;
    for (size_t i = 0; i < subvector_num_; ++i) {
      size_t cur_pre_len = pre_length_[i];
      float* data_ptr = data + cur_pre_len;
      size_t cur_subvec_len = subvector_length_[i];

      float subvec_max_dist = 0;
      float subvec_min_dist = FLT_MAX;
      encode_t best_index = 0;

#if defined(OPT)
      if (cur_subvec_len == 4) {
        __m128 cal_res;
        cal_res = _mm_set1_ps(0);

        __m128 v1;
        __m128 v2;
        __m128 diff;
        v1 = _mm_loadu_ps(data_ptr);

        // 每次处理一个 cluster center
        for (size_t j = 0; j < CLUSTER_NUM; ++j) {
          float res = 0;
          v2 = _mm_loadu_ps(codebook_ptr);
          diff = _mm_sub_ps(v1, v2);
          cal_res = _mm_mul_ps(diff, diff);
          res = sum_four(cal_res);

          if (res < subvec_min_dist) {
            best_index = j;
            subvec_min_dist = res;
          } else if (res > subvec_max_dist) {
            subvec_max_dist = res;
          }
          codebook_ptr += 4;

          dist[dist_index] = res;
          dist_index += 1;
        }
        max_dist += subvec_max_dist;
        min_dist = std::min(min_dist, subvec_min_dist);
        encoded_vector[i] = best_index;
      } else if (cur_subvec_len == 2) {
        // 每次处理 2 个 cluster center
        __m128 cal_res;
        cal_res = _mm_set1_ps(0);

        __m128 v1;
        __m128 v2;
        __m128 diff;

        v1 = _mm_set_ps(data_ptr[1], data_ptr[0], data_ptr[1], data_ptr[0]);
        alignas(16) float tmp_res[4];

        for (size_t j = 0; j < CLUSTER_NUM; j += 2) {
          v2 = _mm_loadu_ps(codebook_ptr);
          diff = _mm_sub_ps(v1, v2);
          cal_res = _mm_mul_ps(diff, diff);
          // 得到 [a+b, c+d, a+b, c+d]
          cal_res = _mm_hadd_ps(cal_res, cal_res);

          _mm_store_ps(tmp_res, cal_res);

          {
            auto cur_res_0 = tmp_res[0];
            if (cur_res_0 < subvec_min_dist) {
              best_index = j + 0;
              subvec_min_dist = cur_res_0;
            } else if (cur_res_0 > subvec_max_dist) {
              subvec_max_dist = cur_res_0;
            }

            auto cur_res_1 = tmp_res[1];
            if (cur_res_1 < subvec_min_dist) {
              best_index = j + 1;
              subvec_min_dist = cur_res_1;
            } else if (cur_res_1 > subvec_max_dist) {
              subvec_max_dist = cur_res_1;
            }
          }

          dist[dist_index] = tmp_res[0];
          dist[dist_index + 1] = tmp_res[1];
          dist_index += 2;

          codebook_ptr += 4;
        }

        max_dist += subvec_max_dist;
        min_dist = std::min(min_dist, subvec_min_dist);
        encoded_vector[i] = best_index;
      } else if (cur_subvec_len == 1) {
        // 每次处理 4 个 cluster center
        __m128 cal_res;
        cal_res = _mm_set1_ps(0);

        __m128 v1;
        __m128 v2;
        __m128 diff;

        v1 = _mm_set1_ps(*data_ptr);
        float PORTABLE_ALIGN32 tmp_res[4];

        for (size_t j = 0; j < CLUSTER_NUM; j += 4) {
          v2 = _mm_loadu_ps(codebook_ptr);
          diff = _mm_sub_ps(v1, v2);
          cal_res = _mm_mul_ps(diff, diff);

          _mm_store_ps(tmp_res, cal_res);

          for (size_t k = 0; k < 4; ++k) {
            auto cur_res = tmp_res[k];
            if (cur_res < subvec_min_dist) {
              best_index = j * 4 + k;
              subvec_min_dist = cur_res;
            } else if (cur_res > subvec_max_dist) {
              subvec_max_dist = cur_res;
            }
          }

          dist[dist_index] = tmp_res[0];
          dist[dist_index + 1] = tmp_res[1];
          dist[dist_index + 2] = tmp_res[2];
          dist[dist_index + 3] = tmp_res[3];
          dist_index += 4;

          codebook_ptr += 4;
        }

        max_dist += subvec_max_dist;
        min_dist = std::min(min_dist, subvec_min_dist);
        encoded_vector[i] = best_index;
      }

#else
      {
        auto* cur_codebook_ptr = codebook_ptr + cur_codebook_prelen;
        for (size_t j = 0; j < CLUSTER_NUM; ++j) {
          auto cur_res = 0;
          for (size_t k = 0; k < cur_subvec_len; ++k) {
            auto t = data_ptr[k] - cur_codebook_ptr[j * cur_subvec_len + k];
            cur_res += t * t;
          }

          if (cur_res < subvec_min_dist) {
            subvec_min_dist = cur_res;
            best_index = j;
          } else if (cur_res > subvec_max_dist) {
            subvec_max_dist = cur_res;
          }

          dist[dist_index] = cur_res;
          dist_index += 1;
        }

        max_dist += subvec_max_dist;
        min_dist = std::min(min_dist, subvec_min_dist);
        encoded_vector[i] = best_index;
      }
#endif

      cur_codebook_prelen += CLUSTER_NUM * cur_subvec_len;
    }

    max_dist -= min_dist;

    auto e_pq_dist = std::chrono::steady_clock::now();
    pq_dist_cost += std::chrono::duration_cast<std::chrono::nanoseconds>(e_pq_dist - s_pq_dist).count();

    auto s_pq_quant = std::chrono::steady_clock::now();
    if (is_query == 1) {
      auto* dist_ptr = dist.data();

      float qscale = 1 / max_dist;
      for (size_t i = 0; i < subvector_num_; ++i) {
        for (size_t j = 0; j < CLUSTER_NUM; ++j) {
          float value = (*dist_ptr - min_dist) * qscale;
          // value = std::min(value, 1.0f);
          *dist_table = (data_t)((double)std::numeric_limits<data_t>::max() * value);
          dist_table++;
          dist_ptr++;
        }
      }

      auto e_pq_quant = std::chrono::steady_clock::now();
      pq_quant_cost += std::chrono::duration_cast<std::chrono::nanoseconds>(e_pq_quant - s_pq_quant).count();

    } else {
      float* dist_ptr = dist.data();
      for (size_t i = 0; i < subvector_num_; ++i) {
        float min_dist = FLT_MAX;
        uint16_t best_index = 0;
        for (size_t j = 0; j < CLUSTER_NUM; ++j, ++dist_ptr) {
          if (*dist_ptr < min_dist) {
            min_dist = *dist_ptr;
            best_index = j;
          }
        }

        encoded_vector[i] = best_index;
      }
      // qmin and qmax are obtained from the `generate_codebooks` function
      dist_ptr = dist.data();
#if defined(FLOAT32)
      memcpy(dist_table, dist_ptr, CLUSTER_NUM * subvector_num_ * sizeof(float));
#else
      for (size_t i = 0; i < subvector_num_; ++i) {
        for (size_t j = 0; j < CLUSTER_NUM; ++j) {
          float value = (*dist_ptr - qmin) / qmax;
          if (value > 1) value = 1;
          *dist_table = (double)std::numeric_limits<data_t>::max() * value;
          dist_table++;
          dist_ptr++;
        }
      }
#endif
    }
  }

  // PCA functions
 protected:
  /**
   * Generate the principal_components from the given dataset
   * @param data_set Pointer to the dataset
   * @param sample_num Number of sampled data points
   */
  void generate_matrix(std::vector<std::vector<float>>& data_set, size_t sample_num) {
    std::random_device rd;
    std::mt19937 g(rd());
    // std::mt19937 g(19260817);
    std::uniform_int_distribution<size_t> dis(0, data_num_ - 1);

    size_t data_dim = data_set_[0].size();
    Eigen::MatrixXf data(sample_num, data_dim);
#pragma omp parallel for schedule(dynamic) num_threads(NUM_THREADS)
    for (int i = 0; i < sample_num; ++i) {
      size_t idx = dis(g);
      Eigen::Map<Eigen::VectorXf> row(data_set[idx].data(), data_dim);
      data.row(i) = row.transpose();
    }

    // Calculate the mean vector of the data points
    data_mean_ = data.colwise().mean();
    // Calculate the centralized matrix of the data points
    data.rowwise() -= data_mean_.transpose();
    // Calculate the covariance matrix of the data points
    Eigen::MatrixXf covariance_matrix = (data.adjoint() * data) / float(sample_num - 1);

    // Perform eigenvalue decomposition on the covariance matrix to obtain eigenvalues and eigenvectors
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> eigensolver(covariance_matrix);

    auto eigenvalues = eigensolver.eigenvalues().reverse();
    auto eigenvectors = eigensolver.eigenvectors().rowwise().reverse();

    principal_components = eigenvectors;

#if defined(USE_PCA_OPTIMAL)
    float total_var = eigenvalues.sum();
    std::vector<float> explained_variance_ratio;
    std::vector<float> cumulative_variance_ratio;

    float acc_var = 0;
    int count = 0;
    int cur_subvec_index = 0;

    float remain_var = total_var;
    int remain_subvector_num = subvector_num_;
    float target_var = remain_var / remain_subvector_num;

    for (int i = 0; i < eigenvalues.size(); ++i) {
      acc_var += eigenvalues[i];
      count += 1;
      std::cout << "dim " << i << ", var: " << eigenvalues[i] << ", acc_var: " << acc_var << std::endl;

      float next_acc_var = acc_var;
      if (i < eigenvalues.size() - 1) {
        next_acc_var += eigenvalues[i + 1];
      }

      if ((next_acc_var >= target_var || i == eigenvalues.size() - 1)) {
        subvector_length_[cur_subvec_index] = count;

        std::cout << "sub_vec " << cur_subvec_index << ", len: " << count << ", acc_var: " << acc_var
                  << ", thres: " << target_var << std::endl;
        std::cout << std::endl;

        remain_subvector_num -= 1;
        remain_var -= acc_var;
        target_var = remain_var / remain_subvector_num;

        acc_var = 0;
        count = 0;
        cur_subvec_index += 1;
      }
    }

    int acc_len = 0;
    for (int i = 0; i < subvector_num_; ++i) {
      acc_len += subvector_length_[i];
      std::cout << "sub_vec " << i << ", len: " << subvector_length_[i] << ", acc_len: " << acc_len
                << std::endl;
    }

    pre_length_[0] = 0;
    for (int i = 1; i < subvector_num_; ++i) {
      pre_length_[i] = pre_length_[i - 1] + subvector_length_[i - 1];
    }

    // Eigen::VectorXf eigenvalues = eigensolver.eigenvalues().reverse();
    // // Calculate the proportion of each eigenvalue to the total sum of eigenvalues, that is, the variance
    // contribution ratio Eigen::VectorXf explained_variance_ratio = eigenvalues / eigenvalues.sum();
    // // Calculate the cumulative variance contribution ratio
    // Eigen::VectorXf cumulative_variance = explained_variance_ratio;

    // // float sum = 0, res_sum = 0;
    // // int len = 0, res_len = subvector_num_;
    // // for (size_t i = 0; i < PRINCIPAL_DIM; ++i) {
    // //     res_sum += cumulative_variance[data_dim_ - i - 1];
    // // }
    // // for (size_t i = 0; i < PRINCIPAL_DIM; ++i) {
    // //     sum += cumulative_variance[data_dim_ - i - 1];
    // //     len ++;
    // //     if (sum * res_len >= res_sum) {
    // //         subvector_length_[subvector_num_ - res_len] = len;
    // //         res_sum -= sum;
    // //         sum = len = 0;
    // //         res_len --;
    // //         if (res_len == 1) {
    // //             subvector_length_[subvector_num_ - 1] = PRINCIPAL_DIM - i - 1;
    // //             break;
    // //         }
    // //     }
    // // }
    // // for (int i = 0; i < subvector_num_ / 2; ++i) {
    // //     std::swap(subvector_length_[i], subvector_length_[subvector_num_ - i]);
    // // }
    // // pre_length_[0] = 0;
    // // for (int i = 1; i < subvector_num_; ++i) {
    // //     pre_length_[i] = pre_length_[i - 1] + subvector_length_[i - 1];
    // // }

    // // for (int i = 0; i < subvector_num_; ++i) {
    // //     std::cout << "subvec " << i << ", len: " << subvector_length_[i] << ", pre_len: " <<
    // pre_length_[i] << std::endl;
    // // }

#else
    // If the USE_PCA_OPTIMAL is not enabled, set the length of each subvector to the same value.
    size_t length = PRINCIPAL_DIM / subvector_num_;
    for (size_t i = 0; i < subvector_num_; ++i) {
      subvector_length_[i] = length;
      pre_length_[i] = i * length;
    }
#endif
  }

  /**
   * Perform PCA encoding on the given dataset
   * @param data_set Pointer to the dataset to be encoded
   */
  void pcaEncode(std::vector<std::vector<float>>& data_set) {
    size_t data_num = data_set.size();
    size_t data_dim = data_set[0].size();

    Eigen::MatrixXf data(data_num, data_dim);
#pragma omp parallel for schedule(dynamic) num_threads(NUM_THREADS)
    for (size_t i = 0; i < data_num; ++i) {
      Eigen::VectorXf row = Eigen::Map<Eigen::VectorXf>(data_set[i].data(), data_dim);
      // Center the data
      data.row(i) = row - data_mean_;
    }
    // PCA
    data = data * principal_components;

    std::vector<std::vector<float>>(data_num, std::vector<float>(PRINCIPAL_DIM)).swap(data_set);
#pragma omp parallel for schedule(dynamic) num_threads(NUM_THREADS)
    for (size_t i = 0; i < data_num; ++i) {
      for (size_t j = 0; j < PRINCIPAL_DIM; j++) {
        data_set[i][j] = data(i, j);
      }
    }
  }

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
  float qmin, qmax;   // The min and max bounds of SQ

  size_t* pre_length_;         // The prefix sum of subvector_length_
  size_t* subvector_length_;   // Dimension of each subvector
                               // When USE_PCA_OPTIMAL is enabled, the dimensions of the subvectors may not
                               // be equal
  Eigen::VectorXf data_mean_;  // Mean of data
  Eigen::MatrixXf principal_components;  // Principal components
};
