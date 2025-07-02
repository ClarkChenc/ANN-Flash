#pragma once

#include <alloca.h>
#include <cstdlib>
// #include <gperftools/malloc_extension.h>

#include "solve_strategy.h"
#include "../space/space_flash.h"
#include "../../third_party/hnswlib/hnswalg_flash_v3.h"

using Eigen::MatrixXf;
using Eigen::VectorXf;

class FlashStrategy_V3 : public SolveStrategy {
 public:
  FlashStrategy_V3(std::string source_path,
                   std::string query_path,
                   std::string codebooks_path,
                   std::string index_path)
      : SolveStrategy(source_path, query_path, codebooks_path, index_path) {
    data_path_ = source_path;
    subvector_num_ = SUBVECTOR_NUM;
    cluster_num_ = CLUSTER_NUM;
    // sample_num_ = std::max((size_t)(data_num_ * 0.1), SAMPLE_NUM);
    sample_num_ = (size_t)(data_num_ * 0.1);
    ori_dim = data_dim_;
    pre_length_ = (size_t*)malloc(subvector_num_ * sizeof(size_t));
    subvector_length_ = (size_t*)malloc(subvector_num_ * sizeof(size_t));
  }

  ~FlashStrategy_V3() {}

  void solve() {
    // With PQ CLUSTER_NUM set to 16, each cluster can be represented using 4 bits.
    // This allows storing two subvectors in a single byte, effectively saving space.

    hnswlib::FlashSpace<data_t> flash_space(subvector_num_);
    hnswlib::HierarchicalNSWFlash_V3<data_t, data_t>* hnsw;

    // Malloc
    Eigen::setNbThreads(NUM_THREADS);
    // To save memory and avoid excessive malloc calls during vector encoding, we allocate space for each
    // thread separately.
    char** thread_encoded_vector = (char**)malloc(NUM_THREADS * sizeof(char*));
    for (int i = 0; i < NUM_THREADS; ++i) {
      thread_encoded_vector[i] = (char*)aligned_alloc(
          64, subvector_num_ * cluster_num_ * sizeof(data_t) + subvector_num_ * sizeof(encode_t));
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

      auto& codebooks = hnswlib::flash_v3_codebooks_;
      codebooks = (float*)malloc(CLUSTER_NUM * ori_dim * sizeof(float));

      for (int i = 0, ptr = 0; i < CLUSTER_NUM; ++i) {
        for (int j = 0; j < ori_dim; ++j, ++ptr) {
          in.read(reinterpret_cast<char*>(&codebooks[ptr]), sizeof(float));
        }
      }

      auto& dist = hnswlib::flash_v3_dist_;
      dist = (data_t*)malloc(SUBVECTOR_NUM * CLUSTER_NUM * CLUSTER_NUM * sizeof(data_t));
      for (int i = 0; i < SUBVECTOR_NUM * CLUSTER_NUM * CLUSTER_NUM; ++i) {
        in.read(reinterpret_cast<char*>(&dist[i]), sizeof(data_t));
      }

      if (std::filesystem::exists(index_path_)) {
        std::cout << "load index from " << index_path_ << std::endl;
        hnsw = new hnswlib::HierarchicalNSWFlash_V3<data_t, data_t>(&flash_space, index_path_, subvector_num_,
                                                                    cluster_num_);

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
      auto s_encode_data_pca = std::chrono::steady_clock::now();
      generate_matrix(data_set_, sample_num_);
      pcaEncode(data_set_);

      auto e_encode_data_pca = std::chrono::steady_clock::now();
      std::cout << "pca encode data cost: " << time_cost(s_encode_data_pca, e_encode_data_pca) << " (ms)\n";
#else
      // If PCA is not used, simply initialize these variables
      size_t length = ori_dim / subvector_num_;
      for (int i = 0; i < subvector_num_; ++i) {
        subvector_length_[i] = length;
        pre_length_[i] = i * length;
      }
#endif
      // Generate/Read PQ's codebooks
      auto s_gen = std::chrono::system_clock::now();
      generate_codebooks(data_set_, sample_num_);
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

        auto& codebooks = hnswlib::flash_v3_codebooks_;
        for (int i = 0, ptr = 0; i < CLUSTER_NUM; ++i) {
          for (int j = 0; j < ori_dim; ++j, ++ptr) {
            out.write(reinterpret_cast<char*>(&codebooks[ptr]), sizeof(float));
          }
        }

        auto& dist = hnswlib::flash_v3_dist_;
        for (int i = 0; i < SUBVECTOR_NUM * CLUSTER_NUM * CLUSTER_NUM; ++i) {
          out.write(reinterpret_cast<char*>(&dist[i]), sizeof(data_t));
        }
      }
    }

    // Build index
    if (need_build_index) {
      std::cout << "build index to " << index_path_ << std::endl;

      auto s_build = std::chrono::system_clock::now();
      hnsw = new hnswlib::HierarchicalNSWFlash_V3<data_t, data_t>(
          &flash_space, data_num_, M_, ef_construction_, subvector_num_, cluster_num_);
      // Encode data with PQ and SQ and add point
#pragma omp parallel for schedule(dynamic) num_threads(NUM_THREADS)
      for (size_t i = 0; i < data_num_; ++i) {
        char* encoded_data = thread_encoded_vector[omp_get_thread_num()];
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

    int64_t rerank_cost = 0;
    int64_t knn_cost = 0;
    int64_t collect_cost = 0;
    int64_t pq_cost = 0;

#if defined(ADSAMPLING)
    hnswlib::init_ratio();
#endif

#if defined(USE_PCA)
    pcaEncode(query_set_);
#endif
    hnsw->setEf(EF_SEARCH);

    auto s_solve = std::chrono::steady_clock::now();
    for (size_t k = 0; k < REPEATED_COUNT; ++k) {
      // #pragma omp parallel for schedule(dynamic) num_threads(NUM_THREADS)
      for (size_t i = 0; i < query_num_; ++i) {
        // Encode query with PQ
        auto s_pq_cost = std::chrono::steady_clock::now();
        // char* encoded_query = thread_encoded_vector[omp_get_thread_num()];
        char* encoded_query = thread_encoded_vector[0];
        pqEncode(query_set_[i].data(),
                 (encode_t*)(encoded_query + subvector_num_ * CLUSTER_NUM * sizeof(data_t)),
                 (data_t*)encoded_query, true);
        auto e_pq_cost = std::chrono::steady_clock::now();
        pq_cost += std::chrono::duration_cast<std::chrono::milliseconds>(e_pq_cost - s_pq_cost).count();

        //         // search
        // #if defined(RERANK)
        //         size_t rerank_topk = K * 3;

        //         if (K < 10) {
        //           rerank_topk = K + 10;
        //         }

        //         auto s_knn_cost = std::chrono::steady_clock::now();
        //         std::priority_queue<std::pair<data_t, hnswlib::labeltype>> tmp =
        //             hnsw->searchKnn(encoded_query, rerank_topk);
        //         auto e_knn_cost = std::chrono::steady_clock::now();
        //         knn_cost += time_cost(s_knn_cost, e_knn_cost);

        //         std::priority_queue<std::pair<float, hnswlib::labeltype>,
        //                             std::vector<std::pair<float, hnswlib::labeltype>>, std::greater<>>
        //             result;

        //         if (need_debug && i == 0) {
        //           std::cout << "search rerank res: " << std::endl;
        //         }

        //         auto s_rerank = std::chrono::steady_clock::now();
        //         while (!tmp.empty()) {
        //           float res = 0;
        //           const auto& top_item = tmp.top();
        //           if (need_debug && i == 0) {
        //             std::cout << "[" << (data_t)top_item.first << ", " << top_item.second << "]" << "\t";
        //           }

        //           size_t data_id = top_item.second;
        //           for (int j = 0; j < ori_dim; ++j) {
        //             float t = data_set_[data_id][j] - query_set_[i][j];
        //             res += t * t;
        //           }
        //           result.emplace(res, data_id);
        //           tmp.pop();
        //         }
        //         if (need_debug && i == 0) {
        //           std::cout << std::endl;
        //         }
        //         auto e_rerank = std::chrono::steady_clock::now();
        //         rerank_cost += time_cost(s_rerank, e_rerank);
        // #else
        //         std::priority_queue<std::pair<data_t, hnswlib::labeltype>> result =
        //         hnsw->searchKnn(encoded_query, K);
        // #endif
        //         if (need_debug && i == 0) {
        //           std::cout << "search topk res: " << std::endl;
        //         }

        //         auto s_collect = std::chrono::steady_clock::now();
        //         while (!result.empty() && knn_results_[i].size() < K) {
        //           knn_results_[i].emplace_back(result.top().second);
        //           if (need_debug && i == 0) {
        //             std::cout << "[" << result.top().first << ", " << result.top().second << "]" << "\t";
        //           }

        //           result.pop();
        //         }
        //         if (need_debug && i == 0) {
        //           std::cout << std::endl;
        //         }

        //         while (knn_results_[i].size() < K) {
        //           knn_results_[i].emplace_back(-1);
        //         }
        //         auto e_collect = std::chrono::steady_clock::now();
        //         collect_cost += time_cost(s_collect, e_collect);
      }
    }
    auto e_solve = std::chrono::steady_clock::now();
    auto solve_cost = std::chrono::duration_cast<std::chrono::milliseconds>(e_solve - s_solve).count();

    std::cout << "solve cost: " << (solve_cost) << " (ms)" << std::endl;
    std::cout << "rerank_cost: " << rerank_cost << " (ms)" << std::endl;
    std::cout << "pq cost : " << pq_cost << " (ms)" << std::endl;
    std::cout << "search_base_layer_cost: " << (hnsw->search_base_layer_st_cost) << " (ms)" << std::endl;
    std::cout << "search_upper_layer_cost: " << (hnsw->search_upper_layer_cost) << " (ms)" << std::endl;
    std::cout << "knn_cost: " << knn_cost << " (ms)" << std::endl;
    std::cout << "collect_cost: " << collect_cost << " (ms)" << std::endl;
    std::cout << "metric_hops: " << (hnsw->metric_hops / REPEATED_COUNT / query_num_)
              << ", metric_distance_computations: "
              << (hnsw->metric_distance_computations / REPEATED_COUNT / query_num_) << std::endl;

    for (int i = 0; i < NUM_THREADS; ++i) {
      free(thread_encoded_vector[i]);
    }
    free(pre_length_);
    free(subvector_length_);
    free(thread_encoded_vector);
    free(hnswlib::flash_v3_codebooks_);
    free(hnswlib::flash_v3_dist_);
  };

 protected:
  /**
   * Generate codebooks for PQ, compute the distance table, and then perform SQ on the table
   * @param data_set_ Pointer to the dataset
   * @param sample_num Number of sampled data points
   */
  void generate_codebooks(std::vector<std::vector<float>>& data_set_, size_t sample_num) {
    // Sample sample_num data points from the range [0, data_num_)
    std::vector<size_t> subset_data(sample_num_);
    std::random_device rd;
    std::mt19937 g(rd());
    // std::mt19937 g(19260817);
    std::uniform_int_distribution<size_t> dis(0, data_num_ - 1);
    for (size_t i = 0; i < sample_num; ++i) {
      subset_data[i] = dis(g);
    }

    auto& codebooks = hnswlib::flash_v3_codebooks_;
    codebooks = (float*)malloc(CLUSTER_NUM * ori_dim * sizeof(float));
    // Iterate through each subvector
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

      // Store each cluster center from the cluster center matrix into the codebook.
      for (int r = 0; r < centroid_matrix.rows(); ++r) {
        Eigen::VectorXf row = centroid_matrix.row(r);
        std::copy(row.data(), row.data() + row.size(), codebooks + r * ori_dim + pre_length_[i]);
      }
    }

    // Calculate the distance table between the clusters of each subvector
    hnswlib::flash_v3_dist_ = (data_t*)malloc(SUBVECTOR_NUM * CLUSTER_NUM * CLUSTER_NUM * sizeof(data_t));
    data_t* dist_ptr = hnswlib::flash_v3_dist_;
    float* fdist = (float*)malloc(SUBVECTOR_NUM * CLUSTER_NUM * CLUSTER_NUM * sizeof(float));
    float* fdist_ptr = fdist;
    qmin = FLT_MAX;
    qmax = 0;
    for (size_t i = 0; i < subvector_num_; ++i) {
      float max_dist = 0;
      for (size_t j1 = 0; j1 < CLUSTER_NUM; ++j1) {
        for (size_t j2 = 0; j2 < CLUSTER_NUM; ++j2) {
          VectorXf v1 = Eigen::Map<VectorXf>(codebooks + j1 * ori_dim + pre_length_[i], subvector_length_[i]);
          VectorXf v2 = Eigen::Map<VectorXf>(codebooks + j2 * ori_dim + pre_length_[i], subvector_length_[i]);
          *fdist_ptr = (v1 - v2).squaredNorm();
          qmin = std::min(*fdist_ptr, qmin);
          max_dist = std::max(*fdist_ptr, max_dist);
          fdist_ptr++;
        }
      }
      qmax += max_dist;
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
    int n_samples = data.rows();
    int n_features = data.cols();

    MatrixXf centers(k, n_features);
    VectorXf min_distances = VectorXf::Constant(n_samples, numeric_limits<float>::max());

    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, n_samples - 1);

    // Step 1: 随机选择第一个中心
    int first_idx = dis(gen);
    centers.row(0) = data.row(first_idx);

    for (int c = 1; c < k; ++c) {
      // Step 2: 更新每个点到最近中心的距离
      for (int i = 0; i < n_samples; ++i) {
        float dist = (data.row(i) - centers.row(c - 1)).squaredNorm();
        if (dist < min_distances(i)) {
          min_distances(i) = dist;
        }
      }

      // Step 3: 按距离平方作为权重选下一个中心
      float dist_sum = min_distances.sum();
      uniform_real_distribution<float> dist_pick(0, dist_sum);
      float r = dist_pick(gen);

      float acc = 0;
      int next_idx = 0;
      for (; next_idx < n_samples; ++next_idx) {
        acc += min_distances(next_idx);
        if (acc >= r) break;
      }

      // Step 4: 选择该点作为新中心
      centers.row(c) = data.row(next_idx);
    }

    return centers;
  }

  /**
   * Perform k-means clustering on the given dataset
   * @param data_set Pointer to the dataset
   * @param cluster_num Number of clusters
   * @param max_iterations Maximum number of iterations
   * @return Returns the cluster center matrix
   */
  MatrixXf kMeans(const MatrixXf& data_set, size_t cluster_num, size_t max_iterations) {
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
  void pqEncode(float* data, encode_t* encoded_vector, data_t* dist_table, int is_query = 1) {
    // todo: 每次 encode 都申请，浪费 cpu
    // float* dist = (float *)malloc(CLUSTER_NUM * subvector_num_ * sizeof(float));

    float* dist = (float*)alloca(CLUSTER_NUM * subvector_num_ * sizeof(float));

    // std::unique_ptr<float, decltype(&std::free)> dist_ptr(dist, &std::free);
    // Calculate the distance from each subvector to each cluster center.
    for (size_t i = 0; i < subvector_num_; ++i) {
      size_t cur_pre_len = pre_length_[i];
      float* data_ptr = data + cur_pre_len;
      size_t cur_subvec_len = subvector_length_[i];

      __m128 cal_res;
      __m128 v1;
      __m128 v2;
      __m128 diff;
      for (size_t j = 0; j < CLUSTER_NUM; ++j) {
        float res = 0;
        cal_res = _mm_set1_ps(0);
        float* codebook_ptr = hnswlib::flash_v3_codebooks_ + j * ori_dim + cur_pre_len;

        if (cur_subvec_len == 4) {
          float t0 = data_ptr[0] - codebook_ptr[0];
          float t1 = data_ptr[0 + 1] - codebook_ptr[0 + 1];
          float t2 = data_ptr[0 + 2] - codebook_ptr[0 + 2];
          float t3 = data_ptr[0 + 3] - codebook_ptr[0 + 3];
          res = t0 * t0 + t1 * t1 + t2 * t2 + t3 * t3;

          // v1 = _mm_loadu_ps(data_ptr);
          // v2 = _mm_loadu_ps(codebook_ptr);
          // diff = _mm_sub_ps(v1, v2);
          // cal_res = _mm_mul_ps(diff, diff);
          // res = sum_four(cal_res);
        } else if (cur_subvec_len == 2) {
          float t0 = data_ptr[0] - codebook_ptr[0];
          float t1 = data_ptr[0 + 1] - codebook_ptr[0 + 1];
          res = t0 * t0 + t1 * t1;

          // v1 = _mm_loadu_ps(data_ptr);
          // v2 = _mm_loadu_ps(codebook_ptr);
          // diff = _mm_sub_ps(v1, v2);
          // cal_res = _mm_mul_ps(diff, diff);
          // res = sum_first_two(cal_res);

        } else {
          // Calculate the sum of the squared distances between the subvector and the cluster center
          for (size_t k = 0; k < subvector_length_[i]; ++k) {
            float t = data_ptr[k] - codebook_ptr[k];
            res += t * t;
          }
        }

        dist[i * CLUSTER_NUM + j] = res;
      }
    }

    if (is_query == 1) {
      float* dist_ptr = dist;
      float qmin = FLT_MAX, qmax = 0;
      // Iterate through each subvector to find the minimum and maximum distances.
      for (size_t i = 0; i < subvector_num_; ++i) {
        float min_dist = FLT_MAX, max_dist = 0;
        uint16_t best_index = 0;
        // Iterate through each cluster center to find the cluster center corresponding to the minimum
        // distance.
        for (size_t j = 0; j < CLUSTER_NUM; ++j, ++dist_ptr) {
          if (*dist_ptr < min_dist) {
            min_dist = *dist_ptr;
            best_index = j;
          }
          if (*dist_ptr > max_dist) {
            max_dist = *dist_ptr;
          }
        }
        // Update global minimum and maximum distance
        qmin = std::min(qmin, min_dist);
        qmax += max_dist;

        encoded_vector[i] = best_index;
      }
      qmax -= qmin;
      dist_ptr = dist;
#if defined(FLOAT32)
      memcpy(dist_table, dist_ptr, CLUSTER_NUM * subvector_num_ * sizeof(float));
#else
      for (size_t i = 0; i < subvector_num_; ++i) {
        for (size_t j = 0; j < CLUSTER_NUM; ++j) {
          float value = (*dist_ptr - qmin) / qmax;
          if (value > 1) value = 1;
          *dist_table = (data_t)((double)std::numeric_limits<data_t>::max() * value);
          dist_table++;
          dist_ptr++;
        }
      }
#endif
    } else {
      float* dist_ptr = dist;
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
      dist_ptr = dist;
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
    // free(dist);
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
  size_t cluster_num_{0};
  size_t sample_num_{0};

  size_t ori_dim{0};  // The original dim of data before PCA
  float qmin, qmax;   // The min and max bounds of SQ

  size_t* pre_length_;         // The prefix sum of subvector_length_
  size_t* subvector_length_;   // Dimension of each subvector
                               // When USE_PCA_OPTIMAL is enabled, the dimensions of the subvectors may not be
                               // equal
  Eigen::VectorXf data_mean_;  // Mean of data
  Eigen::MatrixXf principal_components;  // Principal components
};
