#pragma once

#include <alloca.h>
#include <cstdlib>
// #include <gperftools/malloc_extension.h>

#include "solve_strategy.h"
#include "../space/space_flash.h"
#include "../../third_party/hnswlib/hnswalg_flash_v2.h"

using Eigen::MatrixXf;
using Eigen::VectorXf;

class FlashV2Strategy: public SolveStrategy {
public:
    FlashV2Strategy(std::string source_path, std::string query_path, std::string codebooks_path, std::string index_path):
        SolveStrategy(source_path, query_path, codebooks_path, index_path) {

        data_path_ = source_path;
        subvector_num_ = SUBVECTOR_NUM;
        sample_num_ = std::min(SAMPLE_NUM, (size_t)data_num_);
        ori_dim = data_dim_;
        pre_length_ = (size_t *)malloc(subvector_num_ * sizeof(size_t));
        subvector_length_ = (size_t *)malloc(subvector_num_ * sizeof(size_t));
    }

    ~FlashV2Strategy() {
    }

    void solve() {        
        hnswlib::data_dim_ = subvector_num_;
        hnswlib::L2Space l2_index_space(data_dim_);
        hnswlib::FlashSpace flash_search_space(subvector_num_);

        hnswlib::HierarchicalNSWFlashV2<float, data_t>* hnsw;

        // Malloc
        Eigen::setNbThreads(NUM_THREADS);
        // To save memory and avoid excessive malloc calls during vector encoding, we allocate space for each thread separately.
        uint8_t **thread_encoded_vector = (uint8_t **)malloc(NUM_THREADS * sizeof(uint8_t *));
        for (int i = 0; i < NUM_THREADS; ++i) {
            thread_encoded_vector[i] = (uint8_t *)aligned_alloc(64, SUBVECTOR_NUM * CLUSTER_NUM * sizeof(data_t) + ori_dim * sizeof(float) + subvector_num_ * sizeof(data_t));
        }

        // Create index
        // If the index is already saved, load it from the file system
        bool need_build_index = true;
        if (std::filesystem::exists(codebooks_path_)) {
            std::cout << "load codebooks from " << codebooks_path_ << std::endl;
            std::ifstream in(codebooks_path_, std::ios::binary);

            in.read(reinterpret_cast<char*>(&qmin), sizeof(float));
            in.read(reinterpret_cast<char*>(&qmax), sizeof(float));
            for (int i = 0; i < subvector_num_; ++i) {
                in.read(reinterpret_cast<char*>(&pre_length_[i]), sizeof(size_t));
            }
            for (int i = 0; i < subvector_num_; ++i) {
                in.read(reinterpret_cast<char*>(&subvector_length_[i]), sizeof(size_t));
            }

            auto& codebooks = hnswlib::flash_codebooks_;
            codebooks = (float *)malloc(CLUSTER_NUM * data_dim_ * sizeof(float));

            for (int i = 0, ptr = 0; i < CLUSTER_NUM; ++i) {
                for (int j = 0; j < data_dim_; ++j, ++ptr) {
                    in.read(reinterpret_cast<char*>(&codebooks[ptr]), sizeof(float));
                }
            }

            auto& dist = hnswlib::flash_dist_;
            dist = (data_t *)malloc(SUBVECTOR_NUM * CLUSTER_NUM * CLUSTER_NUM * sizeof(data_t));
            for (int i = 0; i < SUBVECTOR_NUM * CLUSTER_NUM * CLUSTER_NUM; ++i) {
                in.read(reinterpret_cast<char*>(&dist[i]), sizeof(data_t));
            }


            if (std::filesystem::exists(index_path_)) {
                need_build_index = false;
                std::cout << "load index from " << index_path_ << std::endl;
                hnsw = new hnswlib::HierarchicalNSWFlashV2<float, data_t>(&l2_index_space, &flash_search_space, index_path_);
                hnsw->showNeighborDegree();
            } 
        } else {
            std::cout << "generate codebooks to " << codebooks_path_ << std::endl;

            size_t length = ori_dim / subvector_num_;
            for (int i = 0; i < subvector_num_; ++i) {
                subvector_length_[i] = length;
                pre_length_[i] = i * length;
            }

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

                // save pq info
                out.write(reinterpret_cast<char*>(&qmin), sizeof(float));
                out.write(reinterpret_cast<char*>(&qmax), sizeof(float));
                for (int i = 0; i < subvector_num_; ++i) {
                    out.write(reinterpret_cast<char*>(&pre_length_[i]), sizeof(size_t));
                }
                for (int i = 0; i < subvector_num_; ++i) {
                    out.write(reinterpret_cast<char*>(&subvector_length_[i]), sizeof(size_t));
                }

                auto& codebooks = hnswlib::flash_codebooks_;
                for (int i = 0, ptr = 0; i < CLUSTER_NUM; ++i) {
                    for (int j = 0; j < data_dim_; ++j, ++ptr) {
                        out.write(reinterpret_cast<char*>(&codebooks[ptr]), sizeof(float));
                    }
                }

                auto& dist = hnswlib::flash_dist_;
                for (int i = 0; i < SUBVECTOR_NUM * CLUSTER_NUM * CLUSTER_NUM; ++i) {
                    out.write(reinterpret_cast<char*>(&dist[i]), sizeof(data_t));
                }
            }
        }

        // Build index
        if (need_build_index) {
            std::cout << "build index to " << index_path_ << std::endl;

            auto s_build = std::chrono::system_clock::now();
            hnsw = new hnswlib::HierarchicalNSWFlashV2<float, data_t>(&l2_index_space, &flash_search_space, data_num_, M_, ef_construction_);
            // Encode data with PQ and SQ and add point
#pragma omp parallel for schedule(dynamic) num_threads(NUM_THREADS)
            for (size_t i = 0; i < data_num_; ++i) {
                uint8_t *encoded_data = thread_encoded_vector[omp_get_thread_num()];
                pqEncode(data_set_[i].data(), (char*)((char*)encoded_data + subvector_num_ * CLUSTER_NUM * sizeof(data_t)), (data_t *)encoded_data, false);
                hnsw->addPoint(encoded_data, i);

                if (i % 100000 == 0) {
                    std::cout << "add point: " << i << std::endl;
                } 
            }
            auto e_build = std::chrono::system_clock::now();
            std::cout << "build cost: " << time_cost(s_build, e_build) << " (ms)\n";
            hnsw->showNeighborDegree();

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
        hnsw->setEf(EF_SEARCH);
        ReadData(data_path_, data_set_, data_num_, data_dim_);

        // search 
        auto s_solve = std::chrono::system_clock::now();
        for (size_t k = 0; k < REPEATED_COUNT; ++k) {
            #pragma omp parallel for schedule(dynamic) num_threads(NUM_THREADS)
            for (size_t i = 0; i < query_num_; ++i) {
                // Encode query with PQ
                uint8_t *encoded_query = thread_encoded_vector[omp_get_thread_num()];
                pqEncode(query_set_[i].data(), (char*)(encoded_query + subvector_num_ * CLUSTER_NUM * sizeof(data_t)), (data_t *)encoded_query);
    
                // search
    #if defined(RERANK)
                std::priority_queue<std::pair<data_t, hnswlib::labeltype>> tmp = hnsw->searchKnn(encoded_query, K * 2);
                std::priority_queue<std::pair<float, hnswlib::labeltype>, std::vector<std::pair<float, hnswlib::labeltype>>, std::greater<>> result;
    
                if (need_debug && i == 0) {
                    std::cout << "search quantized res: " << std::endl;
                }
    
                while (!tmp.empty()) {
                    float res = 0;
                    const auto& top_item = tmp.top();
                    if (need_debug && i == 0) {
                        std::cout << "[" << (int)top_item.first << ", " << top_item.second << "]" << "\t";
                    }
    
                    size_t data_id = top_item.second;
                    for (int j = 0; j < data_dim_; ++j) {
                        float t = data_set_[data_id][j] - query_set_[i][j];
                        res += t * t;
                    }
                    result.emplace(res, data_id);
                    tmp.pop();
                }
                if(need_debug && i == 0) {
                    std::cout << std::endl;
                }
    #else
                std::priority_queue<std::pair<data_t, hnswlib::labeltype>> result = hnsw->searchKnn(encoded_query, K);
    #endif
                if (need_debug && i == 0) {
                    std::cout << "search real res: " << std::endl;
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
        std::cout << "metric_hops: " << (hnsw->metric_hops / REPEATED_COUNT / query_num_) << ", metric_distance_computations: " << (hnsw->metric_distance_computations / REPEATED_COUNT / query_num_)   << std::endl;

        for (int i = 0; i < NUM_THREADS; ++i) {
            free(thread_encoded_vector[i]);
        }
        free(pre_length_);
        free(subvector_length_);
        free(thread_encoded_vector);
        free(hnswlib::flash_codebooks_);
        free(hnswlib::flash_dist_);
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

        auto& codebooks = hnswlib::flash_codebooks_;
        codebooks = (float *)malloc(CLUSTER_NUM * data_dim_ * sizeof(float));
        // Iterate through each subvector
        for (size_t i = 0; i < subvector_num_; ++i) {
            MatrixXf subvector_data(sample_num, subvector_length_[i]);
            for (size_t j = 0; j < sample_num; ++j) {
                // Map the subvectors in the dataset to the rows of the Eigen matrix
                // The dimension of subvecotr[i] is in range [pre_length_[i], pre_length_[i] + subvector_length_[i])
                subvector_data.row(j) = Eigen::Map<VectorXf>(data_set_[subset_data[j]].data() + pre_length_[i], subvector_length_[i]);
            }
            // Perform k-means clustering on the subvector data to obtain the cluster center matrix.
            MatrixXf centroid_matrix = kMeans(subvector_data, CLUSTER_NUM, MAX_ITERATIONS);

            // Store each cluster center from the cluster center matrix into the codebook.
            for (int r = 0; r < centroid_matrix.rows(); ++r) {
                Eigen::VectorXf row = centroid_matrix.row(r);
                std::copy(row.data(), row.data() + row.size(), codebooks + r * data_dim_ + pre_length_[i]);
            }
        }

        // Calculate the distance table between the clusters of each subvector
        hnswlib::flash_dist_ = (data_t *)malloc(SUBVECTOR_NUM * CLUSTER_NUM * CLUSTER_NUM * sizeof(data_t));
        data_t *dist_ptr = hnswlib::flash_dist_;
        float *fdist = (float *)malloc(SUBVECTOR_NUM * CLUSTER_NUM * CLUSTER_NUM * sizeof(float));
        float *fdist_ptr = fdist;
        qmin = FLT_MAX;
        qmax = 0;
        for (size_t i = 0; i < subvector_num_; ++i) {
            float max_dist = 0;
            for (size_t j1 = 0; j1 < CLUSTER_NUM; ++j1) {
                for (size_t j2 = 0; j2 < CLUSTER_NUM; ++j2) {
                    VectorXf v1 = Eigen::Map<VectorXf>(codebooks + j1 * data_dim_ + pre_length_[i], subvector_length_[i]);
                    VectorXf v2 = Eigen::Map<VectorXf>(codebooks + j2 * data_dim_ + pre_length_[i], subvector_length_[i]);
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
              if (acc >= r)
                  break;
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

            // Update the cluster centers, calculating the mean of all data points in each cluster as the new cluster center.
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
                    new_centroids.row(j) = data_set.row(dis(gen)); // Reinitialize a random centroid if no points are assigned
                }
            }

            if (iter % 10 == 0) {
                // std::cout << "diff: " << (centroids - new_centroids).norm() << std::endl;
            }

            if (new_centroids.isApprox(centroids, 1e-3)) {
                // std::cout << "Converged at iteration " << iter << std::endl;
                break; // Convergence check
            }

            centroids = new_centroids;
        }
        progressBar(max_iterations, max_iterations, startTime);
        return centroids;
    }

    /**
     * Perform PQ encoding on the given data and compute the distance table between the encoded vectors and the original data.
     * Then, perform SQ encoding on the distance table with an upper bound of the sum of the maximum distance from each subvector.
     * When encoding base data, the distance table with qmin and qmax remains stable.
     * When encoding query data, the distance table with qmin and qmax needs to be recalculated.
     * @param data Pointer to the data to be encoded
     * @param encoded_vector Pointer to the encoded vector
     * @param dist_table Pointer to the distance table
     * @param is_query Flag indicating whether the data is a query: 1 for query data, 0 for non-query data
     */
    void pqEncode(float *data, char *encoded_vector, data_t *dist_table, bool is_query = true) {
        float *dist = (float *)alloca(CLUSTER_NUM * subvector_num_ * sizeof(float));
        // Calculate the distance from each subvector to each cluster center.
        // todo: 是否可以优化？
        float* index_data = (float*)encoded_vector;
        data_t* pq_data = (data_t*)(encoded_vector + ori_dim * sizeof(float));

        memcpy(index_data, data, ori_dim * sizeof(float));

        for (size_t i = 0; i < subvector_num_; ++i) {
            for (size_t j = 0; j < CLUSTER_NUM; ++j) {
                float res = 0;
                // Calculate the sum of the squared distances between the subvector and the cluster center
                for (size_t k = 0; k < subvector_length_[i]; ++k) {
                    float t = data[pre_length_[i] + k] - hnswlib::flash_codebooks_[j * data_dim_ + pre_length_[i] + k];
                    res += t * t;
                }
                dist[i * CLUSTER_NUM + j] = res;
            }
        }

        if (is_query) {
            float *dist_ptr = dist;
            float qmin = FLT_MAX, qmax = 0;
            // Iterate through each subvector to find the minimum and maximum distances.
            for (size_t i = 0; i < subvector_num_; ++i) {
                float min_dist = FLT_MAX, max_dist = 0;
                data_t best_index = 0;
                // Iterate through each cluster center to find the cluster center corresponding to the minimum distance.
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
                pq_data[i] = best_index;
            }
            qmax -= qmin;
            dist_ptr = dist;
            // Perform SQ encoding on the distance table.
            for (size_t i = 0; i < subvector_num_; ++i) {
                for (size_t j = 0; j < CLUSTER_NUM; ++j) {
                    float value = (*dist_ptr - qmin) / qmax;
                    if (value > 1) value = 1;
                    *dist_table = (double)std::numeric_limits<data_t>::max() * value;
                    dist_table++;
                    dist_ptr++;
                }
            }
        } else {
            float *dist_ptr = dist;
            for (size_t i = 0; i < subvector_num_; ++i) {
                float min_dist = FLT_MAX;
                data_t best_index = 0;
                for (size_t j = 0; j < CLUSTER_NUM; ++j, ++dist_ptr) {
                    if (*dist_ptr < min_dist) {
                        min_dist = *dist_ptr;
                        best_index = j;
                    }
                }
                pq_data[i] = best_index;
            }
            // qmin and qmax are obtained from the `generate_codebooks` function
            dist_ptr = dist;
            for (size_t i = 0; i < subvector_num_; ++i) {
                for (size_t j = 0; j < CLUSTER_NUM; ++j) {
                    float value = (*dist_ptr - qmin) / qmax;
                    if (value > 1) value = 1;
                    *dist_table = (double)std::numeric_limits<data_t>::max() * value;
                    dist_table++;
                    dist_ptr++;
                }
            }
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
        principal_components = eigensolver.eigenvectors();
        // Sort the eigenvalues in descending order
        principal_components = principal_components.rowwise().reverse();
        principal_components.conservativeResize(Eigen::NoChange, PRINCIPAL_DIM);


        if (true)
        {
            Eigen::VectorXf eigenvalues = eigensolver.eigenvalues();
            Eigen::VectorXf sorted_eigenvalues = eigenvalues.reverse();

            float total_variance = sorted_eigenvalues.sum();
            float retained_variance = sorted_eigenvalues.head(PRINCIPAL_DIM).sum();
            float explained_ratio = retained_variance / total_variance;
            std::cout << "PCA explained: variance ratio: " << explained_ratio << ", dim: " << PRINCIPAL_DIM << std::endl;
        }


#if defined(USE_PCA_OPTIMAL)
        Eigen::VectorXf eigenvalues = eigensolver.eigenvalues();
        // Calculate the proportion of each eigenvalue to the total sum of eigenvalues, that is, the variance contribution ratio
        Eigen::VectorXf explained_variance_ratio = eigenvalues / eigenvalues.sum();
        // Calculate the cumulative variance contribution ratio
        Eigen::VectorXf cumulative_variance = explained_variance_ratio;

        // If USE_PCA_OPTIMAL is enabled, dynamically adjust the length of each subvector based on the cumulative variance contribution rate.
        // Greedily control the subvector lengths to ensure the sum of cumulative variance contribution rate is balanced.
        subvector_length_.resize(subvector_num_); 
        float sum = 0, res_sum = 0;
        int len = 0, res_len = subvector_num_;
        for (size_t i = 0; i < PRINCIPAL_DIM; ++i) {
            res_sum += cumulative_variance[data_dim_ - i - 1];
        }
        for (size_t i = 0; i < PRINCIPAL_DIM; ++i) {
            sum += cumulative_variance[data_dim_ - i - 1];
            len ++;
            if (sum * res_len >= res_sum) {
                subvector_length_[subvector_num_ - res_len] = len;
                res_sum -= sum;
                sum = len = 0;
                res_len --;
                if (res_len == 1) {
                    subvector_length_[subvector_num_ - 1] = PRINCIPAL_DIM - i - 1;
                    break;
                }
            }
        }
        for (int i = 0; i < subvector_num_ / 2; ++i) {
            std::swap(subvector_length_[i], subvector_length_[subvector_num_ - i]);
        }
        pre_length_[0] = 0;
        for (int i = 1; i < subvector_length_.size(); ++i) {
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
    size_t sample_num_{0};
                   
    size_t ori_dim{0};                         // The original dim of data before PCA
    size_t pca_dim{0};
    float qmin, qmax;                       // The min and max bounds of SQ

    size_t *pre_length_;                    // The prefix sum of subvector_length_
    size_t *subvector_length_;              // Dimension of each subvector
                                            // When USE_PCA_OPTIMAL is enabled, the dimensions of the subvectors may not be equal
    Eigen::VectorXf data_mean_;             // Mean of data
    Eigen::MatrixXf principal_components;   // Principal components
};
