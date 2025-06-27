#pragma once

#include "core.h"
#include "util.h"
#include "../../third_party/hnswlib/hnswalg.h"

template<typename T>
void debug_vec(const std::vector<T>& vec) {
    std::cout << "[";
    for (const auto& v : vec) {
        std::cout << v << ", ";
    }
    std::cout << "]" << std::endl;
}

class SolveStrategy {
public:
    SolveStrategy(std::string source_path, std::string query_path, std::string codebooks_path, std::string index_path) {
        M_ = M;
        ef_construction_ = EF_CONSTRUCTION;
        ef_search_ = EF_SEARCH;
        K_ = K;

        codebooks_path_ = codebooks_path;
        index_path_ = index_path;

        // Read data and query
        ReadData(query_path, query_set_, query_num_, query_dim_);

#if defined(TRACE_SEARCH)
    query_num_ = 1;
#endif

        knn_results_.resize(query_num_);
        data_dim_ = query_dim_;

        if (!std::filesystem::exists(codebooks_path_) || !std::filesystem::exists(index_path_)) {
            ReadData(source_path, data_set_, data_num_, data_dim_);
        }

        std::cout << "strategy init done" << std::endl;
    }

    virtual void solve() = 0;

    void read_knn(std::string knn_path) {
        uint32_t num, dim;
        ReadData(knn_path, knn_results_, num, dim);
    }

    void save_knn(std::string knn_path) {
        WriteData(knn_path, knn_results_);
    }

    void recall(std::string gt_path) {
        // Read ground truth
        uint32_t gt_num, gt_dim;
        std::vector<std::vector<uint32_t>> gt_set;
        ReadData(gt_path, gt_set, gt_num, gt_dim);

        // Calculate recall
        int hit = 0;
        size_t dim = data_dim_;
        for (int i = 0; i < query_num_; ++i) {
            auto& knn = knn_results_[i];
            // auto& truth_knn = gt_set[i];
            std::vector<uint32_t> truth_knn;

            truth_knn.insert(truth_knn.end(), gt_set[i].begin(), gt_set[i].begin() + K);

            // fetch the top-K ground truth
            // std::vector<std::pair<float, uint32_t>> knn_with_dist;
            // for (auto gt : gt_set[i]) {
            //     knn_with_dist.emplace_back(std::make_pair(hnswlib::L2Sqr(query_set_[i].data(), data_set_[gt].data(), &dim), gt));
            // }
            // sort(knn_with_dist.begin(), knn_with_dist.end());
            // truth_knn.clear();
            // for (int j = 0; j < K; ++j) {
            //     truth_knn.emplace_back(knn_with_dist[j].second);
            // }
        
#if defined(DEBUG_LOG)
           if (i == 0) {
                //std::cout << "query: " << i << ", total size: " << query_set_.size() << std::endl;
                //debug_vec(query_set_[i]);
                //std::cout << std::endl;

                // std::cout << "ground truth knn: " << std::endl;
                // for (int j = 0; j < knn_with_dist.size(); ++j) {
                //     std::cout << "[" << knn_with_dist[j].first << ", " << knn_with_dist[j].second << "]\t";
                // }
                // std::cout << std::endl;
                std::cout << std::endl;

                std::cout << "topk ground truth knn: " << std::endl;
                for (int j = 0; j < truth_knn.size(); ++j) {
                    std::cout << truth_knn[j] << "\t";
                }
                std::cout << std::endl;

                std::cout << "knn search res: " << std::endl;
                for (int j = 0; j < knn.size(); ++j) {
                    std::cout << knn[j] << "\t";
                }
                std::cout << std::endl;
            }
#endif

            std::sort(knn.begin(), knn.end());
            std::sort(truth_knn.begin(), truth_knn.end());

            std::vector<uint32_t> intersection;
            std::set_intersection(knn.begin(), knn.end(), truth_knn.begin(), truth_knn.end(), std::back_inserter(intersection));
            hit += static_cast<int>(intersection.size());

            if (i == 0) {
                std::cout << "intersection size: " << intersection.size() << std::endl;
            }
        }

        float recall = hit * 1.0f / (query_num_ * K);
        std::cout << "Recall: " << recall << std::endl;
    }

protected:
    // data
    std::vector<std::vector<float>> data_set_;
    uint32_t data_num_;
    uint32_t data_dim_;
    size_t M_;
    size_t ef_construction_;

    // query
    std::vector<std::vector<float>> query_set_;
    uint32_t query_num_;
    uint32_t query_dim_;
    size_t ef_search_;

    // knn_results
    std::vector<std::vector<uint32_t>> knn_results_;
    size_t K_;

    std::string codebooks_path_;
    std::string index_path_;
};
