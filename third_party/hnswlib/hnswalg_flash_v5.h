#pragma once

#include <stdlib.h>
#include <atomic>
#include <cassert>
#include <functional>
#include <iostream>
#include <list>
#include <memory>
#include <random>
#include <unordered_set>
#include <alloca.h>

#include "flash_lib.h"
#include <cstdlib>

// #include "se/txt2vid_se/ann_engine/third_party/se_hnswlib/hnswlib.h"
// #include "folly/container/F14Map.h"
// #include "se/txt2vid_se/ann_engine/third_party/se_hnswlib/flash_lib.h"
// #include "se/txt2vid_se/ann_engine/third_party/se_hnswlib/visited_list_pool.h"

namespace hnswlib {
// typedef uint32_t tableint;
// typedef uint32_t linklistsizeint;
// typedef size_t labeltype;

template <typename dist_t>
class HnswFlash {
 public:
  constexpr static size_t max_label_op_locks = 65536;
  static const unsigned char DELETE_MARK = 0x01;
  constexpr static float default_rerank_ratio = 1.2f;

  FlashSpaceInterface<dist_t>* space_ = nullptr;
  std::string location_;
  bool is_mutable_ = true;

  size_t M_{0};
  size_t maxM_{0};
  size_t maxM0_{0};
  size_t ef_{0}, ef_construction_{0};
  double mult_{0}, rev_size_{0};
  float rerank_ratio_{1.0};
  size_t kmeans_train_round_{0};
  int maxlevel_{0};

  tableint enterpoint_node_{0};

  size_t max_elements_{0};
  mutable std::atomic<size_t> cur_element_count_{0};

  bool has_deletions_{false};
  mutable std::atomic<size_t> num_delete_{0};

  size_t size_links_per_element_{0};
  size_t size_links_level0_{0};
  size_t size_data_per_element_{0};
  size_t encode_data_size_{0};
  size_t raw_data_size_{0};
  size_t size_raw_data_per_element_{0};
  size_t data_dim_{0};
  size_t subspace_num_{0};
  size_t cluster_num_{0};
  size_t cluster_num_sqr_{0};

  size_t offset_data_{0};
  size_t offset_label_{0};

  size_t offset_encode_query_data_{0};
  size_t offset_raw_query_data_{0};

  // pq parameters
  float pq_max_{0};
  float pq_min_{0};

  char* data_level0_memory_ = nullptr;
  char** linkLists_ = nullptr;
  std::vector<int> element_levels_;

  float* raw_data_table_;

  // subspace_1, cluster_1, cluster_2, ..
  // subspace_2, cluster_1, cluster_2, ...
  float* pq_codebooks_;

  // subspace_i:
  //    -       cluster_1,  cluster_2, ...
  // cluster_1  dis_1_1,    dis_1_2, ...
  // cluster_2  dis_2_1,    dis_2_2, ...
  pq_dist_t* pq_center_dis_table_;

  VisitedListPool* visited_list_pool_{nullptr};

  std::mutex global;
  mutable std::vector<std::mutex> link_list_locks_;
  mutable std::vector<std::mutex> label_op_locks_;

  mutable std::mutex label_lookup_lock_;
  std::unordered_map<labeltype, tableint> label_lookup_;

  std::default_random_engine level_generator_;
  std::default_random_engine update_probability_generator_;

  PQ_FUNC pq_encode_func_{nullptr};
  RERANK_FUNC<dist_t> rerank_func_{nullptr};

  bool allow_update_point_{false};

  // todo: 之后考虑对 quantizer 的支持
  // quantizer_t* quantizer_ = nullptr;

  mutable std::atomic<size_t> metric_distance_computations{0};
  mutable std::atomic<size_t> metric_hops{0};

 protected:
  struct CompareByFirstLess {
    constexpr bool operator()(std::pair<dist_t, tableint> const& a,
                              std::pair<dist_t, tableint> const& b) const noexcept {
      return a.first < b.first;
    }

    constexpr bool operator()(std::pair<pq_dist_t, tableint> const& a,
                              std::pair<pq_dist_t, tableint> const& b) const noexcept {
      return a.first < b.first;
    }
  };

  struct CompareByFirstGreater {
    constexpr bool operator()(std::pair<dist_t, tableint> const& a,
                              std::pair<dist_t, tableint> const& b) const noexcept {
      return a.first > b.first;
    }

    constexpr bool operator()(std::pair<pq_dist_t, tableint> const& a,
                              std::pair<pq_dist_t, tableint> const& b) const noexcept {
      return a.first > b.first;
    }
  };

 public:
  HnswFlash(FlashSpaceInterface<dist_t>* s) {}

  HnswFlash(FlashSpaceInterface<dist_t>* s,
            const std::string& location,
            size_t max_elements = 0,
            bool is_mutable = true)
      : is_mutable_(is_mutable) {
    std::ifstream input(location, std::ios::binary);

    space_ = s;
    location_ = location;
    input.exceptions(std::ios_base::failbit | std::ios_base::badbit);
    if (!input.is_open()) throw std::runtime_error("Cannot open file");

    loadIndex(input, s, max_elements);
    input.close();
  }

  HnswFlash(FlashSpaceInterface<dist_t>* s, std::ifstream& input) {
    location_ = "UNKNOWN";
    loadIndex(input, s);
  }

  HnswFlash(FlashSpaceInterface<dist_t>* s,
            size_t max_elements,
            size_t M = 16,
            size_t ef_construction = 200,
            size_t random_seed = 100)
      : link_list_locks_(max_elements), label_op_locks_(max_label_op_locks), element_levels_(max_elements) {
    space_ = s;

    max_elements_ = max_elements;
    cur_element_count_ = 0;
    has_deletions_ = false;

    data_dim_ = s->get_data_dim();
    subspace_num_ = s->get_subspace_num();
    cluster_num_ = s->get_cluster_num();
    cluster_num_sqr_ = cluster_num_ * cluster_num_;

    encode_data_size_ = s->get_encode_data_size();
    raw_data_size_ = s->get_raw_data_size();
    pq_encode_func_ = s->get_pq_encode_func();
    rerank_func_ = s->get_rerank_func();

    M_ = M;
    maxM_ = M_;
    maxM0_ = M_ * 2;
    ef_construction_ = std::max(ef_construction, M_);
    ef_ = 10;

    level_generator_.seed(random_seed);
    update_probability_generator_.seed(random_seed + 1);

    size_links_per_element_ = sizeof(linklistsizeint) + maxM_ * sizeof(tableint);
    size_links_level0_ = sizeof(linklistsizeint) + maxM0_ * sizeof(tableint);
    size_data_per_element_ = size_links_level0_ + encode_data_size_ + sizeof(labeltype);
    size_raw_data_per_element_ = s->get_data_dim() * sizeof(float);

    offset_data_ = size_links_level0_;
    offset_label_ = size_links_level0_ + encode_data_size_;

    offset_encode_query_data_ = subspace_num_ * cluster_num_ * sizeof(pq_dist_t);
    offset_raw_query_data_ = offset_encode_query_data_ + subspace_num_ * sizeof(encode_t);

    data_level0_memory_ = (char*)aligned_alloc(64, max_elements_ * size_data_per_element_);
    if (data_level0_memory_ == nullptr) {
      throw std::runtime_error("Not enough memory");
    }
    memset(data_level0_memory_, 0, max_elements_ * size_data_per_element_);

    pq_codebooks_ = (float*)aligned_alloc(64, cluster_num_ * raw_data_size_);
    memset(pq_codebooks_, 0, cluster_num_ * raw_data_size_);

    pq_center_dis_table_ =
        (pq_dist_t*)aligned_alloc(64, subspace_num_ * cluster_num_ * cluster_num_ * sizeof(pq_dist_t));
    memset(pq_center_dis_table_, 0, subspace_num_ * cluster_num_ * cluster_num_ * sizeof(pq_dist_t));

    raw_data_table_ = (float*)aligned_alloc(64, max_elements * size_raw_data_per_element_);
    memset(raw_data_table_, 0, max_elements * size_raw_data_per_element_);

    visited_list_pool_ = new VisitedListPool(1, max_elements);

    // initializations for special treatment of the first node
    enterpoint_node_ = -1;
    maxlevel_ = -1;

    linkLists_ = (char**)malloc(sizeof(void*) * max_elements_);
    if (linkLists_ == nullptr) {
      throw std::runtime_error("Not enough memory: HnswFlash failed to allocate linklists");
    }

    mult_ = 1 / log(1.0 * M_);
    rev_size_ = 1.0 / mult_;

    rerank_ratio_ = default_rerank_ratio;
  }

  ~HnswFlash() {
    // LOG(INFO) << "~HnswFlash";
    FreeHeapData();
  }

  void FreeHeapData() {
    free(data_level0_memory_);
    data_level0_memory_ = nullptr;

    for (tableint i = 0; i < cur_element_count_; i++) {
      if (linkLists_ != nullptr) {
        if (element_levels_[i] > 0) free(linkLists_[i]);
      }
      if (i % 1000000 == 0) {
        usleep(10000);
      }
    }
    free(linkLists_);
    linkLists_ = nullptr;

    cur_element_count_ = 0;

    free(pq_codebooks_);
    pq_codebooks_ = nullptr;

    free(pq_center_dis_table_);
    pq_center_dis_table_ = nullptr;

    free(raw_data_table_);
    raw_data_table_ = nullptr;

    delete visited_list_pool_;
    visited_list_pool_ = nullptr;
  }

  size_t GetDimension() {
    return data_dim_;
  }

  uint64_t getDocCount() {
    return cur_element_count_.load();
  }

  inline labeltype getExternalLabel(tableint internal_id) const {
    labeltype return_label;
    memcpy(&return_label, (data_level0_memory_ + internal_id * size_data_per_element_ + offset_label_),
           sizeof(labeltype));
    return return_label;
  }

  inline void setExternalLabel(tableint internal_id, labeltype label) const {
    memcpy((data_level0_memory_ + internal_id * size_data_per_element_ + offset_label_), &label,
           sizeof(labeltype));
  }

  inline labeltype* getExternalLabeLp(tableint internal_id) const {
    return (labeltype*)(data_level0_memory_ + internal_id * size_data_per_element_ + offset_label_);
  }

  inline encode_t* getDataByInternalId(tableint internal_id) const {
    return (encode_t*)(data_level0_memory_ + internal_id * size_data_per_element_ + offset_data_);
  }

  inline float* getRawDataByInternalId(tableint internal_id) const {
    return (float*)(raw_data_table_ + internal_id * data_dim_);
  }

  int getRandomLevel(double reverse_size) {
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double r = -log(distribution(level_generator_)) * reverse_size;
    return (int)r;
  }

  linklistsizeint* get_linklist0(tableint internal_id) const {
    return (linklistsizeint*)(data_level0_memory_ + internal_id * size_data_per_element_);
  };

  linklistsizeint* get_linklist(tableint internal_id, int level) const {
    return level == 0 ? get_linklist0(internal_id)
                      : (linklistsizeint*)(linkLists_[internal_id] + (level - 1) * size_links_per_element_);
  };

  void setEf(size_t ef) {
    ef_ = ef;
  }

  void setKmeansTrainRound(size_t max_kmeans_train_round) {
    kmeans_train_round_ = max_kmeans_train_round;
  }

  void setRerankRatio(float rerank_ratio) {
    if (rerank_ratio > 1.0f) {
      rerank_ratio_ = rerank_ratio;
    }
  }

  void changeAllowUpdatePoint(bool expect) {
    allow_update_point_ = expect;
  }

  // void resizeVisListPool(size_t vt_siz) {
  //   visited_list_pool_->resize(vt_siz);
  // }

  std::vector<float> getRawDataByLabel(labeltype label) {
    tableint label_c;
    auto it = label_lookup_.find(label);
    if (it == label_lookup_.end() || isMarkedDeleted(it->second)) {
      return std::vector<float>();
    }
    label_c = it->second;

    std::vector<float> data(data_dim_);
    memcpy(data.data(), getRawDataByInternalId(label_c), raw_data_size_);

    return data;
  }

  std::vector<float> getFloatDataByLabel(labeltype label) {
    return getRawDataByLabel(label);
  }

  bool exist(labeltype label) {
    auto it = label_lookup_.find(label);
    if (it != label_lookup_.end()) {
      return !isMarkedDeleted(it->second);
    }
    return false;
  }

  void markDelete(labeltype label) {
    has_deletions_ = true;
    auto search = label_lookup_.find(label);
    if (search == label_lookup_.end()) {
      throw std::runtime_error("Label not found");
    }
    markDeletedInternal(search->second);
  }

  void markDeletedInternal(tableint internalId) {
    unsigned char* ll_cur = ((unsigned char*)get_linklist0(internalId)) + 2;
    *ll_cur |= DELETE_MARK;
  }

  void unmarkDeletedInternal(tableint internalId) {
    unsigned char* ll_cur = ((unsigned char*)get_linklist0(internalId)) + 2;
    *ll_cur &= ~DELETE_MARK;
  }

  bool isMarkedDeleted(tableint internalId) const {
    unsigned char* ll_cur = ((unsigned char*)get_linklist0(internalId)) + 2;
    return *ll_cur & DELETE_MARK;
  }

  unsigned short int getListCount(linklistsizeint* ptr) const {
    return *((unsigned short int*)ptr);
  }

  void setListCount(linklistsizeint* ptr, unsigned short int size) const {
    *((unsigned short int*)(ptr)) = *((unsigned short int*)&size);
  }

  int horizontal_add_epi32(__m128i v) const {
    __m128i sum = _mm_hadd_epi32(v, v);  // [x0+x1, x2+x3, x0+x1, x2+x3]
    sum = _mm_hadd_epi32(sum, sum);      // [x0+x1+x2+x3, ...]
    return _mm_cvtsi128_si32(sum);
  }

  int horizontal_add_epi32(__m256i v) const {
    // 将 __m256i 拆成两个 __m128i
    __m128i lo = _mm256_castsi256_si128(v);       // 低 128 位
    __m128i hi = _mm256_extracti128_si256(v, 1);  // 高 128 位

    // 对低高两个部分分别水平加
    lo = _mm_add_epi32(lo, hi);   // 合并为一个 __m128i（前四加后四）
    lo = _mm_hadd_epi32(lo, lo);  // [a0+a1, a2+a3, ...]
    lo = _mm_hadd_epi32(lo, lo);  // [a0+a1+a2+a3, ...]

    return _mm_cvtsi128_si32(lo);
  }

  inline pq_dist_t get_subspace_dis(size_t subspace_index, encode_t i, encode_t j) const {
    return pq_center_dis_table_[subspace_index * cluster_num_sqr_ + i * cluster_num_ + j];
  }

  pq_dist_t get_c2c_dis(const encode_t* p_encode1, const encode_t* p_encode2) const {
    encode_t* ptr_encode1 = (encode_t*)p_encode1;
    encode_t* ptr_encode2 = (encode_t*)p_encode2;

    pq_dist_t dis = 0;
    for (size_t i = 0; i < subspace_num_; ++i) {
      dis += get_subspace_dis(i, ptr_encode1[i], ptr_encode2[i]);
    }

    return dis;
  }

  pq_dist_t get_pq_dis(const void* p_vec1, const void* p_vec2) const {
    pq_dist_t dis = 0;

    pq_dist_t* ptr_vec1 = (pq_dist_t*)p_vec1;
    encode_t* ptr_vec2 = (encode_t*)p_vec2;

    __m128i sum = _mm_setzero_si128();
    __m128i v1;
    __m128i v2;
    __m128i tmp;
    for (size_t i = 0; i < subspace_num_; i += 8) {
      v1 = _mm_set_epi32(ptr_vec1[ptr_vec2[0]], ptr_vec1[1 * cluster_num_ + ptr_vec2[1]],
                         ptr_vec1[2 * cluster_num_ + ptr_vec2[2]], ptr_vec1[3 * cluster_num_ + ptr_vec2[3]]);
      v2 = _mm_set_epi32(ptr_vec1[4 * cluster_num_ + ptr_vec2[4]], ptr_vec1[5 * cluster_num_ + ptr_vec2[5]],
                         ptr_vec1[6 * cluster_num_ + ptr_vec2[6]], ptr_vec1[7 * cluster_num_ + ptr_vec2[7]]);

      tmp = _mm_add_epi32(v1, v2);
      sum = _mm_add_epi32(sum, tmp);
      ptr_vec1 += 8 * cluster_num_;
      ptr_vec2 += 8;
    }
    dis = horizontal_add_epi32(sum);

    return dis;
  }

  void get_pq_dist_batch(const void* result,
                         size_t num,
                         const void* qp_dis_table,
                         const void* pq_encodes) const {
    pq_dist_t* res = (pq_dist_t*)result;
    memset(res, 0, num * sizeof(pq_dist_t));

    pq_dist_t* ptr_qp_dis_table = (pq_dist_t*)qp_dis_table;
    encode_t* ptr_pq_encodes = (encode_t*)pq_encodes;

    for (size_t i = 0; i < num; ++i) {
      res[i] = get_pq_dis(ptr_qp_dis_table, ptr_pq_encodes + i * subspace_num_);
    }
  }

  std::priority_queue<std::pair<pq_dist_t, tableint>,
                      std::vector<std::pair<pq_dist_t, tableint>>,
                      CompareByFirstLess>
  searchBaseLayer(tableint ep_id, const void* data_point, int layer) {
    VisitedList* vl = visited_list_pool_->getFreeVisitedList();
    vl_type* visited_array = vl->mass;
    vl_type visited_array_tag = vl->curV;

    std::priority_queue<std::pair<pq_dist_t, tableint>, std::vector<std::pair<pq_dist_t, tableint>>,
                        CompareByFirstLess>
        top_candidates;
    std::priority_queue<std::pair<pq_dist_t, tableint>, std::vector<std::pair<pq_dist_t, tableint>>,
                        CompareByFirstGreater>
        candidateSet;

    pq_dist_t lowerBound;
    if (!isMarkedDeleted(ep_id)) {
      pq_dist_t dist = get_pq_dis(data_point, getDataByInternalId(ep_id));

      top_candidates.emplace(dist, ep_id);
      lowerBound = dist;
      candidateSet.emplace(dist, ep_id);
    } else {
      lowerBound = std::numeric_limits<pq_dist_t>::max();
      candidateSet.emplace(lowerBound, ep_id);
    }
    visited_array[ep_id] = visited_array_tag;

    while (!candidateSet.empty()) {
      std::pair<pq_dist_t, tableint> curr_el_pair = candidateSet.top();
      if ((curr_el_pair.first) > lowerBound && top_candidates.size() >= ef_construction_) {
        break;
      }
      candidateSet.pop();

      tableint cur_node_id = curr_el_pair.second;
      std::unique_lock<std::mutex> lock(link_list_locks_[cur_node_id]);

      int* data = (int*)get_linklist(cur_node_id, layer);
      linklistsizeint size = getListCount((linklistsizeint*)data);
      tableint* datal = (tableint*)(data + 1);

      for (size_t j = 0; j < size; j++) {
        tableint candidate_id = *(datal + j);

        if (visited_array[candidate_id] == visited_array_tag) {
          continue;
        }

        visited_array[candidate_id] = visited_array_tag;
        auto* currObj1 = getDataByInternalId(candidate_id);
        pq_dist_t dist1 = get_pq_dis(data_point, currObj1);

        if (top_candidates.size() < ef_construction_ || dist1 < lowerBound) {
          candidateSet.emplace(dist1, candidate_id);

          if (!isMarkedDeleted(candidate_id)) {
            top_candidates.emplace(dist1, candidate_id);

            if (top_candidates.size() > ef_construction_) {
              top_candidates.pop();
            }
            lowerBound = top_candidates.top().first;
          }
        }
      }
    }
    visited_list_pool_->releaseVisitedList(vl);

    return top_candidates;
  }

  template <bool has_deletions, bool collect_metrics = false>
  std::priority_queue<std::pair<pq_dist_t, tableint>,
                      std::vector<std::pair<pq_dist_t, tableint>>,
                      CompareByFirstLess>
  searchBaseLayerST(tableint ep_id, const void* data_point, size_t ef) const {
    VisitedList* vl = visited_list_pool_->getFreeVisitedList();
    vl_type* visited_array = vl->mass;
    vl_type visited_array_tag = vl->curV;

    std::priority_queue<std::pair<pq_dist_t, tableint>, std::vector<std::pair<pq_dist_t, tableint>>,
                        CompareByFirstLess>
        top_candidates;
    std::priority_queue<std::pair<pq_dist_t, tableint>, std::vector<std::pair<pq_dist_t, tableint>>,
                        CompareByFirstGreater>
        candidate_set;

    pq_dist_t lowerBound;
    if (!has_deletions || !isMarkedDeleted(ep_id)) {
      pq_dist_t dist = get_pq_dis(data_point, getDataByInternalId(ep_id));

      lowerBound = dist;
      top_candidates.emplace(dist, ep_id);
      candidate_set.emplace(dist, ep_id);
    } else {
      lowerBound = std::numeric_limits<pq_dist_t>::max();
      candidate_set.emplace(lowerBound, ep_id);
    }

    visited_array[ep_id] = visited_array_tag;

    // pre allocate for neighbor encode datas
    thread_local std::vector<encode_t> neighbor_encode_datas(maxM0_ * subspace_num_);
    while (!candidate_set.empty()) {
      std::pair<pq_dist_t, tableint> current_node_pair = candidate_set.top();
      if ((current_node_pair.first) > lowerBound && top_candidates.size() == ef) {
        break;
      }
      candidate_set.pop();

      tableint current_node_id = current_node_pair.second;
      linklistsizeint* data = (linklistsizeint*)get_linklist0(current_node_id);
      linklistsizeint size = getListCount((linklistsizeint*)data);

      // collect neighbors data
      tableint* datal = (tableint*)(data + 1);
      size_t to_visit_count = 0;
      for (size_t i = 0; i < size; ++i) {
        tableint candidate_id = datal[i];
        if (visited_array[candidate_id] == visited_array_tag) {
          continue;
        }

        const encode_t* neighbor_encode_data = getDataByInternalId(candidate_id);
        __builtin_memcpy(neighbor_encode_datas.data() + to_visit_count * subspace_num_, neighbor_encode_data,
                         subspace_num_ * sizeof(encode_t));
        to_visit_count += 1;
      }
      pq_dist_t* dist_list = (pq_dist_t*)alloca(to_visit_count * sizeof(pq_dist_t));
      get_pq_dist_batch(dist_list, to_visit_count, data_point, neighbor_encode_datas.data());

      if (collect_metrics) {
        metric_hops++;
        metric_distance_computations += to_visit_count;
      }

      for (size_t i = 0, to_visit_idx = 0; i < size; ++i) {
        int candidate_id = datal[i];

        if (!(visited_array[candidate_id] == visited_array_tag)) {
          visited_array[candidate_id] = visited_array_tag;

          pq_dist_t dist = dist_list[to_visit_idx];
          to_visit_idx += 1;

          if (top_candidates.size() < ef || dist < lowerBound) {
            candidate_set.emplace(dist, candidate_id);

            if (!has_deletions || !isMarkedDeleted(candidate_id)) {
              top_candidates.emplace(dist, candidate_id);
              if (top_candidates.size() > ef) top_candidates.pop();
              lowerBound = top_candidates.top().first;
            }
          }
        }
      }
    }
    visited_list_pool_->releaseVisitedList(vl);

    return top_candidates;
  }

  void getNeighborsByHeuristic2(std::priority_queue<std::pair<pq_dist_t, tableint>,
                                                    std::vector<std::pair<pq_dist_t, tableint>>,
                                                    CompareByFirstLess>& top_candidates,
                                const size_t M) {
    if (top_candidates.size() < M) {
      return;
    }

    std::priority_queue<std::pair<pq_dist_t, tableint>, std::vector<std::pair<pq_dist_t, tableint>>,
                        CompareByFirstGreater>
        queue_closest;
    std::vector<std::pair<pq_dist_t, tableint>> return_list;

    while (top_candidates.size() > 0) {
      queue_closest.emplace(top_candidates.top().first, top_candidates.top().second);
      top_candidates.pop();
    }

    while (queue_closest.size()) {
      if (return_list.size() >= M) break;

      std::pair<pq_dist_t, tableint> curent_pair = queue_closest.top();
      pq_dist_t pq_dist_to_query = curent_pair.first;
      queue_closest.pop();

      bool good = true;
      for (std::pair<pq_dist_t, tableint> second_pair : return_list) {
        pq_dist_t curdist =
            get_c2c_dis(getDataByInternalId(second_pair.second), getDataByInternalId(curent_pair.second));

        if (curdist < pq_dist_to_query) {
          good = false;
          break;
        }
      }
      if (good) {
        return_list.push_back(curent_pair);
      }
    }

    for (std::pair<pq_dist_t, tableint> curent_pair : return_list) {
      top_candidates.emplace(curent_pair.first, curent_pair.second);
    }
  }

  tableint mutuallyConnectNewElement(const void* data_point,
                                     tableint cur_c,
                                     std::priority_queue<std::pair<pq_dist_t, tableint>,
                                                         std::vector<std::pair<pq_dist_t, tableint>>,
                                                         CompareByFirstLess>& top_candidates,
                                     int level,
                                     bool isUpdate) {
    size_t Mcurmax = level ? maxM_ : maxM0_;
    getNeighborsByHeuristic2(top_candidates, Mcurmax);
    if (top_candidates.size() > Mcurmax) {
      throw std::runtime_error("Should be not be more than Mcurmax candidates returned by the heuristic");
    }

    std::vector<tableint> selectedNeighbors;
    selectedNeighbors.reserve(Mcurmax);
    while (top_candidates.size() > 0) {
      selectedNeighbors.push_back(top_candidates.top().second);
      top_candidates.pop();
    }
    tableint next_closest_entry_point = selectedNeighbors.back();

    // add neighbors to the current element
    {
      std::unique_lock<std::mutex> lock(link_list_locks_[cur_c], std::defer_lock);
      // protect the link list of the current element
      if (isUpdate) {
        lock.lock();
      }

      linklistsizeint* ll_cur = get_linklist(cur_c, level);
      if (!isUpdate && *ll_cur) {
        throw std::runtime_error("The newly inserted element should have blank link list");
      }

      setListCount(ll_cur, selectedNeighbors.size());
      tableint* datal = (tableint*)(ll_cur + 1);
      for (size_t idx = 0; idx < selectedNeighbors.size(); ++idx) {
        if (datal[idx] && !isUpdate) {
          throw std::runtime_error("Possible memory corruption");
        }
        if (level > element_levels_[selectedNeighbors[idx]]) {
          throw std::runtime_error("Trying to make a link on a non-existent level");
        }

        datal[idx] = selectedNeighbors[idx];
      }
    }

    // inverse add connection for the selected neighbors
    for (size_t idx = 0; idx < selectedNeighbors.size(); ++idx) {
      std::unique_lock<std::mutex> lock(link_list_locks_[selectedNeighbors[idx]]);

      linklistsizeint* ll_other = get_linklist(selectedNeighbors[idx], level);
      size_t sz_link_list_other = getListCount(ll_other);

      if (sz_link_list_other > Mcurmax) {
        throw std::runtime_error("Bad value of sz_link_list_other");
      }
      if (selectedNeighbors[idx] == cur_c) {
        throw std::runtime_error("Trying to connect an element to itself");
      }
      if (level > element_levels_[selectedNeighbors[idx]]) {
        throw std::runtime_error("Trying to make a link on a non-existent level");
      }

      tableint* datal = (tableint*)(ll_other + 1);
      bool is_cur_c_present = false;
      if (isUpdate) {
        for (size_t j = 0; j < sz_link_list_other; j++) {
          if (datal[j] == cur_c) {
            is_cur_c_present = true;
            break;
          }
        }
      }

      // If cur_c is already present in the neighboring connections of `selectedNeighbors[idx]` then no need
      // to modify any connections or run the heuristics.
      if (!is_cur_c_present) {
        if (sz_link_list_other < Mcurmax) {
          datal[sz_link_list_other] = cur_c;
          setListCount(ll_other, sz_link_list_other + 1);
        } else {
          // finding the "weakest" element to replace it with the new one
          pq_dist_t d_max =
              get_c2c_dis(getDataByInternalId(cur_c), getDataByInternalId(selectedNeighbors[idx]));

          std::priority_queue<std::pair<pq_dist_t, tableint>, std::vector<std::pair<pq_dist_t, tableint>>,
                              CompareByFirstLess>
              candidates;
          candidates.emplace(d_max, cur_c);

          for (size_t j = 0; j < sz_link_list_other; j++) {
            auto dis =
                get_c2c_dis(getDataByInternalId(datal[j]), getDataByInternalId(selectedNeighbors[idx]));
            candidates.emplace(dis, datal[j]);
          }
          getNeighborsByHeuristic2(candidates, Mcurmax);

          int indx = 0;
          while (candidates.size() > 0) {
            datal[indx] = candidates.top().second;
            candidates.pop();
            indx++;
          }
          setListCount(ll_other, indx);
        }
      }
    }

    return next_closest_entry_point;
  }

  void updatePoint(const void* dataPoint, tableint internalId, float updateNeighborProbability) {
    // update encode data
    char* dst_encode_data = (char*)getDataByInternalId(internalId);
    encode_t* encode_data = (encode_t*)((char*)dataPoint + offset_encode_query_data_);
    memcpy(dst_encode_data, encode_data, encode_data_size_);

    // update raw data
    char* dst_raw_data = (char*)raw_data_table_ + internalId * raw_data_size_;
    float* raw_data = (float*)((char*)dataPoint + offset_raw_query_data_);
    memcpy(dst_raw_data, raw_data, raw_data_size_);

    int maxLevelCopy = maxlevel_;
    tableint entryPointCopy = enterpoint_node_;
    // If point to be updated is entry point and graph just contains single element then just return.
    if (entryPointCopy == internalId && cur_element_count_ == 1) return;

    int elemLevel = element_levels_[internalId];
    std::uniform_real_distribution<float> distribution(0.0, 1.0);
    for (int layer = 0; layer <= elemLevel; layer++) {
      std::unordered_set<tableint> sCand;
      std::unordered_set<tableint> sNeigh;
      std::vector<tableint> listOneHop = getConnectionsWithLock(internalId, layer);
      if (listOneHop.size() == 0) continue;

      sCand.insert(internalId);

      for (auto&& elOneHop : listOneHop) {
        sCand.insert(elOneHop);

        if (distribution(update_probability_generator_) > updateNeighborProbability) continue;

        // 这些一跳的点才会重建链接
        sNeigh.insert(elOneHop);

        std::vector<tableint> listTwoHop = getConnectionsWithLock(elOneHop, layer);
        for (auto&& elTwoHop : listTwoHop) {
          sCand.insert(elTwoHop);
        }
      }

      // 一跳里随机选择里一些链接点
      for (auto&& neigh : sNeigh) {
        std::priority_queue<std::pair<pq_dist_t, tableint>, std::vector<std::pair<pq_dist_t, tableint>>,
                            CompareByFirstLess>
            candidates;
        int size = sCand.find(neigh) == sCand.end() ? sCand.size() : sCand.size() - 1;
        int elementsToKeep = std::min(int(ef_construction_), size);
        // 一跳的节点与所有二跳的节点计算距离, 找出其中的top N
        for (auto&& cand : sCand) {
          if (cand == neigh) continue;
          pq_dist_t dis = get_c2c_dis(getDataByInternalId(neigh), getDataByInternalId(cand));

          if (candidates.size() < elementsToKeep) {
            candidates.emplace(dis, cand);
          } else {
            if (dis < candidates.top().first) {
              candidates.pop();
              candidates.emplace(dis, cand);
            }
          }
        }

        // candidates 就是一跳的点与所有二二跳的点，之间距离最近的N个点
        // 启发式的查找这些点之间两两距离最近的点，从一个中心点出发，记录和他距离最近的
        // Retrieve neighbours using heuristic and set connections.
        getNeighborsByHeuristic2(candidates, layer == 0 ? maxM0_ : maxM_);

        {
          std::unique_lock<std::mutex> lock(link_list_locks_[neigh]);
          linklistsizeint* ll_cur;
          ll_cur = get_linklist(neigh, layer);
          int candSize = candidates.size();
          setListCount(ll_cur, candSize);
          tableint* datal = (tableint*)(ll_cur + 1);
          for (size_t idx = 0; idx < candSize; idx++) {
            datal[idx] = candidates.top().second;
            candidates.pop();
          }
        }
      }
    }

    repairConnectionsForUpdate(dataPoint, entryPointCopy, internalId, elemLevel, maxLevelCopy);
  };

  void repairConnectionsForUpdate(const void* dataPoint,
                                  tableint entryPointInternalId,
                                  tableint dataPointInternalId,
                                  int dataPointLevel,
                                  int maxLevel) {
    tableint currObj = entryPointInternalId;
    if (dataPointLevel < maxLevel) {
      pq_dist_t curdist = get_pq_dis(dataPoint, getDataByInternalId(currObj));

      for (int level = maxLevel; level > dataPointLevel; level--) {
        bool changed = true;
        while (changed) {
          changed = false;
          unsigned int* data;
          std::unique_lock<std::mutex> lock(link_list_locks_[currObj]);
          data = get_linklist(currObj, level);
          int size = getListCount(data);
          tableint* datal = (tableint*)(data + 1);
#ifdef USE_SSE
          _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
#endif
          for (int i = 0; i < size; i++) {
#ifdef USE_SSE
            _mm_prefetch(getDataByInternalId(*(datal + i + 1)), _MM_HINT_T0);
#endif
            tableint cand = datal[i];
            pq_dist_t d = get_pq_dis(dataPoint, getDataByInternalId(cand));
            if (d < curdist) {
              curdist = d;
              currObj = cand;
              changed = true;
            }
          }
        }
      }
    }

    if (dataPointLevel > maxLevel) {
      throw std::runtime_error("Level of item to be updated cannot be bigger than max level");
    }

    for (int level = dataPointLevel; level >= 0; level--) {
      std::priority_queue<std::pair<pq_dist_t, tableint>, std::vector<std::pair<pq_dist_t, tableint>>,
                          CompareByFirstLess>
          topCandidates = searchBaseLayer(currObj, dataPoint, level);

      std::priority_queue<std::pair<pq_dist_t, tableint>, std::vector<std::pair<pq_dist_t, tableint>>,
                          CompareByFirstLess>
          filteredTopCandidates;
      while (topCandidates.size() > 0) {
        if (topCandidates.top().second != dataPointInternalId)
          filteredTopCandidates.push(topCandidates.top());

        topCandidates.pop();
      }

      if (filteredTopCandidates.size() > 0) {
        bool epDeleted = isMarkedDeleted(entryPointInternalId);
        if (epDeleted) {
          auto dis = get_pq_dis(dataPoint, getDataByInternalId(entryPointInternalId));
          filteredTopCandidates.emplace(dis, entryPointInternalId);

          if (filteredTopCandidates.size() > ef_construction_) {
            filteredTopCandidates.pop();
          }
        }

        currObj =
            mutuallyConnectNewElement(dataPoint, dataPointInternalId, filteredTopCandidates, level, true);
      }
    }
  }

  std::vector<tableint> getConnectionsWithLock(tableint internalId, int level) {
    std::unique_lock<std::mutex> lock(link_list_locks_[internalId]);
    unsigned int* data = get_linklist(internalId, level);
    int size = getListCount(data);
    std::vector<tableint> result(size);
    tableint* ll = (tableint*)(data + 1);
    memcpy(result.data(), ll, size * sizeof(tableint));
    return result;
  };

  void addPoint(const void* data_point, labeltype label) {
    thread_local std::vector<char> data_internal(subspace_num_ * cluster_num_ * sizeof(pq_dist_t) +
                                                 encode_data_size_ + raw_data_size_);
    pq_encode_func_(pq_codebooks_, pq_min_, pq_max_, subspace_num_, cluster_num_, data_dim_,
                    (float*)data_point, (encode_t*)(data_internal.data() + offset_encode_query_data_),
                    (pq_dist_t*)data_internal.data(), false);
    __builtin_memcpy(data_internal.data() + offset_raw_query_data_, (char*)data_point, raw_data_size_);

    addPoint(data_internal.data(), label, -1);
  }

  tableint addPoint(const void* data_point, labeltype label, int level) {
    tableint cur_c = 0;
    {
      std::unique_lock<std::mutex> templock_curr(label_lookup_lock_);
      auto search = label_lookup_.find(label);
      if (search != label_lookup_.end()) {
        // 更新操作将影响add性能
        if (!allow_update_point_) {
          return -1;
        }
        tableint existingInternalId = search->second;
        templock_curr.unlock();

        updatePoint(data_point, existingInternalId, 1.0);
        return existingInternalId;
      }

      if (cur_element_count_ >= max_elements_) {
        throw std::runtime_error("The number of elements exceeds the specified limit");
      };

      cur_c = cur_element_count_;
      cur_element_count_++;
      label_lookup_[label] = cur_c;
    }

    // Take update lock to prevent race conditions on an element with insertion/update at the same time.
    std::unique_lock<std::mutex> lock_el_update(label_op_locks_[(cur_c & (max_label_op_locks - 1))]);
    std::unique_lock<std::mutex> lock_el(link_list_locks_[cur_c]);
    int curlevel = getRandomLevel(mult_);
    if (level > 0) curlevel = level;

    element_levels_[cur_c] = curlevel;

    std::unique_lock<std::mutex> templock(global);
    int maxlevelcopy = maxlevel_;
    if (curlevel <= maxlevelcopy) {
      templock.unlock();
    }
    tableint currObj = enterpoint_node_;
    tableint enterpoint_copy = enterpoint_node_;

    memset(data_level0_memory_ + cur_c * size_data_per_element_, 0, size_data_per_element_);

    // set label / encode data / raw data
    memcpy(getExternalLabeLp(cur_c), &label, sizeof(labeltype));

    char* dst_encode_data = (char*)getDataByInternalId(cur_c);
    encode_t* encode_data = (encode_t*)((char*)data_point + offset_encode_query_data_);
    memcpy(dst_encode_data, encode_data, encode_data_size_);

    float* dst_raw_data = raw_data_table_ + cur_c * data_dim_;
    float* raw_data = (float*)((char*)data_point + offset_raw_query_data_);
    memcpy(dst_raw_data, raw_data, raw_data_size_);

    if (curlevel) {
      linkLists_[cur_c] = (char*)malloc(size_links_per_element_ * curlevel + 1);
      if (linkLists_[cur_c] == nullptr) {
        throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
      }
      memset(linkLists_[cur_c], 0, size_links_per_element_ * curlevel + 1);
    }

    if ((signed)currObj != -1) {
      if (curlevel < maxlevelcopy) {
        pq_dist_t curdist = get_pq_dis(data_point, getDataByInternalId(currObj));

        for (int level = maxlevelcopy; level > curlevel; level--) {
          bool changed = true;
          while (changed) {
            changed = false;
            unsigned int* data;
            std::unique_lock<std::mutex> lock(link_list_locks_[currObj]);
            data = get_linklist(currObj, level);
            int size = getListCount(data);

            tableint* datal = (tableint*)(data + 1);
            for (int i = 0; i < size; i++) {
              tableint cand = datal[i];
              if (cand < 0 || cand > max_elements_) {
                throw std::runtime_error("cand error");
              }

              pq_dist_t d = get_pq_dis(data_point, getDataByInternalId(cand));
              if (d < curdist) {
                curdist = d;
                currObj = cand;
                changed = true;
              }
            }
          }
        }
      }

      bool epDeleted = isMarkedDeleted(enterpoint_copy);
      for (int level = std::min(curlevel, maxlevelcopy); level >= 0; level--) {
        if (level > maxlevelcopy || level < 0) {
          throw std::runtime_error("Level error");
        }

        std::priority_queue<std::pair<pq_dist_t, tableint>, std::vector<std::pair<pq_dist_t, tableint>>,
                            CompareByFirstLess>
            top_candidates = searchBaseLayer(currObj, data_point, level);
        if (epDeleted) {
          auto dist = get_pq_dis(data_point, getDataByInternalId(enterpoint_copy));
          top_candidates.emplace(dist, enterpoint_copy);
          if (top_candidates.size() > ef_construction_) top_candidates.pop();
        }
        currObj = mutuallyConnectNewElement(data_point, cur_c, top_candidates, level, false);
      }
    } else {
      // Do nothing for the first element
      enterpoint_node_ = 0;
      maxlevel_ = curlevel;
    }

    // Releasing lock for the maximum level
    if (curlevel > maxlevelcopy) {
      enterpoint_node_ = cur_c;
      maxlevel_ = curlevel;
    }
    return cur_c;
  };

  std::priority_queue<std::pair<pq_dist_t, tableint>> searchKnnIndex(const void* query_data_internal,
                                                                     size_t k) const {
    std::priority_queue<std::pair<pq_dist_t, tableint>> result;
    if (cur_element_count_ == 0) return result;

    tableint currObj = enterpoint_node_;
    pq_dist_t curdist = get_pq_dis(query_data_internal, getDataByInternalId(enterpoint_node_));

    thread_local std::vector<encode_t> neighbor_encode_datas(maxM_ * subspace_num_);
    for (int level = maxlevel_; level > 0; level--) {
      bool changed = true;
      while (changed) {
        changed = false;

        linklistsizeint* data = (linklistsizeint*)get_linklist(currObj, level);
        linklistsizeint size = getListCount(data);
        metric_hops++;
        metric_distance_computations += size;

        tableint* datal = (tableint*)(data + 1);

        // collect neighbor datas
        for (size_t i = 0; i < size; ++i) {
          tableint cand = datal[i];
          const encode_t* neighbor_data = (encode_t*)getDataByInternalId(cand);
          __builtin_memcpy(neighbor_encode_datas.data() + i * subspace_num_, neighbor_data,
                           subspace_num_ * sizeof(encode_t));
        }

        pq_dist_t* dist_list = (pq_dist_t*)alloca(size * sizeof(pq_dist_t));
        get_pq_dist_batch(dist_list, size, query_data_internal, neighbor_encode_datas.data());

        for (int i = 0; i < size; i++) {
          tableint cand = datal[i];
          pq_dist_t d = dist_list[i];

          if (d < curdist) {
            curdist = d;
            currObj = cand;
            changed = true;
          }
        }
      }
    }

    std::priority_queue<std::pair<pq_dist_t, tableint>, std::vector<std::pair<pq_dist_t, tableint>>,
                        CompareByFirstLess>
        top_candidates;
    if (has_deletions_) {
      top_candidates = searchBaseLayerST<true, true>(currObj, query_data_internal, std::max(ef_, k));
    } else {
      top_candidates = searchBaseLayerST<false, true>(currObj, query_data_internal, std::max(ef_, k));
    }

    while (top_candidates.size() > k) {
      top_candidates.pop();
    }
    while (top_candidates.size() > 0) {
      std::pair<pq_dist_t, tableint> rez = top_candidates.top();
      result.push(std::pair<pq_dist_t, tableint>(rez.first, rez.second));
      top_candidates.pop();
    }

    return result;
  };

  std::priority_queue<std::pair<dist_t, labeltype>> searchKnn(const void* query_data, size_t k) const {
    thread_local std::vector<char> query_data_internal(subspace_num_ * cluster_num_ * sizeof(pq_dist_t) +
                                                       encode_data_size_);
    memset(query_data_internal.data(), 0, query_data_internal.size());
    pq_encode_func_(pq_codebooks_, pq_min_, pq_max_, subspace_num_, cluster_num_, data_dim_,
                    (float*)query_data, (encode_t*)(query_data_internal.data() + offset_encode_query_data_),
                    (pq_dist_t*)query_data_internal.data(), true);

    size_t rerank_top_k = k * rerank_ratio_;
    rerank_top_k = std::max(rerank_top_k, ef_);

    auto rerank_topk_cands = searchKnnIndex(query_data_internal.data(), rerank_top_k);
    std::priority_queue<std::pair<dist_t, labeltype>> ret;
    auto size = std::min(k, rerank_topk_cands.size());
    for (size_t i = 0; i < size; ++i) {
      auto& top = rerank_topk_cands.top();
      auto id = top.second;
      float* data_raw_emb = (float*)getRawDataByInternalId(id);
      auto dist = rerank_func_(query_data, data_raw_emb, &data_dim_);

      ret.emplace(dist, getExternalLabel(id));
      rerank_topk_cands.pop();
    }

    while (!rerank_topk_cands.empty()) {
      auto& top = rerank_topk_cands.top();
      dist_t dist = rerank_func_(query_data, getRawDataByInternalId(top.second), &data_dim_);
      if (dist < ret.top().first) {
        ret.pop();
        ret.emplace(dist, getExternalLabel(top.second));
      }
      rerank_topk_cands.pop();
    }

    return ret;
  }

  std::priority_queue<std::pair<dist_t, labeltype>> bruceForceSearchKnn(const void* query_data,
                                                                        size_t k) const {
    return bruceForceSearchKnnInner(query_data, k);
  }

  std::priority_queue<std::pair<dist_t, labeltype>> bruceForceSearchKnnInner(const void* query_data,
                                                                             size_t k) const {
    std::priority_queue<std::pair<dist_t, labeltype>> topResults;
    if (cur_element_count_ == 0) return topResults;
    int id = 0;
    for (; id < k && id < cur_element_count_; id++) {
      dist_t dist = rerank_func_(query_data, getRawDataByInternalId(id), &data_dim_);
      topResults.push(std::pair<dist_t, labeltype>(dist, getExternalLabel(id)));
    }

    dist_t lastdist = topResults.top().first;
    for (; id < cur_element_count_; id++) {
      dist_t dist = rerank_func_(query_data, getRawDataByInternalId(id), &data_dim_);

      if (dist <= lastdist) {
        topResults.push(std::pair<dist_t, labeltype>(dist, getExternalLabel(id)));
        if (topResults.size() > k) topResults.pop();
        lastdist = topResults.top().first;
      }
    }
    return topResults;
  };

  std::priority_queue<std::pair<pq_dist_t, tableint>> bruceForceSearchInPQ(const void* query_data, size_t k) {
    thread_local std::vector<char> query_data_internal(subspace_num_ * cluster_num_ * sizeof(pq_dist_t) +
                                                       encode_data_size_);
    memset(query_data_internal.data(), 0, query_data_internal.size());
    pq_encode_func_(pq_codebooks_, pq_min_, pq_max_, subspace_num_, cluster_num_, data_dim_,
                    (float*)query_data, (encode_t*)(query_data_internal.data() + offset_encode_query_data_),
                    (pq_dist_t*)query_data_internal.data(), true);

    std::priority_queue<std::pair<pq_dist_t, tableint>, std::vector<std::pair<pq_dist_t, tableint>>>
        topResults;
    if (cur_element_count_ == 0) return topResults;

    int id = 0;
    for (; id < cur_element_count_; id++) {
      pq_dist_t pq_dist = get_pq_dis(query_data_internal.data(), getDataByInternalId(id));
      topResults.push(std::pair<pq_dist_t, labeltype>(pq_dist, getExternalLabel(id)));
    }

    while (topResults.size() > k) {
      topResults.pop();
    }

    return topResults;
  }

  Eigen::MatrixXf kMeanspp_init(const Eigen::MatrixXf& data, int k) {
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

      if (new_centroids.isApprox(centroids, 1e-5)) {
        break;
      }
      centroids = new_centroids;
    }

    return centroids;
  }

  void train(int n, const float* x) {
    size_t subspace_len = data_dim_ / subspace_num_;
    size_t pre_subspace_size = 0;

    // generate codebook
    for (size_t i = 0; i < subspace_num_; ++i) {
      std::cout << "begin kMeans for subspace: (" << i + 1 << " / " << subspace_num_ << ")" << std::endl;
      Eigen::MatrixXf subspace_data(n, subspace_len);
      size_t cur_subspace_prelen = i * subspace_len;
      for (size_t j = 0; j < n; ++j) {
        float* cur_emb = const_cast<float*>(x) + j * data_dim_;
        subspace_data.row(j) = Eigen::Map<Eigen::VectorXf>(cur_emb + cur_subspace_prelen, subspace_len);
      }

      Eigen::MatrixXf centroid_matrix = kMeans(subspace_data, cluster_num_, kmeans_train_round_);
      auto* cur_codebook_ptr = pq_codebooks_ + pre_subspace_size;

      for (size_t j = 0; j < cluster_num_; ++j) {
        Eigen::VectorXf row = centroid_matrix.row(j);
        __builtin_memcpy(cur_codebook_ptr + j * subspace_len, row.data(), subspace_len * sizeof(float));
      }

      pre_subspace_size += cluster_num_ * subspace_len;
    }

    // get quantize param
    pre_subspace_size = 0;
    pq_max_ = 0;
    pq_min_ = std::numeric_limits<pq_dist_t>::max();

    float* tmp_table = (float*)malloc(subspace_num_ * cluster_num_ * cluster_num_ * sizeof(float));
    memset(tmp_table, 0, subspace_num_ * cluster_num_ * cluster_num_ * sizeof(float));
    float* ptr_tmp_table = tmp_table;
    for (size_t i = 0; i < subspace_num_; ++i) {
      float max_dis = 0;
      auto* cur_codebook_ptr = pq_codebooks_ + pre_subspace_size;

      for (size_t c1 = 0; c1 < cluster_num_; ++c1) {
        for (size_t c2 = 0; c2 < cluster_num_; ++c2) {
          if (c1 == c2) {
            ptr_tmp_table += 1;
            continue;
          }

          *ptr_tmp_table = rerank_func_(cur_codebook_ptr + c1 * subspace_len,
                                        cur_codebook_ptr + c2 * subspace_len, &subspace_len);
          pq_min_ = std::min(pq_min_, *ptr_tmp_table);
          max_dis = std::max(max_dis, *ptr_tmp_table);
          ptr_tmp_table += 1;
        }
      }

      pq_max_ += max_dis;
      pre_subspace_size += cluster_num_ * subspace_len;
    }
    pq_max_ -= pq_min_;

    ptr_tmp_table = tmp_table;
    pq_dist_t* ptr_pq_center_dis_table_ = pq_center_dis_table_;
    for (size_t i = 0; i < subspace_num_; ++i) {
      for (size_t c1 = 0; c1 < cluster_num_; ++c1) {
        for (size_t c2 = 0; c2 < cluster_num_; ++c2) {
          float ratio = (*ptr_tmp_table - pq_min_) / pq_max_;
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
  };

  void saveIndex(const std::string& location) {
    location_ = location;
    std::ofstream output(location, std::ios::binary);
    output.exceptions(std::ios_base::failbit | std::ios_base::badbit);
    std::streampos position;

    writeBinaryPOD(output, M_);
    writeBinaryPOD(output, maxM_);
    writeBinaryPOD(output, maxM0_);
    writeBinaryPOD(output, ef_construction_);
    writeBinaryPOD(output, mult_);
    writeBinaryPOD(output, maxlevel_);

    writeBinaryPOD(output, enterpoint_node_);
    writeBinaryPOD(output, max_elements_);
    writeBinaryPOD(output, cur_element_count_);

    writeBinaryPOD(output, size_links_per_element_);
    writeBinaryPOD(output, size_links_level0_);
    writeBinaryPOD(output, size_data_per_element_);
    writeBinaryPOD(output, size_raw_data_per_element_);

    writeBinaryPOD(output, offset_data_);
    writeBinaryPOD(output, offset_label_);

    writeBinaryPOD(output, pq_max_);
    writeBinaryPOD(output, pq_min_);

    std::cout << "save hnsw-flash: location: " << location_ << ", maxlevel_:" << maxlevel_
              << ", max_elements_: " << max_elements_ << ", cur_element_count_: " << cur_element_count_
              << ", size_data_per_element_: " << size_data_per_element_
              << ", enterpoint_node_: " << enterpoint_node_ << ", pq_max_: " << pq_max_
              << ", pq_min_: " << pq_min_ << std::endl;

    output.write(data_level0_memory_, cur_element_count_ * size_data_per_element_);
    output.write((char*)raw_data_table_, cur_element_count_ * raw_data_size_);
    output.write((char*)pq_codebooks_, cluster_num_ * raw_data_size_);
    output.write((char*)pq_center_dis_table_,
                 subspace_num_ * cluster_num_ * cluster_num_ * sizeof(pq_dist_t));

    for (size_t i = 0; i < cur_element_count_; i++) {
      unsigned int linkListSize = element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
      writeBinaryPOD(output, linkListSize);
      if (linkListSize) output.write(linkLists_[i], linkListSize);
    }

    output.close();
  }

  void loadIndex(std::ifstream& input, FlashSpaceInterface<dist_t>* s, size_t max_elements_i = 0) {
    try {
      space_ = s;
      data_dim_ = space_->get_data_dim();
      subspace_num_ = space_->get_subspace_num();
      cluster_num_ = space_->get_cluster_num();
      cluster_num_sqr_ = cluster_num_ * cluster_num_;
      encode_data_size_ = space_->get_encode_data_size();
      raw_data_size_ = space_->get_raw_data_size();
      pq_encode_func_ = space_->get_pq_encode_func();
      rerank_func_ = space_->get_rerank_func();

      offset_encode_query_data_ = subspace_num_ * cluster_num_ * sizeof(pq_dist_t);
      offset_raw_query_data_ = offset_encode_query_data_ + subspace_num_ * sizeof(encode_t);

      // get file size:
      input.seekg(0, input.end);
      std::streampos total_filesize = input.tellg();

      input.seekg(0, input.beg);

      readBinaryPOD(input, M_);
      readBinaryPOD(input, maxM_);
      readBinaryPOD(input, maxM0_);
      readBinaryPOD(input, ef_construction_);
      readBinaryPOD(input, mult_);
      rev_size_ = 1.0 / mult_;
      readBinaryPOD(input, maxlevel_);
      rerank_ratio_ = default_rerank_ratio;

      readBinaryPOD(input, enterpoint_node_);
      readBinaryPOD(input, max_elements_);
      readBinaryPOD(input, cur_element_count_);

      size_t max_elements = max_elements_i;
      if (max_elements < cur_element_count_) max_elements = max_elements_;
      max_elements_ = max_elements;

      readBinaryPOD(input, size_links_per_element_);
      readBinaryPOD(input, size_links_level0_);
      readBinaryPOD(input, size_data_per_element_);
      readBinaryPOD(input, size_raw_data_per_element_);

      readBinaryPOD(input, offset_data_);
      readBinaryPOD(input, offset_label_);

      readBinaryPOD(input, pq_max_);
      readBinaryPOD(input, pq_min_);

      std::cout << "init hnsw-flash: location: " << location_
                << ", total file size: " << (1.0f * total_filesize / 1024 / 1024) << " (MB)"
                << ", maxlevel_:" << maxlevel_ << ", max_elements_: " << max_elements_
                << ", cur_element_count_: " << cur_element_count_
                << ", size_data_per_element_: " << size_data_per_element_
                << ", enterpoint_node_: " << enterpoint_node_ << ", pq_max_: " << pq_max_
                << ", pq_min_: " << pq_min_ << std::endl;

      {
        /// Optional - check if index is ok:
        auto pos = input.tellg();
        input.seekg((cur_element_count_ * size_data_per_element_) + (cur_element_count_ * raw_data_size_) +
                        (cluster_num_ * raw_data_size_) +
                        (subspace_num_ * cluster_num_ * cluster_num_ * sizeof(pq_dist_t)),
                    input.cur);

        for (size_t i = 0; i < cur_element_count_; i++) {
          if (input.tellg() < 0 || input.tellg() >= total_filesize) {
            throw std::runtime_error("Index seems to be corrupted or unsupported");
          }

          unsigned int linkListSize;
          readBinaryPOD(input, linkListSize);
          if (linkListSize != 0) {
            input.seekg(linkListSize, input.cur);
          }
        }

        // throw exception if it either corrupted or old index
        if (input.tellg() != total_filesize)
          throw std::runtime_error("Index seems to be corrupted or unsupported");

        input.clear();
        input.seekg(pos, input.beg);
      }

      data_level0_memory_ = (char*)aligned_alloc(64, max_elements * size_data_per_element_);
      if (data_level0_memory_ == nullptr) {
        throw std::runtime_error("Not enough memory: loadIndex failed to allocate data_level0_memory_");
      }
      if (input.read(data_level0_memory_, cur_element_count_ * size_data_per_element_)) {
      } else {
        std::cout << "[failed] load to data_level0_memory_, size: "
                  << cur_element_count_ * size_data_per_element_ << ", status:" << input.bad() << input.fail()
                  << std::endl;
        return;
      }

      size_t raw_data_table_size = cur_element_count_ * raw_data_size_;
      raw_data_table_ = (float*)aligned_alloc(64, raw_data_table_size);
      if (raw_data_table_ == nullptr) {
        throw std::runtime_error("Not enough memory: loadIndex failed to allocate raw_data_table_");
      }
      if (input.read((char*)raw_data_table_, raw_data_table_size)) {
      } else {
        std::cout << "[failed] load to raw_data_table_, size: " << raw_data_table_size
                  << ", status: " << input.bad() << input.fail() << std::endl;
        return;
      }

      size_t pq_codebooks_size = cluster_num_ * raw_data_size_;
      pq_codebooks_ = (float*)aligned_alloc(64, pq_codebooks_size);
      if (pq_codebooks_ == nullptr) {
        throw std::runtime_error("Not enough memory: loadIndex failed to allocate pq_codebooks_");
      }
      if (input.read((char*)pq_codebooks_, pq_codebooks_size)) {
      } else {
        std::cout << "[failed] load to pq_codebooks_, size: " << pq_codebooks_size
                  << ", status: " << input.bad() << input.fail() << std::endl;
        return;
      }

      size_t pq_center_dis_table_size = subspace_num_ * cluster_num_ * cluster_num_ * sizeof(pq_dist_t);
      pq_center_dis_table_ = (pq_dist_t*)aligned_alloc(64, pq_center_dis_table_size);
      if (pq_center_dis_table_ == nullptr) {
        throw std::runtime_error("Not enough memory: loadIndex failed to allocate pq_center_dis_table_");
      }
      if (input.read((char*)pq_center_dis_table_, pq_center_dis_table_size)) {
      } else {
        std::cout << "[failed] load to pq_center_dis_table_, size: " << pq_center_dis_table_size
                  << ", status: " << input.bad() << input.fail() << std::endl;
        return;
      }

      std::vector<std::mutex>(max_elements).swap(link_list_locks_);
      std::vector<std::mutex>(max_label_op_locks).swap(label_op_locks_);

      if (is_mutable_) {
        visited_list_pool_ = new VisitedListPool(1, max_elements);
      } else {
        visited_list_pool_ = new VisitedListPool(1, cur_element_count_);
      }

      if (visited_list_pool_ == nullptr) throw std::runtime_error("Not enough memory: visited_list_pool_");

      linkLists_ = (char**)malloc(sizeof(void*) * max_elements);
      if (linkLists_ == nullptr) {
        throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklists");
      }
      memset(linkLists_, 0, sizeof(void*) * max_elements);

      element_levels_ = std::vector<int>(max_elements);
      rev_size_ = 1.0 / mult_;
      ef_ = 10;
      has_deletions_ = false;
      for (size_t i = 0; i < cur_element_count_; i++) {
        if (isMarkedDeleted(i)) has_deletions_ = true;

        label_lookup_[getExternalLabel(i)] = i;
        unsigned int linkListSize;
        readBinaryPOD(input, linkListSize);
        if (linkListSize == 0) {
          element_levels_[i] = 0;
          linkLists_[i] = nullptr;
        } else {
          element_levels_[i] = linkListSize / size_links_per_element_;
          linkLists_[i] = (char*)malloc(linkListSize);
          if (linkLists_[i] == nullptr)
            throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklist");
          if (!input.read(linkLists_[i], linkListSize)) {
            std::cout << "[failed] read linkLists_ " << i << ",gid:" << getExternalLabel(i)
                      << ",linkListSize:" << linkListSize << std::endl;
            continue;
          }
        }
      }

      // ::google::FlushLogFiles(::google::INFO);
      // ::google::FlushLogFiles(::google::ERROR);
    } catch (const std::exception& ex) {
      std::cout << "[CORE DEUBG] LOAD HNSW ERROR " << location_ << "," << ex.what() << std::endl;
      // this->~HnswFlash();  // NOTE 这样会 core
      FreeHeapData();  // 回收资源
      std::cout << "[CORE DEUBG] LOAD HNSW ERROR throw" << std::endl;
      // ::google::FlushLogFiles(::google::INFO);
      // ::google::FlushLogFiles(::google::ERROR);
      throw;
    }
    return;
  }
};

}  // namespace hnswlib
