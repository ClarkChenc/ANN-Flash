#pragma once

#include "visited_list_pool.h"
#include "hnswlib.h"
#include <atomic>
#include <random>
#include <stdlib.h>
#include <assert.h>
#include <unordered_set>
#include <list>
#include <memory>
#include <absl/container/flat_hash_map.h>
#include <mutex>
#include <shared_mutex>
// #include "utils.h"

#include <cstdlib>

namespace hnswlib {

#define ls(x) ((x >> 4) & 0x0F)
#define rs(x) (x & 0x0F)

typedef unsigned int tableint;
typedef unsigned int linklistsizeint;

float* flash_codebooks_v4_;
// subvector_id, cluster_id, cluster_id
data_t* flash_dist_v4_;

size_t subvec_num_v4_;

inline data_t* get_dist_v4(int i, int j, int k) {
  return (flash_dist_v4_ + i * CLUSTER_NUM2 + j * CLUSTER_NUM + k);
}

#if defined(ADSAMPLING)
float ratio[SUBVECTOR_NUM + 1];
void init_ratio() {
  size_t D = SUBVECTOR_NUM;
  float epsilon0 = ADSAMPLING_EPSILON;
  for (int i = 1; i <= SUBVECTOR_NUM; ++i) {
    ratio[i] = 1.0 * i / D * (1.0 + epsilon0 / std::sqrt(i)) * (1.0 + epsilon0 / std::sqrt(i));
  }
}
#endif

template <typename dist_t>
class HierarchicalNSWFlash_V4 : public AlgorithmInterface<dist_t> {
 public:
  static const tableint MAX_LABEL_OPERATION_LOCKS = 65536;
  static const unsigned char DELETE_MARK = 0x01;

  size_t max_elements_{0};
  mutable std::atomic<size_t> cur_element_count{0};

  size_t size_data_per_element_{0};
  size_t size_links_per_element_{0};
  mutable std::atomic<size_t> num_deleted_{0};  // number of deleted elements
  size_t M_{0};
  size_t maxM_{0};
  size_t maxM0_{0};
  size_t ef_construction_{0};
  size_t ef_{0};

  double mult_{0.0}, revSize_{0.0};
  int maxlevel_{0};

  std::unique_ptr<VisitedListPool> visited_list_pool_{nullptr};

  // Locks operations with element by label value
  mutable std::vector<std::mutex> label_op_locks_;

  std::mutex global;
  std::vector<std::mutex> link_list_locks_;

  tableint enterpoint_node_{0};

  size_t size_links_level0_{0};
  size_t offsetData_{0}, offsetLevel0_{0}, label_offset_{0};

  // new offset parameters
  size_t offsetLinklist_{0}, offsetLinklist0_{0};          // pre-blank to align the memory
  size_t offsetLinklistData_{0}, offsetLinklistData0_{0};  // offset to the LinksData

  //    mutable std::unique_ptr<NeighborDataCachePool<dist_t>> layer0_neighbor_cache_pool_{nullptr};

  char* data_level0_memory_{nullptr};
  char** linkLists_{nullptr};
  std::vector<int> element_levels_;  // keeps level of each element

  size_t data_size_{0};

  DISTFUNC<dist_t> fstdistfunc_;
  void* dist_func_param_{nullptr};

  mutable std::mutex label_lookup_lock;  // lock for label_lookup_
  std::unordered_map<labeltype, tableint> label_lookup_;

  std::default_random_engine level_generator_;
  std::default_random_engine update_probability_generator_;

  mutable std::atomic<long> metric_distance_computations{0};
  mutable std::atomic<long> metric_hops{0};

  mutable int64_t knn_upper_layer_cost{0};
  mutable int64_t knn_base_layer_cost{0};

  bool allow_replace_deleted_ =
      false;  // flag to replace deleted elements (marked as deleted) during insertions

  std::mutex deleted_elements_lock;               // lock for deleted_elements
  std::unordered_set<tableint> deleted_elements;  // contains internal ids of deleted elements

  HierarchicalNSWFlash_V4(SpaceInterface<dist_t>* s) {}

  HierarchicalNSWFlash_V4(SpaceInterface<dist_t>* s,
                          const std::string& location,
                          bool nmslib = false,
                          size_t max_elements = 0,
                          bool allow_replace_deleted = false)
      : allow_replace_deleted_(allow_replace_deleted) {
    loadIndex(location, s, max_elements);
  }

  HierarchicalNSWFlash_V4(SpaceInterface<dist_t>* s,
                          size_t max_elements,
                          size_t M = 16,
                          size_t ef_construction = 200,
                          size_t random_seed = 100,
                          bool allow_replace_deleted = false)
      : label_op_locks_(MAX_LABEL_OPERATION_LOCKS)
      , link_list_locks_(max_elements)
      , element_levels_(max_elements)
      , allow_replace_deleted_(allow_replace_deleted) {
    max_elements_ = max_elements;
    num_deleted_ = 0;

    // data_size_ = s->get_data_size();
    data_size_ = CLUSTER_NUM * sizeof(encode_t);

    fstdistfunc_ = s->get_dist_func();
    dist_func_param_ = s->get_dist_func_param();
    if (M <= 10000) {
      M_ = M;
    } else {
      HNSWERR << "warning: M parameter exceeds 10000 which may lead to adverse effects." << std::endl;
      HNSWERR << "         Cap to 10000 will be applied for the rest of the processing." << std::endl;
      M_ = 10000;
    }
    maxM_ = M_;
    maxM0_ = M_ * 2;
    ef_construction_ = std::max(ef_construction, M_);
    ef_ = 10;

    level_generator_.seed(random_seed);
    update_probability_generator_.seed(random_seed + 1);

    //        layer0_neighbor_cache_pool_ = std::unique_ptr<NeighborDataCachePool<dist_t>>(new
    //        NeighborDataCachePool<dist_t>(1, PQ_LINK_LRU_SIZE, maxM0_ * SUBVECTOR_NUM));

#if defined(PQLINK_STORE)
    size_links_level0_ = maxM0_ * sizeof(tableint) + maxM0_ * data_size_ + sizeof(linklistsizeint);
    size_data_per_element_ = size_links_level0_ + data_size_ + sizeof(labeltype);
    offsetLinklist0_ = 0;
    offsetLinklistData0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);  // + tim;
    offsetData_ = size_links_level0_;
    label_offset_ = offsetData_ + data_size_;
#else
    size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
#endif

#if defined(USE_PREFETCH)
    size_links_level0_ = ((size_links_level0_ + 63) / 64) * 64;
#endif

    offsetData_ = size_links_level0_;
    label_offset_ = size_links_level0_ + data_size_;
    offsetLinklist0_ = 0;
    size_data_per_element_ = size_links_level0_ + data_size_ + sizeof(labeltype);
#if defined(USE_PREFETCH)
    size_data_per_element_ = ((size_data_per_element_ + 63) / 64) * 64;
#endif

    std::cout << "data_size_: " << data_size_ << std::endl;
    std::cout << "layer0 memory per data element: " << size_data_per_element_ << std::endl;
    data_level0_memory_ = (char*)aligned_alloc(64, max_elements_ * size_data_per_element_);
    if (data_level0_memory_ == nullptr) throw std::runtime_error("Not enough memory");

    cur_element_count = 0;
    visited_list_pool_ = std::unique_ptr<VisitedListPool>(new VisitedListPool(1, max_elements));

    // initializations for special treatment of the first node
    enterpoint_node_ = -1;
    maxlevel_ = -1;

    linkLists_ = (char**)malloc(sizeof(char*) * max_elements_);
    if (linkLists_ == nullptr)
      throw std::runtime_error("Not enough memory: HierarchicalNSW failed to allocate linklists");

#if defined(PQLINK_STORE)
    offsetLinklist_ = 0;
    offsetLinklistData_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);
    size_links_per_element_ = maxM_ * sizeof(tableint) + maxM_ * data_size_ + sizeof(linklistsizeint);
    std::cout << "link per memory: " << size_links_per_element_ << std::endl;
#else
    offsetLinklist_ = 0;
    size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);
#if defined(USE_PREFETCH)
    size_links_per_element_ = ((size_links_per_element_ + 63) / 64) * 64;
#endif
    std::cout << "level0+ link memory per elemnt: " << size_links_per_element_ << std::endl;
#endif
    mult_ = 1 / log(1.0 * M_);
    revSize_ = 1.0 / mult_;
  }

  ~HierarchicalNSWFlash_V4() {
    clear();
  }

  void clear() {
    free(data_level0_memory_);
    data_level0_memory_ = nullptr;
    for (tableint i = 0; i < cur_element_count; i++) {
      if (element_levels_[i] > 0) {
        free(linkLists_[i]);
      }
    }
    free(linkLists_);
    linkLists_ = nullptr;
    cur_element_count = 0;
    visited_list_pool_.reset(nullptr);
  }

  struct CompareByFirstLess {
    constexpr bool operator()(std::pair<dist_t, tableint> const& a,
                              std::pair<dist_t, tableint> const& b) const noexcept {
      // update more strict check function to make the result same
      return a.first < b.first || a.first == b.first && a.second < b.second;
      // return a.first < b.first;
    }
  };

  struct CompareByFirstGreater {
    constexpr bool operator()(std::pair<dist_t, tableint> const& a,
                              std::pair<dist_t, tableint> const& b) const noexcept {
      return a.first > b.first || a.first == b.first && a.second < b.second;
    }
  };

  void setEf(size_t ef) {
    ef_ = ef;
  }

  inline std::mutex& getLabelOpMutex(labeltype label) const {
    // calculate hash
    size_t lock_id = label & (MAX_LABEL_OPERATION_LOCKS - 1);
    return label_op_locks_[lock_id];
  }

  inline labeltype getExternalLabel(tableint internal_id) const {
    labeltype return_label;
    memcpy(&return_label, (data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_),
           sizeof(labeltype));
    return return_label;
  }

  inline void setExternalLabel(tableint internal_id, labeltype label) const {
    memcpy((data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), &label,
           sizeof(labeltype));
  }

  inline labeltype* getExternalLabeLp(tableint internal_id) const {
    return (labeltype*)(data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_);
  }

  inline char* getDataByInternalId(tableint internal_id) const {
    return (data_level0_memory_ + internal_id * size_data_per_element_ + offsetData_);
  }

#if defined(PQLINK_STORE)
  inline char* getLinksData(tableint internal_id, int level = 0) const {
    return level == 0
               ? (data_level0_memory_ + internal_id * size_data_per_element_ + offsetLinklistData0_)
               : (linkLists_[internal_id] + (level - 1) * size_links_per_element_ + offsetLinklistData_);
  }
#endif

  /*
      Method
      for each node, set consecutive two dimensions in one Byte
      the first 16 Bytes are "the 0th and 1st dim of node 0", "the 0th and 1st dim of node 1", ..., "the 0th
     and 1st dim of node 15" the second 16 Bytes are "the 2nd and 3rd dim of node 0", "the 2nd and 3rd dim of
     node 1", ..., "the 2nd and 3rd dim of node 15" and so on ... each 16 vectors set together

      Data Posision
      level0: set the links data after the origin data
      level0+: set the links data after the link list in each level
  */
  void setLinksData(const void* data, size_t idx, tableint neighbor_id, int level) const {
    uint8_t* p_data = (uint8_t*)data - sizeof(linklistsizeint) +
                      (level == 0 ? offsetLinklistData0_ : offsetLinklistData_) +
                      (idx * SUBVECTOR_NUM * sizeof(uint16_t));
    uint8_t* data_point = (uint8_t*)getDataByInternalId(neighbor_id);
    memcpy(p_data, data_point, SUBVECTOR_NUM * sizeof(uint16_t));
  }

  int getRandomLevel(double reverse_size) {
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double r = -log(distribution(level_generator_)) * reverse_size;
    return (int)r;
  }

  size_t getMaxElements() {
    return max_elements_;
  }

  size_t getCurrentElementCount() {
    return cur_element_count;
  }

  size_t getDeletedCount() {
    return num_deleted_;
  }

  void countOutDegrees(std::vector<std::vector<linklistsizeint>>& out_degrees) {
    out_degrees.resize(max_elements_);
    for (int i = 0; i < max_elements_; i++) {
      out_degrees[i].resize(element_levels_[i] + 1);
    }

    for (int i = 0; i < max_elements_; i++) {
      for (int level = 0; level <= element_levels_[i]; level++) {
        linklistsizeint* ll_cur;
        if (level == 0) {
          ll_cur = get_linklist0(i);
        } else {
          ll_cur = get_linklist(i, level);
        }
        out_degrees[i][level] = *ll_cur;
      }
    }
  }

  std::priority_queue<std::pair<dist_t, tableint>,
                      std::vector<std::pair<dist_t, tableint>>,
                      CompareByFirstLess>
  searchBaseLayer(tableint ep_id, const void* data_point, int layer) {
    VisitedList* vl = visited_list_pool_->getFreeVisitedList();
    vl_type* visited_array = vl->mass;
    vl_type visited_array_tag = vl->curV;

    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                        CompareByFirstLess>
        top_candidates;
#ifdef ALL_POSITIVE_NUMBER
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                        CompareByFirstGreater>
        candidateSet;
#else
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                        CompareByFirstLess>
        candidateSet;
#endif
    dist_t lowerBound;
    if (!isMarkedDeleted(ep_id)) {
      dist_t dist = flash_l2sqr_dist(data_point, getDataByInternalId(ep_id));

      top_candidates.emplace(dist, ep_id);
      lowerBound = dist;
#ifdef ALL_POSITIVE_NUMBER
      candidateSet.emplace(dist, ep_id);
#else
      candidateSet.emplace(-dist, ep_id);
#endif
    } else {
      lowerBound = std::numeric_limits<dist_t>::max();
#ifdef ALL_POSITIVE_NUMBER
      candidateSet.emplace(lowerBound, ep_id);
#else
      candidateSet.emplace(-lowerBound, ep_id);
#endif
    }
    visited_array[ep_id] = visited_array_tag;

    while (!candidateSet.empty()) {
      std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();
#ifdef ALL_POSITIVE_NUMBER
      if ((curr_el_pair.first) > lowerBound && top_candidates.size() == ef_construction_) {
#else
      if ((-curr_el_pair.first) > lowerBound && top_candidates.size() == ef_construction_) {
#endif
        break;
      }
      candidateSet.pop();

      tableint curNodeNum = curr_el_pair.second;

      std::unique_lock<std::mutex> lock(link_list_locks_[curNodeNum]);

      int* data;  // = (int *)(linkList0_ + curNodeNum * size_links_per_element0_);
      if (layer == 0) {
        data = (int*)get_linklist0(curNodeNum);
      } else {
        data = (int*)get_linklist(curNodeNum, layer);
        //                    data = (int *) (linkLists_[curNodeNum] + (layer - 1) * size_links_per_element_);
      }
      size_t size = getListCount((linklistsizeint*)data);
      tableint* datal = (tableint*)(data + 1);

      // #ifdef USE_SSE
      //             _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
      // //             _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
      // //             _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
      // //             _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);
      // #endif
      for (size_t j = 0; j < size; j++) {
        tableint candidate_id = *(datal + j);
        //                    if (candidate_id == 0) continue;
        // #ifdef USE_SSE
        //                  _mm_prefetch((char *) (visited_array + *(datal + j + 1)), _MM_HINT_T0);
        // //                 _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
        // #endif
        if (visited_array[candidate_id] == visited_array_tag) continue;
        visited_array[candidate_id] = visited_array_tag;

#if defined(PQLINK_CALC)
        dist_t dist1 = dist_list[j];
#else
        char* currObj1 = (getDataByInternalId(candidate_id));
        dist_t dist1 = flash_l2sqr_dist(data_point, currObj1);

        // todo:
        // dist_t dist1 = dist_list[j];
#endif

        if (top_candidates.size() < ef_construction_ || lowerBound > dist1) {
#ifdef ALL_POSITIVE_NUMBER
          candidateSet.emplace(dist1, candidate_id);
#else
          candidateSet.emplace(-dist1, candidate_id);
#endif
          // #ifdef USE_SSE
          //                     _mm_prefetch(getDataByInternalId(candidateSet.top().second), _MM_HINT_T0);
          // #endif

          if (!isMarkedDeleted(candidate_id)) top_candidates.emplace(dist1, candidate_id);

          if (top_candidates.size() > ef_construction_) top_candidates.pop();

          if (!top_candidates.empty()) lowerBound = top_candidates.top().first;
        }
      }
#if defined(PQLINK_CALC)
      // free(dist_list);
#endif
    }
    visited_list_pool_->releaseVisitedList(vl);

    return top_candidates;
  }

  // bare_bone_search means there is no check for deletions and stop condition is ignored in return of extra
  // performance
  template <bool bare_bone_search = true, bool collect_metrics = false>
  std::priority_queue<std::pair<dist_t, tableint>,
                      std::vector<std::pair<dist_t, tableint>>,
                      CompareByFirstLess>
  searchBaseLayerST(tableint ep_id,
                    const void* data_point,
                    size_t ef,
                    BaseFilterFunctor* isIdAllowed = nullptr,
                    BaseSearchStopCondition<dist_t>* stop_condition = nullptr) const {
    VisitedList* vl = visited_list_pool_->getFreeVisitedList();
    // NeighborDataCache<dist_t>* layer0_neighbor_cache =
    // layer0_neighbor_cache_pool_->getFreeCache(omp_get_thread_num());

    vl_type* visited_array = vl->mass;
    vl_type visited_array_tag = vl->curV;

    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                        CompareByFirstLess>
        top_candidates;
#ifdef ALL_POSITIVE_NUMBER
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                        CompareByFirstGreater>
        candidate_set;
#else
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                        CompareByFirstLess>
        candidate_set;
#endif

    dist_t lowerBound;
    if (bare_bone_search ||
        (!isMarkedDeleted(ep_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(ep_id))))) {
      char* ep_data = getDataByInternalId(ep_id);
      dist_t dist = flash_l2sqr_dist(data_point, ep_data);

      lowerBound = dist;
      top_candidates.emplace(dist, ep_id);
      if (!bare_bone_search && stop_condition) {
        stop_condition->add_point_to_result(getExternalLabel(ep_id), ep_data, dist);
      }
#ifdef ALL_POSITIVE_NUMBER
      candidate_set.emplace(dist, ep_id);
#else
      candidate_set.emplace(-dist, ep_id);
#endif
    } else {
      lowerBound = std::numeric_limits<dist_t>::max();
#ifdef ALL_POSITIVE_NUMBER
      candidate_set.emplace(lowerBound, ep_id);
#else
      candidate_set.emplace(-lowerBound, ep_id);
#endif
    }

    visited_array[ep_id] = visited_array_tag;

#if defined(TRACE_SEARCH)
    constexpr bool need_trace = true;
#else
    constexpr bool need_trace = false;
#endif
    if (need_trace) {
      std::cout << "begin to search layer0: " << std::endl;
    }
    while (!candidate_set.empty()) {
      std::pair<dist_t, tableint> current_node_pair = candidate_set.top();
#ifdef ALL_POSITIVE_NUMBER
      dist_t candidate_dist = current_node_pair.first;
#else
      dist_t candidate_dist = -current_node_pair.first;
#endif

      bool flag_stop_search;
      if (bare_bone_search) {
        flag_stop_search = candidate_dist > lowerBound;

      } else {
        if (stop_condition) {
          flag_stop_search = stop_condition->should_stop_search(candidate_dist, lowerBound);
        } else {
          flag_stop_search = candidate_dist > lowerBound && top_candidates.size() == ef;
#if defined(VBASE)
          flag_stop_search = midian > lowerBound && top_candidates.size() == ef;
#endif
        }
      }
      if (flag_stop_search) {
        break;
      }
      candidate_set.pop();

      tableint current_node_id = current_node_pair.second;
      int* data = (int*)get_linklist0(current_node_id);
      size_t size = getListCount((linklistsizeint*)data);
      //                bool cur_node_deleted = isMarkedDeleted(current_node_id);
      if (collect_metrics) {
        metric_hops++;
        metric_distance_computations += size;
      }

      if (need_trace) {
        std::cout << "enter _point: " << getExternalLabel(current_node_id) << ", dis: " << candidate_dist
                  << ", size: " << size << ", cur_hops: " << metric_hops
                  << ", cur_comp: " << metric_distance_computations << std::endl;
      }

      // #ifdef USE_SSE
      //             _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
      //             _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
      //             _mm_prefetch(data_level0_memory_ + (*(data + 1)) * size_data_per_element_ + offsetData_,
      //             _MM_HINT_T0); _mm_prefetch((char *) (data + 2), _MM_HINT_T0);
      // #endif
      tableint* datal = (tableint*)(data + 1);

      encode_t* neighbors_data = (encode_t*)alloca(size * SUBVECTOR_NUM * sizeof(encode_t));
      const size_t data_size = SUBVECTOR_NUM * sizeof(encode_t);
      for (int k = 0; k < size; ++k) {
        tableint neighbor_id = datal[k];

        const encode_t* neighbor_data = (encode_t*)getDataByInternalId(neighbor_id);
        encode_t* dst = neighbors_data + k * SUBVECTOR_NUM;
        for (int m = 0; m < SUBVECTOR_NUM; m += 2) {
          dst[m] = neighbor_data[m];
          dst[m + 1] = neighbor_data[m + 1];
        }

        // memcpy(neighbors_data + k * SUBVECTOR_NUM, neighbor_data, data_size);
      }
      dist_t* dist_list = (dist_t*)alloca(maxM0_ * sizeof(dist_t));
      PqLinkL2Sqr(dist_list, data_point, neighbors_data, size, 0, lowerBound);

      for (size_t j = 0; j < size; j++) {
        int candidate_id = datal[j];
        //                    if (candidate_id == 0) continue;
        // #ifdef USE_SSE
        //                 _mm_prefetch((char *) (visited_array + *(data + j + 1)), _MM_HINT_T0);
        //                 _mm_prefetch(data_level0_memory_ + (*(data + j + 1)) * size_data_per_element_ +
        //                 offsetData_,
        //                                 _MM_HINT_T0);  ////////////
        // #endif
        if (!(visited_array[candidate_id] == visited_array_tag)) {
          visited_array[candidate_id] = visited_array_tag;

          // char* currObj1 = getDataByInternalId(candidate_id);
          // auto dist = flash_l2sqr_dist(data_point, currObj1);

          auto dist = dist_list[j];

          if (need_trace) {
            std::cout << "(" << getExternalLabel(candidate_id) << "," << dist << "), ";
          }

          bool flag_consider_candidate;
          if (!bare_bone_search && stop_condition) {
            flag_consider_candidate = stop_condition->should_consider_candidate(dist, lowerBound);
          } else {
            flag_consider_candidate = top_candidates.size() < ef || lowerBound > dist;
          }

          if (flag_consider_candidate) {
#ifdef ALL_POSITIVE_NUMBER
            candidate_set.emplace(dist, candidate_id);
#else
            candidate_set.emplace(-dist, candidate_id);
#endif
            // #ifdef USE_SSE
            //                         _mm_prefetch(data_level0_memory_ + candidate_set.top().second *
            //                         size_data_per_element_ +
            //                                         offsetLevel0_,  ///////////
            //                                         _MM_HINT_T0);  ////////////////////////
            // #endif

            if (bare_bone_search || (!isMarkedDeleted(candidate_id) &&
                                     ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(candidate_id))))) {
              top_candidates.emplace(dist, candidate_id);
              if (!bare_bone_search && stop_condition) {
                // stop_condition->add_point_to_result(getExternalLabel(candidate_id), currObj1, dist);
              }
            }

            bool flag_remove_extra = false;
            if (!bare_bone_search && stop_condition) {
              flag_remove_extra = stop_condition->should_remove_extra();
            } else {
              flag_remove_extra = top_candidates.size() > ef;
            }
            while (flag_remove_extra) {
              tableint id = top_candidates.top().second;
              top_candidates.pop();
              if (!bare_bone_search && stop_condition) {
                stop_condition->remove_point_from_result(getExternalLabel(id), getDataByInternalId(id), dist);
                flag_remove_extra = stop_condition->should_remove_extra();
              } else {
                flag_remove_extra = top_candidates.size() > ef;
              }
            }

            if (!top_candidates.empty()) lowerBound = top_candidates.top().first;
          }
        }
      }

      if (need_trace) {
        std::cout << std::endl;
        std::cout << std::endl;
      }
    }

    visited_list_pool_->releaseVisitedList(vl);
    return top_candidates;
  }

  void getNeighborsByHeuristic2(std::priority_queue<std::pair<dist_t, tableint>,
                                                    std::vector<std::pair<dist_t, tableint>>,
                                                    CompareByFirstLess>& top_candidates,
                                const size_t M) {
    if (top_candidates.size() < M) {
      return;
    }

#ifdef ALL_POSITIVE_NUMBER
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                        CompareByFirstGreater>
        queue_closest;
#else
    std::priority_queue<std::pair<dist_t, tableint>> queue_closest;
#endif
    std::vector<std::pair<dist_t, tableint>> return_list;
    while (top_candidates.size() > 0) {
#ifdef ALL_POSITIVE_NUMBER
      queue_closest.emplace(top_candidates.top().first, top_candidates.top().second);
#else
      queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
#endif
      top_candidates.pop();
    }

    while (queue_closest.size()) {
      if (return_list.size() >= M) break;
      std::pair<dist_t, tableint> curent_pair = queue_closest.top();
#ifdef ALL_POSITIVE_NUMBER
      dist_t dist_to_query = curent_pair.first;
#else
      dist_t dist_to_query = -curent_pair.first;
#endif
      queue_closest.pop();
      bool good = true;
      // todo: 可以整体一起做
      for (std::pair<dist_t, tableint> second_pair : return_list) {
        dist_t curdist =
            PqSdcL2Sqr(getDataByInternalId(second_pair.second), getDataByInternalId(curent_pair.second));

        if (curdist < dist_to_query) {
          good = false;
          break;
        }
      }
      if (good) {
        return_list.push_back(curent_pair);
      }
    }

    for (std::pair<dist_t, tableint> curent_pair : return_list) {
#ifdef ALL_POSITIVE_NUMBER
      top_candidates.emplace(curent_pair.first, curent_pair.second);
#else
      top_candidates.emplace(-curent_pair.first, curent_pair.second);
#endif
    }
  }

  linklistsizeint* get_linklist0(tableint internal_id) const {
    return (linklistsizeint*)(data_level0_memory_ + internal_id * size_data_per_element_ + offsetLinklist0_);
  }

  linklistsizeint* get_linklist0(tableint internal_id, char* data_level0_memory_) const {
    return (linklistsizeint*)(data_level0_memory_ + internal_id * size_data_per_element_ + offsetLinklist0_);
  }

  linklistsizeint* get_linklist(tableint internal_id, int level) const {
    return level == 0 ? get_linklist0(internal_id)
                      : (linklistsizeint*)(linkLists_[internal_id] + (level - 1) * size_links_per_element_ +
                                           offsetLinklist_);
  }

  linklistsizeint* get_linklist_at_level(tableint internal_id, int level) const {
    return level == 0 ? get_linklist0(internal_id) : get_linklist(internal_id, level);
  }

  tableint mutuallyConnectNewElement(const void* data_point,
                                     tableint cur_c,
                                     std::priority_queue<std::pair<dist_t, tableint>,
                                                         std::vector<std::pair<dist_t, tableint>>,
                                                         CompareByFirstLess>& top_candidates,
                                     int level,
                                     bool isUpdate) {
    size_t Mcurmax = level ? maxM_ : maxM0_;
    getNeighborsByHeuristic2(top_candidates, Mcurmax);
    if (top_candidates.size() > Mcurmax)
      throw std::runtime_error("Should be not be more than M_ candidates returned by the heuristic");

    std::vector<tableint> selectedNeighbors;
    selectedNeighbors.reserve(Mcurmax);
    while (top_candidates.size() > 0) {
      selectedNeighbors.push_back(top_candidates.top().second);
      top_candidates.pop();
    }

    tableint next_closest_entry_point = selectedNeighbors.back();

    {
      // lock only during the update
      // because during the addition the lock for cur_c is already acquired
      std::unique_lock<std::mutex> lock(link_list_locks_[cur_c], std::defer_lock);
      if (isUpdate) {
        lock.lock();
      }
      linklistsizeint* ll_cur;
      if (level == 0)
        ll_cur = get_linklist0(cur_c);
      else
        ll_cur = get_linklist(cur_c, level);

      if (*ll_cur && !isUpdate) {
        throw std::runtime_error("The newly inserted element should have blank link list");
      }
      setListCount(ll_cur, selectedNeighbors.size());
      tableint* data = (tableint*)(ll_cur + 1);
      for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
        if (data[idx] && !isUpdate) throw std::runtime_error("Possible memory corruption");
        if (level > element_levels_[selectedNeighbors[idx]])
          throw std::runtime_error("Trying to make a link on a non-existent level");

        data[idx] = selectedNeighbors[idx];
#if defined(PQLINK_STORE)
        setLinksData(data, idx, selectedNeighbors[idx], level);
#endif
      }
    }

    for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
      std::unique_lock<std::mutex> lock(link_list_locks_[selectedNeighbors[idx]]);

      linklistsizeint* ll_other;
      if (level == 0)
        ll_other = get_linklist0(selectedNeighbors[idx]);
      else
        ll_other = get_linklist(selectedNeighbors[idx], level);

      size_t sz_link_list_other = getListCount(ll_other);

      if (sz_link_list_other > Mcurmax) throw std::runtime_error("Bad value of sz_link_list_other");
      if (selectedNeighbors[idx] == cur_c) throw std::runtime_error("Trying to connect an element to itself");
      if (level > element_levels_[selectedNeighbors[idx]])
        throw std::runtime_error("Trying to make a link on a non-existent level");

      tableint* data = (tableint*)(ll_other + 1);

      bool is_cur_c_present = false;
      if (isUpdate) {
        for (size_t j = 0; j < sz_link_list_other; j++) {
          if (data[j] == cur_c) {
            is_cur_c_present = true;
            break;
          }
        }
      }

      // If cur_c is already present in the neighboring connections of `selectedNeighbors[idx]` then no need
      // to modify any connections or run the heuristics.
      if (!is_cur_c_present) {
        if (sz_link_list_other < Mcurmax) {
          data[sz_link_list_other] = cur_c;
#if defined(PQLINK_STORE)
          setLinksData(data, sz_link_list_other, cur_c, level);
#endif
          setListCount(ll_other, sz_link_list_other + 1);
        } else {
          // finding the "weakest" element to replace it with the new one
          dist_t d_max = PqSdcL2Sqr(getDataByInternalId(cur_c), getDataByInternalId(selectedNeighbors[idx]));

          // Heuristic:
          std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                              CompareByFirstLess>
              candidates;
          candidates.emplace(d_max, cur_c);

          // todo: 可以整体一起做
          for (size_t j = 0; j < sz_link_list_other; j++) {
#if defined(PQLINK_CALC) && !defined(SAVE_MEMORY)
            candidates.emplace(dist_list[j], data[j]);
#else
            candidates.emplace(
                PqSdcL2Sqr(getDataByInternalId(data[j]), getDataByInternalId(selectedNeighbors[idx])),
                data[j]);
#endif
          }
#if defined(PQLINK_CALC) && !defined(SAVE_MEMORY)
          // free(dist_list);
#endif
          getNeighborsByHeuristic2(candidates, Mcurmax);

          int indx = 0;
          while (candidates.size() > 0) {
            data[indx] = candidates.top().second;
#if defined(PQLINK_STORE)
            setLinksData(data, indx, candidates.top().second, level);
#endif
            candidates.pop();
            indx++;
          }

          setListCount(ll_other, indx);
        }
      }
    }

    return next_closest_entry_point;
  }

  void resizeIndex(size_t new_max_elements) {
    if (new_max_elements < cur_element_count)
      throw std::runtime_error("Cannot resize, max element is less than the current number of elements");

    visited_list_pool_.reset(new VisitedListPool(1, new_max_elements));

    element_levels_.resize(new_max_elements);

    std::vector<std::mutex>(new_max_elements).swap(link_list_locks_);

    // Reallocate base layer
    char* data_level0_memory_new =
        (char*)realloc(data_level0_memory_, new_max_elements * size_data_per_element_);
    if (data_level0_memory_new == nullptr)
      throw std::runtime_error("Not enough memory: resizeIndex failed to allocate base layer");
    data_level0_memory_ = data_level0_memory_new;

    // Reallocate all other layers
    char** linkLists_new = (char**)realloc(linkLists_, sizeof(void*) * new_max_elements);
    if (linkLists_new == nullptr)
      throw std::runtime_error("Not enough memory: resizeIndex failed to allocate other layers");
    linkLists_ = linkLists_new;

    max_elements_ = new_max_elements;
  }

  size_t indexFileSize() const {
    size_t size = 0;

    size += sizeof(max_elements_);
    size += sizeof(cur_element_count);
    size += sizeof(M_);
    size += sizeof(maxM_);
    size += sizeof(maxM0_);
    size += sizeof(ef_construction_);
    size += sizeof(mult_);
    size += sizeof(maxlevel_);
    size += sizeof(enterpoint_node_);

    size += sizeof(size_data_per_element_);
    size += sizeof(size_links_level0_);
    size += sizeof(size_links_per_element_);

    size += sizeof(offsetLevel0_);
    size += sizeof(offsetLinklist0_);
    size += sizeof(offsetLinklist_);
    size += sizeof(offsetLinklistData0_);
    size += sizeof(offsetLinklistData_);
    size += sizeof(offsetData_);
    size += sizeof(label_offset_);

    size += cur_element_count * size_data_per_element_;

    for (size_t i = 0; i < cur_element_count; i++) {
      unsigned int linkListSize = element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
      size += sizeof(linkListSize);
      size += linkListSize;
    }
    return size;
  }

  void saveIndex(const std::string& location) {
    std::ofstream output(location, std::ios::binary);
    std::streampos position;

    writeBinaryPOD(output, max_elements_);
    writeBinaryPOD(output, cur_element_count);
    writeBinaryPOD(output, M_);
    writeBinaryPOD(output, maxM_);
    writeBinaryPOD(output, maxM0_);
    writeBinaryPOD(output, ef_construction_);
    writeBinaryPOD(output, mult_);
    writeBinaryPOD(output, maxlevel_);
    writeBinaryPOD(output, enterpoint_node_);

    writeBinaryPOD(output, size_data_per_element_);
    writeBinaryPOD(output, size_links_level0_);
    writeBinaryPOD(output, size_links_per_element_);

    writeBinaryPOD(output, offsetLevel0_);
    writeBinaryPOD(output, offsetLinklist0_);
    writeBinaryPOD(output, offsetLinklist_);
    writeBinaryPOD(output, offsetLinklistData0_);
    writeBinaryPOD(output, offsetLinklistData_);
    writeBinaryPOD(output, offsetData_);
    writeBinaryPOD(output, label_offset_);

    output.write(data_level0_memory_, cur_element_count * size_data_per_element_);

    for (size_t i = 0; i < cur_element_count; i++) {
      unsigned int linkListSize = element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
      writeBinaryPOD(output, linkListSize);
      if (linkListSize) output.write(linkLists_[i], linkListSize);
    }
    output.close();
  }

  void loadIndex(const std::string& location, SpaceInterface<dist_t>* s, size_t max_elements_i = 0) {
    std::ifstream input(location, std::ios::binary);

    if (!input.is_open()) throw std::runtime_error("Cannot open file");

    clear();
    // get file size:
    input.seekg(0, input.end);
    std::streampos total_filesize = input.tellg();
    input.seekg(0, input.beg);

    readBinaryPOD(input, max_elements_);
    readBinaryPOD(input, cur_element_count);

    size_t max_elements = max_elements_;
    if (cur_element_count < max_elements_) max_elements = cur_element_count;
    max_elements_ = max_elements;

    readBinaryPOD(input, M_);
    readBinaryPOD(input, maxM_);
    readBinaryPOD(input, maxM0_);
    readBinaryPOD(input, ef_construction_);
    readBinaryPOD(input, mult_);
    readBinaryPOD(input, maxlevel_);
    readBinaryPOD(input, enterpoint_node_);

    readBinaryPOD(input, size_data_per_element_);
    std::cout << "size_data_per_element: " << size_data_per_element_ << std::endl;
    readBinaryPOD(input, size_links_level0_);
    readBinaryPOD(input, size_links_per_element_);

    readBinaryPOD(input, offsetLevel0_);
    readBinaryPOD(input, offsetLinklist0_);
    readBinaryPOD(input, offsetLinklist_);
    readBinaryPOD(input, offsetLinklistData0_);
    readBinaryPOD(input, offsetLinklistData_);
    readBinaryPOD(input, offsetData_);
    readBinaryPOD(input, label_offset_);

    std::cout << "offset_data: " << offsetData_ << std::endl;
    std::cout << "offset_label: " << label_offset_ << std::endl;

    // data_size_ = s->get_data_size();
    data_size_ = CLUSTER_NUM * sizeof(encode_t);

    fstdistfunc_ = s->get_dist_func();
    dist_func_param_ = s->get_dist_func_param();

    std::cout << "data_size: " << data_size_ << std::endl;

    auto pos = input.tellg();

    /// Optional - check if index is ok:
    input.seekg(cur_element_count * size_data_per_element_, input.cur);
    for (size_t i = 0; i < cur_element_count; i++) {
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
    /// Optional check end

    input.seekg(pos, input.beg);

    data_level0_memory_ = (char*)aligned_alloc(64, max_elements * size_data_per_element_);
    if (data_level0_memory_ == nullptr)
      throw std::runtime_error("Not enough memory: loadIndex failed to allocate level0");
    input.read(data_level0_memory_, cur_element_count * size_data_per_element_);

    std::vector<std::mutex>(max_elements).swap(link_list_locks_);
    std::vector<std::mutex>(MAX_LABEL_OPERATION_LOCKS).swap(label_op_locks_);

    visited_list_pool_.reset(new VisitedListPool(1, max_elements));

    linkLists_ = (char**)malloc(sizeof(void*) * max_elements);
    if (linkLists_ == nullptr)
      throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklists");
    element_levels_ = std::vector<int>(max_elements);
    revSize_ = 1.0 / mult_;
    ef_ = 10;
    for (size_t i = 0; i < cur_element_count; i++) {
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
        input.read(linkLists_[i], linkListSize);
      }
    }

    for (size_t i = 0; i < cur_element_count; i++) {
      if (isMarkedDeleted(i)) {
        num_deleted_ += 1;
        if (allow_replace_deleted_) deleted_elements.insert(i);
      }
    }

    input.close();
#if defined(DEBUG_LOG)
    std::vector<std::vector<linklistsizeint>> stat_out_degrees;
    countOutDegrees(stat_out_degrees);

    std::unordered_map<int /*outdegree*/, int /*count*/> mp;
    for (const auto& level_out_degrees : stat_out_degrees) {
      mp[level_out_degrees[0]]++;
    }

    std::cout << "count outdegree: " << std::endl;
    for (const auto& [key, val] : mp) {
      std::cout << "[" << key << ", " << val << "], ";
    }
    std::cout << std::endl;
#endif

    return;
  }

  template <typename data_t>
  std::vector<data_t> getDataByLabel(labeltype label) const {
    // lock all operations with element by label
    std::unique_lock<std::mutex> lock_label(getLabelOpMutex(label));

    std::unique_lock<std::mutex> lock_table(label_lookup_lock);
    auto search = label_lookup_.find(label);
    if (search == label_lookup_.end() || isMarkedDeleted(search->second)) {
      throw std::runtime_error("Label not found");
    }
    tableint internalId = search->second;
    lock_table.unlock();

    char* data_ptrv = getDataByInternalId(internalId);
    size_t dim = *((size_t*)dist_func_param_);
    std::vector<data_t> data;
    data_t* data_ptr = (data_t*)data_ptrv;
    for (size_t i = 0; i < dim; i++) {
      data.push_back(*data_ptr);
      data_ptr += 1;
    }
    return data;
  }

  /*
   * Marks an element with the given label deleted, does NOT really change the current graph.
   */
  void markDelete(labeltype label) {
    // lock all operations with element by label
    std::unique_lock<std::mutex> lock_label(getLabelOpMutex(label));

    std::unique_lock<std::mutex> lock_table(label_lookup_lock);
    auto search = label_lookup_.find(label);
    if (search == label_lookup_.end()) {
      throw std::runtime_error("Label not found");
    }
    tableint internalId = search->second;
    lock_table.unlock();

    markDeletedInternal(internalId);
  }

  /*
   * Uses the last 16 bits of the memory for the linked list size to store the mark,
   * whereas maxM0_ has to be limited to the lower 16 bits, however, still large enough in almost all cases.
   */
  void markDeletedInternal(tableint internalId) {
    assert(internalId < cur_element_count);
    if (!isMarkedDeleted(internalId)) {
      unsigned char* ll_cur = ((unsigned char*)get_linklist0(internalId)) + 2;
      *ll_cur |= DELETE_MARK;
      num_deleted_ += 1;
      if (allow_replace_deleted_) {
        std::unique_lock<std::mutex> lock_deleted_elements(deleted_elements_lock);
        deleted_elements.insert(internalId);
      }
    } else {
      throw std::runtime_error("The requested to delete element is already deleted");
    }
  }

  /*
   * Removes the deleted mark of the node, does NOT really change the current graph.
   *
   * Note: the method is not safe to use when replacement of deleted elements is enabled,
   *  because elements marked as deleted can be completely removed by addPoint
   */
  void unmarkDelete(labeltype label) {
    // lock all operations with element by label
    std::unique_lock<std::mutex> lock_label(getLabelOpMutex(label));

    std::unique_lock<std::mutex> lock_table(label_lookup_lock);
    auto search = label_lookup_.find(label);
    if (search == label_lookup_.end()) {
      throw std::runtime_error("Label not found");
    }
    tableint internalId = search->second;
    lock_table.unlock();

    unmarkDeletedInternal(internalId);
  }

  /*
   * Remove the deleted mark of the node.
   */
  void unmarkDeletedInternal(tableint internalId) {
    assert(internalId < cur_element_count);
    if (isMarkedDeleted(internalId)) {
      unsigned char* ll_cur = ((unsigned char*)get_linklist0(internalId)) + 2;
      *ll_cur &= ~DELETE_MARK;
      num_deleted_ -= 1;
      if (allow_replace_deleted_) {
        std::unique_lock<std::mutex> lock_deleted_elements(deleted_elements_lock);
        deleted_elements.erase(internalId);
      }
    } else {
      throw std::runtime_error("The requested to undelete element is not deleted");
    }
  }

  /*
   * Checks the first 16 bits of the memory to see if the element is marked deleted.
   */
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

  /*
   * Adds point. Updates the point if it is already in the index.
   * If replacement of deleted elements is enabled: replaces previously deleted point if any, updating it with
   * new point
   */
  void addPoint(const void* data_point, labeltype label, bool replace_deleted = false) {
    if ((allow_replace_deleted_ == false) && (replace_deleted == true)) {
      throw std::runtime_error("Replacement of deleted elements is disabled in constructor");
    }

    // lock all operations with element by label
    std::unique_lock<std::mutex> lock_label(getLabelOpMutex(label));
    if (!replace_deleted) {
      addPoint(data_point, label, -1);
      return;
    }

    std::cout << "leave replace_deleted" << std::endl;

    // check if there is vacant place
    tableint internal_id_replaced;
    std::unique_lock<std::mutex> lock_deleted_elements(deleted_elements_lock);
    bool is_vacant_place = !deleted_elements.empty();
    if (is_vacant_place) {
      internal_id_replaced = *deleted_elements.begin();
      deleted_elements.erase(internal_id_replaced);
    }
    lock_deleted_elements.unlock();

    // if there is no vacant place then add or update point
    // else add point to vacant place
    if (!is_vacant_place) {
      addPoint(data_point, label, -1);
    } else {
      // we assume that there are no concurrent operations on deleted element
      labeltype label_replaced = getExternalLabel(internal_id_replaced);
      setExternalLabel(internal_id_replaced, label);

      std::unique_lock<std::mutex> lock_table(label_lookup_lock);
      label_lookup_.erase(label_replaced);
      label_lookup_[label] = internal_id_replaced;
      lock_table.unlock();

      unmarkDeletedInternal(internal_id_replaced);
      updatePoint(data_point, internal_id_replaced, 1.0);
    }
  }

  void updatePoint(const void* dataPoint, tableint internalId, float updateNeighborProbability) {
    // update the feature vector associated with existing point with new vector
    memcpy(getDataByInternalId(internalId), dataPoint, data_size_);

    int maxLevelCopy = maxlevel_;
    tableint entryPointCopy = enterpoint_node_;
    // If point to be updated is entry point and graph just contains single element then just return.
    if (entryPointCopy == internalId && cur_element_count == 1) return;

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

        sNeigh.insert(elOneHop);

        std::vector<tableint> listTwoHop = getConnectionsWithLock(elOneHop, layer);
        for (auto&& elTwoHop : listTwoHop) {
          sCand.insert(elTwoHop);
        }
      }

      for (auto&& neigh : sNeigh) {
        // if (neigh == internalId)
        //     continue;

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                            CompareByFirstLess>
            candidates;
        size_t size = sCand.find(neigh) == sCand.end()
                          ? sCand.size()
                          : sCand.size() - 1;  // sCand guaranteed to have size >= 1
        size_t elementsToKeep = std::min(ef_construction_, size);
        for (auto&& cand : sCand) {
          if (cand == neigh) continue;
          dist_t distance = PqSdcL2Sqr(getDataByInternalId(neigh), getDataByInternalId(cand));

          if (candidates.size() < elementsToKeep) {
            candidates.emplace(distance, cand);
          } else {
            if (distance < candidates.top().first) {
              candidates.pop();
              candidates.emplace(distance, cand);
            }
          }
        }

        // Retrieve neighbours using heuristic and set connections.
        getNeighborsByHeuristic2(candidates, layer == 0 ? maxM0_ : maxM_);

        {
          std::unique_lock<std::mutex> lock(link_list_locks_[neigh]);
          linklistsizeint* ll_cur;
          ll_cur = get_linklist_at_level(neigh, layer);
          size_t candSize = candidates.size();
          setListCount(ll_cur, candSize);
          tableint* data = (tableint*)(ll_cur + 1);
          for (size_t idx = 0; idx < candSize; idx++) {
            data[idx] = candidates.top().second;
            candidates.pop();
          }
        }
      }
    }

    repairConnectionsForUpdate(dataPoint, entryPointCopy, internalId, elemLevel, maxLevelCopy);
  }

  void repairConnectionsForUpdate(const void* dataPoint,
                                  tableint entryPointInternalId,
                                  tableint dataPointInternalId,
                                  int dataPointLevel,
                                  int maxLevel) {
    tableint currObj = entryPointInternalId;
    if (dataPointLevel < maxLevel) {
      dist_t curdist = flash_l2sqr_dist(dataPoint, getDataByInternalId(currObj));

      for (int level = maxLevel; level > dataPointLevel; level--) {
        bool changed = true;
        while (changed) {
          changed = false;
          unsigned int* data;
          std::unique_lock<std::mutex> lock(link_list_locks_[currObj]);
          data = get_linklist_at_level(currObj, level);
          int size = getListCount(data);
          tableint* datal = (tableint*)(data + 1);
          // #ifdef USE_SSE
          //                     _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
          // #endif
          for (int i = 0; i < size; i++) {
            // #ifdef USE_SSE
            //                         _mm_prefetch(getDataByInternalId(*(datal + i + 1)), _MM_HINT_T0);
            // #endif
            tableint cand = datal[i];
            dist_t d = flash_l2sqr_dist(dataPoint, getDataByInternalId(cand));
            if (d < curdist) {
              curdist = d;
              currObj = cand;
              changed = true;
            }
          }
        }
      }
    }

    if (dataPointLevel > maxLevel)
      throw std::runtime_error("Level of item to be updated cannot be bigger than max level");

    for (int level = dataPointLevel; level >= 0; level--) {
      std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                          CompareByFirstLess>
          topCandidates = searchBaseLayer(currObj, dataPoint, level);

      std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                          CompareByFirstLess>
          filteredTopCandidates;
      while (topCandidates.size() > 0) {
        if (topCandidates.top().second != dataPointInternalId)
          filteredTopCandidates.push(topCandidates.top());

        topCandidates.pop();
      }

      // Since element_levels_ is being used to get `dataPointLevel`, there could be cases where
      // `topCandidates` could just contains entry point itself. To prevent self loops, the `topCandidates` is
      // filtered and thus can be empty.
      if (filteredTopCandidates.size() > 0) {
        bool epDeleted = isMarkedDeleted(entryPointInternalId);
        if (epDeleted) {
          auto dist = flash_l2sqr_dist(dataPoint, getDataByInternalId(entryPointInternalId));
          filteredTopCandidates.emplace(dist, entryPointInternalId);
          if (filteredTopCandidates.size() > ef_construction_) filteredTopCandidates.pop();
        }

        currObj =
            mutuallyConnectNewElement(dataPoint, dataPointInternalId, filteredTopCandidates, level, true);
      }
    }
  }

  std::vector<tableint> getConnectionsWithLock(tableint internalId, int level) {
    std::unique_lock<std::mutex> lock(link_list_locks_[internalId]);
    unsigned int* data = get_linklist_at_level(internalId, level);
    int size = getListCount(data);
    std::vector<tableint> result(size);
    tableint* ll = (tableint*)(data + 1);
    memcpy(result.data(), ll, size * sizeof(tableint));
    return result;
  }

  tableint addPoint(const void* data_point, labeltype label, int level) {
    tableint cur_c = 0;
    {
      // Checking if the element with the same label already exists
      // if so, updating it *instead* of creating a new element.
      std::unique_lock<std::mutex> lock_table(label_lookup_lock);
      auto search = label_lookup_.find(label);
      if (search != label_lookup_.end()) {
        std::cout << "get repeated doc: " << label << std::endl;
        tableint existingInternalId = search->second;
        if (allow_replace_deleted_) {
          if (isMarkedDeleted(existingInternalId)) {
            throw std::runtime_error(
                "Can't use addPoint to update deleted elements if replacement of deleted elements is "
                "enabled.");
          }
        }
        lock_table.unlock();

        if (isMarkedDeleted(existingInternalId)) {
          unmarkDeletedInternal(existingInternalId);
        }
        updatePoint(data_point, existingInternalId, 1.0);

        return existingInternalId;
      }

      if (cur_element_count >= max_elements_) {
        throw std::runtime_error("The number of elements exceeds the specified limit");
      }

      cur_c = cur_element_count;
      cur_element_count++;
      label_lookup_[label] = cur_c;
    }

    std::unique_lock<std::mutex> lock_el(link_list_locks_[cur_c]);

    // todo
    int curlevel = getRandomLevel(mult_);
    // int curlevel  = 0;
    if (level > 0) curlevel = level;

    element_levels_[cur_c] = curlevel;

    std::unique_lock<std::mutex> templock(global);
    int maxlevelcopy = maxlevel_;
    if (curlevel <= maxlevelcopy) templock.unlock();
    tableint currObj = enterpoint_node_;
    tableint enterpoint_copy = enterpoint_node_;

    // return cur_c;

    memset(data_level0_memory_ + cur_c * size_data_per_element_ + offsetLevel0_, 0, size_data_per_element_);

    // Initialisation of the data and label
    memcpy(getExternalLabeLp(cur_c), &label, sizeof(labeltype));
    memcpy(getDataByInternalId(cur_c), (dist_t*)data_point + SUBVECTOR_NUM * CLUSTER_NUM, data_size_);

#if !defined(SAVE_MEMORY)
    set_dist_table(data_point, cur_c);
    return cur_c;
#endif

    if (curlevel) {
      static uint64_t total_alloc_size = 0;
      static uint64_t total_count = 0;
      total_count++;
      // +1 prevents code analyzers report out of bound when doing prefetching
      // which means not much need
      linkLists_[cur_c] = (char*)malloc(size_links_per_element_ * curlevel);
      total_alloc_size += size_links_per_element_ * curlevel;

      if (total_count % 1000 == 0) {
        std::cout << "allocated linklist memory: " << "curlevel: " << curlevel
                  << ", total size: " << (total_alloc_size / 1024 / 1024.0f) << "M"
                  << ", total_count: " << total_count << std::endl;
      }

      if (linkLists_[cur_c] == nullptr)
        throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
      memset(linkLists_[cur_c], 0, size_links_per_element_ * curlevel);
    }
    if ((signed)currObj != -1) {
      if (curlevel < maxlevelcopy) {
        dist_t curdist = flash_l2sqr_dist(data_point, getDataByInternalId(currObj));
        for (int level = maxlevelcopy; level > curlevel; level--) {
          bool changed = true;
          while (changed) {
            changed = false;
            unsigned int* data;
            std::unique_lock<std::mutex> lock(link_list_locks_[currObj]);
            data = get_linklist(currObj, level);
            int size = getListCount(data);

            tableint* datal = (tableint*)(data + 1);
#if defined(PQLINK_CALC)
            dist_t* dist_list = (dist_t*)malloc(maxM_ * sizeof(dist_t));
            std::unique_ptr<dist_t, decltype(&std::free)> p_dist_list(dist_list, &std::free);

            PqLinkL2Sqr(dist_list, data_point, getLinksData(currObj, level), size, level);
#endif
            for (int i = 0; i < size; i++) {
              tableint cand = datal[i];
              if (cand < 0 || cand > max_elements_) throw std::runtime_error("cand error");
#if defined(PQLINK_CALC)
              dist_t d = dist_list[i];
#else
              dist_t d = flash_l2sqr_dist(data_point, getDataByInternalId(cand));
#endif

              if (d < curdist) {
                curdist = d;
                currObj = cand;
                changed = true;
              }
            }
#if defined(PQLINK_CALC)
            // free(dist_list);
#endif
          }
        }
      }
      bool epDeleted = isMarkedDeleted(enterpoint_copy);
      for (int level = std::min(curlevel, maxlevelcopy); level >= 0; level--) {
        if (level > maxlevelcopy || level < 0)  // possible?
          throw std::runtime_error("Level error");

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                            CompareByFirstLess>
            top_candidates = searchBaseLayer(currObj, data_point, level);

        // std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
        // CompareByFirstLess> top_candidates;
        if (epDeleted) {
          auto dist = flash_l2sqr_dist(data_point, getDataByInternalId(enterpoint_copy));
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
  }

  std::priority_queue<std::pair<dist_t, labeltype>>
  searchKnn(const void* query_data, size_t k, BaseFilterFunctor* isIdAllowed = nullptr) const {
    std::priority_queue<std::pair<dist_t, labeltype>> result;
    if (cur_element_count == 0) return result;

    auto s_knn_upper_layer = std::chrono::steady_clock::now();
    tableint currObj = enterpoint_node_;
    dist_t curdist = flash_l2sqr_dist(query_data, getDataByInternalId(enterpoint_node_));

#if defined(TRACE_SEARCH)
    constexpr bool need_trace = true;
#else
    constexpr bool need_trace = false;
#endif

    if (need_trace) {
      std::cout << "before level0: " << std::endl;
    }

    for (int level = maxlevel_; level > 0; level--) {
      bool changed = true;
      while (changed) {
        changed = false;
        unsigned int* data;

        data = (unsigned int*)get_linklist(currObj, level);
        int size = getListCount(data);

        metric_hops++;
        metric_distance_computations += size;

        if (need_trace) {
          std::cout << "enter point: " << getExternalLabel(currObj) << ", dis: " << curdist
                    << ", size: " << size << ", level: " << level << ", cur_hops: " << metric_hops
                    << ", cur_comp: " << metric_distance_computations << std::endl;
        }

        tableint* datal = (tableint*)(data + 1);
        // dist_t* neighbors_data = (dist_t*)alloca(size * SUBVECTOR_NUM * sizeof(dist_t));
        // for (int k = 0; k < size; ++k) {
        //   tableint neighbor_id = datal[k];
        //   dist_t* neighbor_data = (dist_t*)getDataByInternalId(neighbor_id);
        //   memcpy(neighbors_data + k * SUBVECTOR_NUM, neighbor_data, SUBVECTOR_NUM * sizeof(dist_t));
        // }

        // dist_t* dist_list = (dist_t*)alloca(maxM0_ * sizeof(dist_t));
        // PqLinkL2Sqr(dist_list, query_data, neighbors_data, size, level, curdist);

        for (int i = 0; i < size; i++) {
          tableint cand = datal[i];
          if (cand < 0 || cand > max_elements_) throw std::runtime_error("cand error");
          dist_t d = flash_l2sqr_dist(query_data, getDataByInternalId(cand));

          if (need_trace) {
            std::cout << "(" << getExternalLabel(cand) << ", " << d << "), ";
          }

          if (d < curdist) {
            curdist = d;
            currObj = cand;
            changed = true;
          }
        }

        if (need_trace) {
          std::cout << std::endl;
          std::cout << std::endl;
        }
      }
    }
    auto e_knn_upper_layer = std::chrono::steady_clock::now();
    knn_upper_layer_cost += time_cost(s_knn_upper_layer, e_knn_upper_layer);

    auto s_knn_base_layer = std::chrono::steady_clock::now();
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                        CompareByFirstLess>
        top_candidates;
    bool bare_bone_search = !num_deleted_ && !isIdAllowed;
    if (bare_bone_search) {
      top_candidates = searchBaseLayerST<true, true>(currObj, query_data, k, isIdAllowed);
    } else {
      top_candidates = searchBaseLayerST<false, true>(currObj, query_data, k, isIdAllowed);
    }

    while (top_candidates.size() > k) {
      top_candidates.pop();
    }
    while (top_candidates.size() > 0) {
      std::pair<dist_t, tableint> rez = top_candidates.top();
      result.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
      top_candidates.pop();
    }
    auto e_knn_base_layer = std::chrono::steady_clock::now();
    knn_base_layer_cost += time_cost(s_knn_base_layer, e_knn_base_layer);

    return result;
  }

  std::vector<std::pair<dist_t, labeltype>> searchStopConditionClosest(
      const void* query_data,
      BaseSearchStopCondition<dist_t>& stop_condition,
      BaseFilterFunctor* isIdAllowed = nullptr) const {
    std::vector<std::pair<dist_t, labeltype>> result;
    if (cur_element_count == 0) return result;

    tableint currObj = enterpoint_node_;
    dist_t curdist = flash_l2sqr_dist(query_data, getDataByInternalId(enterpoint_node_));

    for (int level = maxlevel_; level > 0; level--) {
      bool changed = true;
      while (changed) {
        changed = false;
        unsigned int* data;

        data = (unsigned int*)get_linklist(currObj, level);
        int size = getListCount(data);
        metric_hops++;
        metric_distance_computations += size;

        tableint* datal = (tableint*)(data + 1);
#if defined(PQLINK_CALC)
        dist_t* dist_list = (dist_t*)malloc(maxM_ * sizeof(dist_t));
        std::unique_ptr<dist_t, decltype(&std::free)> p_dist_list(dist_list, &std::free);

        PqLinkL2Sqr(dist_list, query_data, getLinksData(currObj, level), size, level);
#endif
        for (int i = 0; i < size; i++) {
          tableint cand = datal[i];
          if (cand < 0 || cand > max_elements_) throw std::runtime_error("cand error");
#if defined(PQLINK_CALC)
          dist_t d = dist_list[i];
#else
          dist_t d = flash_l2sqr_dist(query_data, getDataByInternalId(cand));
#endif
          if (d < curdist) {
            curdist = d;
            currObj = cand;
            changed = true;
          }
        }
#if defined(PQLINK_CALC)
        // free(dist_list);
#endif
      }
    }

    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                        CompareByFirstLess>
        top_candidates;
    top_candidates = searchBaseLayerST<false>(currObj, query_data, 0, isIdAllowed, &stop_condition);

    size_t sz = top_candidates.size();
    result.resize(sz);
    while (!top_candidates.empty()) {
      result[--sz] = top_candidates.top();
      top_candidates.pop();
    }

    stop_condition.filter_results(result);

    return result;
  }

  void checkIntegrity() {
    int connections_checked = 0;
    std::vector<int> inbound_connections_num(cur_element_count, 0);
    for (int i = 0; i < cur_element_count; i++) {
      for (int l = 0; l <= element_levels_[i]; l++) {
        linklistsizeint* ll_cur = get_linklist_at_level(i, l);
        int size = getListCount(ll_cur);
        tableint* data = (tableint*)(ll_cur + 1);
        std::unordered_set<tableint> s;
        for (int j = 0; j < size; j++) {
          assert(data[j] < cur_element_count);
          assert(data[j] != i);
          inbound_connections_num[data[j]]++;
          s.insert(data[j]);
          connections_checked++;
        }
        assert(s.size() == size);
      }
    }
    if (cur_element_count > 1) {
      int min1 = inbound_connections_num[0], max1 = inbound_connections_num[0];
      for (int i = 0; i < cur_element_count; i++) {
        assert(inbound_connections_num[i] > 0);
        min1 = std::min(inbound_connections_num[i], min1);
        max1 = std::max(inbound_connections_num[i], max1);
      }
      std::cout << "Min inbound: " << min1 << ", Max inbound:" << max1 << "\n";
    }
    std::cout << "integrity ok, checked " << connections_checked << " connections\n";
  }

  /**
   * Calculate the squared Euclidean distance between a point and all its neighbors.
   * All calculations utilize [PSHUFB](https://www.felixcloutier.com/x86/pshufb),
   * where shuffling is constrained to 128-bit lanes with SSE, AVX/AVX2, or AVX512.

   * - The order of subvectors is rearranged for each data point.
   * - For AVX/AVX2 with 256-bit length, the first 4 subvectors are stored in the order [0, 2, 1, 3].
   * - Specifically, in one iteration, we load 256 bits from LinksData, which contains the encoded data for
   neighbor 0 and neighbor 1:
   *   - The first 128 bits contain subvector 0 data in the high (left) 4 bits of each byte and subvector 2
   data in the low (right) 4 bits.
   *   - The second 128 bits contain subvector 1 data in the high (left) 4 bits of each byte and subvector 3
   data in the low (right) 4 bits.
   * - The distance table is then shuffled:
   *   - The distance table for subvectors 0 and 1 is loaded into `qdists[0]`, while subvectors 2 and 3 are
   loaded into `qdists[1]`.
   *   - Shuffling is constrained within 128-bit lanes. The first 128 bits are shuffled with `qdists[0]`, and
   the second 128 bits with `qdists[1]`.
   * - A low_mask is used to extract the low 4 bits of each byte in the 128-bit lane, representing subvectors
   2 and 3.
   * - `qdists[1]` directly provides the result for these subvectors.
   * - A right shift by 4 bits extracts the low 4 bits of each byte, representing subvectors 0 and 1.
   * - Finally, the two 128-bit lanes are summed to produce the final result.

   * @param result Pointer to a distance vector. Save the distance betwenn the points with all its neighbors
   in align.
   * @param pVect1v Pointer to a distance table. The distance table contains CLUSTER_NUM distances for each
   subvector.
   * @param pVect2v Pointer to a distance table. The encoded data of all neighbors.
   * @param qty_ptr Pointer to the dimension of the vectors
   * @param level Level of the node since layer0 has 2M neighbors while others only have M neighbors
   */
  void PqLinkL2Sqr(const void* result,
                   const void* pVect1v,
                   const void* pVect2v,
                   size_t qty,
                   size_t level = 0,
                   dist_t dis = 0xff) const {
    dist_t* res = (dist_t*)result;
#if defined(RUN_WITH_SSE) && defined(INT8) && !defined(FORBID_RUN)
    const __m128i low_mask = _mm_set1_epi8(0x0F);
    const __m128i* qdists = reinterpret_cast<const __m128i*>(pVect1v);  // qdists[i] for table of dim i
    const __m128i* part = reinterpret_cast<const __m128i*>(pVect2v);    // part for iterating data

    size_t tmp = SUBVECTOR_NUM / BATCH;
    for (int i = 0; i < BLOCKS; ++i) {
      if (qty <= VECTORS_PER_BLOCK * i) break;
      __m128i comps = _mm_loadu_si128(part);
      __m128i masked = _mm_and_si128(comps, low_mask);
      __m128i twolane_sum =
          _mm_add_epi8(_mm_shuffle_epi8(qdists[1], masked),
                       _mm_shuffle_epi8(qdists[0], _mm_and_si128(_mm_srli_epi64(comps, 4), low_mask)));
      part++;

      for (size_t j = 2; j < tmp; j += 2) {
        comps = _mm_loadu_si128(part);
        masked = _mm_and_si128(comps, low_mask);
        twolane_sum = _mm_add_epi8(
            twolane_sum,
            _mm_add_epi8(_mm_shuffle_epi8(qdists[j | 1], masked),
                         _mm_shuffle_epi8(qdists[j], _mm_and_si128(_mm_srli_epi64(comps, 4), low_mask))));
        part++;
      }
      _mm_storeu_si128(reinterpret_cast<__m128i*>(res + i * VECTORS_PER_BLOCK), twolane_sum);
    }
#elif defined(RUN_WITH_AVX) && defined(INT8) && !defined(FORBID_RUN)
    const __m256i low_mask = _mm256_set1_epi8(0x0F);
    const __m256i* qdists =
        reinterpret_cast<const __m256i*>(pVect1v);  // qdists[i] for table of dim i
                                                    // the distance table are aligned alloc by 32-byte
    const __m256i* part = reinterpret_cast<const __m256i*>(pVect2v);  // part for iterating data

    size_t tmp = SUBVECTOR_NUM / BATCH;
    for (size_t i = 0; i < BLOCKS; ++i) {
      if (qty <= VECTORS_PER_BLOCK * i) break;
      __m256i comps;
      __m256i twolane_sum = _mm256_setzero_si256();
      int tim = 0;

      for (size_t j = 0; j < tmp; j += 2) {
        comps = _mm256_loadu_si256(part);
        twolane_sum = _mm256_add_epi8(
            twolane_sum,
            _mm256_add_epi8(
                _mm256_shuffle_epi8(qdists[j | 1], _mm256_and_si256(comps, low_mask)),
                _mm256_shuffle_epi8(qdists[j], _mm256_and_si256(_mm256_srli_epi64(comps, 4), low_mask))));
#if defined(ADSAMPLING)
        tim += 4;
        if (tim % ADSAMPLING_DELTA_D == 0) {
          dist_t dis_ratio = static_cast<dist_t>(std::min(255, (int)(dis * ratio[tim])));

          // std::cout << (int)dis << " " << tim << " " << ratio[tim] << " " << (int)dis_ratio << " " <<
          // std::endl;
          __m128i data =
              _mm_adds_epi8(_mm256_castsi256_si128(twolane_sum), _mm256_extracti128_si256(twolane_sum, 1));

          // if dis_ratio >= any of data, then continue to calculate
          __m128i cmp_result = _mm_subs_epu8(_mm_set1_epi8(dis_ratio), data);
          if (_mm_testz_si128(cmp_result, cmp_result)) {
            std::fill(res + i * VECTORS_PER_BLOCK, res + (i + 1) * VECTORS_PER_BLOCK,
                      std::numeric_limits<dist_t>::max());
            // memset(res + i * VECTORS_PER_BLOCK, 0xFF, sizeof(dist_t) * VECTORS_PER_BLOCK);
            part += (tmp - j) / 2;
            break;
          }
        }
#endif
        ++part;
      }

      _mm_storeu_si128(
          reinterpret_cast<__m128i*>(res + i * VECTORS_PER_BLOCK),
          _mm_adds_epi8(_mm256_castsi256_si128(twolane_sum), _mm256_extracti128_si256(twolane_sum, 1)));
    }
#elif defined(RUN_WITH_AVX512) && defined(INT8) && !defined(FORBID_RUN)
  const __m512i low_mask = _mm512_set1_epi8(0x0F);
  const __m512i* qdists = reinterpret_cast<const __m512i*>(pVect1v);
  const __m512i* part = reinterpret_cast<const __m512i*>(pVect2v);  // part for iterating data

  size_t tmp = SUBVECTOR_NUM / BATCH;
  for (size_t i = 0; i < BLOCKS; ++i) {
    if (qty <= VECTORS_PER_BLOCK * i) break;
    __m512i comps;
    __m512i twolane_sum = _mm512_setzero_si512();

    for (size_t j = 0; j < tmp; j += 2) {
      comps = _mm512_loadu_si512(part);
      twolane_sum = _mm512_add_epi8(
          twolane_sum,
          _mm512_add_epi8(
              _mm512_shuffle_epi8(qdists[j | 1], _mm512_and_si512(comps, low_mask)),
              _mm512_shuffle_epi8(qdists[j], _mm512_and_si512(_mm512_srli_epi64(comps, 4), low_mask))));
      ++part;
    }

    _mm_storeu_si128(reinterpret_cast<__m128i*>(res + i * VECTORS_PER_BLOCK),
                     _mm_adds_epi8(_mm_adds_epi8(_mm512_extracti32x4_epi32(twolane_sum, 0),
                                                 _mm512_extracti32x4_epi32(twolane_sum, 1)),
                                   _mm_adds_epi8(_mm512_extracti32x4_epi32(twolane_sum, 2),
                                                 _mm512_extracti32x4_epi32(twolane_sum, 3))));
  }
#else
  {
    encode_t* pVect2 = (encode_t*)pVect2v;
    memset(res, 0, qty * sizeof(dist_t));

    // for (int i = 0; i < qty; ++i) {
    //   dist_t* pVect1 = (dist_t*)pVect1v;

    //   for (int j = 0; j < SUBVECTOR_NUM; ++j) {
    //     res[i] += *(pVect1 + (*pVect2));

    //     pVect1 += CLUSTER_NUM;
    //     pVect2++;
    //   }
    // }

    for (size_t i = 0; i < qty; ++i) {
      res[i] = flash_l2sqr_dist(pVect1v, pVect2 + i * SUBVECTOR_NUM);
    }
  }

#endif
  }

  data_t ADSamplingFlashL2Sqr(const void* pVect1v,
                              const void* pVect2v,
                              const void* qty_ptr,
                              data_t dis = 0xff) {
    data_t* pVect1 = (data_t*)pVect1v;    // distance table
    uint8_t* pVect2 = (uint8_t*)pVect2v;  // encoded data
    size_t qty = *((size_t*)qty_ptr);

    data_t res = 0;
    int tim = 0;
    int dim = qty * 2;
    if (CLUSTER_NUM == 16) {
      int tmp = qty / BATCH;
      for (int i = 0; i < tmp; ++i) {
        for (int j = 0; j < BATCH; ++j) {
          res += *(pVect1 + ls(*(pVect2 + j)));
          pVect1 += CLUSTER_NUM;
        }
        for (int j = 0; j < BATCH; ++j) {
          res += *(pVect1 + rs(*(pVect2 + j)));
          pVect1 += CLUSTER_NUM;
        }
        pVect2 += BATCH;

        tim += 4;
        if (tim % ADSAMPLING_DELTA_D == 0) {
          if (res >= dis * adsampling::ratio(dim, tim)) return res * dim / tim;
        }
      }
    } else {
      int tmp = qty;
      for (int i = 0; i < tmp; ++i) {
        res += *(pVect1 + (*pVect2));
        pVect1 += CLUSTER_NUM;
        pVect2++;
      }
    }
    return (res);
  }

  int horizontal_add_epi32(__m128i v) const {
    __m128i sum = _mm_hadd_epi32(v, v);  // [x0+x1, x2+x3, x0+x1, x2+x3]
    sum = _mm_hadd_epi32(sum, sum);      // [x0+x1+x2+x3, ...]
    return _mm_cvtsi128_si32(sum);
  }

  dist_t flash_l2sqr_dist(const void* p_vec1, const void* p_vec2) const {
    dist_t ret = 0;

    dist_t* ptr_vec1 = (dist_t*)p_vec1;
    encode_t* ptr_vec2 = (encode_t*)p_vec2;

    __m128i sum = _mm_setzero_si128();
    __m128i v1;
    __m128i v2;
    __m128i tmp;
    for (size_t i = 0; i < subvec_num_v4_; i += 8) {
      v1 = _mm_set_epi32(ptr_vec1[ptr_vec2[0]], ptr_vec1[1 * CLUSTER_NUM + ptr_vec2[1]],
                         ptr_vec1[2 * CLUSTER_NUM + ptr_vec2[2]], ptr_vec1[3 * CLUSTER_NUM + ptr_vec2[3]]);
      v2 = _mm_set_epi32(ptr_vec1[4 * CLUSTER_NUM + ptr_vec2[4]], ptr_vec1[5 * CLUSTER_NUM + ptr_vec2[5]],
                         ptr_vec1[6 * CLUSTER_NUM + ptr_vec2[6]], ptr_vec1[7 * CLUSTER_NUM + ptr_vec2[7]]);

      tmp = _mm_add_epi32(v1, v2);
      sum = _mm_add_epi32(sum, tmp);
      ptr_vec1 += 8 * CLUSTER_NUM;
      ptr_vec2 += 8;
    }
    ret = horizontal_add_epi32(sum);

    return ret;
  }

  dist_t PqSdcL2Sqr(const void* Vec1, const void* Vec2) const {
    encode_t* pVect1 = (encode_t*)Vec1;  // distance encode
    encode_t* pVect2 = (encode_t*)Vec2;  // distance encode

    dist_t res = 0;
    for (int i = 0; i < subvec_num_v4_; ++i) {
      res += *get_dist_v4(i, *pVect1, *pVect2);
      pVect1++;
      pVect2++;
    }

    return res;
  }
};
}  // namespace hnswlib
