#pragma once

#include <assert.h>
#include <stdlib.h>

#include <atomic>
#include <list>
#include <memory>
#include <random>
#include <unordered_map>
#include <unordered_set>

#include "hnswlib.h"
#include "visited_list_pool.h"

namespace hnswlib {
typedef unsigned int tableint;
typedef unsigned int linklistsizeint;

template <typename dist_t>
class HierarchicalNSW_V2 : public AlgorithmInterface<dist_t> {
 public:
  static const tableint MAX_LABEL_OPERATION_LOCKS = 65536;
  static const unsigned char DELETE_MARK = 0x01;

  size_t max_elements_{0};
  mutable std::atomic<size_t> cur_element_count_{0};  // current number of elements
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
  size_t offsetData_{0}, offsetLevel0_{0}, label_offset_{0}, linkdata_offset_{0};

  char* data_level0_memory_{nullptr};
  char** linkLists_{nullptr};
  std::vector<int> element_levels_;  // keeps level of each element

  size_t data_size_{0};
  size_t subvec_num_{0};

  DISTFUNC_EXTRA<dist_t> fstdistfunc_;
  void* dist_func_param_{nullptr};

  mutable std::mutex label_lookup_lock;  // lock for label_lookup_
  std::unordered_map<labeltype, tableint> label_lookup_;

  std::default_random_engine level_generator_;
  std::default_random_engine update_probability_generator_;

  mutable std::atomic<long> metric_distance_computations{0};
  mutable std::atomic<long> metric_hops{0};

  bool allow_replace_deleted_ = false;  // flag to replace deleted elements
                                        // (marked as deleted) during insertions

  std::mutex deleted_elements_lock;               // lock for deleted_elements
  std::unordered_set<tableint> deleted_elements;  // contains internal ids of deleted elements

 public:
  struct CandInfo {
    tableint id{0};
    dist_t dist{0};
    float avg_subvec_dis{0};
    // std::vector<float> subvec_dis;
    float subvec_dis[8];

    bool operator<(CandInfo const& rhs) const noexcept {
      if (dist != rhs.dist) {
        return dist < rhs.dist;
      }
      return false;
    }
  };

 public:
  HierarchicalNSW_V2(SpaceInterface<dist_t>* s) {}

  HierarchicalNSW_V2(SpaceInterface<dist_t>* s,
                     const std::string& location,
                     bool nmslib = false,
                     size_t max_elements = 0,
                     bool allow_replace_deleted = false)
      : allow_replace_deleted_(allow_replace_deleted) {
    loadIndex(location, s, max_elements);
  }

  HierarchicalNSW_V2(SpaceInterface<dist_t>* s,
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
    data_size_ = s->get_data_size();
    fstdistfunc_ = s->get_dist_func_extra();
    subvec_num_ = *(size_t*)s->get_dist_func_param_extra();

    dist_func_param_ = s->get_dist_func_param();
    if (M <= 10000) {
      M_ = M;
    } else {
      HNSWERR << "warning: M parameter exceeds 10000 which may lead to adverse "
                 "effects."
              << std::endl;
      HNSWERR << "         Cap to 10000 will be applied for the rest of the "
                 "processing."
              << std::endl;
      M_ = 10000;
    }
    maxM_ = M_;
    maxM0_ = M_ * 2;
    ef_construction_ = std::max(ef_construction, M_);
    ef_ = 10;

    level_generator_.seed(random_seed);
    update_probability_generator_.seed(random_seed + 1);

    size_links_level0_ = sizeof(linklistsizeint) + maxM0_ * sizeof(tableint) + maxM0_ * sizeof(int16_t);
    size_data_per_element_ = size_links_level0_ + data_size_ + sizeof(labeltype);

    linkdata_offset_ = sizeof(linklistsizeint) + maxM0_ * sizeof(tableint);
    offsetData_ = size_links_level0_;
    label_offset_ = size_links_level0_ + data_size_;
    offsetLevel0_ = 0;

    data_level0_memory_ = (char*)malloc(max_elements_ * size_data_per_element_);
    if (data_level0_memory_ == nullptr) throw std::runtime_error("Not enough memory");

    cur_element_count_ = 0;

    visited_list_pool_ = std::unique_ptr<VisitedListPool>(new VisitedListPool(1, max_elements));

    // initializations for special treatment of the first node
    enterpoint_node_ = -1;
    maxlevel_ = -1;

    linkLists_ = (char**)malloc(sizeof(void*) * max_elements_);
    if (linkLists_ == nullptr)
      throw std::runtime_error(
          "Not enough memory: HierarchicalNSW_V2 failed "
          "to allocate linklists");
    size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);
    mult_ = 1 / log(1.0 * M_);
    revSize_ = 1.0 / mult_;
  }

  ~HierarchicalNSW_V2() {
    clear();
  }

  void clear() {
    free(data_level0_memory_);
    data_level0_memory_ = nullptr;
    for (tableint i = 0; i < cur_element_count_; i++) {
      if (element_levels_[i] > 0) free(linkLists_[i]);
    }
    free(linkLists_);
    linkLists_ = nullptr;
    cur_element_count_ = 0;
    visited_list_pool_.reset(nullptr);
  }

  struct CompareByFirst {
    constexpr bool operator()(std::pair<dist_t, CandInfo> const& a,
                              std::pair<dist_t, CandInfo> const& b) const noexcept {
      if (a.first != b.first) {
        return a.first < b.first;
      }
      return a.second.id < b.second.id;
    }
  };

  void setEf(size_t ef) {
    ef_ = ef;
  }

  inline void set_search_bits(int16_t* search_bits, int i) const {
    *search_bits |= (1 << i);
  }

  inline bool can_search(int16_t* search_bits_a, const int16_t* search_bits_b) const {
    return ((*search_bits_a) & (*search_bits_b)) != 0;
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

  inline int16_t* getLinkDataByInternalId(tableint internal_id) const {
    return (int16_t*)(data_level0_memory_ + internal_id * size_data_per_element_ + linkdata_offset_);
  }

  inline char* getDataByInternalId(tableint internal_id) const {
    return (data_level0_memory_ + internal_id * size_data_per_element_ + offsetData_);
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
    return cur_element_count_;
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

  std::priority_queue<std::pair<dist_t, CandInfo>, std::vector<std::pair<dist_t, CandInfo>>, CompareByFirst>
  searchBaseLayer(tableint ep_id, const void* data_point, int layer) {
    VisitedList* vl = visited_list_pool_->getFreeVisitedList();
    vl_type* visited_array = vl->mass;
    vl_type visited_array_tag = vl->curV;

    std::priority_queue<std::pair<dist_t, CandInfo>, std::vector<std::pair<dist_t, CandInfo>>, CompareByFirst>
        top_candidates;
    std::priority_queue<std::pair<dist_t, CandInfo>, std::vector<std::pair<dist_t, CandInfo>>, CompareByFirst>
        candidateSet;
    dist_t lowerBound;
    if (!isMarkedDeleted(ep_id)) {
      float* subvec_dis = (float*)alloca(subvec_num_ * sizeof(float));
      dist_t dist =
          fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_, &subvec_num_, subvec_dis);
      CandInfo cand_info{
          .id = ep_id,
          .dist = dist,
          .avg_subvec_dis = dist / subvec_num_,
      };
      memcpy(cand_info.subvec_dis, subvec_dis, subvec_num_ * sizeof(float));

      top_candidates.emplace(std::make_pair(dist, cand_info));
      lowerBound = dist;
      candidateSet.emplace(std::make_pair(-dist, cand_info));
    } else {
      lowerBound = std::numeric_limits<dist_t>::max();

      CandInfo cand_info{
          .id = ep_id, .dist = lowerBound, .avg_subvec_dis = 0.0f,
          // .subvec_dis = std::vector<float>(subvec_num_, 0.0f)
      };
      memset(cand_info.subvec_dis, 0, subvec_num_ * sizeof(float));
      candidateSet.emplace(std::make_pair(0, cand_info));
    }
    visited_array[ep_id] = visited_array_tag;

    while (!candidateSet.empty()) {
      std::pair<dist_t, CandInfo> curr_el_pair = candidateSet.top();
      if ((-curr_el_pair.first) > lowerBound && top_candidates.size() == ef_construction_) {
        break;
      }
      candidateSet.pop();

      tableint curNodeNum = curr_el_pair.second.id;

      std::unique_lock<std::mutex> lock(link_list_locks_[curNodeNum]);

      int* data;  // = (int *)(linkList0_ + curNodeNum *
                  // size_links_per_element0_);
      if (layer == 0) {
        data = (int*)get_linklist0(curNodeNum);
      } else {
        data = (int*)get_linklist(curNodeNum, layer);
        //                    data = (int *) (linkLists_[curNodeNum] +
        //                    (layer - 1) * size_links_per_element_);
      }
      size_t size = getListCount((linklistsizeint*)data);
      tableint* datal = (tableint*)(data + 1);
#ifdef USE_SSE
      _mm_prefetch((char*)(visited_array + *(data + 1)), _MM_HINT_T0);
      _mm_prefetch((char*)(visited_array + *(data + 1) + 64), _MM_HINT_T0);
      _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
      _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);
#endif

      for (size_t j = 0; j < size; j++) {
        tableint candidate_id = *(datal + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
        _mm_prefetch((char*)(visited_array + *(datal + j + 1)), _MM_HINT_T0);
        _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
#endif
        if (visited_array[candidate_id] == visited_array_tag) continue;
        visited_array[candidate_id] = visited_array_tag;
        char* currObj1 = (getDataByInternalId(candidate_id));

        float* subvec_dis = (float*)alloca(subvec_num_ * sizeof(float));
        dist_t dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_, &subvec_num_, subvec_dis);
        CandInfo cand_info{
            .id = candidate_id, .dist = dist1, .avg_subvec_dis = dist1 / subvec_num_,
            // .subvec_dis = std::vector<float>(subvec_dis, subvec_dis +
            // subvec_num_)
        };
        memcpy(cand_info.subvec_dis, subvec_dis, subvec_num_ * sizeof(float));

        if (top_candidates.size() < ef_construction_ || dist1 < lowerBound) {
          candidateSet.emplace(std::make_pair(-dist1, cand_info));
#ifdef USE_SSE
          _mm_prefetch(getDataByInternalId(candidateSet.top().second.id), _MM_HINT_T0);
#endif

          if (!isMarkedDeleted(candidate_id)) {
            top_candidates.push(std::make_pair(dist1, cand_info));
          }

          if (top_candidates.size() > ef_construction_) {
            top_candidates.pop();
          }

          if (!top_candidates.empty()) {
            lowerBound = top_candidates.top().first;
          }
        }
      }
    }
    visited_list_pool_->releaseVisitedList(vl);

    return top_candidates;
  }

  // bare_bone_search means there is no check for deletions and stop condition
  // is ignored in return of extra performance
  template <bool bare_bone_search = true, bool collect_metrics = false>
  std::priority_queue<std::pair<dist_t, CandInfo>, std::vector<std::pair<dist_t, CandInfo>>, CompareByFirst>
  searchBaseLayerST(tableint ep_id,
                    const void* data_point,
                    size_t ef,
                    BaseFilterFunctor* isIdAllowed = nullptr,
                    BaseSearchStopCondition<dist_t>* stop_condition = nullptr) const {
    VisitedList* vl = visited_list_pool_->getFreeVisitedList();
    vl_type* visited_array = vl->mass;
    vl_type visited_array_tag = vl->curV;

    std::priority_queue<std::pair<dist_t, CandInfo>, std::vector<std::pair<dist_t, CandInfo>>, CompareByFirst>
        top_candidates;
    std::priority_queue<std::pair<dist_t, CandInfo>, std::vector<std::pair<dist_t, CandInfo>>, CompareByFirst>
        candidate_set;
#if defined(TRACE_SEARCH)
    constexpr bool need_trace = true;
#else
    constexpr bool need_trace = false;
#endif
    if (need_trace) {
      std::cout << "begin level0 search: " << std::endl;
    }

    dist_t lowerBound;
    if (bare_bone_search ||
        (!isMarkedDeleted(ep_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(ep_id))))) {
      char* ep_data = getDataByInternalId(ep_id);

      float* subvec_dis = (float*)alloca(subvec_num_ * sizeof(float));
      dist_t dist = fstdistfunc_(data_point, ep_data, dist_func_param_, &subvec_num_, subvec_dis);
      lowerBound = dist;

      CandInfo cand_info{
          .id = ep_id, .dist = dist, 
          // .avg_subvec_dis = dist,
          .avg_subvec_dis = dist / subvec_num_,
          // .subvec_dis = std::vector<float>(subvec_dis, subvec_dis +
          // subvec_num_)
      };
      memcpy(cand_info.subvec_dis, subvec_dis, subvec_num_ * sizeof(float));

      top_candidates.emplace(std::make_pair(dist, cand_info));
      if (!bare_bone_search && stop_condition) {
        stop_condition->add_point_to_result(getExternalLabel(ep_id), ep_data, dist);
      }
      candidate_set.emplace(std::make_pair(-dist, cand_info));
    } else {
      lowerBound = std::numeric_limits<dist_t>::max();
      CandInfo cand_info{
          .id = ep_id,
          .dist = lowerBound,
          .avg_subvec_dis = lowerBound,
      };

      candidate_set.emplace(std::make_pair(-lowerBound, cand_info));
    }

    visited_array[ep_id] = visited_array_tag;

    while (!candidate_set.empty()) {
      std::pair<dist_t, CandInfo> current_node_pair = candidate_set.top();
      dist_t candidate_dist = -current_node_pair.first;

      bool flag_stop_search;
      if (bare_bone_search) {
        flag_stop_search = candidate_dist > lowerBound;
      } else {
        if (stop_condition) {
          flag_stop_search = stop_condition->should_stop_search(candidate_dist, lowerBound);
        } else {
          flag_stop_search = candidate_dist > lowerBound && top_candidates.size() == ef;
        }
      }
      if (flag_stop_search) {
        break;
      }
      candidate_set.pop();

      tableint current_node_id = current_node_pair.second.id;
      int* data = (int*)get_linklist0(current_node_id);
      size_t size = getListCount((linklistsizeint*)data);
      //                bool cur_node_deleted =
      //                isMarkedDeleted(current_node_id);
      if (collect_metrics) {
        metric_hops++;
        // metric_distance_computations+=size;
      }

#ifdef USE_SSE
      _mm_prefetch((char*)(visited_array + *(data + 1)), _MM_HINT_T0);
      _mm_prefetch((char*)(visited_array + *(data + 1) + 64), _MM_HINT_T0);
      _mm_prefetch(data_level0_memory_ + (*(data + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
      _mm_prefetch((char*)(data + 2), _MM_HINT_T0);
#endif

      std::vector<std::tuple<int, float, uint16_t, float>> debug_neighbor_list;

      int16_t parent_search_bits = 0;

      size_t actual_computation = 0;
      for (size_t j = 0; j < subvec_num_; j++) {
        if (current_node_pair.second.subvec_dis[j] <= current_node_pair.second.avg_subvec_dis) {
          set_search_bits(&parent_search_bits, j);
        }
      }

      if (need_trace) {
        std::cout << "enter_point: " << getExternalLabel(current_node_id) << ", dis: " << candidate_dist
                  << ", search_bits:" << parent_search_bits << ", cand size: " << size
                  << ", cur_hops:" << metric_hops << ", cur_comp: " << metric_distance_computations
                  << ", top_cand_size: " << top_candidates.size() << ", lower_bound: " << lowerBound
                  << std::endl;
      }

      // auto* neighbors_link_data =
      // getLinkDataByInternalId(current_node_id);
      auto* neighbors_link_data = (int16_t*)((char*)data + linkdata_offset_);
      for (size_t j = 0; j < size; j++) {
        tableint candidate_id = *(data + j);
        auto* cur_link_data = neighbors_link_data + j;

        if (!can_search(&parent_search_bits, cur_link_data)) {
          // visited_array[candidate_id] = visited_array_tag;
          continue;
        }
        if ((visited_array[candidate_id] == visited_array_tag)) {
          continue;
        }

        visited_array[candidate_id] = visited_array_tag;

#ifdef USE_SSE
        _mm_prefetch((char*)(visited_array + *(data + j)), _MM_HINT_T0);
        _mm_prefetch(data_level0_memory_ + (*(data + j)) * size_data_per_element_ + offsetData_,
                     _MM_HINT_T0);  ////////////
#endif

        actual_computation += 1;

        char* currObj1 = (getDataByInternalId(candidate_id));

        float* subvec_dis_query = (float*)alloca(subvec_num_ * sizeof(float));
        dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_, &subvec_num_, subvec_dis_query);
        CandInfo cand_info{
            .id = candidate_id,
            .dist = dist,
            .avg_subvec_dis = dist / subvec_num_,
        };
        memcpy(cand_info.subvec_dis, subvec_dis_query, subvec_num_ * sizeof(float));

        if (need_trace) {
          dist_t enter_dist = fstdistfunc_(getDataByInternalId(current_node_id), currObj1, dist_func_param_,
                                           &subvec_num_, nullptr);
          debug_neighbor_list.push_back(
              std::make_tuple(getExternalLabel(candidate_id), dist, *cur_link_data, enter_dist));
        }

        bool flag_consider_candidate = false;
        if (!bare_bone_search && stop_condition) {
          flag_consider_candidate = stop_condition->should_consider_candidate(dist, lowerBound);
        } else {
          flag_consider_candidate = top_candidates.size() < ef || lowerBound > dist;
        }

        if (flag_consider_candidate) {
          candidate_set.emplace(std::make_pair(-dist, cand_info));
#ifdef USE_SSE
          _mm_prefetch(data_level0_memory_ + candidate_set.top().second.id * size_data_per_element_ +
                           offsetLevel0_,  ///////////
                       _MM_HINT_T0);       ////////////////////////
#endif

          if (bare_bone_search || (!isMarkedDeleted(candidate_id) &&
                                   ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(candidate_id))))) {
            top_candidates.emplace(std::make_pair(dist, cand_info));
            if (!bare_bone_search && stop_condition) {
              stop_condition->add_point_to_result(getExternalLabel(candidate_id), currObj1, dist);
            }
          }

          bool flag_remove_extra = false;
          if (!bare_bone_search && stop_condition) {
            flag_remove_extra = stop_condition->should_remove_extra();
          } else {
            flag_remove_extra = top_candidates.size() > ef;
          }
          while (flag_remove_extra) {
            tableint id = top_candidates.top().second.id;
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

      metric_distance_computations += actual_computation;

      if (need_trace) {
        std::sort(
            debug_neighbor_list.begin(), debug_neighbor_list.end(),
            [](const std::tuple<int, float, int16_t, float>& a,
               const std::tuple<int, float, int16_t, float>& b) { return std::get<3>(a) < std::get<3>(b); });

        for (int i = 0; i < debug_neighbor_list.size(); ++i) {
          std::cout << "(" << std::get<0>(debug_neighbor_list[i]) << ", "
                    << std::get<1>(debug_neighbor_list[i]) << ", " << std::get<2>(debug_neighbor_list[i])
                    << ", " << std::get<3>(debug_neighbor_list[i]) << ") ";
        }

        std::cout << std::endl;
        std::cout << std::endl;
      }
    }

    visited_list_pool_->releaseVisitedList(vl);
    return top_candidates;
  }

  void getNeighborsByHeuristic2(std::priority_queue<std::pair<dist_t, CandInfo>,
                                                    std::vector<std::pair<dist_t, CandInfo>>,
                                                    CompareByFirst>& top_candidates,
                                const size_t M) {
    if (top_candidates.size() < M) {
      return;
    }

    std::priority_queue<std::pair<dist_t, CandInfo>> queue_closest;
    std::vector<std::pair<dist_t, CandInfo>> return_list;
    while (top_candidates.size() > 0) {
      queue_closest.emplace(std::make_pair(-top_candidates.top().first, top_candidates.top().second));
      top_candidates.pop();
    }

    while (queue_closest.size()) {
      if (return_list.size() >= M) break;
      std::pair<dist_t, CandInfo> curent_pair = queue_closest.top();
      dist_t dist_to_query = -curent_pair.first;
      queue_closest.pop();
      bool good = true;

      for (std::pair<dist_t, CandInfo> second_pair : return_list) {
        // float* subvec_dis = (float*)alloca(subvec_num_ * sizeof(float));
        dist_t curdist =
            fstdistfunc_(getDataByInternalId(second_pair.second.id),
                         getDataByInternalId(curent_pair.second.id), dist_func_param_, &subvec_num_, nullptr);
        if (curdist < dist_to_query) {
          good = false;
          /*
          tableint* data = get_linklist0(second_pair.second.id);
          size_t size = *(linklistsizeint*)data;
          tableint* neighbor_ids = (tableint *)(data + 1);
          bool is_find = false;
          for (size_t i = 0; i < size; ++i) {
              if (neighbor_ids[i] == curent_pair.second.id) {
                  is_find = true;
                  break;
              }
          }
          if (!is_find) {
              good = true;
          }
          */
          break;
        }
      }
      if (good) {
        return_list.push_back(curent_pair);
      }
    }

    for (std::pair<dist_t, CandInfo> curent_pair : return_list) {
      top_candidates.emplace(std::make_pair(-curent_pair.first, curent_pair.second));
    }
  }

  linklistsizeint* get_linklist0(tableint internal_id) const {
    return (linklistsizeint*)(data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
  }

  linklistsizeint* get_linklist0(tableint internal_id, char* data_level0_memory_) const {
    return (linklistsizeint*)(data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
  }

  linklistsizeint* get_linklist(tableint internal_id, int level) const {
    return (linklistsizeint*)(linkLists_[internal_id] + (level - 1) * size_links_per_element_);
  }

  linklistsizeint* get_linklist_at_level(tableint internal_id, int level) const {
    return level == 0 ? get_linklist0(internal_id) : get_linklist(internal_id, level);
  }

  tableint mutuallyConnectNewElement(const void* data_point,
                                     tableint cur_c,
                                     std::priority_queue<std::pair<dist_t, CandInfo>,
                                                         std::vector<std::pair<dist_t, CandInfo>>,
                                                         CompareByFirst>& top_candidates,
                                     int level,
                                     bool isUpdate) {
    size_t Mcurmax = level ? maxM_ : maxM0_;
    getNeighborsByHeuristic2(top_candidates, M_);
    if (top_candidates.size() > M_)
      throw std::runtime_error(
          "Should be not be more than M_ candidates "
          "returned by the heuristic");

    std::vector<CandInfo> selectedNeighbors;
    selectedNeighbors.reserve(Mcurmax);
    while (top_candidates.size() > 0) {
      selectedNeighbors.push_back(top_candidates.top().second);
      top_candidates.pop();
    }

    CandInfo next_closest_entry_point = selectedNeighbors.back();
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
        if (level > element_levels_[selectedNeighbors[idx].id])
          throw std::runtime_error("Trying to make a link on a non-existent level");

        data[idx] = selectedNeighbors[idx].id;
      }

      // set linkdata for level 0
      if (level == 0) {
        auto* link_data = getLinkDataByInternalId(cur_c);
        for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
          const auto& cand_info = selectedNeighbors[idx];
          auto* cur_link_data = link_data + idx;

          for (int i = 0; i < subvec_num_; i++) {
            if (cand_info.subvec_dis[i] <= cand_info.avg_subvec_dis) {
              set_search_bits((int16_t*)cur_link_data, i);
            }
          }
        }
      }
    }

    for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
      auto cur_select_neighbor = selectedNeighbors[idx];
      std::unique_lock<std::mutex> lock(link_list_locks_[cur_select_neighbor.id]);

      linklistsizeint* ll_other;
      if (level == 0)
        ll_other = get_linklist0(cur_select_neighbor.id);
      else
        ll_other = get_linklist(cur_select_neighbor.id, level);

      size_t sz_link_list_other = getListCount(ll_other);

      if (sz_link_list_other > Mcurmax) throw std::runtime_error("Bad value of sz_link_list_other");
      if (selectedNeighbors[idx].id == cur_c)
        throw std::runtime_error("Trying to connect an element to itself");
      if (level > element_levels_[selectedNeighbors[idx].id])
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

      // If cur_c is already present in the neighboring connections of
      // `selectedNeighbors[idx]` then no need to modify any connections or
      // run the heuristics.
      if (!is_cur_c_present) {
        if (sz_link_list_other < Mcurmax) {
          data[sz_link_list_other] = cur_c;
          setListCount(ll_other, sz_link_list_other + 1);

          // set linkdata for level 0
          if (level == 0) {
            auto* link_data = getLinkDataByInternalId(cur_select_neighbor.id) + sz_link_list_other;
            *link_data = 0;  // reset link data for the new neighbor

            for (int i = 0; i < subvec_num_; i++) {
              if (cur_select_neighbor.subvec_dis[i] <= cur_select_neighbor.avg_subvec_dis) {
                set_search_bits((int16_t*)(link_data), i);
              }
            }
          }
        } else {
          // finding the "weakest" element to replace it with the new one
          float* subvec_dis = (float*)alloca(subvec_num_ * sizeof(float));
          dist_t d_max = fstdistfunc_(getDataByInternalId(cur_c), getDataByInternalId(cur_select_neighbor.id),
                                      dist_func_param_, &subvec_num_, subvec_dis);

          // todo: 这里的距离已经算过了，可以不用算
          // Heuristic:
          std::priority_queue<std::pair<dist_t, CandInfo>, std::vector<std::pair<dist_t, CandInfo>>,
                              CompareByFirst>
              candidates;
          CandInfo cand_info{
              .id = cur_c, .dist = d_max, .avg_subvec_dis = d_max / subvec_num_,
              // .subvec_dis = std::vector<float>(subvec_dis, subvec_dis +
              // subvec_num_)
          };
          memcpy(cand_info.subvec_dis, subvec_dis, subvec_num_ * sizeof(float));
          candidates.emplace(std::make_pair(d_max, cand_info));

          for (size_t j = 0; j < sz_link_list_other; j++) {
            float* subvec_dis = (float*)alloca(subvec_num_ * sizeof(float));
            auto dis = fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(cur_select_neighbor.id),
                                    dist_func_param_, &subvec_num_, subvec_dis);

            CandInfo data_j{
                .id = data[j], .dist = dis, .avg_subvec_dis = dis / subvec_num_,
                // .subvec_dis = std::vector<float>(subvec_dis,
                // subvec_dis + subvec_num_)
            };
            memcpy(data_j.subvec_dis, subvec_dis, subvec_num_ * sizeof(float));
            candidates.emplace(std::make_pair(dis, data_j));
          }

          getNeighborsByHeuristic2(candidates, Mcurmax);
          // std::cout << "invert Heuristic2 connect : id: " <<
          // cur_select_neighbor.id << std::endl;

          int indx = 0;
          while (candidates.size() > 0) {
            const auto& candidates_top = candidates.top().second;
            data[indx] = candidates_top.id;
            if (level == 0) {
              auto* cur_link_data = getLinkDataByInternalId(cur_select_neighbor.id) + indx;
              *cur_link_data = 0;
              for (int i = 0; i < subvec_num_; i++) {
                if (candidates_top.subvec_dis[i] <= candidates_top.avg_subvec_dis) {
                  set_search_bits((int16_t*)cur_link_data, i);
                }
              }
            }

            candidates.pop();
            indx++;
          }

          setListCount(ll_other, indx);
        }
      }
    }

    return next_closest_entry_point.id;
  }

  void resizeIndex(size_t new_max_elements) {
    if (new_max_elements < cur_element_count_)
      throw std::runtime_error(
          "Cannot resize, max element is less than the current number of "
          "elements");

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
    size += sizeof(offsetLevel0_);
    size += sizeof(max_elements_);
    size += sizeof(cur_element_count_);
    size += sizeof(size_data_per_element_);
    size += sizeof(label_offset_);
    size += sizeof(offsetData_);
    size += sizeof(maxlevel_);
    size += sizeof(enterpoint_node_);
    size += sizeof(maxM_);

    size += sizeof(maxM0_);
    size += sizeof(M_);
    size += sizeof(mult_);
    size += sizeof(ef_construction_);

    size += cur_element_count_ * size_data_per_element_;

    for (size_t i = 0; i < cur_element_count_; i++) {
      unsigned int linkListSize = element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
      size += sizeof(linkListSize);
      size += linkListSize;
    }
    return size;
  }

  void saveIndex(const std::string& location) {
    std::ofstream output(location, std::ios::binary);
    std::streampos position;

    writeBinaryPOD(output, offsetLevel0_);
    writeBinaryPOD(output, max_elements_);
    writeBinaryPOD(output, cur_element_count_);
    writeBinaryPOD(output, size_data_per_element_);
    writeBinaryPOD(output, linkdata_offset_);
    writeBinaryPOD(output, label_offset_);
    writeBinaryPOD(output, offsetData_);
    writeBinaryPOD(output, maxlevel_);
    writeBinaryPOD(output, enterpoint_node_);
    writeBinaryPOD(output, maxM_);

    writeBinaryPOD(output, maxM0_);
    writeBinaryPOD(output, M_);
    writeBinaryPOD(output, mult_);
    writeBinaryPOD(output, ef_construction_);

    output.write(data_level0_memory_, cur_element_count_ * size_data_per_element_);

    for (size_t i = 0; i < cur_element_count_; i++) {
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

    readBinaryPOD(input, offsetLevel0_);
    readBinaryPOD(input, max_elements_);
    readBinaryPOD(input, cur_element_count_);

    size_t max_elements = max_elements_i;
    if (max_elements < cur_element_count_) max_elements = max_elements_;
    max_elements_ = max_elements;
    readBinaryPOD(input, size_data_per_element_);
    readBinaryPOD(input, linkdata_offset_);
    readBinaryPOD(input, label_offset_);
    readBinaryPOD(input, offsetData_);
    readBinaryPOD(input, maxlevel_);
    readBinaryPOD(input, enterpoint_node_);

    readBinaryPOD(input, maxM_);
    readBinaryPOD(input, maxM0_);
    readBinaryPOD(input, M_);
    readBinaryPOD(input, mult_);
    readBinaryPOD(input, ef_construction_);

    data_size_ = s->get_data_size();
    fstdistfunc_ = s->get_dist_func_extra();
    dist_func_param_ = s->get_dist_func_param();
    subvec_num_ = *(size_t*)s->get_dist_func_param_extra();

    auto pos = input.tellg();

    /// Optional - check if index is ok:
    input.seekg(cur_element_count_ * size_data_per_element_, input.cur);
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
    /// Optional check end

    input.seekg(pos, input.beg);

    data_level0_memory_ = (char*)malloc(max_elements * size_data_per_element_);
    if (data_level0_memory_ == nullptr)
      throw std::runtime_error("Not enough memory: loadIndex failed to allocate level0");
    input.read(data_level0_memory_, cur_element_count_ * size_data_per_element_);

    size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);

    size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
    std::vector<std::mutex>(max_elements).swap(link_list_locks_);
    std::vector<std::mutex>(MAX_LABEL_OPERATION_LOCKS).swap(label_op_locks_);

    visited_list_pool_.reset(new VisitedListPool(1, max_elements));

    linkLists_ = (char**)malloc(sizeof(void*) * max_elements);
    if (linkLists_ == nullptr)
      throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklists");
    element_levels_ = std::vector<int>(max_elements);
    revSize_ = 1.0 / mult_;
    ef_ = 10;
    for (size_t i = 0; i < cur_element_count_; i++) {
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

    for (size_t i = 0; i < cur_element_count_; i++) {
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
   * Marks an element with the given label deleted, does NOT really change the
   * current graph.
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
   * Uses the last 16 bits of the memory for the linked list size to store the
   * mark, whereas maxM0_ has to be limited to the lower 16 bits, however,
   * still large enough in almost all cases.
   */
  void markDeletedInternal(tableint internalId) {
    assert(internalId < cur_element_count_);
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
   * Removes the deleted mark of the node, does NOT really change the current
   * graph.
   *
   * Note: the method is not safe to use when replacement of deleted elements
   * is enabled, because elements marked as deleted can be completely removed
   * by addPoint
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
    assert(internalId < cur_element_count_);
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
   * Checks the first 16 bits of the memory to see if the element is marked
   * deleted.
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
   * If replacement of deleted elements is enabled: replaces previously deleted
   * point if any, updating it with new point
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
    // If point to be updated is entry point and graph just contains single
    // element then just return.
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

        sNeigh.insert(elOneHop);

        std::vector<tableint> listTwoHop = getConnectionsWithLock(elOneHop, layer);
        for (auto&& elTwoHop : listTwoHop) {
          sCand.insert(elTwoHop);
        }
      }

      for (auto&& neigh : sNeigh) {
        // if (neigh == internalId)
        //     continue;

        std::priority_queue<std::pair<dist_t, CandInfo>, std::vector<std::pair<dist_t, CandInfo>>,
                            CompareByFirst>
            candidates;
        size_t size =
            sCand.find(neigh) == sCand.end() ? sCand.size() : sCand.size() - 1;  // sCand guaranteed
                                                                                 // to have size >= 1
        size_t elementsToKeep = std::min(ef_construction_, size);
        for (auto&& cand : sCand) {
          if (cand == neigh) continue;

          float* subvec_dis = (float*)alloca(subvec_num_ * sizeof(float));
          dist_t distance = fstdistfunc_(getDataByInternalId(neigh), getDataByInternalId(cand),
                                         dist_func_param_, &subvec_num_, subvec_dis);

          CandInfo cand_info{
              .id = cand, .dist = distance, .avg_subvec_dis = distance / subvec_num_,
              // .subvec_dis = std::vector<float>(subvec_dis, subvec_dis +
              // subvec_num_)
          };
          memcpy(cand_info.subvec_dis, subvec_dis, subvec_num_ * sizeof(float));

          if (candidates.size() < elementsToKeep) {
            candidates.emplace(std::make_pair(distance, cand_info));
          } else {
            if (distance < candidates.top().first) {
              candidates.pop();
              candidates.emplace(std::make_pair(distance, cand_info));
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

          std::vector<CandInfo> add_cands;
          add_cands.reserve(candSize);
          for (size_t idx = 0; idx < candSize; idx++) {
            const auto& cand_info_top = candidates.top().second;
            data[idx] = cand_info_top.id;

            add_cands.push_back(cand_info_top);
            candidates.pop();
          }

          if (layer == 0) {
            auto* link_data = getLinkDataByInternalId(neigh);
            for (size_t idx = 0; idx < add_cands.size(); idx++) {
              auto* cur_link_data = link_data + idx;
              const auto& cand_info = add_cands[idx];

              for (int i = 0; i < subvec_num_; i++) {
                if (cand_info.subvec_dis[i] < cand_info.avg_subvec_dis) {
                  set_search_bits((int16_t*)cur_link_data, i);
                }
              }
            }
          }
        }
      }
    }

    repairConnectionsForUpdate(dataPoint, entryPointCopy, internalId, elemLevel, maxLevelCopy);
  }

  // todo
  void repairConnectionsForUpdate(const void* dataPoint,
                                  tableint entryPointInternalId,
                                  tableint dataPointInternalId,
                                  int dataPointLevel,
                                  int maxLevel) {
    tableint currObj = entryPointInternalId;
    if (dataPointLevel < maxLevel) {
      dist_t curdist =
          fstdistfunc_(dataPoint, getDataByInternalId(currObj), dist_func_param_, &subvec_num_, nullptr);
      for (int level = maxLevel; level > dataPointLevel; level--) {
        bool changed = true;
        while (changed) {
          changed = false;
          unsigned int* data;
          std::unique_lock<std::mutex> lock(link_list_locks_[currObj]);
          data = get_linklist_at_level(currObj, level);
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
            dist_t d =
                fstdistfunc_(dataPoint, getDataByInternalId(cand), dist_func_param_, &subvec_num_, nullptr);
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
      std::priority_queue<std::pair<dist_t, CandInfo>, std::vector<std::pair<dist_t, CandInfo>>,
                          CompareByFirst>
          topCandidates = searchBaseLayer(currObj, dataPoint, level);

      std::priority_queue<std::pair<dist_t, CandInfo>, std::vector<std::pair<dist_t, CandInfo>>,
                          CompareByFirst>
          filteredTopCandidates;
      while (topCandidates.size() > 0) {
        if (topCandidates.top().second.id != dataPointInternalId)
          filteredTopCandidates.push(topCandidates.top());

        topCandidates.pop();
      }

      // Since element_levels_ is being used to get `dataPointLevel`, there
      // could be cases where `topCandidates` could just contains entry point
      // itself. To prevent self loops, the `topCandidates` is filtered and
      // thus can be empty.
      if (filteredTopCandidates.size() > 0) {
        bool epDeleted = isMarkedDeleted(entryPointInternalId);
        if (epDeleted) {
          float* subvec_dis = (float*)alloca(subvec_num_ * sizeof(float));
          auto dis = fstdistfunc_(dataPoint, getDataByInternalId(entryPointInternalId), dist_func_param_,
                                  &subvec_num_, subvec_dis);

          CandInfo cand_info{
              .id = entryPointInternalId, .dist = dis, .avg_subvec_dis = dis / subvec_num_,
              // .subvec_dis = std::vector<float>(subvec_dis, subvec_dis +
              // subvec_num_)
          };
          filteredTopCandidates.emplace(std::make_pair(dis, cand_info));
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
        tableint existingInternalId = search->second;
        if (allow_replace_deleted_) {
          if (isMarkedDeleted(existingInternalId)) {
            throw std::runtime_error(
                "Can't use addPoint to update "
                "deleted elements if replacement "
                "of deleted elements is enabled.");
          }
        }
        lock_table.unlock();

        if (isMarkedDeleted(existingInternalId)) {
          unmarkDeletedInternal(existingInternalId);
        }
        updatePoint(data_point, existingInternalId, 1.0);

        return existingInternalId;
      }

      if (cur_element_count_ >= max_elements_) {
        throw std::runtime_error("The number of elements exceeds the specified limit");
      }

      cur_c = cur_element_count_;
      cur_element_count_++;
      label_lookup_[label] = cur_c;
    }

    std::unique_lock<std::mutex> lock_el(link_list_locks_[cur_c]);
    int curlevel = getRandomLevel(mult_);
    if (level > 0) curlevel = level;

    element_levels_[cur_c] = curlevel;

    std::unique_lock<std::mutex> templock(global);
    int maxlevelcopy = maxlevel_;
    if (curlevel <= maxlevelcopy) templock.unlock();
    tableint currObj = enterpoint_node_;
    tableint enterpoint_copy = enterpoint_node_;

    memset(data_level0_memory_ + cur_c * size_data_per_element_ + offsetLevel0_, 0, size_data_per_element_);

    // Initialisation of the data and label
    memcpy(getExternalLabeLp(cur_c), &label, sizeof(labeltype));
    memcpy(getDataByInternalId(cur_c), data_point, data_size_);

    if (curlevel) {
      linkLists_[cur_c] = (char*)malloc(size_links_per_element_ * curlevel + 1);
      if (linkLists_[cur_c] == nullptr)
        throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
      memset(linkLists_[cur_c], 0, size_links_per_element_ * curlevel + 1);
    }

    if ((signed)currObj != -1) {
      if (curlevel < maxlevelcopy) {
        dist_t curdist =
            fstdistfunc_(data_point, getDataByInternalId(currObj), dist_func_param_, &subvec_num_, nullptr);
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
              if (cand < 0 || cand > max_elements_) throw std::runtime_error("cand error");

              // float* subvec_dis = (float*)alloca(subvec_num_ *
              // sizeof(float));
              dist_t d = fstdistfunc_(data_point, getDataByInternalId(cand), dist_func_param_, &subvec_num_,
                                      nullptr);
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
        if (level > maxlevelcopy || level < 0)  // possible?
          throw std::runtime_error("Level error");

        std::priority_queue<std::pair<dist_t, CandInfo>, std::vector<std::pair<dist_t, CandInfo>>,
                            CompareByFirst>
            top_candidates = searchBaseLayer(currObj, data_point, level);
        if (epDeleted) {
          float* subvec_dis = (float*)alloca(subvec_num_ * sizeof(float));
          float dist = fstdistfunc_(data_point, getDataByInternalId(enterpoint_copy), dist_func_param_,
                                    &subvec_num_, subvec_dis);

          CandInfo cand_info{
              .id = enterpoint_copy, .dist = dist, .avg_subvec_dis = dist / subvec_num_,
              // .subvec_dis = std::vector<float>(subvec_dis, subvec_dis +
              // subvec_num_)
          };
          memcpy(cand_info.subvec_dis, subvec_dis, subvec_num_ * sizeof(float));

          top_candidates.emplace(std::make_pair(dist, cand_info));
          if (top_candidates.size() > ef_construction_) {
            top_candidates.pop();
          }
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
    if (cur_element_count_ == 0) return result;

    tableint currObj = enterpoint_node_;
    dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_,
                                  &subvec_num_, nullptr);

#if defined(TRACE_SEARCH)
    constexpr bool need_trace = true;
#else
    constexpr bool need_trace = false;
#endif
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
          std::cout << "enter_point: " << currObj << ", dis: " << curdist << ", cand size: " << size
                    << ", level: " << level << ", cur_hops: " << metric_hops
                    << ", cur_comp: " << metric_distance_computations << std::endl;
        }

        tableint* datal = (tableint*)(data + 1);
        for (int i = 0; i < size; i++) {
          tableint cand = datal[i];
          if (cand < 0 || cand > max_elements_) throw std::runtime_error("cand error");

          // float* subvec_dis = (float*)alloca(subvec_num_ *
          // sizeof(float));
          dist_t d =
              fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_, &subvec_num_, nullptr);
          if (need_trace) {
            std::cout << "" << "(" << getExternalLabel(cand) << ":" << d << "), ";
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

    std::priority_queue<std::pair<dist_t, CandInfo>, std::vector<std::pair<dist_t, CandInfo>>, CompareByFirst>
        top_candidates;
    bool bare_bone_search = !num_deleted_ && !isIdAllowed;
    if (bare_bone_search) {
      top_candidates = searchBaseLayerST<true, true>(currObj, query_data, k + EF_SEARCH, isIdAllowed);
    } else {
      top_candidates = searchBaseLayerST<false, true>(currObj, query_data, k + EF_SEARCH, isIdAllowed);
    }

    while (top_candidates.size() > k) {
      top_candidates.pop();
    }
    while (top_candidates.size() > 0) {
      std::pair<dist_t, CandInfo> rez = top_candidates.top();
      result.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second.id)));
      top_candidates.pop();
    }
    return result;
  }

  std::vector<std::pair<dist_t, labeltype>> searchStopConditionClosest(
      const void* query_data,
      BaseSearchStopCondition<dist_t>& stop_condition,
      BaseFilterFunctor* isIdAllowed = nullptr) const {
    std::vector<std::pair<dist_t, labeltype>> result;
    if (cur_element_count_ == 0) return result;

    tableint currObj = enterpoint_node_;
    dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_,
                                  &subvec_num_, nullptr);

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
        for (int i = 0; i < size; i++) {
          tableint cand = datal[i];
          if (cand < 0 || cand > max_elements_) throw std::runtime_error("cand error");
          dist_t d =
              fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_, &subvec_num_, nullptr);

          if (d < curdist) {
            curdist = d;
            currObj = cand;
            changed = true;
          }
        }
      }
    }

    std::priority_queue<std::pair<dist_t, CandInfo>, std::vector<std::pair<dist_t, CandInfo>>, CompareByFirst>
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
    std::vector<int> inbound_connections_num(cur_element_count_, 0);
    for (int i = 0; i < cur_element_count_; i++) {
      for (int l = 0; l <= element_levels_[i]; l++) {
        linklistsizeint* ll_cur = get_linklist_at_level(i, l);
        int size = getListCount(ll_cur);
        tableint* data = (tableint*)(ll_cur + 1);
        std::unordered_set<tableint> s;
        for (int j = 0; j < size; j++) {
          assert(data[j] < cur_element_count_);
          assert(data[j] != i);
          inbound_connections_num[data[j]]++;
          s.insert(data[j]);
          connections_checked++;
        }
        assert(s.size() == size);
      }
    }
    if (cur_element_count_ > 1) {
      int min1 = inbound_connections_num[0], max1 = inbound_connections_num[0];
      for (int i = 0; i < cur_element_count_; i++) {
        assert(inbound_connections_num[i] > 0);
        min1 = std::min(inbound_connections_num[i], min1);
        max1 = std::max(inbound_connections_num[i], max1);
      }
      std::cout << "Min inbound: " << min1 << ", Max inbound:" << max1 << "\n";
    }
    std::cout << "integrity ok, checked " << connections_checked << " connections\n";
  }
};
}  // namespace hnswlib
