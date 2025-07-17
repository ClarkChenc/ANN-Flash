#pragma once

#include <cstdint>
#include <cstddef>
#include <xmmintrin.h>
#include <immintrin.h>
#include <pmmintrin.h>

namespace hnswlib {

typedef uint8_t encode_t;
typedef uint16_t pq_dist_t;

using PQ_ENCODE_FUNC = void (*)(float* codebook,
                                float qmin,
                                float qmax,
                                size_t subspace_num,
                                size_t cluster_nun,
                                size_t data_dim,
                                float* data,
                                encode_t* encode_vec,
                                pq_dist_t* pq_dist_table,
                                bool is_query);

using DIS_FUNC = float (*)(const void*, const void*, const void*);

inline float sum_four(__m128 v) {
  __m128 sum1 = _mm_hadd_ps(v, v);        // [a+b, c+d, a+b, c+d]
  __m128 sum2 = _mm_hadd_ps(sum1, sum1);  //[a+b+c+d, a+b+c+d, a+b+c+d, a+b+c+d]
  return _mm_cvtss_f32(sum2);             // return fisrt element
}

inline float sum_first_two(__m128 v) {
  __m128 sum = _mm_add_ss(v, _mm_shuffle_ps(v, v, 0x55));
  return _mm_cvtss_f32(sum);
}

template <typename data_t>
class FlashSpaceInterface {
 public:
  size_t subspace_num_{0};
  size_t cluster_num_{0};
  size_t data_dim_{0};

 public:
  FlashSpaceInterface() = default;

  explicit FlashSpaceInterface(size_t subspace_num, size_t cluster_num, size_t data_dim)
      : subspace_num_(subspace_num), cluster_num_(cluster_num), data_dim_(data_dim) {}

  FlashSpaceInterface(const FlashSpaceInterface& rhs) {
    subspace_num_ = rhs.subspace_num_;
    cluster_num_ = rhs.cluster_num_;
    data_dim_ = rhs.data_dim_;
  }

  virtual ~FlashSpaceInterface() {}

  virtual size_t get_encode_data_size() {
    return subspace_num_ * sizeof(encode_t);
  }

  inline size_t get_subspace_num() {
    return subspace_num_;
  }

  inline size_t get_cluster_num() {
    return cluster_num_;
  }

  inline size_t get_data_dim() {
    return data_dim_;
  }

  inline size_t get_raw_data_size() {
    return data_dim_ * sizeof(data_t);
  }

  virtual PQ_ENCODE_FUNC get_pq_encode_func() const = 0;

  virtual DIS_FUNC get_dis_func() const = 0;

  virtual DIS_FUNC get_dis_func_with_quantizer() const = 0;
};

}  // namespace hnswlib