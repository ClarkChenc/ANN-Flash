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
                                size_t subspace_num,
                                size_t cluster_nun,
                                size_t data_dim,
                                float* data,
                                encode_t* encode_vec,
                                pq_dist_t* pq_dist_table);

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

static void PqEncodeWithSSE(float* codebook,
                            size_t subspace_num,
                            size_t cluster_num,
                            size_t data_dim,
                            float* data,
                            encode_t* encode_vector,
                            pq_dist_t* dist_table) {
  thread_local std::vector<float> raw_dist_table(subspace_num * cluster_num);
  float* codebook_ptr = codebook;

  size_t dist_table_index = 0;
  float min_dist = std::numeric_limits<float>::max(), max_dist = 0;
  size_t subspace_len = data_dim / subspace_num;

  // 填充 raw_dist_table
  for (size_t i = 0; i < subspace_num; ++i) {
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
      for (size_t j = 0; j < cluster_num; ++j) {
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
      for (size_t j = 0; j < cluster_num; j += 2) {
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
      for (size_t j = 0; j < cluster_num; j += 4) {
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
    max_dist += (subspace_max_dist - subspace_min_dist);
    encode_vector[i] = best_index;
  }

  auto* raw_dist_table_ptr = raw_dist_table.data();
  float qscale = 1 / max_dist;
  for (size_t i = 0; i < subspace_num; ++i) {
    for (size_t j = 0; j < cluster_num; ++j) {
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
template <typename data_t>
class FlashSpaceInterface {
 public:
  enum DisType { L2 = 0, IP };

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

  virtual DisType get_dis_type() = 0;

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

  virtual PQ_ENCODE_FUNC get_pq_encode_func() const {
    return &PqEncodeWithSSE;
  }

  virtual DIS_FUNC get_dis_func() const = 0;

  virtual DIS_FUNC get_dis_func_with_quantizer() const = 0;
};

}  // namespace hnswlib