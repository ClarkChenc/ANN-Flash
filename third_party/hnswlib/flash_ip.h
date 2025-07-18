#pragma once

#include "flash_lib.h"
#include "distance_ip.h"
#include "quantizer.h"

#include <vector>
#include <limits>

namespace hnswlib {

template <typename data_t>
static float InnerProductDistFuncSSE(const void* emb1, const void* emb2, const void* dim) {
  return -InnerProduct_SSE((const data_t*)emb1, (const data_t*)emb2, *(size_t*)dim);
}

template <typename data_t>
static float InnerProductDistFuncAVX2(const void* a, const void* b, const void* dim) {
  return -InnerProduct_AVX2((const data_t*)a, (const data_t*)b, *(size_t*)dim);
}

template <typename data_t>
static float InnerProductDistFuncAVX512(const void* a, const void* b, const void* dim) {
  return -InnerProduct_AVX512((const data_t*)a, (const data_t*)b, *(size_t*)dim);
}

// ip 距离统一 x -1，越小表示越相似
template <typename data_t = float>
class FlashIP : public FlashSpaceInterface<data_t> {
 public:
  explicit FlashIP(size_t subspace_num, size_t cluster_num, size_t data_dim)
      : FlashSpaceInterface<data_t>(subspace_num, cluster_num, data_dim) {}

  PQ_ENCODE_FUNC get_pq_encode_func() const override {
    return &FlashIP::PqEncodeWithSSE;
  }

  DIS_FUNC get_dis_func() const override {
    return InnerProductDistFuncSSE<float>;
  }

  DIS_FUNC get_dis_func_with_quantizer() const override {
    return InnerProductDistFuncSSE<data_t>;
  }

  static void PqEncodeWithSSE(float* codebook,
                              float qmin,
                              float qmax,
                              size_t subspace_num,
                              size_t cluster_num,
                              size_t data_dim,
                              float* data,
                              encode_t* encode_vector,
                              pq_dist_t* dist_table,
                              bool is_query) {
    thread_local std::vector<float> raw_dist_table(subspace_num * cluster_num);
    float* codebook_ptr = codebook;

    size_t dist_table_index = 0;
    float min_dist = std::numeric_limits<float>::max(), max_dist = std::numeric_limits<float>::min();
    size_t subspace_len = data_dim / subspace_num;

    // 填充 raw_dist_table
    for (size_t i = 0; i < subspace_num; ++i) {
      float* data_ptr = data + i * subspace_len;
      encode_t best_index = 0;

      float subspace_min_dist = std::numeric_limits<float>::max();
      float subspace_max_dist = std::numeric_limits<float>::min();
      if (subspace_len == 4) {
        __m128 cal_res;
        cal_res = _mm_set1_ps(0);

        __m128 v1;
        __m128 v2;
        v1 = _mm_loadu_ps(data_ptr);
        // 每次处理 4 个 float, 即 1 个 cluster
        for (size_t j = 0; j < cluster_num; ++j) {
          float res = 0;
          v2 = _mm_loadu_ps(codebook_ptr);
          cal_res = _mm_mul_ps(v1, v2);
          res = -sum_four(cal_res);

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

        __m128 a = _mm_set1_ps(data_ptr[0]);  // [a, a, a, a]
        __m128 b = _mm_set1_ps(data_ptr[1]);  // [b, b, b, b]
        v1 = _mm_unpacklo_ps(a, b);           // [a, b, a, b]
        alignas(16) float tmp_res[4];

        // 每次处理 4 个 float, 即 2 个 cluster
        for (size_t j = 0; j < cluster_num; j += 2) {
          v2 = _mm_loadu_ps(codebook_ptr);
          cal_res = _mm_mul_ps(v1, v2);
          cal_res = _mm_hadd_ps(cal_res, cal_res);  // 【a+b, c+d, a+b, c+d】
          _mm_store_ps(tmp_res, cal_res);

          for (size_t k = 0; k < 2; ++k) {
            auto res = -tmp_res[k];
            if (res < subspace_min_dist) {
              subspace_min_dist = res;
              best_index = j + k;
            } else if (res > subspace_max_dist) {
              subspace_max_dist = res;
            }
          }

          raw_dist_table[dist_table_index] = -tmp_res[0];
          raw_dist_table[dist_table_index + 1] = -tmp_res[1];
          dist_table_index += 2;
          codebook_ptr += 4;
        }
      } else if (subspace_len == 1) {
        __m128 cal_res;
        cal_res = _mm_set1_ps(0);

        __m128 v1;
        __m128 v2;

        v1 = _mm_set1_ps(*data_ptr);
        alignas(16) float tmp_res[4];

        // 每次处理 4 个 float, 即 4 个 cluster
        for (size_t j = 0; j < cluster_num; j += 4) {
          v2 = _mm_loadu_ps(codebook_ptr);
          cal_res = _mm_mul_ps(v1, v2);

          _mm_store_ps(tmp_res, cal_res);
          for (size_t k = 0; k < 4; ++k) {
            auto cur_res = -tmp_res[k];
            if (cur_res < subspace_min_dist) {
              subspace_min_dist = cur_res;
              best_index = j + k;
            } else if (cur_res > subspace_max_dist) {
              subspace_max_dist = cur_res;
            }
          }

          raw_dist_table[dist_table_index] = -tmp_res[0];
          raw_dist_table[dist_table_index + 1] = -tmp_res[1];
          raw_dist_table[dist_table_index + 2] = -tmp_res[2];
          raw_dist_table[dist_table_index + 3] = -tmp_res[3];
          dist_table_index += 4;
          codebook_ptr += 4;
        }
      }

      min_dist = std::min(min_dist, subspace_min_dist);
      max_dist += subspace_max_dist - subspace_min_dist;
      encode_vector[i] = best_index;
    }

    // 量化 raw_dist_table，并将结果填充到 dist_table
    // query 使用独立的 qmin 和 qmax
    // index data 使用码本 qmin 和 qmax
    if (!is_query) {
      max_dist = qmax;
      min_dist = qmin;
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
};

}  // namespace hnswlib