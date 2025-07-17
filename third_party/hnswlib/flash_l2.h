#pragma once

#include "flash_lib.h"

#include <vector>
#include <limits>

namespace hnswlib {

class FlashL2 : public FlashSpaceInterface<float> {
 public:
  explicit FlashL2(size_t subspace_num, size_t cluster_num, size_t data_dim)
      : FlashSpaceInterface(subspace_num, cluster_num, data_dim) {}

  PQ_FUNC get_pq_encode_func() const override {
    return &FlashL2::PqEncodeWithSSE;
  }

  RERANK_FUNC<float> get_rerank_func() const override {
    return &FlashL2::RerankWithSSE16;
  }

  static float RerankWithSSE16(const void* emb1, const void* emb2, const void* dim) {
    float* ptr_emb1 = (float*)emb1;
    float* ptr_emb2 = (float*)emb2;
    size_t data_dim = *(size_t*)dim;

    const float* ptr_emb1_end = ptr_emb1 + data_dim;
    __m128 v1, v2, diff, cal_res;
    v1 = _mm_set1_ps(0);
    v2 = _mm_set1_ps(0);
    cal_res = _mm_set1_ps(0);

    // 每次处理 16 个 float
    while (ptr_emb1 < ptr_emb1_end) {
      for (size_t i = 0; i < 4; ++i) {
        v1 = _mm_loadu_ps(ptr_emb1 + i * 4);
        v2 = _mm_loadu_ps(ptr_emb2 + i * 4);
        diff = _mm_sub_ps(v1, v2);
        cal_res = _mm_add_ps(cal_res, _mm_mul_ps(diff, diff));
      }
      ptr_emb1 += 16;
      ptr_emb2 += 16;
    }
    return sum_four(cal_res);
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
    float min_dist = std::numeric_limits<float>::max(), max_dist = 0;
    size_t subspace_len = data_dim / subspace_num;

    // 填充 raw_dist_table
    size_t cur_prelen = 0;
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