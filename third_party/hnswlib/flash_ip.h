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
  using typename FlashSpaceInterface<data_t>::DisType;

  explicit FlashIP(size_t subspace_num, size_t cluster_num, size_t data_dim)
      : FlashSpaceInterface<data_t>(subspace_num, cluster_num, data_dim) {}

  DisType get_dis_type() {
    return DisType::IP;
  }

  DIS_FUNC get_dis_func() const override {
    return InnerProductDistFuncSSE<float>;
  }

  DIS_FUNC get_dis_func_with_quantizer() const override {
    return InnerProductDistFuncSSE<data_t>;
  }
};

}  // namespace hnswlib