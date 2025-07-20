#pragma once

#include "flash_lib.h"
#include "distance_l2.h"
#include "quantizer.h"

#include <vector>
#include <limits>

namespace hnswlib {

template <typename data_t>
static float L2SqrDistFuncSSE(const void* emb1, const void* emb2, const void* dim) {
  return L2Sqr_SSE((const data_t*)emb1, (const data_t*)emb2, *(size_t*)dim);
}

template <typename data_t>
static float L2SqrDistFuncAVX2(const void* a, const void* b, const void* dim) {
  return L2Sqr_AVX2((const data_t*)a, (const data_t*)b, *(size_t*)dim);
}

template <typename data_t>
static float L2SqrDistFuncAVX512(const void* a, const void* b, const void* dim) {
  return L2Sqr_AVX512((const data_t*)a, (const data_t*)b, *(size_t*)dim);
}

template <typename data_t = float>
class FlashL2 : public FlashSpaceInterface<data_t> {
 public:
  using typename FlashSpaceInterface<data_t>::DisType;

  explicit FlashL2(size_t subspace_num, size_t cluster_num, size_t data_dim)
      : FlashSpaceInterface<data_t>(subspace_num, cluster_num, data_dim) {}

  DisType get_dis_type() {
    return DisType::L2;
  }

  DIS_FUNC get_dis_func() const override {
    return L2SqrDistFuncSSE<float>;
  }

  DIS_FUNC get_dis_func_with_quantizer() const override {
    return L2SqrDistFuncSSE<data_t>;
  }
};

}  // namespace hnswlib