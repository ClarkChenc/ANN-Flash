#pragma once
#include "hnswlib.h"

namespace hnswlib {

static float L2Sqr(const void* pVect1v,
                   const void* pVect2v,
                   const void* qty_ptr,
                   const void* = nullptr,
                   void* = nullptr) {
  float* pVect1 = (float*)pVect1v;
  float* pVect2 = (float*)pVect2v;
  size_t qty = *((size_t*)qty_ptr);

  float res = 0;
  for (size_t i = 0; i < qty; i++) {
    float t = *pVect1 - *pVect2;
    pVect1++;
    pVect2++;
    res += t * t;
  }
  return (res);
}

#if defined(USE_AVX512)

// Favor using AVX512 if available.
static float L2SqrSIMD16ExtAVX512(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
  float* pVect1 = (float*)pVect1v;
  float* pVect2 = (float*)pVect2v;
  size_t qty = *((size_t*)qty_ptr);
  float PORTABLE_ALIGN64 TmpRes[16];
  size_t qty16 = qty >> 4;

  const float* pEnd1 = pVect1 + (qty16 << 4);

  __m512 diff, v1, v2;
  __m512 sum = _mm512_set1_ps(0);

  while (pVect1 < pEnd1) {
    v1 = _mm512_loadu_ps(pVect1);
    pVect1 += 16;
    v2 = _mm512_loadu_ps(pVect2);
    pVect2 += 16;
    diff = _mm512_sub_ps(v1, v2);
    // sum = _mm512_fmadd_ps(diff, diff, sum);
    sum = _mm512_add_ps(sum, _mm512_mul_ps(diff, diff));
  }

  _mm512_store_ps(TmpRes, sum);
  float res = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7] +
              TmpRes[8] + TmpRes[9] + TmpRes[10] + TmpRes[11] + TmpRes[12] + TmpRes[13] + TmpRes[14] +
              TmpRes[15];

  return (res);
}
#endif

#if defined(USE_AVX)

// Favor using AVX if available.
static float L2SqrSIMD16ExtAVX(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
  float* pVect1 = (float*)pVect1v;
  float* pVect2 = (float*)pVect2v;
  size_t qty = *((size_t*)qty_ptr);
  float PORTABLE_ALIGN32 TmpRes[8];
  size_t qty16 = qty >> 4;

  const float* pEnd1 = pVect1 + (qty16 << 4);

  __m256 diff, v1, v2;
  __m256 sum = _mm256_set1_ps(0);

  while (pVect1 < pEnd1) {
    v1 = _mm256_loadu_ps(pVect1);
    pVect1 += 8;
    v2 = _mm256_loadu_ps(pVect2);
    pVect2 += 8;
    diff = _mm256_sub_ps(v1, v2);
    sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));

    v1 = _mm256_loadu_ps(pVect1);
    pVect1 += 8;
    v2 = _mm256_loadu_ps(pVect2);
    pVect2 += 8;
    diff = _mm256_sub_ps(v1, v2);
    sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
  }

  _mm256_store_ps(TmpRes, sum);
  return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];
}

#endif

#if defined(USE_SSE)

static inline float horizontal_add(__m128 v) {
  __m128 sum1 = _mm_hadd_ps(v, v);        // [a+b, c+d, a+b, c+d]
  __m128 sum2 = _mm_hadd_ps(sum1, sum1);  // [a+b+c+d, a+b+c+d, ...]
  return _mm_cvtss_f32(sum2);             // 取第一个元素
}

static float L2SqrSIMD16ExtSSE(const void* pVect1v,
                               const void* pVect2v,
                               const void* qty_ptr,
                               const void* subvec_size_ptr = nullptr,
                               void* subvec_sum_ptr = nullptr) {
  float* pVect1 = (float*)pVect1v;
  float* pVect2 = (float*)pVect2v;
  size_t qty = *((size_t*)qty_ptr);
  float* subvec_sum = (float*)subvec_sum_ptr;

  size_t qty16 = qty >> 4;
  qty16 = qty16 << 4;

  const float* pEnd1 = pVect1 + qty16;

  __m128 diff, v1, v2;
  __m128 sum = _mm_set1_ps(0);

  while (pVect1 < pEnd1) {
    // 每一遍计算 16 个 float

    _mm_prefetch((char*)(pVect1 + 64), _MM_HINT_T0);
    _mm_prefetch((char*)(pVect2 + 64), _MM_HINT_T0);
    // v1 = _mm_loadu_ps(pVect1);
    // pVect1 += 4;
    // v2 = _mm_loadu_ps(pVect2);
    // pVect2 += 4;
    // diff = _mm_sub_ps(v1, v2);
    // subvec_sum_x[cur_subvec_index] = _mm_add_ps(subvec_sum_x[cur_subvec_index], _mm_mul_ps(diff, diff));

    // v1 = _mm_loadu_ps(pVect1);
    // pVect1 += 4;
    // v2 = _mm_loadu_ps(pVect2);
    // pVect2 += 4;
    // diff = _mm_sub_ps(v1, v2);
    // subvec_sum_x[cur_subvec_index] = _mm_add_ps(subvec_sum_x[cur_subvec_index], _mm_mul_ps(diff, diff));

    // v1 = _mm_loadu_ps(pVect1);
    // pVect1 += 4;
    // v2 = _mm_loadu_ps(pVect2);
    // pVect2 += 4;
    // diff = _mm_sub_ps(v1, v2);
    // subvec_sum_x[cur_subvec_index] = _mm_add_ps(subvec_sum_x[cur_subvec_index], _mm_mul_ps(diff, diff));

    // v1 = _mm_loadu_ps(pVect1);
    // pVect1 += 4;
    // v2 = _mm_loadu_ps(pVect2);
    // pVect2 += 4;
    // diff = _mm_sub_ps(v1, v2);
    // subvec_sum_x[cur_subvec_index] = _mm_add_ps(subvec_sum_x[cur_subvec_index], _mm_mul_ps(diff, diff));

    for (int i = 0; i < 4; ++i) {
      v1 = _mm_loadu_ps(pVect1);
      pVect1 += 4;
      v2 = _mm_loadu_ps(pVect2);
      pVect2 += 4;
      diff = _mm_sub_ps(v1, v2);
      sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
    }
  }

  // _mm_store_ps(TmpRes, sum);
  return horizontal_add(sum);
}
#endif

#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
static DISTFUNC_EXTRA<float> L2SqrSIMD16Ext_extra = L2SqrSIMD16ExtSSE;

// static float
// L2SqrSIMD16ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
//     size_t qty = *((size_t *) qty_ptr);
//     size_t qty16 = qty >> 4 << 4;
//     float res = L2SqrSIMD16Ext(pVect1v, pVect2v, &qty16);
//     float *pVect1 = (float *) pVect1v + qty16;
//     float *pVect2 = (float *) pVect2v + qty16;

//     size_t qty_left = qty - qty16;
//     float res_tail = L2Sqr(pVect1, pVect2, &qty_left);
//     return (res + res_tail);
// }
// #endif

// #if defined(USE_SSE)
// static float
// L2SqrSIMD4Ext(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
//     float PORTABLE_ALIGN32 TmpRes[8];
//     float *pVect1 = (float *) pVect1v;
//     float *pVect2 = (float *) pVect2v;
//     size_t qty = *((size_t *) qty_ptr);

//     size_t qty4 = qty >> 2;

//     const float *pEnd1 = pVect1 + (qty4 << 2);

//     __m128 diff, v1, v2;
//     __m128 sum = _mm_set1_ps(0);

//     while (pVect1 < pEnd1) {
//         v1 = _mm_loadu_ps(pVect1);
//         pVect1 += 4;
//         v2 = _mm_loadu_ps(pVect2);
//         pVect2 += 4;
//         diff = _mm_sub_ps(v1, v2);
//         sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
//     }
//     _mm_store_ps(TmpRes, sum);
//     return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
// }

// static float
// L2SqrSIMD4ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
//     size_t qty = *((size_t *) qty_ptr);
//     size_t qty4 = qty >> 2 << 2;

//     float res = L2SqrSIMD4Ext(pVect1v, pVect2v, &qty4);
//     size_t qty_left = qty - qty4;

//     float *pVect1 = (float *) pVect1v + qty4;
//     float *pVect2 = (float *) pVect2v + qty4;
//     float res_tail = L2Sqr(pVect1, pVect2, &qty_left);

//     return (res + res_tail);
// }
#endif

class L2Space_V2 : public SpaceInterface<float> {
  DISTFUNC_EXTRA<float> fstdistfunc_;
  size_t data_size_;
  size_t dim_;
  size_t subvec_size_;

 public:
  L2Space_V2(size_t dim, size_t subvec_size = 1) {
    fstdistfunc_ = L2Sqr;
#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
#if defined(USE_AVX512)
    if (AVX512Capable())
      L2SqrSIMD16Ext = L2SqrSIMD16ExtAVX512;
    else if (AVXCapable())
      L2SqrSIMD16Ext = L2SqrSIMD16ExtAVX;
#elif defined(USE_AVX)
    if (AVXCapable()) L2SqrSIMD16Ext = L2SqrSIMD16ExtAVX;
#endif

    // if (dim % 16 == 0)
    //     fstdistfunc_ = L2SqrSIMD16Ext;
    // else if (dim % 4 == 0)
    //     fstdistfunc_ = L2SqrSIMD4Ext;
    // else if (dim > 16)
    //     fstdistfunc_ = L2SqrSIMD16ExtResiduals;
    // else if (dim > 4)
    //     fstdistfunc_ = L2SqrSIMD4ExtResiduals;
    fstdistfunc_ = L2SqrSIMD16Ext_extra;
#endif
    dim_ = dim;
    data_size_ = dim * sizeof(float);
    subvec_size_ = subvec_size;
  }

  size_t get_data_size() {
    return data_size_;
  }

  virtual DISTFUNC<float> get_dist_func() {
    return nullptr;
  }

  DISTFUNC_EXTRA<float> get_dist_func_extra() {
    return fstdistfunc_;
  }

  void* get_dist_func_param() {
    return &dim_;
  }

  void* get_dist_func_param_extra() {
    return &subvec_size_;
  }

  ~L2Space_V2() {}
};
}  // namespace hnswlib
