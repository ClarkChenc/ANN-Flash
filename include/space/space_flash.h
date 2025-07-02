#pragma once

#include "adsampling.h"
#include "core.h"

namespace hnswlib {

#define ls(x) ((x >> 4) & 0x0F)
#define rs(x) (x & 0x0F)

/**
 * Calculate the squared Euclidean distance between two vectors
 * @param pVect1v Pointer to a distance table. The distance table contains CLUSTER_NUM distances for each
 * subvector.
 * @param pVect2v Pointer to encoded data. The encoded data contains the cluster indices of two subvectors,
 * with each index stored in the high 4 bits and low 4 bits, respectively.
 * @param qty_ptr Pointer to the dimension of the vectors
 * @return The squared Euclidean distance between the two vectors
 */

template <typename dist_t>
class FlashSpace : public SpaceInterface<dist_t> {
  DISTFUNC<dist_t> fstdistfunc_;
  size_t data_size_;
  size_t dim_;

 public:
  FlashSpace(size_t dim) {
    fstdistfunc_ = FlashL2Sqr;
    dim_ = dim;
    data_size_ = dim * sizeof(encode_t);
  }

  size_t get_data_size() {
    return data_size_;
  }

  DISTFUNC<dist_t> get_dist_func() {
    return fstdistfunc_;
  }

  void* get_dist_func_param() {
    return &dim_;
  }

  static dist_t FlashL2Sqr(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    dist_t* pVect1 = (dist_t*)pVect1v;      // distance table
    encode_t* pVect2 = (encode_t*)pVect2v;  // encoded data
    size_t qty = *((size_t*)qty_ptr);

    dist_t res = 0;
    for (size_t i = 0; i < qty; ++i) {
      res += pVect1[*pVect2];
      pVect1 += CLUSTER_NUM;
      pVect2++;
    }

    return res;
  }
};

}  // namespace hnswlib
