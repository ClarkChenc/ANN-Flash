#pragma once

#include <stdint.h>
#include <x86intrin.h>

#define HNSWLIB_vnni __attribute__((target("avx512vnni,avx512vl,avx512bw,avx512f,avx512dq")))
#define HNSWLIB_avx512 __attribute__((target("avx512f,avx512dq,avx512bw,avx512vl")))

class Bfp16 {
 private:
  union {
    uint16_t u16;
    struct {
      uint16_t mantissa : 7;
      uint16_t exponent : 8;
      uint16_t sign : 1;
    };
  } storage;

  typedef union {
    float value;
    struct {
      uint32_t mantissa : 23;
      uint32_t exponent : 8;
      uint32_t sign : 1;
    };
    uint16_t u16s[2];
  } Fp32Format;

 public:
  inline Bfp16() {}

  template <typename T>
  inline Bfp16(const T& x) {
    *this = x;
  }

  template <typename T>
  inline void operator=(const T& x) {
    Fp32Format fp32;
    fp32.value = float(x);
    if (fp32.u16s[0] & 0x8000) {
      fp32.u16s[0] = 0;
      fp32.value *= 129.0f / 128.0f;
    }
    storage.u16 = fp32.u16s[1];
  }

  template <typename T>
  inline operator T() const {
    Fp32Format fp32;
    fp32.u16s[0] = 0;
    fp32.u16s[1] = storage.u16;
    return T(fp32.value);
  }
};

typedef Bfp16 bfp16_t;

class Sefp16 {
 private:
  union {
    uint16_t u16;
    struct {
      uint16_t mantissa : 10;
      uint16_t exponent : 5;
      uint16_t sign : 1;
    };
  } storage;

  typedef union {
    uint32_t u;
    float f;
    struct {
      uint32_t mantissa : 23;
      uint32_t exponent : 8;
      uint32_t sign : 1;
    };
  } Fp32Format;

 public:
  inline Sefp16() {}

  // FIXME: not correct here
  // only support fp32 <=> fp16
  template <typename T>
  inline Sefp16(const T& x) {
    *this = x;
  }

  template <typename T>
  inline void operator=(const T& x) {
    Fp32Format f;
    f.f = x;
    //
    Fp32Format f32infty = {255 << 23};
    Fp32Format f16infty = {31 << 23};
    Fp32Format magic = {15 << 23};
    uint sign_mask = 0x80000000u;
    uint round_mask = ~0xfffu;
    storage.u16 = {0};

    uint sign = f.u & sign_mask;
    f.u ^= sign;

    // NOTE all the integer compares in this function can be safely
    // compiled into signed compares since all operands are below
    // 0x80000000. Important if you want fast straight SSE2 code
    // (since there's no unsigned PCMPGTD).

    if (f.u >= f32infty.u)                                 // Inf or NaN (all exponent bits set)
      storage.u16 = (f.u > f32infty.u) ? 0x7e00 : 0x7c00;  // NaN->qNaN and Inf->Inf
    else                                                   // (De)normalized number or zero
    {
      f.u &= round_mask;
      f.f *= magic.f;
      f.u -= round_mask;
      if (f.u > f16infty.u) f.u = f16infty.u;  // Clamp to signed infinity if overflowed

      storage.u16 = f.u >> 13;  // Take the bits!
    }

    storage.u16 |= sign >> 16;
  }

  template <typename T>
  inline operator T() const {
    static const Fp32Format magic = {(254 - 15) << 23};
    static const Fp32Format was_infnan = {(127 + 16) << 23};
    Fp32Format o;

    o.u = (storage.u16 & 0x7fff) << 13;  // exponent/mantissa bits
    o.f *= magic.f;                      // exponent adjust
    if (o.f >= was_infnan.f)             // make sure Inf/NaN survive
      o.u |= 255 << 23;
    o.u |= (storage.u16 & 0x8000) << 16;  // sign bit
    return T(o.f);
  }
};

typedef Sefp16 sefp16_t;

// some magic number defined here to prevent reassign during executing
static __m128i __m128_mnosign = _mm_set1_epi32(0x7fff);
static __m128i __m128_magic = _mm_set1_epi32(((254 - 15) << 23));
static __m128i __m128_was_infnan = _mm_set1_epi32(0x7bff);
static __m128i __m128_exp_infnan = _mm_set1_epi32((255 << 23));

// need set 16bit pre-zero for each 32bit fp16
static inline __m128 half_to_float_SSE(const __m128i& h) {
  __m128i expmant = _mm_and_si128(__m128_mnosign, h);
  __m128i justsign = _mm_xor_si128(h, expmant);

  __m128i shifted = _mm_slli_epi32(expmant, 13);
  __m128 scaled = _mm_mul_ps(_mm_castsi128_ps(shifted), *(const __m128*)&__m128_magic);
  __m128i expmant2 = expmant;  // copy (just here for counting purposes)
  __m128i b_wasinfnan = _mm_cmpgt_epi32(expmant2, __m128_was_infnan);
  __m128i sign = _mm_slli_epi32(justsign, 16);
  __m128 infnanexp = _mm_and_ps(_mm_castsi128_ps(b_wasinfnan), _mm_castsi128_ps(__m128_exp_infnan));
  __m128 sign_inf = _mm_or_ps(_mm_castsi128_ps(sign), infnanexp);
  return _mm_or_ps(scaled, sign_inf);
  // ~11 SSE2 ops.
}

static __m256i __m256_mnosign = _mm256_set1_epi32(0x7fff);
static __m256i __m256_magic = _mm256_set1_epi32(((254 - 15) << 23));
static __m256i __m256_was_infnan = _mm256_set1_epi32(0x7bff);
static __m256i __m256_exp_infnan = _mm256_set1_epi32((255 << 23));

// need set 16bit pre-zero for each 32bit fp16
static inline __m256 half_to_float_AVX2(const __m256i& h) {
  __m256i expmant = _mm256_and_si256(__m256_mnosign, h);
  __m256i justsign = _mm256_xor_si256(h, expmant);

  __m256i shifted = _mm256_slli_epi32(expmant, 13);
  __m256 scaled = _mm256_mul_ps(_mm256_castsi256_ps(shifted), *(const __m256*)&__m256_magic);
  __m256i expmant2 = expmant;  // copy (just here for counting purposes)
  __m256i b_wasinfnan = _mm256_cmpgt_epi32(expmant2, __m256_was_infnan);
  __m256i sign = _mm256_slli_epi32(justsign, 16);
  __m256 infnanexp = _mm256_and_ps(_mm256_castsi256_ps(b_wasinfnan), _mm256_castsi256_ps(__m256_exp_infnan));
  __m256 sign_inf = _mm256_or_ps(_mm256_castsi256_ps(sign), infnanexp);
  return _mm256_or_ps(scaled, sign_inf);
}
