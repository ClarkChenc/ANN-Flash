#pragma once

#include <iomanip>
#include <sstream>

#include "define.h"

namespace hnswlib {

template <typename input_t>
struct Codec {
  static void encode(float x, input_t* code, int i) {
    return;
  }

  static float decode(const input_t* code, int i) {
    return code[i];
  }

  static float getParams() {
    return 1.0f;
  }
};

template <>
struct Codec<int8_t> {
  static void encode(float x, int8_t* code, int i) {
    // code[i] = int8_t(-128.0f + 255.0f * x);  // previous encode shows the residual not 0
    static constexpr uint32_t bytes_str = 0b01000011011111111111111111111111;
    int32_t trans = int32_t(*(reinterpret_cast<const float*>(&bytes_str)) * x) -
                    128;  // base on IEEE 754 0b01000011011111111111111111111111 is
                          // the least large number smaller than 256
    code[i] = int8_t(trans);
  }

  static float decode(const int8_t* code, int i) {
    // int32_t trans = code[i] + 128;
    // return (code[i] + 128.0f + 0.5f) / 255.0f;
    static constexpr uint32_t bytes_str = 0b01000011011111111111111111111111;
    return (code[i] + 128.0f + 0.5f) / *(reinterpret_cast<const float*>(&bytes_str));
  }

  static float getParams() {
    return 255.0f;
  }
};

template <>
struct Codec<int16_t> {
  static void encode(float x, int16_t* code, int i) {
    code[i] = int16_t(-32768.0f + 65535.0f * x);
  }

  static float decode(const int16_t* code, int i) {
    return (code[i] + 32768.0f + 0.5) / 65535.0f;
  }

  static float getParams() {
    return 65535.0f;
  }
};

class BasicQuantizer {
 public:
  virtual const void* EncodeVectorDim(const void* inx, size_t decode_dim) const = 0;

  virtual const void* EncodeVector(const void* inx) const = 0;

  virtual void DecodeVector(const void* code, void* x) const = 0;

  virtual void setQuantizerParams(const char* params) = 0;

  virtual float DecodeDistance(float code_dist) = 0;

  virtual void train(int n, const float* x) = 0;

  virtual std::string GetQuantizerParams() = 0;

  virtual float DecodeItem(const void* code) = 0;

  virtual void DecodeVectorDim(const void* code, void* x, size_t decode_dim) const = 0;

  virtual void SetCpuFlag(uint32_t cpu_flag = 0) {
    cpu_flag_ = cpu_flag;
  }

  virtual ~BasicQuantizer() {}

  uint32_t cpu_flag_ = 0;
};

template <typename T>
class NoneQuantizer : public BasicQuantizer {
 public:
  using Type = T;

 public:
  NoneQuantizer(size_t _dim) : dim(_dim) {};

 public:
  const void* EncodeVectorDim(const void* inx, size_t decode_dim) const override {
    return inx;
  }

  const void* EncodeVector(const void* x) const override {
    return x;
  }

  void deleteConvertedVector(const Type* tx) const {}

  void DecodeVector(const void* code, void* x) const override {
    float* fx = static_cast<float*>(x);
    const float* fb = static_cast<const float*>(code);
    for (size_t i = 0; i < dim; i++) {
      fx[i] = fb[i];
    }
  }

  void DecodeVectorDim(const void* code, void* x, size_t decode_dim) const override {
    float* fx = static_cast<float*>(x);
    const float* fb = static_cast<const float*>(code);
    for (size_t i = 0; i < decode_dim; i++) {
      fx[i] = fb[i];
    }
  }

  float DecodeItem(const void* code) override {
    const float* fb = static_cast<const float*>(code);
    return *fb;
  }

  void setQuantizerParams(const char* params) override {}

  void train(int n, const float* x) override {};

  float DecodeDistance(float code_dist) override {
    return code_dist;
  }

  std::string GetQuantizerParams() override {
    return "";
  }

  float GetA() {
    return 1.0f;
  }

 private:
  size_t dim = 0;
};

// 标量量化相关文档：https://docs.corp.kuaishou.com/d/home/fcADKP5ujFHyGoTrJrqAsArSO?ro=true
template <typename T>
class ScalarQuantizer : public BasicQuantizer {
 public:
  using Type = T;

 public:
  ScalarQuantizer(size_t input_dim) : dim(input_dim) {
    // std::cerr << "construction new sq, dim is " << dim << std::endl;
    // lowest 太大了（ -FLT_MAX ），因此将值域减少一半
    vmin = std::numeric_limits<float>::lowest() / 2.0f;
    vdiff = std::numeric_limits<float>::max();
  };

  const void* EncodeVectorDim(const void* inx, size_t decode_dim) const override {
    // std::cerr << "[SQP_DEBUG] sq vmin: " << vmin << ", vdiff: " << vdiff << ", dim:" << dim << std::endl;
    const float* x = reinterpret_cast<const float*>(inx);
    T* tx = new T[decode_dim];
    float params = 0.0f;
    for (int i = 0; i < decode_dim; ++i) {
      params = (x[i] - vmin) / vdiff;
      if (params <= 0.0f) {
        // std::cerr << "[SQ_DEBUG] trigger 0.0" << std::endl;
        params = 0.0f;
      } else if (params >= 1.0f) {
        // std::cerr << "[SQ_DEBUG] trigger 1.0" << std::endl;
        params = 1.0f;
      }
      Codec<T>::encode(params, tx, i);
    }
    return tx;
  }

  const void* EncodeVector(const void* inx) const override {
    // std::cerr << "[SQP_DEBUG] sq vmin: " << vmin << ", vdiff: " << vdiff << ", dim:" << dim << std::endl;
    const float* x = reinterpret_cast<const float*>(inx);
    T* tx = new T[dim];
    float params = 0.0f;
    for (int i = 0; i < dim; ++i) {
      params = (x[i] - vmin) / vdiff;
      if (params <= 0.0f) {
        // std::cerr << "[SQ_DEBUG] trigger 0.0" << std::endl;
        params = 0.0f;
      } else if (params >= 1.0f) {
        // std::cerr << "[SQ_DEBUG] trigger 1.0" << std::endl;
        params = 1.0f;
      }
      Codec<T>::encode(params, tx, i);
    }
    return tx;
  }

  void deleteConvertedVector(const T* tx) const {
    delete[] tx;
  }

  void DecodeVector(const void* code, void* x) const override {
    float* fx = static_cast<float*>(x);
    for (size_t i = 0; i < dim; i++) {
      float xi = Codec<T>::decode((const T*)code, i);
      fx[i] = vmin + xi * vdiff;
    }
  }

  void DecodeVectorDim(const void* code, void* x, size_t decode_dim) const override {
    float* fx = static_cast<float*>(x);
    for (size_t i = 0; i < decode_dim; i++) {
      float xi = Codec<T>::decode((const T*)code, i);
      fx[i] = vmin + xi * vdiff;
    }
  };

  float DecodeItem(const void* code) override {
    float xi = Codec<T>::decode((const T*)code, 0);
    return vmin + xi * vdiff;
  }

  void setQuantizerParams(const char* params) override {
    // std::cerr << "[SQP_DEBUG] set quantizer params begin, params is: " << params << std::endl;
    std::string description(params);
    char* ptr;

    for (char* tok = strtok_r(&description[0], " ,", &ptr); tok; tok = strtok_r(nullptr, " ,", &ptr)) {
      // std::cerr << "[SQP_DEBUG] get one param" << std::endl;
      char name[100];
      double val;
      int ret = sscanf(tok, "%99[^=]=%lf", name, &val);
      if (ret != 2) {
        std::cerr << "[SQP_DEBUG] set quantizer params error" << std::endl;
        continue;
      }
      std::string name_str(name);
      if (name_str == "min") {
        vmin = val;
        need_train = false;
      } else if (name_str == "diff") {
        vdiff = val;
        need_train = false;
      } else if (name_str == "diff_scale") {
        std::cerr << "[SQP_DEBUG] set scale " << val << std::endl;
        diff_scale = val;
      } else if (name_str == "normalize_diff") {
        // 归一化IP距离（cosine），需要使线性变换中的 B==0
        if (val) {
          normalize_diff = true;
        }
      } else {
        // std::cerr << "[SQP_DEBUG] name " << name << std::endl;
      }
    }

    UpdateParamA();
    // std::cerr << "[SQP_DEBUG] sq set quantizer params vmin: " << vmin << ", vdiff: " << vdiff
    //           << ", vdiff_scale:" << diff_scale << ",A" << A << std::endl;
  };

  // 去除计算出的距离中的参数（只适用于 L2距离 和 归一化之后的IP距离）
  float DecodeDistance(float code_dist) override {
    return code_dist / A;
  }

  void train(int n, const float* x) override {
    if (!need_train) {
      return;
    }
    if (n == 0) {
      // std::cerr << "[SQP_DEBUG] train set size if 0, sq tarin result vmin: " << vmin << ", vdiff: " <<
      // vdiff
      //           << std::endl;
      return;
    }

    float train_min = std::numeric_limits<float>::max();
    float train_max = std::numeric_limits<float>::lowest();
    for (int sample = 0; sample < n; ++sample) {
      for (int idx = 0; idx < dim; ++idx) {
        train_min = std::min(train_min, x[sample * dim + idx]);
        train_max = std::max(train_max, x[sample * dim + idx]);
      }
    }

    float train_diff = train_max - train_min;
    train_diff *= diff_scale;
    train_min -= train_diff;
    train_max += train_diff;
    train_diff = train_max - train_min;
    // std::cerr << train_diff << " " << train_max << "  " << train_min << " " << diff_scale << std::endl;
    // 如果 diff 手动设置为0，则表示是归一化IP距离（cosine），需要线性变换 y=Ax+B 中的
    // B==0，才能使得IP距离保持正比 使得min max 互为相反数即可
    if (normalize_diff) {
      train_min = std::max(std::fabs(train_min), std::fabs(train_max));
      train_diff = 2.0 * train_min;
      train_min *= -1.0;
    }

    if (train_diff <= 1e-6) {
      // 训练集合的 min 与 max 差距过小，使用默认值
    } else {
      vdiff = train_diff;
      vmin = train_min;
    }

    UpdateParamA();

    need_train = false;
    // std::cerr << "[SQP_DEBUG] sq tarin result vmin: " << vmin << ", vdiff: " << vdiff << ", A:" << A
    //           << std::endl;
  }

  std::string GetQuantizerParams() override {
    std::stringstream ss;
    ss << std::setprecision(15) << "min=" << vmin << ",diff=" << vdiff << ",diff_scale=" << diff_scale;
    return ss.str();
  }

 private:
  void UpdateParamA() {
    A = Codec<T>::getParams() / vdiff;
    A *= A;
  }

 private:
  size_t dim = 0;
  bool normalize_diff = false;

 public:
  float vmin;
  float vdiff;
  float diff_scale = 0.0f;
  bool need_train = true;
  float A = 0;

 public:
  float GetA() {
    return A;
  }
};

template <>
inline void ScalarQuantizer<int8_t>::DecodeVector(const void* code, void* x) const {
  float* fx = static_cast<float*>(x);
  size_t decode_dim = dim;
  const int8_t* f_code = static_cast<const int8_t*>(code);
  uint32_t f_index = 0;
  static __m256 param1 = _mm256_set1_ps(128.5f);
  static constexpr float param2 = 1.0f / 255.999984741f;
  static __m256 param3 = _mm256_set1_ps(param2);
  __m256 param4 = _mm256_set1_ps(vdiff);
  __m256 param5 = _mm256_set1_ps(vmin);
  while (decode_dim >= 8) {
    __m256 ma =
        _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i_u*)(f_code + f_index))));
    ma = _mm256_mul_ps(_mm256_add_ps(ma, param1), param3);
    ma = _mm256_fmadd_ps(ma, param4, param5);
    _mm256_store_ps(fx, ma);
    fx += 8;
    f_index += 8;
    decode_dim -= 8;
  }
  for (size_t i = 0; i < decode_dim; i++) {
    float xi = (float(f_code[i + f_index]) + 128.5f) * param2;
    fx[i] = vmin + xi * vdiff;
  }
  return;
};

template <>
HNSWLIB_avx512 inline void ScalarQuantizer<int8_t>::DecodeVectorDim(const void* code,
                                                                    void* x,
                                                                    size_t decode_dim) const {
  float* fx = static_cast<float*>(x);
  const int8_t* f_code = static_cast<const int8_t*>(code);
  uint32_t f_index = 0;
  if (cpu_flag_ == 1) {
    // implementation for AVX512
    const __m512 param1 = _mm512_set1_ps(128.5f);
    constexpr float param2 = 1.0f / 255.999984741f;
    const __m512 param3 = _mm512_set1_ps(param2);
    __m512 param4 = _mm512_set1_ps(vdiff);
    __m512 param5 = _mm512_set1_ps(vmin);
    while (decode_dim >= 16) {
      __m512 ma =
          _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_load_si128((const __m128i_u*)(f_code + f_index))));
      ma = _mm512_mul_ps(_mm512_add_ps(ma, param1), param3);
      ma = _mm512_fmadd_ps(ma, param4, param5);
      _mm512_store_ps(fx, ma);
      fx += 16;
      f_index += 16;
      decode_dim -= 16;
    }
    for (size_t i = 0; i < decode_dim; i++) {
      float xi = (float(f_code[i + f_index]) + 128.5f) * param2;
      fx[i] = vmin + xi * vdiff;
    }
    return;
  } else if (cpu_flag_ == 0) {
    // implementation for AVX2
    const __m256 param1 = _mm256_set1_ps(128.5f);
    constexpr float param2 = 1.0f / 255.999984741f;
    const __m256 param3 = _mm256_set1_ps(param2);
    __m256 param4 = _mm256_set1_ps(vdiff);
    __m256 param5 = _mm256_set1_ps(vmin);
    while (decode_dim >= 8) {
      __m256 ma =
          _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i_u*)(f_code + f_index))));
      ma = _mm256_mul_ps(_mm256_add_ps(ma, param1), param3);
      ma = _mm256_fmadd_ps(ma, param4, param5);
      _mm256_store_ps(fx, ma);
      fx += 8;
      f_index += 8;
      decode_dim -= 8;
    }
    for (size_t i = 0; i < decode_dim; i++) {
      float xi = (float(f_code[i + f_index]) + 128.5f) * param2;
      fx[i] = vmin + xi * vdiff;
    }
    return;
  }
};

class Bfp16Quantizer : public BasicQuantizer {
 public:
  using Type = Bfp16;

 public:
  Bfp16Quantizer(size_t _dim) : dim(_dim) {};

 public:
  const void* EncodeVectorDim(const void* inx, size_t decode_dim) const override {
    // std::cerr << "[SQP_DEBUG] sq vmin: " << vmin << ", vdiff: " << vdiff << ", dim:" << dim << std::endl;
    const float* x = reinterpret_cast<const float*>(inx);
    Bfp16* tx = new Bfp16[decode_dim];

    for (int i = 0; i < decode_dim; ++i) {
      tx[i] = Bfp16(x[i]);
    }
    return tx;
  }

  const void* EncodeVector(const void* inx) const override {
    // std::cerr << "[SQP_DEBUG] sq vmin: " << vmin << ", vdiff: " << vdiff << ", dim:" << dim << std::endl;
    const float* x = reinterpret_cast<const float*>(inx);
    Bfp16* tx = new Bfp16[dim];

    for (int i = 0; i < dim; ++i) {
      tx[i] = Bfp16(x[i]);
    }
    return tx;
  }

  void deleteConvertedVector(const Bfp16* tx) const {
    delete[] tx;
  }

  void DecodeVector(const void* code_input, void* x) const override {
    float* fx = static_cast<float*>(x);
    Bfp16* code = (Bfp16*)code_input;
    for (size_t i = 0; i < dim; i++) {
      fx[i] = float(code[i]);
    }
  }

  void DecodeVectorDim(const void* code_input, void* x, size_t decode_dim) const override {
    float* fx = static_cast<float*>(x);
    Bfp16* code = (Bfp16*)code_input;
    for (size_t i = 0; i < decode_dim; i++) {
      fx[i] = float(code[i]);
    }
  }

  float DecodeItem(const void* code) override {
    return static_cast<const float*>(code)[0];
  }

  void setQuantizerParams(const char* params) override {}

  void train(int n, const float* x) override {};

  float DecodeDistance(float code_dist) override {
    return code_dist;
  }

  std::string GetQuantizerParams() override {
    return "";
  }

  float GetA() {
    return 1.0f;
  }

 private:
  size_t dim = 0;
};

class Sefp16Quantizer : public BasicQuantizer {
 public:
  using Type = Sefp16;

 public:
  Sefp16Quantizer(size_t _dim) : dim(_dim) {};

 public:
  const void* EncodeVectorDim(const void* inx, size_t decode_dim) const override {
    // std::cerr << "[SQP_DEBUG] sq vmin: " << vmin << ", vdiff: " << vdiff << ", dim:" << dim << std::endl;
    const float* x = reinterpret_cast<const float*>(inx);
    Sefp16* tx = new Sefp16[decode_dim];

    for (int i = 0; i < decode_dim; ++i) {
      tx[i] = Sefp16(x[i]);
    }
    return tx;
  }

  const void* EncodeVector(const void* inx) const override {
    // std::cerr << "[SQP_DEBUG] sq vmin: " << vmin << ", vdiff: " << vdiff << ", dim:" << dim << std::endl;
    const float* x = reinterpret_cast<const float*>(inx);
    Sefp16* tx = new Sefp16[dim];

    for (int i = 0; i < dim; ++i) {
      tx[i] = Sefp16(x[i]);
    }
    return tx;
  }

  void deleteConvertedVector(const Sefp16* tx) const {
    delete[] tx;
  }

  void DecodeVector(const void* code_input, void* x) const override {
    float* fx = static_cast<float*>(x);
    Sefp16* code = (Sefp16*)code_input;
    for (size_t i = 0; i < dim; i++) {
      fx[i] = float(code[i]);
    }
  }

  void DecodeVectorDim(const void* code_input, void* x, size_t decode_dim) const override {
    float* fx = static_cast<float*>(x);
    Sefp16* code = (Sefp16*)code_input;
    for (size_t i = 0; i < decode_dim; i++) {
      fx[i] = float(code[i]);
    }
  }

  float DecodeItem(const void* code) override {
    return static_cast<const float*>(code)[0];
  }

  void setQuantizerParams(const char* params) override {}

  void train(int n, const float* x) override {};

  float DecodeDistance(float code_dist) override {
    return code_dist;
  }

  std::string GetQuantizerParams() override {
    return "";
  }

  float GetA() {
    return 1.0f;
  }

 private:
  size_t dim = 0;
};

using Int8Quantizer = ScalarQuantizer<int8_t>;
using Int16Quantizer = ScalarQuantizer<int16_t>;
using Bfp16Quantizer = Bfp16Quantizer;
using Sefp16Quantizer = Sefp16Quantizer;

}  // namespace hnswlib
