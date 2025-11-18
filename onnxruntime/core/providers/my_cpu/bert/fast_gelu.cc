// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/my_cpu/bert/fast_gelu.h"
#include "core/providers/cpu/tensor/utils.h"
#include <cmath>

namespace onnxruntime {
namespace my_cpu {

template <typename T>
Status FastGelu<T>::Compute(OpKernelContext* context) const {
  // 1. Get input tensor
  const Tensor* input = context->Input<Tensor>(0);
  if (input == nullptr) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Input tensor is null");
  }

  const T* input_data = input->Data<T>();
  const auto& input_shape = input->Shape();
  const size_t count = static_cast<size_t>(input_shape.Size());

  // 2. Allocate output tensor with same shape as input
  Tensor* output = context->Output(0, input_shape);
  T* output_data = output->MutableData<T>();

  // 3. Optional bias input (for BiasGelu fusion - future optimization)
  const Tensor* bias_tensor = context->Input<Tensor>(1);
  const T* bias_data = bias_tensor ? bias_tensor->Data<T>() : nullptr;

  // 4. Compute GELU
  // TODO-OPTIMIZE: [Parallel] Use OpenMP for large tensors (count > 1024)
  // #pragma omp parallel for if(count > 1024)
  if (bias_data) {
    // BiasGelu: GELU(x + bias)
    const size_t bias_size = static_cast<size_t>(bias_tensor->Shape().Size());
    for (size_t i = 0; i < count; ++i) {
      T x = input_data[i] + bias_data[i % bias_size];
      output_data[i] = ComputeGeluValue(x);
    }
  } else {
    // Standard GELU
    ComputeGeluScalar(input_data, output_data, count);
  }

  return Status::OK();
}

// GELU approximation using tanh
// Formula: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
template <typename T>
void FastGelu<T>::ComputeGeluScalar(const T* input, T* output, size_t count) const {
  constexpr T kAlpha = static_cast<T>(0.7978845608028654);  // sqrt(2/π)
  constexpr T kBeta = static_cast<T>(0.044715);
  constexpr T kHalf = static_cast<T>(0.5);

  // TODO-OPTIMIZE: [SIMD] This loop can be vectorized with AVX2
  // Process 8 floats at once, expected speedup: 6-8x
  // See ComputeGeluAVX2() for implementation
  for (size_t i = 0; i < count; ++i) {
    T x = input[i];
    T x_cubed = x * x * x;
    T inner = kAlpha * (x + kBeta * x_cubed);
    T tanh_inner = std::tanh(inner);
    output[i] = kHalf * x * (static_cast<T>(1.0) + tanh_inner);
  }
}

// Helper function for computing single GELU value
template <typename T>
inline T FastGelu<T>::ComputeGeluValue(T x) const {
  constexpr T kAlpha = static_cast<T>(0.7978845608028654);
  constexpr T kBeta = static_cast<T>(0.044715);
  constexpr T kHalf = static_cast<T>(0.5);

  T x_cubed = x * x * x;
  T inner = kAlpha * (x + kBeta * x_cubed);
  T tanh_inner = std::tanh(inner);
  return kHalf * x * (static_cast<T>(1.0) + tanh_inner);
}

// TODO-OPTIMIZE: [SIMD] AVX2 optimized implementation
/*
#ifdef __AVX2__
#include <immintrin.h>

template <>
void FastGelu<float>::ComputeGeluAVX2(const float* input, float* output, size_t count) const {
  const size_t vec_count = count / 8;
  const size_t remainder = count % 8;

  const __m256 kAlpha = _mm256_set1_ps(0.7978845608028654f);
  const __m256 kBeta = _mm256_set1_ps(0.044715f);
  const __m256 kHalf = _mm256_set1_ps(0.5f);
  const __m256 kOne = _mm256_set1_ps(1.0f);

  for (size_t i = 0; i < vec_count; ++i) {
    __m256 x = _mm256_loadu_ps(input + i * 8);
    __m256 x_squared = _mm256_mul_ps(x, x);
    __m256 x_cubed = _mm256_mul_ps(x_squared, x);

    // inner = alpha * (x + beta * x^3)
    __m256 beta_x_cubed = _mm256_mul_ps(kBeta, x_cubed);
    __m256 sum = _mm256_add_ps(x, beta_x_cubed);
    __m256 inner = _mm256_mul_ps(kAlpha, sum);

    // tanh approximation or use fast tanh
    // tanh_val = tanh(inner)
    // For production, use a vectorized tanh approximation

    // output = 0.5 * x * (1 + tanh_val)
    // ...

    _mm256_storeu_ps(output + i * 8, result);
  }

  // Handle remaining elements
  ComputeGeluScalar(input + vec_count * 8, output + vec_count * 8, remainder);
}
#endif
*/

// Register kernel using ONNX_OPERATOR_KERNEL_EX macro
// Using non-TYPED version to avoid template parsing issues in macro
ONNX_OPERATOR_KERNEL_EX(
    FastGelu,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    FastGelu<float>);

// TODO: Add float16 support when needed
// ONNX_OPERATOR_KERNEL_EX(
//     FastGelu,
//     kMSDomain,
//     1,
//     kCpuExecutionProvider,
//     KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<MLFloat16>()),
//     FastGelu<MLFloat16>);

}  // namespace my_cpu
}  // namespace onnxruntime
