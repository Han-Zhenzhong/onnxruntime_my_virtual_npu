// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cuda_runtime.h>
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/shared_inc/cuda_call.h"
#include "core/providers/my_virtual_npu/cuda/fast_gelu_impl.h"

namespace onnxruntime {
namespace my_virtual_npu {
namespace cuda {

// Fast GELU approximation constants
constexpr float A = 0.5f;
constexpr float B = 0.7978845608028654f;  // sqrt(2.0/M_PI)
constexpr float C = 0.035677408136300125f;  // 0.044715 * sqrt(2.0/M_PI)

template <typename T, unsigned TPB>
__global__ void FastGeluKernel(
    const T a,
    const T b,
    const T c,
    int input_length,
    int bias_length,
    const T* input,
    const T* bias,
    T* output) {
  const int idx = blockIdx.x * TPB + threadIdx.x;

  if (idx < input_length) {
    const T x = input[idx];
    const T in = (bias == nullptr) ? x : (T)(x + bias[idx % bias_length]);
    // GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    const T cdf = a + a * _Tanh(in * (c * in * in + b));
    output[idx] = in * cdf;
  }
}

// Optimized half2 kernel for FP16
template <unsigned TPB>
__global__ void FastGeluKernelHalf2(
    const half2 a,
    const half2 b,
    const half2 c,
    int input_length,
    int bias_length,
    const half2* input,
    const half2* bias,
    half2* output) {
  const int idx = blockIdx.x * TPB + threadIdx.x;

  if (idx < input_length) {
    const half2 x = input[idx];
    const half2 in = (bias == nullptr) ? x : (x + bias[idx % bias_length]);
    const half2 cdf = a + a * _Tanh(in * (c * in * in + b));
    output[idx] = in * cdf;
  }
}

// FP32 specialization
template <>
Status LaunchFastGeluKernel<float>(
    cudaStream_t stream,
    int input_length,
    int bias_length,
    const float* input,
    const float* bias,
    float* output) {
  constexpr int blockSize = 256;
  const int gridSize = (input_length + blockSize - 1) / blockSize;

  FastGeluKernel<float, blockSize><<<gridSize, blockSize, 0, stream>>>(
      A, B, C, input_length, bias_length, input, bias, output);

  return CUDA_CALL(cudaGetLastError());
}

// FP64 specialization
template <>
Status LaunchFastGeluKernel<double>(
    cudaStream_t stream,
    int input_length,
    int bias_length,
    const double* input,
    const double* bias,
    double* output) {
  constexpr int blockSize = 256;
  const int gridSize = (input_length + blockSize - 1) / blockSize;

  FastGeluKernel<double, blockSize><<<gridSize, blockSize, 0, stream>>>(
      A, B, C, input_length, bias_length, input, bias, output);

  return CUDA_CALL(cudaGetLastError());
}

// FP16 (half) specialization with half2 optimization
template <>
Status LaunchFastGeluKernel<half>(
    cudaStream_t stream,
    int input_length,
    int bias_length,
    const half* input,
    const half* bias,
    half* output) {
  constexpr int blockSize = 256;

  // Use half2 optimization if data is aligned and we have compute capability >= 7.0
  if ((input_length % 2 == 0) && (bias == nullptr || bias_length % 2 == 0)) {
    const int n = input_length / 2;
    const int gridSize = (n + blockSize - 1) / blockSize;

    const half2 A2 = __floats2half2_rn(A, A);
    const half2 B2 = __floats2half2_rn(B, B);
    const half2 C2 = __floats2half2_rn(C, C);

    const half2* input2 = reinterpret_cast<const half2*>(input);
    const half2* bias2 = reinterpret_cast<const half2*>(bias);
    half2* output2 = reinterpret_cast<half2*>(output);

    FastGeluKernelHalf2<blockSize><<<gridSize, blockSize, 0, stream>>>(
        A2, B2, C2, n, bias_length / 2, input2, bias2, output2);
  } else {
    // Fall back to scalar half
    const int gridSize = (input_length + blockSize - 1) / blockSize;

    FastGeluKernel<half, blockSize><<<gridSize, blockSize, 0, stream>>>(
        A, B, C, input_length, bias_length, input, bias, output);
  }

  return CUDA_CALL(cudaGetLastError());
}

// BFloat16 specialization
template <>
Status LaunchFastGeluKernel<BFloat16>(
    cudaStream_t stream,
    int input_length,
    int bias_length,
    const BFloat16* input,
    const BFloat16* bias,
    BFloat16* output) {
  constexpr int blockSize = 256;
  const int gridSize = (input_length + blockSize - 1) / blockSize;

  FastGeluKernel<BFloat16, blockSize><<<gridSize, blockSize, 0, stream>>>(
      A, B, C, input_length, bias_length, input, bias, output);

  return CUDA_CALL(cudaGetLastError());
}

}  // namespace cuda
}  // namespace my_virtual_npu
}  // namespace onnxruntime
