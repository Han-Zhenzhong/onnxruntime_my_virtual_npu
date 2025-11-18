// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace my_cpu {

/**
 * FastGELU operator - Basic CPU implementation for float
 *
 * Computes: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
 *
 * This is a basic scalar implementation focused on correctness.
 * Optimization opportunities are marked with TODO-OPTIMIZE comments.
 */
class FastGelu final : public OpKernel {
 public:
  FastGelu(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* context) const override;

 private:
  // Basic scalar implementation
  void ComputeGeluScalar(const float* input, float* output, size_t count) const;

  // Helper function for single value
  inline float ComputeGeluValue(float x) const;

  // TODO-OPTIMIZE: [SIMD] AVX2 vectorized implementation
  // Expected speedup: 4-8x for float32
  // void ComputeGeluAVX2(const float* input, float* output, size_t count) const;

  // TODO-OPTIMIZE: [SIMD] SSE implementation for older CPUs
  // void ComputeGeluSSE(const float* input, float* output, size_t count) const;
};

}  // namespace my_cpu
}  // namespace onnxruntime
