// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/my_cpu/bert/fast_gelu.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/graph/constants.h"
#include <cmath>

namespace onnxruntime {
namespace my_cpu {

Status FastGelu::Compute(OpKernelContext* context) const {
  // 1. Get input tensor
  const Tensor* input = context->Input<Tensor>(0);
  if (input == nullptr) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Input tensor is null");
  }

  const float* input_data = input->Data<float>();
  const auto& input_shape = input->Shape();
  const size_t count = static_cast<size_t>(input_shape.Size());

  // 2. Allocate output tensor with same shape as input
  Tensor* output = context->Output(0, input_shape);
  float* output_data = output->MutableData<float>();

  // 3. Optional bias input (for BiasGelu fusion - future optimization)
  const Tensor* bias_tensor = context->Input<Tensor>(1);
  const float* bias_data = bias_tensor ? bias_tensor->Data<float>() : nullptr;

  // 4. Compute GELU
  // TODO-OPTIMIZE: [Parallel] Use OpenMP for large tensors (count > 1024)
  // #pragma omp parallel for if(count > 1024)
  if (bias_data) {
    // BiasGelu: GELU(x + bias)
    const size_t bias_size = static_cast<size_t>(bias_tensor->Shape().Size());
    for (size_t i = 0; i < count; ++i) {
      float x = input_data[i] + bias_data[i % bias_size];
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
void FastGelu::ComputeGeluScalar(const float* input, float* output, size_t count) const {
  constexpr float kAlpha = 0.7978845608028654f;  // sqrt(2/π)
  constexpr float kBeta = 0.044715f;
  constexpr float kHalf = 0.5f;

  // TODO-OPTIMIZE: [SIMD] This loop can be vectorized with AVX2
  // Process 8 floats at once, expected speedup: 6-8x
  for (size_t i = 0; i < count; ++i) {
    float x = input[i];
    float x_cubed = x * x * x;
    float inner = kAlpha * (x + kBeta * x_cubed);
    float tanh_val = std::tanh(inner);
    output[i] = kHalf * x * (1.0f + tanh_val);
  }
}

inline float FastGelu::ComputeGeluValue(float x) const {
  constexpr float kAlpha = 0.7978845608028654f;  // sqrt(2/π)
  constexpr float kBeta = 0.044715f;
  constexpr float kHalf = 0.5f;

  float x_cubed = x * x * x;
  float inner = kAlpha * (x + kBeta * x_cubed);
  float tanh_val = std::tanh(inner);
  return kHalf * x * (1.0f + tanh_val);
}

}  // namespace my_cpu

// Register kernel - must be in onnxruntime namespace, not my_cpu
// The macro creates template specialization that must be in onnxruntime namespace
// Use custom domain to avoid conflict with existing FastGelu in contrib_ops
ONNX_OPERATOR_KERNEL_EX(
    FastGelu,
    kMyCustomDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    my_cpu::FastGelu);

}  // namespace onnxruntime
