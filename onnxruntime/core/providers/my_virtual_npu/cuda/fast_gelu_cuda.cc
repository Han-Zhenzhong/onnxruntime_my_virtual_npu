// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/my_virtual_npu/cuda/fast_gelu_cuda.h"
#include "core/providers/my_virtual_npu/cuda/fast_gelu_impl.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"

namespace onnxruntime {
namespace my_virtual_npu {

template <typename T>
Status FastGeluCuda<T>::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* input = ctx->Input<Tensor>(0);
  const Tensor* bias = ctx->Input<Tensor>(1);  // Optional bias

  const auto& input_shape = input->Shape();
  Tensor* output = ctx->Output(0, input_shape);

  const T* input_data = input->Data<T>();
  T* output_data = output->MutableData<T>();

  int input_length = static_cast<int>(input_shape.Size());
  int bias_length = 0;
  const T* bias_data = nullptr;

  if (bias != nullptr) {
    const auto& bias_shape = bias->Shape();
    bias_length = static_cast<int>(bias_shape.Size());
    bias_data = bias->Data<T>();
  }

  return cuda::LaunchFastGeluKernel<T>(
      Stream(ctx),
      input_length,
      bias_length,
      input_data,
      bias_data,
      output_data);
}

// Explicit template instantiation
template class FastGeluCuda<float>;
template class FastGeluCuda<double>;
template class FastGeluCuda<MLFloat16>;
template class FastGeluCuda<BFloat16>;

}  // namespace my_virtual_npu
}  // namespace onnxruntime
