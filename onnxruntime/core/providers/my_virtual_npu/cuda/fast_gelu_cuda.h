// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace my_virtual_npu {

using namespace onnxruntime::cuda;

template <typename T>
class FastGeluCuda final : public CudaKernel {
 public:
  FastGeluCuda(const OpKernelInfo& op_kernel_info) : CudaKernel(op_kernel_info) {}

  Status ComputeInternal(OpKernelContext* ctx) const override;
};

}  // namespace my_virtual_npu
}  // namespace onnxruntime
