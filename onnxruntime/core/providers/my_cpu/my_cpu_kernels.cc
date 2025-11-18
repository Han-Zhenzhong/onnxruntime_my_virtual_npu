// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "my_cpu/my_cpu_kernels.h"
#include "my_cpu/bert/fast_gelu.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace my_cpu {

// Define operator kernel classes
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSDomain, 1, float, FastGelu);

// TODO: Add more operators as needed
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSDomain, 1, float, SkipLayerNormalization);

Status RegisterMyCpuKernels(KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn function_table[] = {
      // FastGelu operator
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kCpuExecutionProvider, kMSDomain, 1, float, FastGelu)>,

      // TODO-OPTIMIZE: [Fusion] Add fused operators for better performance
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
      //     kCpuExecutionProvider, kMSDomain, 1, float, SkipLayerNormalization)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
      //     kCpuExecutionProvider, kMSDomain, 1, float, BiasGelu)>,
  };

  for (auto& function : function_table) {
    ORT_RETURN_IF_ERROR(kernel_registry.Register(function()));
  }

  return Status::OK();
}

}  // namespace my_cpu
}  // namespace onnxruntime
