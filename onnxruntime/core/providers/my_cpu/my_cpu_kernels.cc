// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/my_cpu/my_cpu_kernels.h"
#include "core/providers/my_cpu/bert/fast_gelu.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace my_cpu {

Status RegisterMyCpuKernels(KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn function_table[] = {
      // FastGelu operator (using non-TYPED macro version)
      // Class is defined in onnxruntime namespace by ONNX_OPERATOR_KERNEL_EX macro
      ::onnxruntime::BuildKernelCreateInfo<::onnxruntime::ONNX_OPERATOR_KERNEL_CLASS_NAME(
          kCpuExecutionProvider, kMSDomain, 1, FastGelu)>,

      // TODO-OPTIMIZE: [Fusion] Add fused operators for better performance
      // BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(
      //     kCpuExecutionProvider, kMSDomain, 1, SkipLayerNormalization)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(
      //     kCpuExecutionProvider, kMSDomain, 1, BiasGelu)>,
  };

  for (auto& function : function_table) {
    ORT_RETURN_IF_ERROR(kernel_registry.Register(function()));
  }

  return Status::OK();
}

}  // namespace my_cpu
}  // namespace onnxruntime
