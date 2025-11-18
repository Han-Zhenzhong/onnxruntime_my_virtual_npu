// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/my_cpu/my_cpu_kernels.h"
#include "core/providers/my_cpu/bert/fast_gelu.h"
#include "core/framework/op_kernel.h"

// Forward declare the kernel class created by ONNX_OPERATOR_KERNEL_EX macro
// This class is defined in fast_gelu.cc using kMyCustomDomain
namespace onnxruntime {
class kCpuExecutionProvider_FastGelu_kMyCustomDomain_ver1;
}

namespace onnxruntime {
namespace my_cpu {

Status RegisterMyCpuKernels(KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn function_table[] = {
      // FastGelu operator (using non-TYPED macro version)
      // Class is forward declared above and defined in fast_gelu.cc using kMyCustomDomain
      ::onnxruntime::BuildKernelCreateInfo<::onnxruntime::kCpuExecutionProvider_FastGelu_kMyCustomDomain_ver1>,

      // TODO-OPTIMIZE: [Fusion] Add fused operators for better performance
      // Use kMyCustomDomain to avoid conflicts with existing operators
      // BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(
      //     kCpuExecutionProvider, kMyCustomDomain, 1, SkipLayerNormalization)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(
      //     kCpuExecutionProvider, kMyCustomDomain, 1, BiasGelu)>,
  };

  for (auto& function : function_table) {
    ORT_RETURN_IF_ERROR(kernel_registry.Register(function()));
  }

  return Status::OK();
}

}  // namespace my_cpu
}  // namespace onnxruntime
