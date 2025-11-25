// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/my_virtual_npu/my_virtual_npu_kernels.h"
#include "core/providers/my_virtual_npu/bert/fast_gelu.h"
#include "core/framework/op_kernel.h"

#ifdef USE_CUDA
#include "core/providers/my_virtual_npu/cuda/fast_gelu_cuda.h"
#endif

// Forward declare the kernel class created by ONNX_OPERATOR_KERNEL_EX macro
// This class is defined in fast_gelu.cc using kMyCustomDomain
namespace onnxruntime {
class kCpuExecutionProvider_FastGelu_kMyCustomDomain_ver1;
}

namespace onnxruntime {
namespace my_virtual_npu {

Status RegisterMyVirtualNpuKernels(KernelRegistry& kernel_registry) {
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

#ifdef USE_CUDA
// CUDA kernel registration
Status RegisterMyVirtualNpuCudaKernels(KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn function_table[] = {
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kCudaExecutionProvider, kMyCustomDomain, 1, float, FastGeluCuda)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kCudaExecutionProvider, kMyCustomDomain, 1, double, FastGeluCuda)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kCudaExecutionProvider, kMyCustomDomain, 1, MLFloat16, FastGeluCuda)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kCudaExecutionProvider, kMyCustomDomain, 1, BFloat16, FastGeluCuda)>,
  };

  for (auto& function : function_table) {
    ORT_RETURN_IF_ERROR(kernel_registry.Register(function()));
  }

  return Status::OK();
}
#endif

}  // namespace my_virtual_npu
}  // namespace onnxruntime
