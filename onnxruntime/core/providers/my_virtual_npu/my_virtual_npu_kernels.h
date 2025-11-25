// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/kernel_registry.h"

namespace onnxruntime {
namespace my_virtual_npu {

/**
 * Register custom virtual NPU kernels for Tiny-GPT2
 *
 * This registers all custom operators in the my_virtual_npu namespace:
 * - FastGelu: Fast GELU activation with tanh approximation (CPU)
 * - (Future) SkipLayerNormalization: Fused residual + layer norm
 * - (Future) BiasGelu: Fused bias + GELU
 */
Status RegisterMyVirtualNpuKernels(KernelRegistry& kernel_registry);

#ifdef USE_CUDA
/**
 * Register CUDA implementations of my_virtual_npu kernels
 *
 * This registers CUDA versions of custom operators:
 * - FastGeluCuda: Fast GELU activation (CUDA optimized with half2 support)
 */
Status RegisterMyVirtualNpuCudaKernels(KernelRegistry& kernel_registry);
#endif

}  // namespace my_virtual_npu
}  // namespace onnxruntime
