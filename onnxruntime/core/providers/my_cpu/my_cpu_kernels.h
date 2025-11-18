// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/kernel_registry.h"

namespace onnxruntime {
namespace my_cpu {

/**
 * Register custom CPU kernels for Tiny-GPT2
 *
 * This registers all custom operators in the my_cpu namespace:
 * - FastGelu: Fast GELU activation with tanh approximation
 * - (Future) SkipLayerNormalization: Fused residual + layer norm
 * - (Future) BiasGelu: Fused bias + GELU
 */
Status RegisterMyCpuKernels(KernelRegistry& kernel_registry);

}  // namespace my_cpu
}  // namespace onnxruntime
