// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"

namespace onnxruntime {
namespace my_virtual_npu {
namespace cuda {

template <typename T>
Status LaunchFastGeluKernel(
    cudaStream_t stream,
    int input_length,
    int bias_length,
    const T* input,
    const T* bias,
    T* output);

}  // namespace cuda
}  // namespace my_virtual_npu
}  // namespace onnxruntime
