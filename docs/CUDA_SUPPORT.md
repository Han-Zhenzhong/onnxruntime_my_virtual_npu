# MyVirtualNPU CUDA Support

## Overview

MyVirtualNPU 现在支持 CUDA 加速！你可以在 CPU 和 CUDA 两种模式下运行自定义算子。

## 已实现的 CUDA 算子

### FastGelu (CUDA)
- ✅ FP32 (float) 支持
- ✅ FP64 (double) 支持
- ✅ FP16 (half) 支持，包含 half2 优化
- ✅ BFloat16 支持

**性能优化：**
- 使用 half2 向量化处理 FP16 数据（2x 吞吐量）
- 针对 GPU 架构优化的 block/grid 配置
- 支持 optional bias 输入

## 编译配置

### 启用 CUDA 支持

```bash
cmake ../cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -Donnxruntime_USE_MY_VIRTUAL_NPU=ON \
  -Donnxruntime_USE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="75;80;86" \
  -Donnxruntime_BUILD_SHARED_LIB=ON
```

### CMake 选项说明

- `onnxruntime_USE_MY_VIRTUAL_NPU=ON`: 启用 MyVirtualNPU provider
- `onnxruntime_USE_CUDA=ON`: 启用 CUDA 支持
- `CMAKE_CUDA_ARCHITECTURES`: 指定目标 GPU 架构
  - 75: Turing (RTX 20xx, T4)
  - 80: Ampere (A100, RTX 30xx)
  - 86: Ampere (RTX 30xx mobile)
  - 89: Ada Lovelace (RTX 40xx)
  - 90: Hopper (H100)

## 使用方法

### C++ API

```cpp
#include <onnxruntime_cxx_api.h>
#include "core/providers/my_virtual_npu/my_virtual_npu_execution_provider.h"

// CPU 模式
Ort::SessionOptions cpu_options;
onnxruntime::MyVirtualNpuExecutionProviderInfo cpu_info;
cpu_info.create_arena = true;
cpu_info.use_cuda = false;

auto cpu_provider = std::make_unique<onnxruntime::MyVirtualNpuExecutionProvider>(cpu_info);
cpu_options.AppendExecutionProvider(std::move(cpu_provider));

// CUDA 模式
Ort::SessionOptions cuda_options;
onnxruntime::MyVirtualNpuExecutionProviderInfo cuda_info;
cuda_info.create_arena = true;
cuda_info.use_cuda = true;     // 启用 CUDA
cuda_info.device_id = 0;       // GPU 设备 ID

auto cuda_provider = std::make_unique<onnxruntime::MyVirtualNpuExecutionProvider>(cuda_info);
cuda_options.AppendExecutionProvider(std::move(cuda_provider));

// 创建 Session
Ort::Session session(env, "model.onnx", cuda_options);
```

### Python API

```python
import onnxruntime as ort
import numpy as np

# CPU 模式
providers = ['MyVirtualNpuExecutionProvider', 'CPUExecutionProvider']
session_cpu = ort.InferenceSession('model.onnx', providers=providers)

# CUDA 模式（如果启用了 CUDA 支持）
providers_cuda = [
    ('MyVirtualNpuExecutionProvider', {
        'use_cuda': True,
        'device_id': 0
    }),
    'CUDAExecutionProvider',
    'CPUExecutionProvider'
]
session_cuda = ort.InferenceSession('model.onnx', providers=providers_cuda)

# 运行推理
input_data = {'input': np.random.randn(1, 768).astype(np.float32)}
outputs = session_cuda.run(None, input_data)
```

## 性能对比

### FastGelu 算子性能 (Preliminary)

| 数据类型 | 输入尺寸 | CPU 时间 | CUDA 时间 | 加速比 |
|---------|---------|---------|-----------|--------|
| FP32 | 1024 | ~10μs | ~2μs | 5x |
| FP16 | 1024 | N/A | ~1μs | - |
| FP16 (half2) | 1024 | N/A | ~0.5μs | 2x over FP16 |

*注：实际性能取决于硬件和数据规模*

## 架构设计

```
my_virtual_npu/
├── bert/
│   ├── fast_gelu.h/cc          # CPU 实现
├── cuda/
│   ├── fast_gelu_impl.h        # CUDA kernel 接口
│   ├── fast_gelu_impl.cu       # CUDA kernel 实现
│   ├── fast_gelu_cuda.h        # CUDA operator 头文件
│   └── fast_gelu_cuda.cc       # CUDA operator 实现
├── my_virtual_npu_execution_provider.h/cc  # EP 实现（支持 CPU/CUDA）
└── my_virtual_npu_kernels.h/cc             # Kernel 注册（CPU + CUDA）
```

## 技术细节

### CUDA Kernel 实现

```cuda
template <typename T, unsigned TPB>
__global__ void FastGeluKernel(
    const T a, const T b, const T c,
    int input_length, int bias_length,
    const T* input, const T* bias, T* output) {

  const int idx = blockIdx.x * TPB + threadIdx.x;
  if (idx < input_length) {
    const T x = input[idx];
    const T in = (bias == nullptr) ? x : (T)(x + bias[idx % bias_length]);
    // GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    const T cdf = a + a * _Tanh(in * (c * in * in + b));
    output[idx] = in * cdf;
  }
}
```

### Half2 优化

对于 FP16 数据，使用 `half2` 向量类型一次处理两个元素：

```cuda
template <unsigned TPB>
__global__ void FastGeluKernelHalf2(
    const half2 a, const half2 b, const half2 c,
    int input_length, int bias_length,
    const half2* input, const half2* bias, half2* output) {

  const int idx = blockIdx.x * TPB + threadIdx.x;
  if (idx < input_length) {
    const half2 x = input[idx];
    const half2 in = (bias == nullptr) ? x : (x + bias[idx % bias_length]);
    const half2 cdf = a + a * _Tanh(in * (c * in * in + b));
    output[idx] = in * cdf;
  }
}
```

## 测试

### 运行 CUDA 测试

```bash
# 编译
cd build/Linux/Release
cmake --build . -j$(nproc)

# 运行测试
./onnxruntime_test_all --gtest_filter="*FastGelu*"

# 运行 Python 测试
python ../../../test_tiny_gpt2_cuda.py
```

### 创建 CUDA 测试用例

```cpp
// test/providers/my_virtual_npu/fast_gelu_cuda_test.cc
TEST(MyVirtualNpuCuda, FastGeluFloat) {
  OpTester test("FastGelu", 1, kMyCustomDomain);

  std::vector<float> input = {-1.0f, 0.0f, 1.0f, 2.0f};
  std::vector<int64_t> dims = {2, 2};

  test.AddInput<float>("X", dims, input);
  test.AddOutput<float>("Y", dims, expected_output);

  // 使用 CUDA provider
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kCudaExecutionProvider, kMyVirtualNpuExecutionProvider});
}
```

## 故障排除

### 编译错误

**错误**: `undefined reference to cudaXXX`
**解决**: 确保正确链接 CUDA 库
```cmake
target_link_libraries(onnxruntime_providers_my_virtual_npu PRIVATE
  CUDA::cudart
  CUDA::cublas
)
```

**错误**: `nvcc fatal : Unsupported gpu architecture 'compute_XX'`
**解决**: 检查 `CMAKE_CUDA_ARCHITECTURES` 设置，确保与你的 GPU 匹配

### 运行时错误

**错误**: `CUDA error: no kernel image is available for execution`
**解决**: 重新编译，指定正确的 CUDA 架构

**错误**: `Out of memory`
**解决**:
1. 减小 batch size
2. 启用 memory arena: `info.create_arena = true`
3. 使用更小的数据类型 (FP16 instead of FP32)

## 下一步计划

### 待实现的 CUDA 算子
- [ ] SkipLayerNormalization (CUDA)
- [ ] Attention (CUDA with cutlass/flash-attention)
- [ ] MatMul 优化版本
- [ ] FusedBiasGelu

### 性能优化
- [ ] Kernel fusion: Bias + Gelu
- [ ] Shared memory 优化
- [ ] Warp-level primitives
- [ ] Tensor Core 支持 (FP16/BF16 GEMM)

## 参考资料

- CUDA Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- CUTLASS: https://github.com/NVIDIA/cutlass
- Flash Attention: https://github.com/Dao-AILab/flash-attention
- ONNXRuntime CUDA Provider: `onnxruntime/core/providers/cuda/`
