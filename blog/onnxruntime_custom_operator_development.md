# ONNXRuntime 自定义算子开发实战：基于 Virtual NPU 的 FastGelu 实现

## 前言

在深度学习模型部署过程中，我们经常需要为特定硬件实现自定义算子以获得更好的性能。本文将详细介绍如何在 ONNXRuntime 中开发自定义算子，以 FastGelu 算子为例，展示从算子注册、内核实现到单元测试、大模型验证的完整流程。

## 项目背景

ONNXRuntime 是微软开源的高性能推理引擎，支持多种硬件后端。本文基于 ONNXRuntime 1.20.0，实现了一个虚拟 NPU 执行提供器（my_virtual_npu provider），用于演示自定义算子的开发流程。

**技术栈：**
- ONNXRuntime 1.20.0
- C++17
- CMake 构建系统
- Python 3.10
- 自定义域名：`com.my_virtual_npu`

## 一、架构设计

### 1.1 自定义域与算子注册

为了避免与 ONNXRuntime 内置算子冲突，我们使用自定义域名：

```cpp
// onnxruntime/core/providers/my_virtual_npu/my_virtual_npu_defs.h
namespace onnxruntime {
namespace contrib {

constexpr const char* kMyCustomDomain = "com.my_virtual_npu";

// 注册自定义域的所有算子 Schema
void RegisterMyVirtualNpuSchemas();

}  // namespace contrib
}  // namespace onnxruntime
```

### 1.2 算子 Schema 定义

Schema 定义了算子的接口规范，包括输入输出、类型约束等：

```cpp
// onnxruntime/core/providers/my_virtual_npu/my_virtual_npu_defs.cc
#include <onnx/defs/schema.h>
#include "onnxruntime/core/graph/constants.h"

namespace onnxruntime {
namespace contrib {

static bool my_virtual_npu_schemas_registered = false;

void RegisterMyVirtualNpuSchemas() {
    // 使用静态标志保证只注册一次（幂等性）
    if (my_virtual_npu_schemas_registered) {
        return;
    }

    // 注册自定义域
    ONNX_NAMESPACE::OpSchemaRegistry::DomainToVersionRange domain_to_version;
    domain_to_version[kMyCustomDomain] = std::make_pair(1, 1);
    ONNX_NAMESPACE::RegisterSchema::Register(domain_to_version);

    // 注册 FastGelu 算子 Schema
    ONNX_NAMESPACE::ONNX_OPERATOR_SET_SCHEMA_EX(
        FastGelu,
        kMyCustomDomain,
        1,
        false,  // 不允许重复注册
        OpSchema()
            .SetDoc("Fast Gaussian Error Linear Unit: y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))")
            .Input(0, "X", "Input tensor", "T")
            .Output(0, "Y", "Output tensor", "T")
            .TypeConstraint(
                "T",
                {"tensor(float)", "tensor(float16)"},
                "Constrain input and output types to float tensors.")
            .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
                propagateElemTypeFromInputToOutput(ctx, 0, 0);
                if (hasInputShape(ctx, 0)) {
                    propagateShapeFromInputToOutput(ctx, 0, 0);
                }
            }));

    my_virtual_npu_schemas_registered = true;
}

}  // namespace contrib
}  // namespace onnxruntime
```

**关键点说明：**
- 使用 `ONNX_OPERATOR_SET_SCHEMA_EX` 宏注册 Schema，需要传递 6 个参数
- `false` 参数确保不允许重复注册，避免冲突
- `TypeAndShapeInferenceFunction` 用于类型和形状推导
- 静态标志 `my_virtual_npu_schemas_registered` 保证幂等性

### 1.3 Schema 注册时机

Schema 必须在 ONNXRuntime 初始化时注册：

```cpp
// onnxruntime/core/session/environment.cc
#include "core/providers/my_virtual_npu/my_virtual_npu_defs.h"

Status Environment::Create(std::unique_ptr<logging::LoggingManager> logging_manager,
                          std::unique_ptr<Environment>& environment,
                          const OrtThreadingOptions* tp_options,
                          bool create_global_thread_pools) {
    // ... 其他初始化代码 ...

    // 注册自定义域的 Schema
    contrib::RegisterMyVirtualNpuSchemas();

    // ... 其他初始化代码 ...
}
```

## 二、算子内核实现

### 2.1 FastGelu 数学原理

FastGelu 是 GELU 激活函数的快速近似实现：

$$
\text{GELU}(x) = 0.5 \times x \times \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} \times (x + 0.044715 \times x^3)\right)\right)
$$

该激活函数在 GPT、BERT 等 Transformer 模型中广泛使用。

### 2.2 内核实现代码

```cpp
// onnxruntime/core/providers/my_virtual_npu/nn/fast_gelu.h
#pragma once
#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

template <typename T>
class FastGelu final : public OpKernel {
 public:
  FastGelu(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* context) const override;
};

}  // namespace contrib
}  // namespace onnxruntime
```

```cpp
// onnxruntime/core/providers/my_virtual_npu/nn/fast_gelu.cc
#include "fast_gelu.h"
#include "core/providers/cpu/nn/gelu_approximation.h"

namespace onnxruntime {
namespace contrib {

template <typename T>
Status FastGelu<T>::Compute(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const T* input_data = input->Data<T>();

  Tensor* output = context->Output(0, input->Shape());
  T* output_data = output->MutableData<T>();

  const auto& shape = input->Shape();
  int64_t total_elements = shape.Size();

  // 使用标量计算（可优化为 SIMD）
  for (int64_t i = 0; i < total_elements; i++) {
    output_data[i] = ComputeGeluScalar(input_data[i]);
  }

  return Status::OK();
}

// 显式实例化
template class FastGelu<float>;

}  // namespace contrib
}  // namespace onnxruntime
```

### 2.3 内核注册

```cpp
// onnxruntime/core/providers/my_virtual_npu/my_virtual_npu_kernels.cc
#include "core/framework/op_kernel.h"
#include "nn/fast_gelu.h"
#include "my_virtual_npu_defs.h"

namespace onnxruntime {
namespace contrib {

// 定义内核注册宏
#define REGISTER_MY_VIRTUAL_NPU_KERNEL_TYPED(name, T, builder) \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                        \
      name,                                             \
      kMyCustomDomain,                                  \
      1,                                                \
      T,                                                \
      kCpuExecutionProvider,                            \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      builder)

void RegisterMyVirtualNpuKernels(KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn function_table[] = {
      BuildKernelCreateInfo<REGISTER_MY_VIRTUAL_NPU_KERNEL_TYPED(FastGelu, float, FastGelu<float>)>,
      // 可以添加更多算子...
  };

  for (auto& function : function_table) {
    ORT_THROW_IF_ERROR(kernel_registry.Register(function()));
  }
}

}  // namespace contrib
}  // namespace onnxruntime
```

### 2.4 集成到 CPU ExecutionProvider

```cpp
// onnxruntime/core/providers/cpu/cpu_execution_provider.cc
#include "core/providers/my_virtual_npu/my_virtual_npu_kernels.h"

namespace onnxruntime {

CPUExecutionProvider::CPUExecutionProvider(const CPUExecutionProviderInfo& info)
    : IExecutionProvider{onnxruntime::kCpuExecutionProvider, true} {
  // ... 其他初始化代码 ...

  // 注册自定义 my_virtual_npu 算子内核
  contrib::RegisterMyCpuKernels(*registry_);
}

}  // namespace onnxruntime
```

## 三、编译构建

### 3.1 CMake 配置

```cmake
# onnxruntime/core/providers/my_virtual_npu/CMakeLists.txt
set(my_virtual_npu_sources
  my_virtual_npu_defs.cc
  my_virtual_npu_kernels.cc
  nn/fast_gelu.cc
)

add_library(onnxruntime_providers_my_virtual_npu OBJECT ${my_virtual_npu_sources})
target_include_directories(onnxruntime_providers_my_virtual_npu PRIVATE
  ${ONNXRUNTIME_ROOT}
  ${ONNXRUNTIME_ROOT}/core
)
```

### 3.2 编译命令

```bash
# 配置构建
./build.sh --config Release \
  --parallel \
  --skip_submodule_sync \
  --skip_tests \
  --build_shared_lib

# 编译完成后，库文件位于
# build/Linux/Release/libonnxruntime.so
```

### 3.3 编译优化建议

- 使用 `--parallel` 加速编译
- 开发阶段使用 `--config Debug` 便于调试
- 生产环境使用 `--config Release` 获得最佳性能
- 添加 `--enable_pybind` 支持 Python 绑定

## 四、单元测试

### 4.1 测试框架

ONNXRuntime 使用 Google Test 框架，提供了 `OpTester` 工具类简化算子测试：

```cpp
// onnxruntime/test/providers/my_virtual_npu/nn/fast_gelu_op_test.cc
#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "core/providers/my_virtual_npu/my_virtual_npu_defs.h"

namespace onnxruntime {
namespace test {

// 确保 Schema 已注册
static void EnsureSchemasRegistered() {
    static bool initialized = false;
    if (!initialized) {
        contrib::RegisterMyVirtualNpuSchemas();
        initialized = true;
    }
}

TEST(FastGeluTest, Basic) {
    EnsureSchemasRegistered();

    OpTester test("FastGelu", 1, contrib::kMyCustomDomain);

    // 输入数据
    std::vector<int64_t> dims{2, 3};
    std::vector<float> input_data = {-1.0f, 0.0f, 1.0f, 2.0f, -2.0f, 0.5f};

    // 期望输出（根据 GELU 公式计算）
    std::vector<float> expected_output = {
        -0.15865529f,  // GELU(-1.0)
        0.0f,          // GELU(0.0)
        0.8413447f,    // GELU(1.0)
        1.9545977f,    // GELU(2.0)
        -0.04540223f,  // GELU(-2.0)
        0.34571534f    // GELU(0.5)
    };

    test.AddInput<float>("X", dims, input_data);
    test.AddOutput<float>("Y", dims, expected_output);

    // 运行测试（会自动使用 CPU ExecutionProvider）
    test.Run();
}

TEST(FastGeluTest, LargeInput) {
    EnsureSchemasRegistered();

    OpTester test("FastGelu", 1, contrib::kMyCustomDomain);

    // 测试大张量
    std::vector<int64_t> dims{128, 768};  // 类似 BERT hidden size
    int64_t total = 128 * 768;

    std::vector<float> input_data(total);
    std::vector<float> expected_output(total);

    // 生成测试数据
    for (int64_t i = 0; i < total; i++) {
        float x = (i % 100 - 50) * 0.1f;  // 范围 [-5, 5]
        input_data[i] = x;

        // 计算期望输出
        float x3 = x * x * x;
        float inner = 0.7978845608f * (x + 0.044715f * x3);
        expected_output[i] = 0.5f * x * (1.0f + std::tanh(inner));
    }

    test.AddInput<float>("X", dims, input_data);
    test.AddOutput<float>("Y", dims, expected_output);
    test.Run();
}

TEST(FastGeluTest, EdgeCases) {
    EnsureSchemasRegistered();

    OpTester test("FastGelu", 1, contrib::kMyCustomDomain);

    // 测试边界情况
    std::vector<int64_t> dims{6};
    std::vector<float> input_data = {
        -10.0f,   // 极小值
        10.0f,    // 极大值
        0.0f,     // 零
        -0.0f,    // 负零
        1e-7f,    // 接近零
        -1e-7f    // 接近负零
    };

    std::vector<float> expected_output(6);
    for (size_t i = 0; i < input_data.size(); i++) {
        float x = input_data[i];
        float x3 = x * x * x;
        float inner = 0.7978845608f * (x + 0.044715f * x3);
        expected_output[i] = 0.5f * x * (1.0f + std::tanh(inner));
    }

    test.AddInput<float>("X", dims, input_data);
    test.AddOutput<float>("Y", dims, expected_output);
    test.Run();
}

}  // namespace test
}  // namespace onnxruntime
```

### 4.2 运行测试

```bash
# 编译测试
./build.sh --config Release --build_shared_lib --enable_tests

# 运行所有测试
./build/Linux/Release/onnxruntime_test_all

# 运行特定测试套件
./build/Linux/Release/onnxruntime_test_all --gtest_filter="FastGeluTest.*"

# 运行单个测试
./build/Linux/Release/onnxruntime_test_all --gtest_filter="FastGeluTest.Basic"
```

### 4.3 测试输出示例

```
[==========] Running 3 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 3 tests from FastGeluTest
[ RUN      ] FastGeluTest.Basic
[       OK ] FastGeluTest.Basic (12 ms)
[ RUN      ] FastGeluTest.LargeInput
[       OK ] FastGeluTest.LargeInput (156 ms)
[ RUN      ] FastGeluTest.EdgeCases
[       OK ] FastGeluTest.EdgeCases (8 ms)
[----------] 3 tests from FastGeluTest (176 ms total)

[==========] 3 tests from 1 test suite ran. (176 ms total)
[  PASSED  ] 3 tests.
```

## 五、Python 集成与大模型测试

### 5.1 Python 包安装

```bash
# 方式 1: 从源码构建安装
python setup.py install

# 方式 2: 构建 wheel 包
python tools/ci_build/build.py --build_wheel --config Release
pip install build/Linux/Release/dist/onnxruntime-1.20.0-*.whl

# 方式 3: 开发模式安装
pip install -e . --no-build-isolation
```

### 5.2 验证安装

```python
# check_onnxruntime_version.py
import onnxruntime as ort
import numpy as np

print(f"ONNXRuntime 版本: {ort.__version__}")
print(f"可用的 Execution Providers: {ort.get_available_providers()}")

# 测试自定义算子
def test_custom_fastgelu():
    """测试自定义 FastGelu 算子"""
    # 创建简单的 ONNX 模型（包含 FastGelu）
    import onnx
    from onnx import helper, TensorProto

    # 定义输入输出
    input_tensor = helper.make_tensor_value_info('X', TensorProto.FLOAT, [2, 3])
    output_tensor = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [2, 3])

    # 创建 FastGelu 节点
    fastgelu_node = helper.make_node(
        'FastGelu',
        inputs=['X'],
        outputs=['Y'],
        domain='com.my_virtual_npu'
    )

    # 创建图
    graph = helper.make_graph(
        [fastgelu_node],
        'test_fastgelu',
        [input_tensor],
        [output_tensor]
    )

    # 创建模型
    model = helper.make_model(graph, producer_name='test')

    # 保存模型
    onnx.save(model, 'test_fastgelu.onnx')

    # 运行推理
    session = ort.InferenceSession('test_fastgelu.onnx')

    input_data = np.array([[-1.0, 0.0, 1.0], [2.0, -2.0, 0.5]], dtype=np.float32)
    outputs = session.run(None, {'X': input_data})

    print("输入:", input_data)
    print("输出:", outputs[0])
    print("自定义算子测试通过！")

if __name__ == "__main__":
    test_custom_fastgelu()
```

### 5.3 大模型测试：Tiny-GPT2

```python
# test_tiny_gpt2.py
import onnxruntime as ort
import numpy as np
from transformers import GPT2Tokenizer
import time

def test_tiny_gpt2():
    """测试 Tiny-GPT2 模型（使用自定义 FastGelu 算子）"""

    # 加载模型（假设已将 Gelu 替换为 FastGelu）
    model_path = "models/tiny-gpt2-fastgelu.onnx"

    # 创建会话
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        model_path,
        sess_options=session_options,
        providers=['CPUExecutionProvider']
    )

    # 打印模型信息
    print("=" * 60)
    print("模型输入:")
    for input_meta in session.get_inputs():
        print(f"  {input_meta.name}: {input_meta.shape} ({input_meta.type})")

    print("\n模型输出:")
    for output_meta in session.get_outputs():
        print(f"  {output_meta.name}: {output_meta.shape} ({output_meta.type})")
    print("=" * 60)

    # 准备输入数据
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    text = "Hello, I am a language model"

    # Tokenize
    inputs = tokenizer(text, return_tensors="np")
    input_ids = inputs["input_ids"].astype(np.int64)
    attention_mask = inputs["attention_mask"].astype(np.int64)

    print(f"\n输入文本: {text}")
    print(f"Input IDs shape: {input_ids.shape}")

    # 推理性能测试
    warmup_runs = 3
    test_runs = 10

    print(f"\n预热运行 {warmup_runs} 次...")
    for _ in range(warmup_runs):
        session.run(None, {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        })

    print(f"性能测试 {test_runs} 次...")
    times = []
    for i in range(test_runs):
        start_time = time.perf_counter()
        outputs = session.run(None, {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        })
        end_time = time.perf_counter()
        times.append(end_time - start_time)

    # 统计结果
    avg_time = np.mean(times) * 1000  # 转换为毫秒
    std_time = np.std(times) * 1000
    min_time = np.min(times) * 1000
    max_time = np.max(times) * 1000

    print("\n" + "=" * 60)
    print("性能统计:")
    print(f"  平均推理时间: {avg_time:.2f} ms")
    print(f"  标准差:       {std_time:.2f} ms")
    print(f"  最小值:       {min_time:.2f} ms")
    print(f"  最大值:       {max_time:.2f} ms")
    print(f"  吞吐量:       {1000/avg_time:.2f} samples/sec")
    print("=" * 60)

    # 输出预测结果
    logits = outputs[0]
    print(f"\nLogits shape: {logits.shape}")
    print(f"Logits 统计: min={logits.min():.4f}, max={logits.max():.4f}, mean={logits.mean():.4f}")

    # 生成文本（简单贪婪解码）
    next_token_id = np.argmax(logits[0, -1, :])
    next_token = tokenizer.decode([next_token_id])
    print(f"\n预测的下一个 token: '{next_token}' (ID: {next_token_id})")

    print("\n✅ 大模型测试完成！")

if __name__ == "__main__":
    test_tiny_gpt2()
```

### 5.4 性能对比

```python
# benchmark_fastgelu.py
import onnxruntime as ort
import numpy as np
import time

def benchmark_comparison():
    """对比标准 Gelu 和自定义 FastGelu 的性能"""

    # 测试配置
    batch_size = 32
    seq_length = 128
    hidden_size = 768
    iterations = 100

    input_shape = (batch_size, seq_length, hidden_size)
    input_data = np.random.randn(*input_shape).astype(np.float32)

    # 测试标准 Gelu
    print("测试标准 Gelu...")
    # ... (省略模型创建代码)

    # 测试自定义 FastGelu
    print("测试自定义 FastGelu...")
    # ... (省略模型创建代码)

    print("\n性能对比:")
    print(f"标准 Gelu:    {gelu_time:.2f} ms")
    print(f"自定义 FastGelu: {fastgelu_time:.2f} ms")
    print(f"加速比:       {gelu_time/fastgelu_time:.2f}x")

if __name__ == "__main__":
    benchmark_comparison()
```

## 六、C++ SDK 打包与分发

### 6.1 打包脚本

为了方便其他开发者使用，我们创建了自动打包脚本：

```bash
#!/bin/bash
# pre_package_cpp_sdk.sh
# 将编译好的库、头文件打包到 PreRelease 目录

set -e

echo "📦 打包 ONNXRuntime C++ SDK..."

# 配置
BUILD_DIR="${1:-build/Linux/Release}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RELEASE_DIR="PreRelease_${TIMESTAMP}/cpp"
VERSION="1.20.0-custom"

# 创建目录结构
mkdir -p "$RELEASE_DIR"/{include,lib,bin,examples}

# 复制头文件
echo "📋 复制头文件..."
cp -r include/onnxruntime "$RELEASE_DIR/include/"

# 复制库文件
echo "📚 复制库文件..."
cp "$BUILD_DIR"/libonnxruntime.so* "$RELEASE_DIR/lib/"

# 创建 CMake 配置
cat > "$RELEASE_DIR/ONNXRuntimeConfig.cmake" << 'EOF'
get_filename_component(ONNXRUNTIME_CMAKE_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
set(ONNXRUNTIME_INCLUDE_DIRS "${ONNXRUNTIME_CMAKE_DIR}/include")
set(ONNXRUNTIME_LIBRARIES "${ONNXRUNTIME_CMAKE_DIR}/lib/libonnxruntime.so")

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARIES}"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIRS}"
)
EOF

# 创建示例代码
cat > "$RELEASE_DIR/examples/simple_inference.cpp" << 'EOF'
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <iostream>

int main() {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;
    Ort::Session session(env, "model.onnx", session_options);

    std::cout << "输入节点数: " << session.GetInputCount() << std::endl;
    std::cout << "输出节点数: " << session.GetOutputCount() << std::endl;

    return 0;
}
EOF

echo "✅ 打包完成: $RELEASE_DIR"
```

### 6.2 使用打包的 SDK

```cmake
# 用户的 CMakeLists.txt
cmake_minimum_required(VERSION 3.13)
project(MyApp)

set(CMAKE_CXX_STANDARD 17)

# 找到 ONNXRuntime
set(ONNXRuntime_DIR /path/to/PreRelease_xxx/cpp)
find_package(ONNXRuntime REQUIRED)

# 创建应用
add_executable(my_app main.cpp)
target_link_libraries(my_app onnxruntime)
```

## 七、常见问题与解决方案

### 7.1 Schema 注册问题

**问题：** `No Schema registered for 'FastGelu'!`

**原因：** Schema 未在模型加载前注册

**解决方案：**
```cpp
// 在 Environment::Create() 中添加
contrib::RegisterMyVirtualNpuSchemas();
```

### 7.2 Domain 冲突

**问题：** 与 Microsoft 内置算子冲突

**解决方案：** 使用自定义域名
```cpp
constexpr const char* kMyCustomDomain = "com.my_virtual_npu";
```

### 7.3 Kernel 未找到

**问题：** `Kernel not found: FastGelu`

**原因：** Kernel 未注册到 ExecutionProvider

**解决方案：**
```cpp
// 在 CPUExecutionProvider 构造函数中
contrib::RegisterMyCpuKernels(*registry_);
```

### 7.4 输入输出不匹配

**问题：** `Input count mismatch: expected 2, got 1`

**原因：** Kernel 实现与 Schema 定义不一致

**解决方案：** 确保 Schema 和 Kernel 的输入输出数量、类型完全一致

### 7.5 Python 包安装失败

**问题：** `No such file or directory: 'build/Linux/Release/wheel'`

**原因：** 未正确生成 wheel 包

**解决方案：**
```bash
# 使用 CI 构建工具
python tools/ci_build/build.py --build_wheel --config Release
```

## 八、性能优化建议

### 8.1 SIMD 优化

```cpp
// 使用 AVX2 加速
#include <immintrin.h>

void FastGeluAVX2(const float* input, float* output, int64_t size) {
    for (int64_t i = 0; i < size; i += 8) {
        __m256 x = _mm256_loadu_ps(input + i);
        // ... AVX2 GELU 计算 ...
        _mm256_storeu_ps(output + i, result);
    }
}
```

### 8.2 多线程并行

```cpp
#include "core/platform/threadpool.h"

// 使用 ONNXRuntime 的线程池
context->GetOperatorThreadPool()->ParallelFor(
    total_elements,
    [&](std::ptrdiff_t i) {
        output_data[i] = ComputeGeluScalar(input_data[i]);
    }
);
```

### 8.3 内存优化

- 使用 `MutableData()` 而非 `Data()` 避免不必要的拷贝
- 利用 `AllocatorPtr` 管理大块内存
- 考虑使用 Arena 分配器减少内存碎片

## 九、总结

本文详细介绍了在 ONNXRuntime 中开发自定义算子的完整流程：

1. **架构设计**：自定义域、Schema 注册、Kernel 注册
2. **算子实现**：FastGelu 的数学原理和 C++ 实现
3. **编译构建**：CMake 配置、编译选项
4. **单元测试**：使用 OpTester 进行全面测试
5. **Python 集成**：安装、验证、大模型测试
6. **SDK 打包**：方便分发和使用

通过本文的实践，您可以：
- 理解 ONNXRuntime 的算子注册机制
- 掌握自定义算子的开发流程
- 学会单元测试和性能验证
- 能够将自定义算子集成到实际项目中

## 参考资料

- [ONNXRuntime 官方文档](https://onnxruntime.ai/docs/)
- [ONNX Operator Schemas](https://github.com/onnx/onnx/blob/main/docs/Operators.md)
- [ONNXRuntime 自定义算子指南](https://onnxruntime.ai/docs/reference/operators/add-custom-op.html)
- [GELU 论文](https://arxiv.org/abs/1606.08415)

---

**作者信息**
- GitHub: [onnxruntime_my_virtual_npu](https://github.com/Han-Zhenzhong/onnxruntime_my_virtual_npu)
- 版本: ONNXRuntime 1.20.0
- 最后更新: 2025-11-19

**许可证**
本文档遵循 MIT License，欢迎转载和修改，但请保留原作者信息。
