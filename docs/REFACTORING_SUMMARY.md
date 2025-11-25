# MyVirtualNPU 架构重构总结

## 重构完成时间
2025-11-25

## 重构目标
将 `my_virtual_npu` 从耦合到 CPU Provider 的实现，重构为**独立的 Execution Provider**。

---

## 架构对比

### 重构前（错误架构）❌
```
CPU Execution Provider
  ├─ cpu_execution_provider.cc
  │   └─ #include "my_virtual_npu/my_virtual_npu_kernels.h"  ← 硬编码依赖
  └─ RegisterCPUKernels()
      └─ RegisterMyVirtualNpuKernels()  ← 直接调用

cmake/onnxruntime_providers_cpu.cmake
  └─ 包含 my_virtual_npu 源文件  ← 耦合到 CPU Provider
```

**问题：**
- 违反了 ONNXRuntime 的插件化架构
- CPU Provider 依赖自定义代码
- 无法独立编译、测试、分发
- 不符合其他 EP（CUDA、TensorRT）的设计模式

### 重构后（正确架构）✅
```
MyVirtualNPU Execution Provider (独立)
  ├─ my_virtual_npu_execution_provider.h/cc  ← 独立 EP 接口
  ├─ my_virtual_npu_kernels.h/cc             ← Kernel 注册
  └─ bert/fast_gelu.h/cc                     ← 算子实现

cmake/onnxruntime_providers_my_virtual_npu.cmake  ← 独立 CMake 配置
cmake/CMakeLists.txt
  └─ option(onnxruntime_USE_MY_VIRTUAL_NPU ON)  ← CMake 选项控制
```

**优势：**
- ✅ 完全独立的 Execution Provider
- ✅ 符合 ONNXRuntime 架构规范
- ✅ 可独立编译、测试、分发
- ✅ 易于扩展和维护
- ✅ 与其他 EP（CUDA、TensorRT）一致的设计

---

## 修改文件清单

### 新增文件 ✨
1. **cmake/onnxruntime_providers_my_virtual_npu.cmake**
   - 独立的 Provider 构建配置
   - 定义 `onnxruntime_providers_my_virtual_npu` 静态库

2. **onnxruntime/core/providers/my_virtual_npu/my_virtual_npu_execution_provider.h**
   - EP 接口定义
   - `MyVirtualNpuExecutionProvider` 类

3. **onnxruntime/core/providers/my_virtual_npu/my_virtual_npu_execution_provider.cc**
   - EP 实现
   - Allocator 管理
   - Kernel Registry 注册

4. **onnxruntime/core/providers/my_virtual_npu/USAGE_EXAMPLE.cc**
   - 使用示例代码

### 修改文件 🔧

#### CMake 配置
1. **cmake/CMakeLists.txt**
   - 添加: `option(onnxruntime_USE_MY_VIRTUAL_NPU "Build with MyVirtualNPU" ON)`

2. **cmake/onnxruntime_providers.cmake**
   - 添加: `${PROVIDERS_MY_VIRTUAL_NPU}` 变量
   - 添加: `include(onnxruntime_providers_my_virtual_npu.cmake)`

3. **cmake/onnxruntime.cmake**
   - 添加: `${PROVIDERS_MY_VIRTUAL_NPU}` 到 `onnxruntime_INTERNAL_PROVIDER_LIBRARIES`

4. **cmake/onnxruntime_providers_cpu.cmake**
   - ❌ 移除: `file(GLOB_RECURSE onnxruntime_my_virtual_npu_ops_srcs ...)`
   - ❌ 移除: `list(APPEND onnxruntime_providers_src ${onnxruntime_my_virtual_npu_ops_srcs})`

5. **cmake/onnxruntime_unittests.cmake**
   - 修改: 添加条件 `if(onnxruntime_USE_MY_VIRTUAL_NPU ...)`

#### 核心代码
6. **onnxruntime/core/providers/cpu/cpu_execution_provider.cc**
   - ❌ 移除: `#include "core/providers/my_virtual_npu/my_virtual_npu_kernels.h"`
   - ❌ 移除: `RegisterMyVirtualNpuKernels(kernel_registry)` 调用

7. **include/onnxruntime/core/graph/constants.h**
   - 添加: `constexpr const char* kMyVirtualNpuExecutionProvider = "MyVirtualNpuExecutionProvider";`

#### 文档
8. **onnxruntime/core/providers/my_virtual_npu/README.md**
   - 更新架构说明
   - 添加 CMake 构建选项说明

---

## 构建和使用

### 1. 配置构建
```bash
cd /d/open-source/onnxruntime
mkdir -p build/Linux/Release
cd build/Linux/Release

cmake ../../../cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -Donnxruntime_USE_MY_VIRTUAL_NPU=ON \
  -Donnxruntime_BUILD_SHARED_LIB=ON \
  -Donnxruntime_BUILD_UNIT_TESTS=ON
```

### 2. 编译
```bash
cmake --build . -j$(nproc)
```

### 3. 运行测试
```bash
./onnxruntime_test_all --gtest_filter="*FastGelu*"
```

### 4. C++ 代码中使用
```cpp
#include <onnxruntime_cxx_api.h>
#include "core/providers/my_virtual_npu/my_virtual_npu_execution_provider.h"

Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
Ort::SessionOptions options;

// 启用 MyVirtualNPU Provider
onnxruntime::MyVirtualNpuExecutionProviderInfo info;
info.create_arena = true;

auto provider = std::make_unique<onnxruntime::MyVirtualNpuExecutionProvider>(info);
options.AppendExecutionProvider(std::move(provider));

// 创建 Session
Ort::Session session(env, "model.onnx", options);
```

### 5. Python 中使用
```python
import onnxruntime as ort

# 配置 Session Options
providers = ['MyVirtualNpuExecutionProvider', 'CPUExecutionProvider']
session = ort.InferenceSession('model.onnx', providers=providers)

# 运行推理
outputs = session.run(None, inputs)
```

---

## 验证清单

### 编译验证 ✅
- [ ] `cmake` 配置成功
- [ ] `cmake --build` 编译成功
- [ ] 生成 `libonnxruntime_providers_my_virtual_npu.a`

### 功能验证 ✅
- [ ] 单元测试通过: `onnxruntime_test_all --gtest_filter="*FastGelu*"`
- [ ] Python 集成测试: `test_tiny_gpt2.py`
- [ ] 混合 Provider 测试: `test_mixed_providers.py`

### 架构验证 ✅
- [ ] CPU Provider 不再包含 my_virtual_npu 代码
- [ ] my_virtual_npu 是独立的静态库
- [ ] 可通过 CMake 选项独立启用/禁用

---

## 后续优化方向

### 性能优化
1. **FastGelu 算子**
   - [ ] AVX2 SIMD 向量化
   - [ ] OpenMP 并行化
   - [ ] 内存布局优化

2. **内存管理**
   - [ ] 自定义 Allocator 实现
   - [ ] Memory Pool 优化

### 功能扩展
3. **新增算子**
   - [ ] SkipLayerNorm
   - [ ] Attention
   - [ ] MatMul 优化版本

4. **EP 功能**
   - [ ] Graph 优化 Pass
   - [ ] Kernel 融合支持
   - [ ] 动态形状支持

---

## 关键设计决策

### 为什么独立 EP？
1. **架构一致性**: 与 CUDA、TensorRT 等其他 EP 保持一致
2. **可维护性**: 独立编译、测试、调试
3. **可扩展性**: 易于添加新功能而不影响核心代码
4. **可分发性**: 可以作为插件独立分发

### 为什么默认启用？
```cmake
option(onnxruntime_USE_MY_VIRTUAL_NPU "..." ON)  # 默认 ON
```
- 这是学习和实验性质的 Provider
- 方便开发和测试
- 生产环境可以设置为 OFF

### 内存管理策略
- 使用 `CPUAllocator` 作为后端（与 CPU Provider 共享内存）
- 支持 Memory Arena（可选）
- 将来可以扩展为自定义 Allocator

---

## 参考资料

### ONNXRuntime EP 实现参考
- **CPU Provider**: `onnxruntime/core/providers/cpu/`
- **CUDA Provider**: `onnxruntime/core/providers/cuda/`
- **CoreML Provider**: `onnxruntime/core/providers/coreml/`

### 文档
- ONNXRuntime Execution Provider 接口: `include/onnxruntime/core/framework/execution_provider.h`
- Kernel Registry: `include/onnxruntime/core/framework/kernel_registry.h`
- Build System: `cmake/onnxruntime_providers.cmake`

---

## 联系和支持
- Repository: onnxruntime_my_virtual_npu
- Owner: Han-Zhenzhong
- Branch: main

## 变更日志
- 2025-11-25: 完成架构重构，从 CPU Provider 解耦为独立 EP
