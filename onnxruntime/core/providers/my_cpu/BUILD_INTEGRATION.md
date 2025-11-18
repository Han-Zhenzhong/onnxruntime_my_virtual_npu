# my_cpu 构建系统集成说明

本文档描述了 `my_cpu` 自定义算子如何集成到 ONNX Runtime 主构建系统。

## 📋 集成概述

`my_cpu` 自定义算子已完全集成到 ONNX Runtime 构建系统，与 `contrib_ops` 类似：

- ✅ 源文件自动包含在 `onnxruntime_providers` 静态库中
- ✅ CPU Execution Provider 自动注册 my_cpu 算子
- ✅ 测试文件自动包含在单元测试套件中

## 🔧 修改的文件

### 1. `cmake/onnxruntime_providers_cpu.cmake`

**作用**: 将 my_cpu 源文件编译到 onnxruntime_providers 库

**修改内容**:
```cmake
# 第13-18行：添加源文件扫描
file(GLOB_RECURSE onnxruntime_my_cpu_ops_srcs CONFIGURE_DEPENDS
  "${ONNXRUNTIME_ROOT}/my_cpu/*.h"
  "${ONNXRUNTIME_ROOT}/my_cpu/*.cc"
)

# 第62-67行：添加到 onnxruntime_providers_src
if(onnxruntime_my_cpu_ops_srcs)
  source_group(TREE ${ONNXRUNTIME_ROOT} FILES ${onnxruntime_my_cpu_ops_srcs})
  list(APPEND onnxruntime_providers_src ${onnxruntime_my_cpu_ops_srcs})
  message(STATUS "my_cpu custom operators enabled")
endif()
```

**编译流程**:
```
my_cpu/*.cc → onnxruntime_providers (静态库) → onnxruntime.dll/libonnxruntime.so
```

---

### 2. `onnxruntime/core/providers/cpu/cpu_execution_provider.cc`

**作用**: 注册 my_cpu 算子到 CPU Execution Provider

**修改内容**:
```cpp
// 第13-19行：包含 my_cpu 头文件
#ifndef DISABLE_CONTRIB_OPS
#include "contrib_ops/cpu/cpu_contrib_kernels.h"
#endif

// Custom my_cpu operators
#include "my_cpu/my_cpu_kernels.h"

// 第3817行：注册 my_cpu kernels
Status RegisterCPUKernels(KernelRegistry& kernel_registry) {
  // ... 其他算子注册 ...

  // Register custom my_cpu operators
  ORT_RETURN_IF_ERROR(::onnxruntime::my_cpu::RegisterMyCpuKernels(kernel_registry));

  return Status::OK();
}
```

**注册流程**:
```
CPU EP 初始化 → RegisterCPUKernels() → RegisterMyCpuKernels()
                                       → FastGelu 可用
```

---

### 3. `cmake/onnxruntime_unittests.cmake`

**作用**: 将 my_cpu 测试文件包含到单元测试

**修改内容**:
```cmake
# 第497-507行：添加测试源文件
if(NOT onnxruntime_MINIMAL_BUILD AND NOT onnxruntime_REDUCED_OPS_BUILD)
  file(GLOB_RECURSE onnxruntime_test_my_cpu_src CONFIGURE_DEPENDS
    "${TEST_SRC_DIR}/my_cpu/*.cc"
    "${TEST_SRC_DIR}/my_cpu/*.h"
    )
  if(onnxruntime_test_my_cpu_src)
    list(APPEND onnxruntime_test_providers_src ${onnxruntime_test_my_cpu_src})
    message(STATUS "my_cpu operator tests enabled")
  endif()
endif()
```

**测试流程**:
```
test/my_cpu/*.cc → onnxruntime_test_all → ctest (FastGeluOpTest.*)
```

---

## 🏗️ 编译方法

### 完整构建（推荐）

```bash
# Windows (使用 build.bat)
cd D:\open-source\onnxruntime
.\build.bat --config Release --build_shared_lib --parallel

# Linux/Mac (使用 build.sh)
./build.sh --config Release --build_shared_lib --parallel
```

### 仅构建测试

```bash
# 进入构建目录
cd build/Windows/Release  # Windows
cd build/Linux/Release    # Linux

# 重新构建
cmake --build . --config Release --target onnxruntime_test_all
```

---

## ✅ 验证步骤

### 1. 编译成功检查

```bash
# 构建成功后检查日志中是否有：
# -- my_cpu custom operators enabled
# -- my_cpu operator tests enabled
```

### 2. 运行单元测试

```bash
# Windows
cd build\Windows\Release\Release
.\onnxruntime_test_all.exe --gtest_filter="FastGeluOpTest.*"

# Linux
cd build/Linux/Release
./onnxruntime_test_all --gtest_filter="FastGeluOpTest.*"
```

**预期输出**:
```
[==========] Running 5 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 5 tests from FastGeluOpTest
[ RUN      ] FastGeluOpTest.BasicFloat32
[       OK ] FastGeluOpTest.BasicFloat32 (x ms)
[ RUN      ] FastGeluOpTest.DifferentShapes
[       OK ] FastGeluOpTest.DifferentShapes (x ms)
[ RUN      ] FastGeluOpTest.EdgeCases
[       OK ] FastGeluOpTest.EdgeCases (x ms)
[ RUN      ] FastGeluOpTest.SingleElement
[       OK ] FastGeluOpTest.SingleElement (x ms)
[ RUN      ] FastGeluOpTest.LargeTensor
[       OK ] FastGeluOpTest.LargeTensor (x ms)
[----------] 5 tests from FastGeluOpTest (x ms total)
[==========] 5 tests from 1 test suite ran. (x ms total)
[  PASSED  ] 5 tests.
```

### 3. 检查库符号

```bash
# Linux: 检查 FastGelu 符号是否存在
nm libonnxruntime.so | grep -i gelu

# Windows: 检查 DLL 导出
dumpbin /EXPORTS onnxruntime.dll | findstr Gelu
```

---

## 📂 集成后的构建结构

```
onnxruntime/
├── my_cpu/                          # 源代码目录
│   ├── bert/
│   │   ├── fast_gelu.h              → 编译到 onnxruntime_providers.lib
│   │   └── fast_gelu.cc             → 编译到 onnxruntime_providers.lib
│   ├── my_cpu_kernels.h             → 编译到 onnxruntime_providers.lib
│   └── my_cpu_kernels.cc            → 编译到 onnxruntime_providers.lib
│
├── test/my_cpu/                     # 测试目录
│   └── fast_gelu_op_test.cc         → 编译到 onnxruntime_test_all.exe
│
├── onnxruntime/core/providers/cpu/
│   └── cpu_execution_provider.cc    → 调用 RegisterMyCpuKernels()
│
└── cmake/
    ├── onnxruntime_providers_cpu.cmake   → 包含 my_cpu/*.cc
    └── onnxruntime_unittests.cmake       → 包含 test/my_cpu/*.cc
```

---

## 🔍 集成原理

### 编译时

1. **CMake 配置阶段**:
   - `onnxruntime_providers_cpu.cmake` 扫描 `my_cpu/*.cc`
   - 将 my_cpu 源文件添加到 `onnxruntime_providers_src` 列表

2. **编译阶段**:
   - my_cpu 源文件被编译为 `.o` / `.obj` 目标文件
   - 链接到 `onnxruntime_providers` 静态库
   - 最终链接到 `onnxruntime.dll` / `libonnxruntime.so`

### 运行时

1. **ONNX Runtime 初始化**:
   ```
   InferenceSession::Initialize()
   └─> CPUExecutionProvider::CPUExecutionProvider()
       └─> GetKernelRegistry()
           └─> RegisterCPUKernels()
               └─> my_cpu::RegisterMyCpuKernels()
                   └─> FastGelu kernel 注册到 kMSDomain
   ```

2. **推理执行**:
   ```
   InferenceSession::Run()
   └─> 遇到 FastGelu 节点 (domain="com.microsoft")
       └─> 查找 kernel: kMSDomain + "FastGelu"
           └─> 找到 my_cpu::FastGelu<float>
               └─> 调用 Compute() 执行
   ```

---

## ❓ 常见问题

### Q: my_cpu 源文件会被自动编译吗？
**A**: 是的。只要 `my_cpu/*.cc` 文件存在，就会被自动扫描并包含在构建中。

### Q: 是否需要修改其他 CMakeLists.txt？
**A**: 不需要。`my_cpu/CMakeLists.txt` 仅用于独立构建测试，主构建系统不使用它。

### Q: 如何禁用 my_cpu 算子？
**A**: 有两种方式：
1. 删除或重命名 `my_cpu/` 目录
2. 在 CMake 中添加条件：
   ```cmake
   if(NOT DISABLE_MY_CPU_OPS)
     list(APPEND onnxruntime_providers_src ${onnxruntime_my_cpu_ops_srcs})
   endif()
   ```

### Q: my_cpu 算子是否支持最小化构建？
**A**: 目前 my_cpu 算子在最小化构建中**不会**被包含（与 contrib_ops 一致）。如需支持，需要在 CMake 中调整条件。

---

## 📊 性能影响

| 影响类型 | 描述 | 评估 |
|---------|------|------|
| **编译时间** | 增加 ~200 行 C++ 代码 | 微小 (< 1%) |
| **二进制大小** | FastGelu 实现 + 注册代码 | ~10-20 KB |
| **运行时内存** | KernelRegistry 中的额外条目 | 可忽略 |
| **推理性能** | 仅在使用 FastGelu 时触发 | 无影响（未使用时） |

---

## 🎓 学习要点

### 关键概念

1. **源文件分组**: `GLOB_RECURSE` 扫描目录树
2. **条件编译**: `DISABLE_CONTRIB_OPS` 等宏控制
3. **Kernel Registry**: 运行时算子查找机制
4. **静态库链接**: 多个 `.a`/`.lib` 合并为 `.so`/`.dll`

### 与 contrib_ops 的对比

| 特性 | contrib_ops | my_cpu |
|------|-------------|---------|
| **目录** | `contrib_ops/cpu/` | `my_cpu/` |
| **命名空间** | `onnxruntime::contrib` | `onnxruntime::my_cpu` |
| **Domain** | kMSDomain / kOnnxDomain | kMSDomain |
| **条件编译** | `DISABLE_CONTRIB_OPS` | 无（总是启用） |
| **注册方式** | `RegisterCpuContribKernels()` | `RegisterMyCpuKernels()` |

---

## 🚀 下一步

1. **✅ 编译验证**:
   ```bash
   ./build.bat --config Release --build_shared_lib
   ```

2. **✅ 运行测试**:
   ```bash
   ./build/Windows/Release/Release/onnxruntime_test_all.exe --gtest_filter="FastGeluOpTest.*"
   ```

3. **⏭️ 端到端测试**: 使用 Tiny-GPT2 ONNX 模型测试

4. **⏭️ 性能优化**: 添加 AVX2 / NEON SIMD 实现

---

## 📚 参考

- [ONNX Runtime 构建文档](https://onnxruntime.ai/docs/build/)
- [自定义算子开发指南](https://onnxruntime.ai/docs/reference/operators/add-custom-op.html)
- [CMake file(GLOB) 文档](https://cmake.org/cmake/help/latest/command/file.html#glob)
- [Google Test 过滤器](https://google.github.io/googletest/advanced.html#running-a-subset-of-the-tests)

---

**最后更新**: 2025-11-18
**状态**: ✅ 集成完成，待编译验证
