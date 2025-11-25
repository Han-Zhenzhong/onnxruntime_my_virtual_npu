# MyVirtualNPU CMake 配置问题修复

## 问题诊断

原始 CMake 配置存在以下问题：

### ❌ 问题 1: 使用 GLOB_RECURSE 导致文件收集错误
```cmake
# 原始代码 - 错误
file(GLOB_RECURSE onnxruntime_providers_my_virtual_npu_cc_srcs ...)
list(FILTER ... EXCLUDE REGEX ".*cuda/.*\\.cu$")  # 尝试过滤但不可靠
```

**问题**: `GLOB_RECURSE` 会递归收集所有子目录，包括 `cuda/` 目录。即使用 `FILTER EXCLUDE`，也可能导致混乱。

### ❌ 问题 2: 缺少 CUDA 语言设置
```cmake
# 原始代码 - 缺失
file(GLOB_RECURSE ... "*.cu")
# 没有设置 .cu 文件的语言属性
```

**问题**: CMake 可能不知道 `.cu` 文件需要用 CUDA 编译器编译，导致编译失败。

### ❌ 问题 3: 缺少 CUDA 编译选项
```cmake
# 原始代码 - 缺失
# 没有设置 CUDA 特定的编译选项
```

**问题**: 缺少 `--expt-relaxed-constexpr` 等 CUDA 编译标志，可能导致编译错误。

### ❌ 问题 4: 不必要的 CUDNN 依赖
```cmake
# 原始代码 - 不需要
${onnxruntime_CUDNN_HOME}/include
```

**问题**: FastGelu 不需要 CUDNN，添加这个依赖是多余的。

---

## ✅ 修复方案

### 修复 1: 使用精确的 GLOB 而非 GLOB_RECURSE

```cmake
# 修复后 - 明确指定每个目录
file(GLOB onnxruntime_providers_my_virtual_npu_cc_srcs CONFIGURE_DEPENDS
  "${ONNXRUNTIME_ROOT}/core/providers/my_virtual_npu/*.h"
  "${ONNXRUNTIME_ROOT}/core/providers/my_virtual_npu/*.cc"
  "${ONNXRUNTIME_ROOT}/core/providers/my_virtual_npu/bert/*.h"
  "${ONNXRUNTIME_ROOT}/core/providers/my_virtual_npu/bert/*.cc"
)
# 不会收集 cuda/ 子目录
```

**优势**:
- 明确控制哪些文件被包含
- CPU 和 CUDA 文件完全分离
- 不需要复杂的过滤逻辑

### 修复 2: 显式设置 CUDA 语言

```cmake
# 修复后 - 明确设置 .cu 文件的语言
if(onnxruntime_USE_CUDA)
  file(GLOB onnxruntime_providers_my_virtual_npu_cu_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/my_virtual_npu/cuda/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/my_virtual_npu/cuda/*.cu"
    "${ONNXRUNTIME_ROOT}/core/providers/my_virtual_npu/cuda/*.cc"
  )

  # 关键：设置 CUDA 语言
  set_source_files_properties(${onnxruntime_providers_my_virtual_npu_cu_srcs}
                              PROPERTIES LANGUAGE CUDA)

  list(APPEND onnxruntime_providers_my_virtual_npu_srcs
              ${onnxruntime_providers_my_virtual_npu_cu_srcs})
endif()
```

**优势**:
- CMake 知道使用 nvcc 编译 .cu 文件
- 正确应用 CUDA 编译选项
- 避免链接错误

### 修复 3: 添加 CUDA 编译选项

```cmake
# 修复后 - 添加 CUDA 特定编译选项
if(onnxruntime_USE_CUDA)
  target_include_directories(onnxruntime_providers_my_virtual_npu PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  )

  # 关键：CUDA 编译选项
  if(CMAKE_CUDA_COMPILER)
    target_compile_options(onnxruntime_providers_my_virtual_npu PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )
  endif()
endif()
```

**优势**:
- `--expt-relaxed-constexpr`: 允许在 constexpr 中使用更灵活的语法
- `$<$<COMPILE_LANGUAGE:CUDA>:...>`: 只对 CUDA 文件应用这些选项

### 修复 4: 移除不必要的依赖

```cmake
# 修复后 - 只包含必需的头文件
if(onnxruntime_USE_CUDA)
  target_include_directories(onnxruntime_providers_my_virtual_npu PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    # 移除了 ${onnxruntime_CUDNN_HOME}/include
  )
endif()
```

**优势**:
- 减少不必要的依赖
- 加快编译速度
- 避免潜在的版本冲突

---

## 完整的修复后配置

### CPU-only 构建流程

```cmake
# 当 onnxruntime_USE_CUDA=OFF 时
file(GLOB ... my_virtual_npu/*.cc bert/*.cc)  # 只收集 CPU 文件
onnxruntime_add_static_library(...)           # 创建库
target_link_libraries(... onnxruntime_common) # 链接 CPU 依赖
```

**结果**:
- ✅ 不包含任何 CUDA 代码
- ✅ 不链接 CUDA 库
- ✅ 可在没有 CUDA 的机器上编译

### CPU + CUDA 构建流程

```cmake
# 当 onnxruntime_USE_CUDA=ON 时
file(GLOB ... my_virtual_npu/*.cc bert/*.cc)     # CPU 文件
file(GLOB ... my_virtual_npu/cuda/*.cu *.cc)     # CUDA 文件
set_source_files_properties(... LANGUAGE CUDA)   # 设置语言
onnxruntime_add_static_library(...)              # 创建库（含 CPU + CUDA）
target_compile_options(... --expt-relaxed-constexpr) # CUDA 选项
target_link_libraries(... CUDA::cudart CUDA::cublas) # CUDA 库
```

**结果**:
- ✅ 包含 CPU 和 CUDA 代码
- ✅ .cu 文件用 nvcc 编译
- ✅ 链接 CUDA runtime 和 cublas
- ✅ 支持 FP32/FP16/BFloat16

---

## 验证清单

### ✅ CPU-only 编译验证

```bash
cmake ../cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -Donnxruntime_USE_MY_VIRTUAL_NPU=ON \
  -Donnxruntime_USE_CUDA=OFF

cmake --build . -j$(nproc)
```

**预期**:
- ✅ 只编译 CPU 文件
- ✅ 生成 `libonnxruntime_providers_my_virtual_npu.a`
- ✅ 没有 CUDA 相关错误

### ✅ CPU + CUDA 编译验证

```bash
cmake ../cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -Donnxruntime_USE_MY_VIRTUAL_NPU=ON \
  -Donnxruntime_USE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="75;80;86"

cmake --build . -j$(nproc)
```

**预期**:
- ✅ 编译 CPU 和 CUDA 文件
- ✅ 看到 "MyVirtualNPU: CUDA support enabled" 消息
- ✅ .cu 文件用 nvcc 编译
- ✅ 生成包含 CUDA kernels 的库

---

## 对比：修复前后

| 方面 | 修复前 ❌ | 修复后 ✅ |
|------|---------|----------|
| **文件收集** | GLOB_RECURSE + FILTER | 精确的 GLOB 分离 |
| **CUDA 语言** | 未设置 | set_source_files_properties |
| **CUDA 选项** | 缺失 | --expt-relaxed-constexpr |
| **依赖管理** | 包含 CUDNN | 只包含必需依赖 |
| **可维护性** | 复杂过滤逻辑 | 清晰的分离 |
| **错误提示** | 无明确提示 | message(STATUS ...) |

---

## 参考：ONNXRuntime 官方实现

参考 `cmake/onnxruntime_providers_cuda.cmake`:

```cmake
# CUDA Provider 的实现方式
file(GLOB_RECURSE onnxruntime_providers_cuda_cc_srcs ...)  # C++ 文件
file(GLOB_RECURSE onnxruntime_providers_cuda_cu_srcs ...)  # CUDA 文件
source_group(...)
set(onnxruntime_providers_cuda_src ${cc_srcs} ${cu_srcs})
```

我们的实现遵循相同的模式，但更简单（因为我们没有复杂的 contrib ops）。

---

## 总结

修复的 CMake 配置现在：

1. ✅ **正确分离 CPU 和 CUDA 代码**
2. ✅ **显式设置 CUDA 语言属性**
3. ✅ **添加必要的 CUDA 编译选项**
4. ✅ **移除不必要的依赖**
5. ✅ **提供清晰的构建消息**
6. ✅ **支持 CPU-only 和 CPU+CUDA 两种模式**

现在可以正确编译了！🎉
