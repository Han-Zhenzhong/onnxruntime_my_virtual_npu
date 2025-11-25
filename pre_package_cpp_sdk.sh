#!/bin/bash
# 打包 ONNXRuntime C++ SDK
# 将编译好的库、头文件打包到 Release 目录供 C++ 项目使用

set -e

echo "📦 打包 ONNXRuntime C++ SDK..."

# 配置（可通过环境变量或命令行参数覆盖）
BUILD_DIR="${1:-${BUILD_DIR:-build/Linux/Release}}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RELEASE_DIR="${2:-${RELEASE_DIR:-PreRelease_${TIMESTAMP}/cpp}}"
VERSION="${VERSION:-1.20.0-custom}"

# 检查 BUILD_DIR 是否存在
if [ ! -d "$BUILD_DIR" ]; then
    echo "❌ 错误: 构建目录不存在: $BUILD_DIR"
    echo ""
    echo "可用的构建目录:"
    find build -maxdepth 2 -type d -name "Release" 2>/dev/null || echo "  (未找到)"
    echo ""
    echo "用法: $0 [BUILD_DIR] [RELEASE_DIR]"
    echo "示例: $0 build/Linux/Release PreRelease_20231119_143025/cpp"
    exit 1
fi

echo "📂 使用构建目录: $BUILD_DIR"

# 清理并创建目录结构
echo "🗂️  创建目录结构..."
rm -rf "$RELEASE_DIR"
mkdir -p "$RELEASE_DIR"/{include,lib,bin}

# ============================================
# 1. 复制头文件
# ============================================
echo "📋 复制头文件..."

# 主要的公共 API 头文件
mkdir -p "$RELEASE_DIR/include/onnxruntime/core/session"
cp include/onnxruntime/core/session/onnxruntime_c_api.h "$RELEASE_DIR/include/onnxruntime/core/session/"
cp include/onnxruntime/core/session/onnxruntime_cxx_api.h "$RELEASE_DIR/include/onnxruntime/core/session/"
cp include/onnxruntime/core/session/onnxruntime_cxx_inline.h "$RELEASE_DIR/include/onnxruntime/core/session/"
cp include/onnxruntime/core/session/onnxruntime_run_options_config_keys.h "$RELEASE_DIR/include/onnxruntime/core/session/" 2>/dev/null || true
cp include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h "$RELEASE_DIR/include/onnxruntime/core/session/" 2>/dev/null || true

# Provider 相关头文件
mkdir -p "$RELEASE_DIR/include/onnxruntime/core/providers/cpu"
cp include/onnxruntime/core/providers/cpu/cpu_provider_factory.h "$RELEASE_DIR/include/onnxruntime/core/providers/cpu/" 2>/dev/null || true

# 其他重要头文件
cp -r include/onnxruntime/core/framework "$RELEASE_DIR/include/onnxruntime/core/" 2>/dev/null || true
cp -r include/onnxruntime/core/common "$RELEASE_DIR/include/onnxruntime/core/" 2>/dev/null || true
cp -r include/onnxruntime/core/graph "$RELEASE_DIR/include/onnxruntime/core/" 2>/dev/null || true

# 自定义算子头文件（如果需要用户实现自定义算子）
mkdir -p "$RELEASE_DIR/include/onnxruntime/core/providers/my_virtual_npu"
if [ -d "onnxruntime/core/providers/my_virtual_npu" ]; then
    find onnxruntime/core/providers/my_virtual_npu -name "*.h" -exec cp --parents {} "$RELEASE_DIR/include/" \;
fi

echo "✅ 头文件复制完成"

# ============================================
# 2. 复制库文件
# ============================================
echo "📚 复制库文件..."

# 主库
if [ -f "$BUILD_DIR/libonnxruntime.so" ]; then
    cp "$BUILD_DIR/libonnxruntime.so"* "$RELEASE_DIR/lib/" 2>/dev/null || true
    echo "✅ 复制 libonnxruntime.so"
fi

# 静态库（如果存在）
if [ -f "$BUILD_DIR/libonnxruntime.a" ]; then
    cp "$BUILD_DIR/libonnxruntime.a" "$RELEASE_DIR/lib/"
    echo "✅ 复制 libonnxruntime.a"
fi

# 其他依赖库
cp "$BUILD_DIR"/lib*.so* "$RELEASE_DIR/lib/" 2>/dev/null || true

echo "✅ 库文件复制完成"

# ============================================
# 3. 复制二进制工具（如果需要）
# ============================================
echo "🔧 复制工具..."
if [ -f "$BUILD_DIR/onnxruntime_test_all" ]; then
    cp "$BUILD_DIR/onnxruntime_test_all" "$RELEASE_DIR/bin/" 2>/dev/null || true
fi

# ============================================
# 4. 创建 CMake 配置文件
# ============================================
echo "⚙️  创建 CMake 配置..."
cat > "$RELEASE_DIR/ONNXRuntimeConfig.cmake" << 'EOF'
# ONNXRuntime CMake Configuration

get_filename_component(ONNXRUNTIME_CMAKE_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
set(ONNXRUNTIME_INCLUDE_DIRS "${ONNXRUNTIME_CMAKE_DIR}/include")
set(ONNXRUNTIME_LIBRARIES "${ONNXRUNTIME_CMAKE_DIR}/lib/libonnxruntime.so")

# 创建导入目标
add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARIES}"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIRS}"
)

message(STATUS "Found ONNXRuntime: ${ONNXRUNTIME_CMAKE_DIR}")
EOF

# ============================================
# 5. 创建 pkg-config 文件
# ============================================
echo "⚙️  创建 pkg-config 文件..."
cat > "$RELEASE_DIR/onnxruntime.pc" << EOF
prefix=$(pwd)/$RELEASE_DIR
exec_prefix=\${prefix}
libdir=\${prefix}/lib
includedir=\${prefix}/include

Name: ONNXRuntime
Description: ONNX Runtime - cross-platform ML inference engine
Version: $VERSION
Libs: -L\${libdir} -lonnxruntime
Cflags: -I\${includedir}
EOF

# ============================================
# 6. 创建示例代码
# ============================================
echo "📝 创建示例代码..."
mkdir -p "$RELEASE_DIR/examples"
cat > "$RELEASE_DIR/examples/simple_inference.cpp" << 'EOF'
// ONNXRuntime C++ API 使用示例
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>

int main() {
    // 1. 创建环境
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");

    // 2. 创建会话选项
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    // 3. 创建会话
    Ort::Session session(env, "model.onnx", session_options);

    // 4. 打印输入输出信息
    size_t num_input_nodes = session.GetInputCount();
    size_t num_output_nodes = session.GetOutputCount();

    std::cout << "输入节点数: " << num_input_nodes << std::endl;
    std::cout << "输出节点数: " << num_output_nodes << std::endl;

    return 0;
}
EOF

cat > "$RELEASE_DIR/examples/CMakeLists.txt" << 'EOF'
cmake_minimum_required(VERSION 3.13)
project(ONNXRuntime_Example)

set(CMAKE_CXX_STANDARD 17)

# 找到 ONNXRuntime
find_package(ONNXRuntime REQUIRED PATHS ${CMAKE_CURRENT_SOURCE_DIR}/..)

# 创建示例可执行文件
add_executable(simple_inference simple_inference.cpp)
target_link_libraries(simple_inference onnxruntime)
EOF

# ============================================
# 7. 创建 README
# ============================================
echo "📖 创建 README..."
cat > "$RELEASE_DIR/README.md" << EOF
# ONNXRuntime C++ SDK

版本: $VERSION
编译日期: $(date)

## 目录结构

\`\`\`
cpp/
├── include/          # 头文件
│   └── onnxruntime/
├── lib/              # 库文件
│   ├── libonnxruntime.so
│   └── ...
├── examples/         # 示例代码
├── ONNXRuntimeConfig.cmake  # CMake 配置
└── onnxruntime.pc    # pkg-config 文件
\`\`\`

## 使用方法

### 方式 1: 使用 CMake

在您的 CMakeLists.txt 中：

\`\`\`cmake
# 设置 ONNXRuntime 路径
set(ONNXRuntime_DIR /path/to/PreRelease/cpp)
find_package(ONNXRuntime REQUIRED)

# 链接库
add_executable(your_app main.cpp)
target_link_libraries(your_app onnxruntime)
\`\`\`

### 方式 2: 手动编译

\`\`\`bash
g++ -std=c++17 main.cpp \\
    -I/path/to/PreRelease/cpp/include \\
    -L/path/to/PreRelease/cpp/lib \\
    -lonnxruntime \\
    -o your_app

# 运行时设置库路径
export LD_LIBRARY_PATH=/path/to/PreRelease/cpp/lib:\$LD_LIBRARY_PATH
./your_app
\`\`\`

### 方式 3: 使用 pkg-config

\`\`\`bash
export PKG_CONFIG_PATH=/path/to/PreRelease/cpp:\$PKG_CONFIG_PATH

g++ main.cpp \$(pkg-config --cflags --libs onnxruntime) -o your_app
\`\`\`

## 示例代码

参见 \`examples/\` 目录：

\`\`\`bash
cd examples
mkdir build && cd build
cmake .. -DONNXRuntime_DIR=../..
make
./simple_inference
\`\`\`

## 自定义算子

本版本包含自定义 my_virtual_npu 算子，支持：
- FastGelu (domain: com.my_virtual_npu)

使用方式与标准算子相同，ONNXRuntime 会自动选择正确的实现。

## 依赖

- GCC 7+ 或 Clang 5+
- CMake 3.13+
- glibc 2.17+ (Linux)

## 支持

- GitHub: https://github.com/microsoft/onnxruntime
- 文档: https://onnxruntime.ai/docs/
EOF

# ============================================
# 8. 打印总结
# ============================================
echo ""
echo "✅ 打包完成！"
echo ""
echo "📦 发布包位置: $RELEASE_DIR"
echo ""ls
echo "📁 目录结构:"
tree -L 2 "$RELEASE_DIR" 2>/dev/null || find "$RELEASE_DIR" -maxdepth 2 -type d
echo ""
echo "📊 统计信息:"
echo "  头文件数量: $(find "$RELEASE_DIR/include" -name "*.h" | wc -l)"
echo "  库文件数量: $(find "$RELEASE_DIR/lib" -name "*.so*" -o -name "*.a" | wc -l)"
echo ""
echo "💡 使用说明:"
echo "  1. 将 $RELEASE_DIR 目录复制到目标机器"
echo "  2. 参考 $RELEASE_DIR/README.md 使用"
echo "  3. 运行 examples 中的示例代码测试"
echo ""

# 可选：创建压缩包
read -p "是否创建 tar.gz 压缩包？(y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    TARBALL="onnxruntime-cpp-$VERSION-$(uname -m)-${TIMESTAMP}.tar.gz"
    tar -czf "$TARBALL" -C "$(dirname "$RELEASE_DIR")" "$(basename "$RELEASE_DIR")"
    echo "✅ 压缩包已创建: $TARBALL"
    echo "   大小: $(du -h "$TARBALL" | cut -f1)"
fi
