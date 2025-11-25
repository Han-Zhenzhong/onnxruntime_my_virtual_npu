#!/bin/bash
# Quick build and test script for MyVirtualNPU with CUDA support

set -e

echo "=================================================="
echo "MyVirtualNPU CUDA 快速编译和测试脚本"
echo "=================================================="
echo ""

# 检测CUDA是否可用
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
    echo "✓ CUDA detected: $CUDA_VERSION"
    USE_CUDA=ON
else
    echo "✗ CUDA not found, building CPU-only version"
    USE_CUDA=OFF
fi

# 选择构建模式
echo ""
echo "选择构建模式:"
echo "1) CPU only"
echo "2) CPU + CUDA"
read -p "请选择 (1 or 2): " choice

case $choice in
    1)
        echo "Building CPU-only version..."
        USE_CUDA=OFF
        ;;
    2)
        if [ "$USE_CUDA" = "OFF" ]; then
            echo "错误: CUDA 未安装，无法编译 CUDA 版本"
            exit 1
        fi
        echo "Building CPU + CUDA version..."
        USE_CUDA=ON
        ;;
    *)
        echo "无效选择"
        exit 1
        ;;
esac

# 创建构建目录
BUILD_DIR="build/Linux/Release"
mkdir -p $BUILD_DIR
cd $BUILD_DIR

# 清理旧的 CMake 缓存
echo ""
echo "清理旧的构建文件..."
rm -rf CMakeCache.txt CMakeFiles/

# 配置 CMake
echo ""
echo "配置 CMake..."

if [ "$USE_CUDA" = "ON" ]; then
    cmake ../../../cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -Donnxruntime_USE_MY_VIRTUAL_NPU=ON \
        -Donnxruntime_USE_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="75;80;86" \
        -Donnxruntime_BUILD_SHARED_LIB=ON \
        -Donnxruntime_BUILD_UNIT_TESTS=ON
else
    cmake ../../../cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -Donnxruntime_USE_MY_VIRTUAL_NPU=ON \
        -Donnxruntime_BUILD_SHARED_LIB=ON \
        -Donnxruntime_BUILD_UNIT_TESTS=ON
fi

# 编译
echo ""
echo "编译中..."
cmake --build . -j$(nproc)

# 检查编译结果
if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "✓ 编译成功！"
    echo "=================================================="
    echo ""

    # 显示生成的库
    echo "生成的库文件:"
    if [ "$USE_CUDA" = "ON" ]; then
        ls -lh libonnxruntime*.so 2>/dev/null || true
        echo ""
        echo "CUDA 支持已启用 🚀"
    else
        ls -lh libonnxruntime*.so 2>/dev/null || true
        echo ""
        echo "CPU-only 模式"
    fi

    echo ""
    echo "=================================================="
    echo "下一步:"
    echo "=================================================="
    echo ""
    echo "1. 运行单元测试:"
    echo "   ./onnxruntime_test_all --gtest_filter='*FastGelu*'"
    echo ""
    echo "2. 运行 Python 测试:"
    echo "   cd ../../.."
    echo "   python test_tiny_gpt2.py"
    echo ""
    if [ "$USE_CUDA" = "ON" ]; then
        echo "3. 验证 CUDA 可用:"
        echo "   nvidia-smi"
        echo "   python -c 'import onnxruntime as ort; print(ort.get_available_providers())'"
        echo ""
    fi
    echo "=================================================="
else
    echo ""
    echo "✗ 编译失败"
    exit 1
fi
