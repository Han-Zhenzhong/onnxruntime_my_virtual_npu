#!/bin/bash
# Quick verification script for MyVirtualNPU refactoring

set -e

echo "=========================================="
echo "MyVirtualNPU 架构重构验证脚本"
echo "=========================================="
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

check_pass() {
    echo -e "${GREEN}✓${NC} $1"
}

check_fail() {
    echo -e "${RED}✗${NC} $1"
    exit 1
}

check_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
}

echo "1. 检查文件结构..."

# 检查新增文件
if [ -f "cmake/onnxruntime_providers_my_virtual_npu.cmake" ]; then
    check_pass "独立 CMake 配置文件存在"
else
    check_fail "缺少 cmake/onnxruntime_providers_my_virtual_npu.cmake"
fi

if [ -f "onnxruntime/core/providers/my_virtual_npu/my_virtual_npu_execution_provider.h" ]; then
    check_pass "EP 头文件存在"
else
    check_fail "缺少 my_virtual_npu_execution_provider.h"
fi

if [ -f "onnxruntime/core/providers/my_virtual_npu/my_virtual_npu_execution_provider.cc" ]; then
    check_pass "EP 实现文件存在"
else
    check_fail "缺少 my_virtual_npu_execution_provider.cc"
fi

echo ""
echo "2. 检查解耦情况..."

# 检查 CPU Provider 不再包含 my_virtual_npu
if grep -q "my_virtual_npu" cmake/onnxruntime_providers_cpu.cmake; then
    check_fail "onnxruntime_providers_cpu.cmake 仍然包含 my_virtual_npu 引用"
else
    check_pass "CPU Provider CMake 已清理"
fi

if grep -q "my_virtual_npu_kernels.h" onnxruntime/core/providers/cpu/cpu_execution_provider.cc; then
    check_fail "cpu_execution_provider.cc 仍然包含 my_virtual_npu include"
else
    check_pass "CPU Provider 代码已解耦"
fi

if grep -q "RegisterMyVirtualNpuKernels" onnxruntime/core/providers/cpu/cpu_execution_provider.cc; then
    check_fail "cpu_execution_provider.cc 仍然调用 RegisterMyVirtualNpuKernels"
else
    check_pass "CPU Provider 不再注册 my_virtual_npu kernels"
fi

echo ""
echo "3. 检查 CMake 配置..."

# 检查主 CMake 配置
if grep -q "onnxruntime_USE_MY_VIRTUAL_NPU" cmake/CMakeLists.txt; then
    check_pass "CMake 选项已添加"
else
    check_fail "CMakeLists.txt 缺少 onnxruntime_USE_MY_VIRTUAL_NPU 选项"
fi

if grep -q "PROVIDERS_MY_VIRTUAL_NPU" cmake/onnxruntime_providers.cmake; then
    check_pass "Provider 变量已定义"
else
    check_fail "onnxruntime_providers.cmake 缺少 PROVIDERS_MY_VIRTUAL_NPU"
fi

if grep -q "onnxruntime_providers_my_virtual_npu.cmake" cmake/onnxruntime_providers.cmake; then
    check_pass "Provider CMake 已包含"
else
    check_fail "onnxruntime_providers.cmake 未包含 my_virtual_npu cmake"
fi

echo ""
echo "4. 检查常量定义..."

if grep -q "kMyVirtualNpuExecutionProvider" include/onnxruntime/core/graph/constants.h; then
    check_pass "EP 常量已定义"
else
    check_fail "constants.h 缺少 kMyVirtualNpuExecutionProvider"
fi

echo ""
echo "5. 检查测试配置..."

if grep -q "onnxruntime_USE_MY_VIRTUAL_NPU" cmake/onnxruntime_unittests.cmake; then
    check_pass "单元测试配置已更新"
else
    check_warn "单元测试配置可能需要更新"
fi

echo ""
echo "=========================================="
echo -e "${GREEN}架构验证通过！${NC}"
echo "=========================================="
echo ""
echo "下一步："
echo "1. 清理旧的构建："
echo "   rm -rf build/Linux/Release/CMakeCache.txt build/Linux/Release/CMakeFiles/"
echo ""
echo "2. 重新配置 CMake："
echo "   cd build/Linux/Release"
echo "   cmake ../../../cmake -DCMAKE_BUILD_TYPE=Release -Donnxruntime_USE_MY_VIRTUAL_NPU=ON"
echo ""
echo "3. 编译："
echo "   cmake --build . -j\$(nproc)"
echo ""
echo "4. 运行测试："
echo "   ./onnxruntime_test_all --gtest_filter='*FastGelu*'"
echo ""
