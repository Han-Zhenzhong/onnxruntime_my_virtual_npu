#!/bin/bash
"""
切换到自编译 ONNXRuntime 的脚本
"""

echo "🔧 设置自编译 ONNXRuntime 环境"
echo "================================="

# 创建虚拟环境（如果不存在）
if [ ! -d "venv_custom_ort" ]; then
    echo "📦 创建虚拟环境..."
    python -m venv venv_custom_ort
fi

# 激活虚拟环境
echo "🚀 激活虚拟环境..."
source venv_custom_ort/bin/activate

# 卸载任何现有的 onnxruntime
echo "🗑️  卸载预安装的 onnxruntime..."
pip uninstall onnxruntime onnxruntime-gpu onnxruntime-training -y

# 查找并安装自编译的 wheel
echo "🔍 查找自编译的 ONNXRuntime wheel..."
WHEEL_FILE=$(find build -name "*.whl" | head -1)

if [ -n "$WHEEL_FILE" ]; then
    echo "✅ 找到 wheel 文件: $WHEEL_FILE"
    echo "📦 安装自编译版本..."
    pip install "$WHEEL_FILE"
else
    echo "❌ 未找到 wheel 文件"
    echo "💡 请先运行："
    echo "   ./build.sh --config Release --build_shared_lib --build_wheel --parallel"
    exit 1
fi

# 验证安装
echo ""
echo "🧪 验证安装..."
python -c "
import onnxruntime as ort
print(f'ONNXRuntime 版本: {ort.__version__}')
print(f'安装路径: {ort.__file__}')
print(f'可用提供者: {ort.get_available_providers()}')
"

echo ""
echo "🎉 设置完成！"
echo "现在你可以运行："
echo "  python test_mixed_providers.py"
echo ""
echo "要退出虚拟环境，运行: deactivate"
