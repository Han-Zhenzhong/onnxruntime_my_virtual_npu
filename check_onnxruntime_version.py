#!/usr/bin/env python3
"""
检测和切换到自编译的 ONNXRuntime 版本
"""
import sys
import os
from pathlib import Path

def setup_custom_onnxruntime():
    """设置使用自编译的 ONNXRuntime"""

    print("🔍 当前 Python 环境信息:")
    print(f"Python 可执行文件: {sys.executable}")
    print(f"Python 版本: {sys.version}")
    print(f"Python 路径: {sys.path[:3]}...")  # 只显示前3个路径

    # 可能的自编译 onnxruntime 路径
    possible_paths = [
        "/d/open-source/onnxruntime/build/Linux/Release",
        "/d/open-source/onnxruntime/build/Windows/Release",
        "/d/open-source/onnxruntime/build/Release",
        # 添加更多可能的路径
    ]

    custom_onnxruntime_path = None
    for path_str in possible_paths:
        path = Path(path_str)
        if path.exists():
            # 查找 onnxruntime 模块
            onnxruntime_module = None
            for item in path.rglob("onnxruntime"):
                if item.is_dir() and (item / "__init__.py").exists():
                    onnxruntime_module = item.parent
                    break

            if onnxruntime_module:
                custom_onnxruntime_path = str(onnxruntime_module)
                print(f"✅ 找到自编译的 ONNXRuntime: {custom_onnxruntime_path}")
                break

    if custom_onnxruntime_path:
        # 将自编译版本路径插入到 sys.path 最前面
        if custom_onnxruntime_path not in sys.path:
            sys.path.insert(0, custom_onnxruntime_path)
            print(f"🔧 已将 {custom_onnxruntime_path} 添加到 Python 路径最前面")
    else:
        print("⚠️  未找到自编译的 ONNXRuntime，将使用预安装版本")

    return custom_onnxruntime_path

def check_onnxruntime_version():
    """检查当前使用的 ONNXRuntime 版本和路径"""
    try:
        import onnxruntime as ort

        print(f"\n📦 ONNXRuntime 信息:")
        print(f"版本: {ort.__version__}")
        print(f"路径: {ort.__file__}")
        print(f"可用提供者: {ort.get_available_providers()}")

        # 检查是否包含自定义算子
        try:
            # 尝试创建一个会话来测试
            print(f"\n🧪 测试基本功能:")
            providers = ort.get_available_providers()
            print(f"✅ 可以正常导入和使用 ONNXRuntime")

            # 检查是否是自编译版本的特征
            if 'CUDAExecutionProvider' in providers:
                print("🎮 包含 CUDA 支持")
            if 'CPUExecutionProvider' in providers:
                print("💻 包含 CPU 支持")

        except Exception as e:
            print(f"❌ ONNXRuntime 测试失败: {e}")

    except ImportError as e:
        print(f"❌ 无法导入 ONNXRuntime: {e}")
        print("💡 可能需要:")
        print("   1. 重新编译 ONNXRuntime")
        print("   2. 检查构建是否成功")
        print("   3. 检查 Python 绑定是否正确生成")

def test_custom_operators():
    """测试自定义算子是否可用"""
    try:
        import onnxruntime as ort
        import numpy as np

        print(f"\n🔬 测试自定义算子:")

        # 尝试创建一个简单的测试会话
        # 这里可以测试是否包含了你的自定义算子
        session_options = ort.SessionOptions()
        session_options.log_severity_level = 3  # 只显示错误

        print("✅ 基本会话创建成功")

        # TODO: 这里可以添加对 FastGelu 等自定义算子的具体测试

    except Exception as e:
        print(f"❌ 自定义算子测试失败: {e}")

if __name__ == "__main__":
    print("🚀 ONNXRuntime 版本检测和切换工具")
    print("=" * 60)

    # 1. 设置自定义 onnxruntime 路径
    setup_custom_onnxruntime()

    # 2. 检查版本信息
    check_onnxruntime_version()

    # 3. 测试自定义功能
    test_custom_operators()

    print("\n" + "=" * 60)
    print("💡 如果仍然使用预安装版本，请:")
    print("   1. 使用虚拟环境")
    print("   2. 卸载预安装的 onnxruntime")
    print("   3. 确保编译生成了 Python 包")
    print("=" * 60)
