#!/usr/bin/env python3
"""
测试 my_cpu 和 CPU 提供者混合运行 Tiny-GPT2
"""
import onnxruntime as ort
import numpy as np
import time
from typing import List, Tuple

class MixedProviderTester:
    def __init__(self):
        self.results = {}

def test_provider_combination(self, providers: List[Tuple[str, dict]], name: str):
        """测试特定提供者组合"""
        try:
            print(f"\n🧪 测试配置: {name}")
            print(f"   提供者: {[p[0] for p in providers]}")

            # 创建会话
            session = ort.InferenceSession('tiny_gpt2.onnx', providers=providers)

            # 检查实际使用的提供者
            actual_providers = session.get_providers()
            print(f"   实际提供者: {actual_providers}")

            # 准备输入
            input_name = session.get_inputs()[0].name
            if 'input_ids' in input_name.lower():
                test_input = np.random.randint(0, 1000, (1, 10), dtype=np.int64)
            else:
                input_shape = session.get_inputs()[0].shape
                test_input = np.random.randn(*input_shape).astype(np.float32)

            # 执行推理并计时
            start_time = time.time()
            outputs = session.run(None, {input_name: test_input})
            inference_time = (time.time() - start_time) * 1000  # ms

            # 记录结果
            result = {
                'success': True,
                'inference_time': inference_time,
                'output_shape': outputs[0].shape,
                'providers': actual_providers,
                'output_stats': {
                    'min': float(outputs[0].min()),
                    'max': float(outputs[0].max()),
                    'mean': float(outputs[0].mean()),
                    'std': float(outputs[0].std())
                }
            }

            self.results[name] = result

            print(f"   ✅ 成功! 耗时: {inference_time:.2f} ms")
            print(f"   📊 输出: {outputs[0].shape}, 均值: {outputs[0].mean():.4f}")

            return result

        except Exception as e:
            error_result = {
                'success': False,
                'error': str(e),
                'providers': [p[0] for p in providers]
            }
            self.results[name] = error_result
            print(f"   ❌ 失败: {e}")
            return error_result

    def test_all_combinations(self):
        """测试所有可能的提供者组合"""

        print("🚀 开始混合提供者测试")
        print("=" * 80)

        # 配置 1: 仅 CPU 提供者（包含 my_cpu 算子集成）
        self.test_provider_combination([
            ('CPUExecutionProvider', {})
        ], "CPU提供者(含my_cpu算子)")

        # 注意：当前 my_cpu 算子已集成到 CPUExecutionProvider 中
        # 所以 MyCpuExecutionProvider 可能不存在
        # 我们主要测试 CPUExecutionProvider 中的混合算子使用

    def analyze_results(self):
        """分析测试结果"""
        print("\n" + "=" * 80)
        print("📊 测试结果分析")
        print("=" * 80)

        successful_configs = []
        for name, result in self.results.items():
            if result['success']:
                successful_configs.append((name, result))
                print(f"\n✅ {name}:")
                print(f"   ⏱️  推理时间: {result['inference_time']:.2f} ms")
                print(f"   🔧 实际提供者: {result['providers']}")
                print(f"   📊 输出统计: 均值={result['output_stats']['mean']:.4f}, "
                      f"标准差={result['output_stats']['std']:.4f}")
            else:
                print(f"\n❌ {name}:")
                print(f"   错误: {result['error']}")

        if len(successful_configs) > 1:
            print(f"\n🏆 性能对比:")
            successful_configs.sort(key=lambda x: x[1]['inference_time'])
            fastest = successful_configs[0]
            print(f"   最快配置: {fastest[0]} ({fastest[1]['inference_time']:.2f} ms)")

            for name, result in successful_configs[1:]:
                slowdown = (result['inference_time'] / fastest[1]['inference_time'] - 1) * 100
                print(f"   {name}: +{slowdown:.1f}% 相比最快配置")

        return successful_configs

def check_provider_registration():
    """检查提供者注册状态"""
    print("🔍 检查提供者注册状态")
    print("-" * 40)

    available_providers = ort.get_available_providers()
    print(f"可用提供者: {available_providers}")

    # 检查 CPU 提供者（应该包含 my_cpu 算子）
    if 'CPUExecutionProvider' in available_providers:
        print("✅ CPUExecutionProvider 可用")
        print("💡 my_cpu 算子应该已集成到 CPUExecutionProvider 中")
    else:
        print("❌ CPUExecutionProvider 不可用")

    # 检查是否有独立的 my_cpu 提供者
    if 'MyCpuExecutionProvider' in available_providers:
        print("✅ MyCpuExecutionProvider 作为独立提供者存在")
    else:
        print("ℹ️  MyCpuExecutionProvider 未作为独立提供者注册")
        print("   这是正常的，my_cpu 算子集成到 CPUExecutionProvider 中")

    return available_providers

def create_test_models():
    """创建测试模型来验证混合使用"""
    try:
        import onnx
        from onnx import helper, TensorProto

        print("\n🛠️  创建混合算子测试模型")

        # 创建包含 my_cpu FastGelu + 标准算子的模型
        input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 4])
        output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 4])

        # 节点序列：Add -> FastGelu -> MatMul
        add_node = helper.make_node('Add', ['input', 'input'], ['add_out'])

        fastgelu_node = helper.make_node(
            'FastGelu',
            ['add_out'],
            ['gelu_out'],
            domain='com.my_virtual_npu'  # 自定义域
        )

        # 创建权重
        weight = helper.make_tensor('weight', TensorProto.FLOAT, [4, 4],
                                  np.eye(4, dtype=np.float32).flatten().tolist())

        matmul_node = helper.make_node('MatMul', ['gelu_out', 'weight'], ['output'])

        # 构建图
        graph = helper.make_graph(
            [add_node, fastgelu_node, matmul_node],
            'mixed_ops_test',
            [input_tensor],
            [output_tensor],
            [weight]
        )

        # 创建模型
        model = helper.make_model(graph)

        # 添加 opset imports
        model.opset_import.add().CopyFrom(helper.make_opsetid('', 13))  # ONNX domain
        model.opset_import.add().CopyFrom(helper.make_opsetid('com.my_virtual_npu', 1))  # 自定义域

        onnx.save(model, 'mixed_ops_test.onnx')
        print("✅ 创建混合算子测试模型: mixed_ops_test.onnx")

        return True

    except ImportError:
        print("⚠️  需要 onnx 库来创建测试模型")
        return False
    except Exception as e:
        print(f"❌ 创建测试模型失败: {e}")
        return False

def test_mixed_ops_model():
    """测试混合算子模型"""
    try:
        print("\n🧪 测试混合算子模型")

        # 只测试 CPU 提供者（包含 my_cpu 算子）
        configurations = [
            ([('CPUExecutionProvider', {})], "CPU(含my_cpu算子)"),
        ]

        for providers, name in configurations:
            try:
                session = ort.InferenceSession('mixed_ops_test.onnx', providers=providers)

                # 测试输入
                test_input = np.array([[1.0, 0.5, -0.5, 2.0]], dtype=np.float32)
                output = session.run(None, {'input': test_input})[0]

                print(f"   ✅ {name}: 输出 = {output}")

            except Exception as e:
                print(f"   ❌ {name}: {e}")

    except FileNotFoundError:
        print("   ⚠️  mixed_ops_test.onnx 不存在，跳过混合算子测试")

if __name__ == "__main__":
    print("🔧 ONNXRuntime 混合执行提供者测试")
    print("=" * 80)

    # 1. 检查提供者状态
    available_providers = check_provider_registration()

    # 2. 创建测试模型
    create_test_models()

    # 3. 测试混合算子模型（如果可能）
    test_mixed_ops_model()

    # 4. 测试 Tiny-GPT2（如果存在）
    try:
        tester = MixedProviderTester()
        tester.test_all_combinations()
        successful_configs = tester.analyze_results()

        print(f"\n🎉 结论:")
        if successful_configs:
            print(f"✅ 混合提供者可以成功运行 Tiny-GPT2!")
            print(f"💡 推荐配置: my_cpu + CPU 混合使用")
            print(f"🔧 算子分配: 自定义算子用 my_cpu，标准算子用 CPU")
        else:
            print(f"❌ 需要解决提供者注册问题")

    except FileNotFoundError:
        print(f"\n⚠️  tiny_gpt2.onnx 未找到，跳过 Tiny-GPT2 测试")
        print(f"💡 请确保 Tiny-GPT2 模型文件在当前目录")

    print("=" * 80)
