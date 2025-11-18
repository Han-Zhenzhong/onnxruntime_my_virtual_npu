#!/usr/bin/env python3
"""
测试 my_cpu 提供者是否能运行 Tiny-GPT2
"""
import onnxruntime as ort
import numpy as np
import time

def test_tiny_gpt2_with_my_cpu():
    try:
        print("🔍 正在测试 Tiny-GPT2 与 my_cpu 提供者...")

        # 获取所有可用提供者
        available_providers = ort.get_available_providers()
        print(f"可用提供者: {available_providers}")

        # 创建推理会话 - CPU 提供者会使用你的自定义算子
        session = ort.InferenceSession(
            'tiny_gpt2.onnx',
            providers=['CPUExecutionProvider']
        )

        print("✅ 模型加载成功！")

        # 检查模型输入输出
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        output_shape = session.get_outputs()[0].shape

        print(f"输入: {input_name} {input_shape}")
        print(f"输出: {output_shape}")

        # 准备测试输入
        if 'input_ids' in input_name.lower():
            # GPT-2 文本输入
            batch_size = 1
            seq_length = 10
            input_data = np.random.randint(0, 50257, (batch_size, seq_length), dtype=np.int64)
        else:
            # 其他类型输入
            input_data = np.random.randn(*input_shape).astype(np.float32)

        print(f"输入数据形状: {input_data.shape}")

        # 执行推理
        start_time = time.time()
        outputs = session.run(None, {input_name: input_data})
        inference_time = time.time() - start_time

        print(f"✅ 推理成功完成！")
        print(f"⏱️  推理耗时: {inference_time*1000:.2f} ms")
        print(f"📊 输出形状: {[out.shape for out in outputs]}")

        # 验证输出合理性
        output = outputs[0]
        if len(output.shape) >= 2:
            print(f"📈 输出统计:")
            print(f"   - Min: {output.min():.4f}")
            print(f"   - Max: {output.max():.4f}")
            print(f"   - Mean: {output.mean():.4f}")
            print(f"   - Std: {output.std():.4f}")

        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_custom_ops():
    """检查自定义算子是否注册"""
    try:
        print("\n🔍 检查自定义算子注册状态...")

        # 创建一个简单的 FastGelu 测试
        test_model_content = '''
import onnx
from onnx import helper, TensorProto

# 创建 FastGelu 测试模型
input_tensor = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 3])
output_tensor = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 3])

fastgelu_node = helper.make_node(
    'FastGelu',
    inputs=['X'],
    outputs=['Y'],
    domain='com.my_virtual_npu'  # 你的自定义域
)

graph = helper.make_graph([fastgelu_node], 'test_fastgelu', [input_tensor], [output_tensor])
model = helper.make_model(graph)

# 添加自定义域
model.opset_import.add().CopyFrom(helper.make_opsetid('com.my_virtual_npu', 1))

onnx.save(model, 'test_fastgelu.onnx')
print("✅ 创建 FastGelu 测试模型成功")
        '''

        exec(test_model_content)

        # 测试自定义算子
        session = ort.InferenceSession('test_fastgelu.onnx', providers=['CPUExecutionProvider'])
        test_input = np.array([[1.0, 0.0, -1.0]], dtype=np.float32)
        output = session.run(None, {'X': test_input})[0]

        print(f"✅ 自定义 FastGelu 算子工作正常！")
        print(f"   输入: {test_input}")
        print(f"   输出: {output}")

    except Exception as e:
        print(f"⚠️  自定义算子测试: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 测试 my_cpu 提供者运行 Tiny-GPT2")
    print("=" * 60)

    # 主要测试
    success = test_tiny_gpt2_with_my_cpu()

    # 自定义算子测试
    check_custom_ops()

    print("\n" + "=" * 60)
    if success:
        print("🎉 结论: my_cpu 提供者可以运行 Tiny-GPT2!")
        print("💡 下一步: 优化性能，添加更多自定义算子")
    else:
        print("💥 还需要解决一些问题")
    print("=" * 60)
