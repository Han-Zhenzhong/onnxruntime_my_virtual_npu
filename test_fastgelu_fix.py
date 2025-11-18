#!/usr/bin/env python3
"""
Test script to verify FastGelu domain conflict fix
"""

import numpy as np
import onnxruntime as ort
import onnx
from onnx import helper, TensorProto

def create_fastgelu_model():
    """Create a simple FastGelu model using the custom domain"""
    # Define input
    input_tensor = helper.make_tensor_value_info(
        'X', TensorProto.FLOAT, [2, 3]
    )

    # Define output
    output_tensor = helper.make_tensor_value_info(
        'Y', TensorProto.FLOAT, [2, 3]
    )

    # Create FastGelu node with custom domain
    fastgelu_node = helper.make_node(
        'FastGelu',
        inputs=['X'],
        outputs=['Y'],
        domain='com.my_virtual_npu'  # Use our custom domain
    )

    # Create graph
    graph = helper.make_graph(
        [fastgelu_node],
        'fastgelu_test',
        [input_tensor],
        [output_tensor]
    )

    # Create model
    model = helper.make_model(graph)

    # Add opset imports
    model.opset_import[0].domain = 'com.my_virtual_npu'
    model.opset_import[0].version = 1

    return model

def test_fastgelu_custom_domain():
    """Test that FastGelu with custom domain works"""
    try:
        print("Creating FastGelu model with custom domain...")
        model = create_fastgelu_model()

        # Save model for inspection
        onnx.save(model, 'fastgelu_custom_domain.onnx')
        print("✓ Model created successfully")

        # Test data
        input_data = np.array([
            [-1.0, 0.0, 1.0],
            [-0.5, 0.5, 2.0]
        ], dtype=np.float32)

        print(f"Input shape: {input_data.shape}")
        print(f"Input data:\n{input_data}")

        # This will test if our custom kernel is properly registered
        # and doesn't conflict with the existing contrib_ops FastGelu
        print("Attempting to create inference session...")
        session = ort.InferenceSession('fastgelu_custom_domain.onnx')
        print("✓ Inference session created successfully")

        # Run inference
        print("Running inference...")
        output = session.run(None, {'X': input_data})[0]
        print(f"Output shape: {output.shape}")
        print(f"Output data:\n{output}")

        print("✅ FastGelu custom domain test PASSED!")
        return True

    except Exception as e:
        print(f"❌ FastGelu custom domain test FAILED: {e}")
        return False

def test_contrib_fastgelu_still_works():
    """Test that the original contrib FastGelu still works"""
    try:
        print("\nTesting original contrib FastGelu...")

        # Create model using Microsoft domain (the original)
        input_tensor = helper.make_tensor_value_info(
            'X', TensorProto.FLOAT, [2, 3]
        )

        output_tensor = helper.make_tensor_value_info(
            'Y', TensorProto.FLOAT, [2, 3]
        )

        fastgelu_node = helper.make_node(
            'FastGelu',
            inputs=['X'],
            outputs=['Y'],
            domain='com.microsoft'  # Original Microsoft domain
        )

        graph = helper.make_graph(
            [fastgelu_node],
            'contrib_fastgelu_test',
            [input_tensor],
            [output_tensor]
        )

        model = helper.make_model(graph)
        model.opset_import[0].domain = 'com.microsoft'
        model.opset_import[0].version = 1

        onnx.save(model, 'contrib_fastgelu.onnx')

        input_data = np.array([
            [-1.0, 0.0, 1.0],
            [-0.5, 0.5, 2.0]
        ], dtype=np.float32)

        session = ort.InferenceSession('contrib_fastgelu.onnx')
        output = session.run(None, {'X': input_data})[0]

        print("✓ Original contrib FastGelu still works")
        print(f"Contrib output:\n{output}")

        return True

    except Exception as e:
        print(f"❌ Contrib FastGelu test FAILED: {e}")
        return False

if __name__ == "__main__":
    print("Testing FastGelu domain conflict fix...")
    print("=" * 50)

    success = True

    # Test 1: Our custom domain FastGelu
    success &= test_fastgelu_custom_domain()

    # Test 2: Original contrib FastGelu
    success &= test_contrib_fastgelu_still_works()

    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests PASSED! Domain conflict resolved.")
    else:
        print("💥 Some tests FAILED. Please check the implementation.")
