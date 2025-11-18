#!/usr/bin/env python3
"""
Generate test data for FastGELU operator

This script generates expected output values for FastGELU tests
using PyTorch's GELU implementation as reference.
"""

import numpy as np
import torch
import torch.nn.functional as F


def gelu_reference(x):
    """PyTorch GELU implementation as reference"""
    return F.gelu(torch.tensor(x, dtype=torch.float32)).numpy()


def fast_gelu_tanh_approx(x):
    """
    Fast GELU approximation using tanh (what we implement in C++)
    GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    """
    kAlpha = 0.7978845608028654  # sqrt(2/π)
    kBeta = 0.044715

    x_cubed = x * x * x
    inner = kAlpha * (x + kBeta * x_cubed)
    tanh_val = np.tanh(inner)
    return 0.5 * x * (1.0 + tanh_val)


def generate_test_cases():
    """Generate test cases for FastGELU"""

    test_cases = []

    # Test case 1: Basic functionality
    input1 = np.array([[-1.0, 0.0, 1.0], [-0.5, 0.5, 2.0]], dtype=np.float32)
    output1 = fast_gelu_tanh_approx(input1)
    test_cases.append(("BasicFloat32", input1, output1))

    # Test case 2: Edge cases
    input2 = np.array([-10.0, -0.001, 0.0, 0.001, 10.0], dtype=np.float32)
    output2 = fast_gelu_tanh_approx(input2)
    test_cases.append(("EdgeCases", input2, output2))

    # Test case 3: Different shapes
    input3 = np.array([[[-2.0, -1.0], [0.0, 1.0], [-0.5, 0.25], [0.75, 1.5]]], dtype=np.float32)
    output3 = fast_gelu_tanh_approx(input3)
    test_cases.append(("DifferentShapes", input3, output3))

    return test_cases


def print_cpp_test_data(test_cases):
    """Print test data in C++ format"""
    for name, input_data, output_data in test_cases:
        print(f"\n// Test case: {name}")
        print(f"// Input shape: {input_data.shape}")

        # Print input
        flat_input = input_data.flatten()
        input_str = ", ".join(f"{v:.6f}f" for v in flat_input)
        print(f"std::vector<float> input = {{{input_str}}};")

        # Print expected output
        flat_output = output_data.flatten()
        output_str = ", ".join(f"{v:.6f}f" for v in flat_output)
        print(f"std::vector<float> expected_output = {{{output_str}}};")

        # Print shape
        shape_str = ", ".join(str(d) for d in input_data.shape)
        print(f"std::vector<int64_t> shape = {{{shape_str}}};")


def compare_implementations():
    """Compare PyTorch GELU vs our tanh approximation"""
    print("\n=== Comparing PyTorch GELU vs Tanh Approximation ===\n")

    test_values = [-5.0, -2.0, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 2.0, 5.0]

    print(f"{'x':<8} {'PyTorch':<12} {'Tanh Approx':<12} {'Diff':<12} {'Rel Error':<12}")
    print("-" * 60)

    for x in test_values:
        pytorch_val = gelu_reference(x).item()
        tanh_val = fast_gelu_tanh_approx(x)
        diff = abs(pytorch_val - tanh_val)
        rel_error = diff / (abs(pytorch_val) + 1e-10)

        print(f"{x:<8.2f} {pytorch_val:<12.6f} {tanh_val:<12.6f} {diff:<12.6e} {rel_error:<12.6e}")

    print("\nMax relative error should be < 1e-3 for acceptable approximation")


def generate_tiny_gpt2_tensor():
    """Generate test tensor with Tiny-GPT2 typical dimensions"""
    print("\n=== Tiny-GPT2 Typical Tensor ===\n")

    # Typical hidden state: [batch=1, seq=8, hidden=768]
    batch, seq_len, hidden_size = 1, 8, 768

    # Random input from normal distribution (typical after layer norm)
    np.random.seed(42)
    input_tensor = np.random.randn(batch, seq_len, hidden_size).astype(np.float32)
    output_tensor = fast_gelu_tanh_approx(input_tensor)

    print(f"Input shape: {input_tensor.shape}")
    print(f"Input stats: mean={input_tensor.mean():.6f}, std={input_tensor.std():.6f}")
    print(f"Output stats: mean={output_tensor.mean():.6f}, std={output_tensor.std():.6f}")

    # Check for NaN or Inf
    assert not np.isnan(output_tensor).any(), "Output contains NaN!"
    assert not np.isinf(output_tensor).any(), "Output contains Inf!"
    print("✓ No NaN or Inf in output")


if __name__ == "__main__":
    print("=" * 60)
    print("FastGELU Test Data Generator")
    print("=" * 60)

    # Generate and print test cases
    test_cases = generate_test_cases()
    print("\n=== Test Data for C++ Unit Tests ===")
    print_cpp_test_data(test_cases)

    # Compare implementations
    compare_implementations()

    # Generate typical Tiny-GPT2 tensor
    generate_tiny_gpt2_tensor()

    print("\n" + "=" * 60)
    print("Test data generation complete!")
    print("Copy the output above to fast_gelu_op_test.cc")
    print("=" * 60)
