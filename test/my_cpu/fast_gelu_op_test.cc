// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

// Basic functionality test for FastGelu
TEST(FastGeluTest, BasicFloat32) {
  OpTester test("FastGelu", 1, kMSDomain);

  // Simple test case
  std::vector<int64_t> shape = {2, 3};
  std::vector<float> input = {
      -1.0f, 0.0f, 1.0f,
      -0.5f, 0.5f, 2.0f
  };

  // Expected output computed using reference implementation
  // GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
  std::vector<float> expected_output = {
      -0.158655f, 0.0f, 0.841345f,
      -0.154269f, 0.345735f, 1.954500f
  };

  test.AddInput<float>("X", shape, input);
  test.AddOutput<float>("Y", shape, expected_output);

  // Test CPU execution provider only
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kCpuExecutionProvider});
}

// Test different tensor shapes
TEST(FastGeluTest, DifferentShapes) {
  OpTester test("FastGelu", 1, kMSDomain);

  // Test 3D tensor [1, 4, 2]
  std::vector<int64_t> shape = {1, 4, 2};
  std::vector<float> input = {
      -2.0f, -1.0f,
      0.0f, 1.0f,
      -0.5f, 0.25f,
      0.75f, 1.5f
  };

  // Expected output (computed using reference GELU)
  std::vector<float> expected_output = {
      -0.04550028f, -0.15865529f,
      0.0f, 0.84134471f,
      -0.15426877f, 0.15626901f,
      0.59825796f, 1.39977265f
  };

  test.AddInput<float>("X", shape, input);
  test.AddOutput<float>("Y", shape, expected_output);
  test.Run();
}

// Test edge cases
TEST(FastGeluTest, EdgeCases) {
  OpTester test("FastGelu", 1, kMSDomain);

  std::vector<int64_t> shape = {5};
  std::vector<float> input = {
      -10.0f,   // Large negative
      -0.001f,  // Near zero negative
      0.0f,     // Zero
      0.001f,   // Near zero positive
      10.0f     // Large positive
  };

  // Expected output
  std::vector<float> expected_output = {
      -0.0f,        // GELU(-10) ≈ 0
      -0.0005f,     // GELU(-0.001) ≈ -0.0005
      0.0f,         // GELU(0) = 0
      0.0005f,      // GELU(0.001) ≈ 0.0005
      10.0f         // GELU(10) ≈ 10
  };

  test.AddInput<float>("X", shape, input);
  test.AddOutput<float>("Y", shape, expected_output);
  test.Run();
}

// Test single element
TEST(FastGeluTest, SingleElement) {
  OpTester test("FastGelu", 1, kMSDomain);

  std::vector<int64_t> shape = {1};
  std::vector<float> input = {0.5f};
  std::vector<float> expected_output = {0.345735f};

  test.AddInput<float>("X", shape, input);
  test.AddOutput<float>("Y", shape, expected_output);
  test.Run();
}

// Test large tensor
TEST(FastGeluTest, LargeTensor) {
  OpTester test("FastGelu", 1, kMSDomain);

  // Test with larger tensor size (common in GPT-2: [batch, seq, hidden])
  std::vector<int64_t> shape = {1, 8, 768};
  size_t count = 1 * 8 * 768;

  std::vector<float> input(count);
  std::vector<float> expected_output(count);

  // Fill with some pattern
  for (size_t i = 0; i < count; ++i) {
    float x = static_cast<float>(i % 100) / 50.0f - 1.0f;  // Range [-1, 1]
    input[i] = x;

    // Compute expected GELU value
    constexpr float kAlpha = 0.7978845608028654f;
    constexpr float kBeta = 0.044715f;
    float x_cubed = x * x * x;
    float inner = kAlpha * (x + kBeta * x_cubed);
    float tanh_val = std::tanh(inner);
    expected_output[i] = 0.5f * x * (1.0f + tanh_val);
  }

  test.AddInput<float>("X", shape, input);
  test.AddOutput<float>("Y", shape, expected_output);
  test.Run();
}

// TODO-OPTIMIZE: [Test] Add performance benchmark test
/*
TEST(FastGeluTest, DISABLED_BenchmarkPerformance) {
  // Compare basic implementation vs optimized version
  // Measure throughput (GB/s) and latency (ms)
  // Test different tensor sizes: small, medium, large
}
*/

}  // namespace test
}  // namespace onnxruntime
