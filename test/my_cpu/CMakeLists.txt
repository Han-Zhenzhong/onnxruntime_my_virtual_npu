# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Unit tests for my_cpu custom operators

set(onnxruntime_test_my_cpu_srcs
  ${ONNXRUNTIME_ROOT}/test/my_cpu/fast_gelu_op_test.cc
)

# TODO: Add more test files as operators are implemented
# ${ONNXRUNTIME_ROOT}/test/my_cpu/skip_layer_norm_test.cc

# Add test source files to main test target
# This assumes integration with existing ONNX Runtime test infrastructure
source_group(TREE ${ONNXRUNTIME_ROOT}/test FILES ${onnxruntime_test_my_cpu_srcs})

# The test files will be added to the main test executable
# set(onnxruntime_test_all_srcs ${onnxruntime_test_all_srcs} ${onnxruntime_test_my_cpu_srcs} PARENT_SCOPE)
