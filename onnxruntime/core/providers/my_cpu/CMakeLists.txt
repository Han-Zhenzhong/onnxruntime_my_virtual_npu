# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Custom CPU operators for Tiny-GPT2
# This is a standalone implementation independent of contrib_ops

set(onnxruntime_my_cpu_srcs
  ${ONNXRUNTIME_ROOT}/my_cpu/bert/fast_gelu.cc
  ${ONNXRUNTIME_ROOT}/my_cpu/bert/fast_gelu.h
  ${ONNXRUNTIME_ROOT}/my_cpu/my_cpu_kernels.cc
  ${ONNXRUNTIME_ROOT}/my_cpu/my_cpu_kernels.h
)

# TODO: Add more source files as implemented
# ${ONNXRUNTIME_ROOT}/my_cpu/bert/skip_layer_norm.cc
# ${ONNXRUNTIME_ROOT}/my_cpu/bert/skip_layer_norm.h

# Create static library for custom CPU operators
add_library(onnxruntime_my_cpu STATIC ${onnxruntime_my_cpu_srcs})

# Add include directories
target_include_directories(onnxruntime_my_cpu PRIVATE
  ${ONNXRUNTIME_ROOT}
  ${ONNXRUNTIME_ROOT}/core
)

# Link dependencies
target_link_libraries(onnxruntime_my_cpu PUBLIC
  onnxruntime_common
  onnxruntime_framework
)

# TODO-OPTIMIZE: [SIMD] Enable AVX2 optimization when implemented
# This will be uncommented when AVX2-optimized code is added
# if(MSVC)
#   set_source_files_properties(
#     ${ONNXRUNTIME_ROOT}/my_cpu/bert/fast_gelu.cc
#     PROPERTIES COMPILE_FLAGS "/arch:AVX2"
#   )
# elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
#   set_source_files_properties(
#     ${ONNXRUNTIME_ROOT}/my_cpu/bert/fast_gelu.cc
#     PROPERTIES COMPILE_FLAGS "-mavx2 -mfma"
#   )
# endif()

# TODO-OPTIMIZE: [Parallel] Enable OpenMP when parallel implementation is added
# if(onnxruntime_USE_OPENMP)
#   target_compile_options(onnxruntime_my_cpu PRIVATE ${OpenMP_CXX_FLAGS})
#   target_link_libraries(onnxruntime_my_cpu PRIVATE ${OpenMP_CXX_LIBRARIES})
# endif()

# Set compile options
target_compile_features(onnxruntime_my_cpu PRIVATE cxx_std_17)

# Add to ONNX Runtime sources
set_target_properties(onnxruntime_my_cpu PROPERTIES FOLDER "ONNXRuntime")
