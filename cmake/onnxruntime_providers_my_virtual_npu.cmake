# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# MyVirtualNPU Execution Provider
# This is a custom execution provider for demonstrating how to add custom operators
# Supports both CPU and CUDA implementations

add_definitions(-DUSE_MY_VIRTUAL_NPU=1)

# Collect CPU implementation files (exclude CUDA subdirectory)
file(GLOB onnxruntime_providers_my_virtual_npu_cc_srcs CONFIGURE_DEPENDS
  "${ONNXRUNTIME_ROOT}/core/providers/my_virtual_npu/*.h"
  "${ONNXRUNTIME_ROOT}/core/providers/my_virtual_npu/*.cc"
  "${ONNXRUNTIME_ROOT}/core/providers/my_virtual_npu/bert/*.h"
  "${ONNXRUNTIME_ROOT}/core/providers/my_virtual_npu/bert/*.cc"
)

set(onnxruntime_providers_my_virtual_npu_srcs ${onnxruntime_providers_my_virtual_npu_cc_srcs})

# Add CUDA support if enabled
if(onnxruntime_USE_CUDA)
  # Collect CUDA files
  file(GLOB onnxruntime_providers_my_virtual_npu_cu_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/my_virtual_npu/cuda/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/my_virtual_npu/cuda/*.cu"
    "${ONNXRUNTIME_ROOT}/core/providers/my_virtual_npu/cuda/*.cc"
  )

  # Set CUDA language for .cu files
  set_source_files_properties(${onnxruntime_providers_my_virtual_npu_cu_srcs} PROPERTIES LANGUAGE CUDA)

  list(APPEND onnxruntime_providers_my_virtual_npu_srcs ${onnxruntime_providers_my_virtual_npu_cu_srcs})

  source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_my_virtual_npu_cu_srcs})

  message(STATUS "MyVirtualNPU: CUDA support enabled")
endif()

source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_my_virtual_npu_cc_srcs})

onnxruntime_add_static_library(onnxruntime_providers_my_virtual_npu ${onnxruntime_providers_my_virtual_npu_srcs})

onnxruntime_add_include_to_target(onnxruntime_providers_my_virtual_npu
  onnxruntime_common onnxruntime_framework onnx onnx_proto ${PROTOBUF_LIB} flatbuffers::flatbuffers Boost::mp11 safeint_interface
)

target_include_directories(onnxruntime_providers_my_virtual_npu PRIVATE
  ${ONNXRUNTIME_ROOT}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${eigen_INCLUDE_DIRS}
)

# Add CUDA include directories if CUDA is enabled
if(onnxruntime_USE_CUDA)
  target_include_directories(onnxruntime_providers_my_virtual_npu PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  )

  # Set CUDA compile options
  if(CMAKE_CUDA_COMPILER)
    target_compile_options(onnxruntime_providers_my_virtual_npu PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )
  endif()
endif()

add_dependencies(onnxruntime_providers_my_virtual_npu onnx ${onnxruntime_EXTERNAL_DEPENDENCIES})

target_link_libraries(onnxruntime_providers_my_virtual_npu PRIVATE
  onnxruntime_common
  onnxruntime_framework
  onnx
  onnx_proto
  ${PROTOBUF_LIB}
  flatbuffers::flatbuffers
)

# Link CUDA libraries if CUDA is enabled
if(onnxruntime_USE_CUDA)
  target_link_libraries(onnxruntime_providers_my_virtual_npu PRIVATE
    CUDA::cudart
    CUDA::cublas
  )
endif()

set_target_properties(onnxruntime_providers_my_virtual_npu PROPERTIES
  FOLDER "ONNXRuntime"
  LINKER_LANGUAGE CXX
)

if(APPLE)
  set_property(TARGET onnxruntime_providers_my_virtual_npu APPEND_STRING PROPERTY LINK_FLAGS "-Xlinker -exported_symbols_list ${ONNXRUNTIME_ROOT}/core/providers/my_virtual_npu/exported_symbols.lst")
elseif(UNIX)
  set_property(TARGET onnxruntime_providers_my_virtual_npu APPEND_STRING PROPERTY LINK_FLAGS "-Xlinker --version-script=${ONNXRUNTIME_ROOT}/core/providers/my_virtual_npu/version_script.lds -Xlinker --gc-sections")
elseif(WIN32)
  set_property(TARGET onnxruntime_providers_my_virtual_npu APPEND_STRING PROPERTY LINK_FLAGS "-DEF:${ONNXRUNTIME_ROOT}/core/providers/my_virtual_npu/symbols.def")
endif()

install(TARGETS onnxruntime_providers_my_virtual_npu
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        FRAMEWORK DESTINATION ${CMAKE_INSTALL_BINDIR})
