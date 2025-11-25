// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/my_virtual_npu/my_virtual_npu_execution_provider.h"
#include "core/providers/my_virtual_npu/my_virtual_npu_kernels.h"
#include "core/framework/allocator.h"
#include "core/framework/kernel_registry.h"
#include "core/session/onnxruntime_cxx_api.h"

#ifdef USE_CUDA
#include "core/providers/cuda/cuda_allocator.h"
#include "core/providers/cuda/cuda_fence.h"
#endif

namespace onnxruntime {

constexpr const char* MY_VIRTUAL_NPU = "MyVirtualNpu";

MyVirtualNpuExecutionProvider::MyVirtualNpuExecutionProvider(const MyVirtualNpuExecutionProviderInfo& info)
    : IExecutionProvider{onnxruntime::kMyVirtualNpuExecutionProvider, true},
      info_(info) {

#ifdef USE_CUDA
  if (info.use_cuda && info.device_id >= 0) {
    // CUDA allocator setup
    AllocatorCreationInfo cuda_memory_info{
        [device_id = info.device_id](int) {
          return std::make_unique<CUDAAllocator>(
              device_id,
              OrtMemoryInfo(MY_VIRTUAL_NPU, OrtAllocatorType::OrtDeviceAllocator, OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, device_id)));
        },
        device_id,
        info.create_arena};

    InsertAllocator(CreateAllocator(cuda_memory_info));

    // CUDA pinned memory allocator for CPU-GPU transfers
    AllocatorCreationInfo cuda_pinned_memory_info{
        [device_id = info.device_id](int) {
          return std::make_unique<CUDAPinnedAllocator>(
              device_id,
              OrtMemoryInfo(MY_VIRTUAL_NPU, OrtAllocatorType::OrtDeviceAllocator, OrtDevice(OrtDevice::GPU, OrtDevice::MemType::CUDA_PINNED, device_id)));
        },
        device_id,
        info.create_arena};

    InsertAllocator(CreateAllocator(cuda_pinned_memory_info));
  } else
#endif
  {
    // CPU allocator setup (default)
    AllocatorCreationInfo default_memory_info{
        [](int) {
          return std::make_unique<CPUAllocator>(OrtMemoryInfo(MY_VIRTUAL_NPU, OrtAllocatorType::OrtDeviceAllocator));
        },
        0,
        info.create_arena};

    InsertAllocator(CreateAllocator(default_memory_info));

    AllocatorCreationInfo cpu_memory_info{
        [](int) {
          return std::make_unique<CPUAllocator>(
              OrtMemoryInfo(MY_VIRTUAL_NPU, OrtAllocatorType::OrtDeviceAllocator, OrtDevice(), 0, OrtMemTypeCPUOutput));
        },
        0,
        info.create_arena};

    InsertAllocator(CreateAllocator(cpu_memory_info));
  }
}

MyVirtualNpuExecutionProvider::~MyVirtualNpuExecutionProvider() {}

std::shared_ptr<KernelRegistry> MyVirtualNpuExecutionProvider::GetKernelRegistry() const {
  static std::shared_ptr<KernelRegistry> kernel_registry = []() {
    auto registry = std::make_shared<KernelRegistry>();
    // Register CPU kernels
    ORT_THROW_IF_ERROR(my_virtual_npu::RegisterMyVirtualNpuKernels(*registry));

#ifdef USE_CUDA
    // Register CUDA kernels if CUDA is available
    ORT_THROW_IF_ERROR(my_virtual_npu::RegisterMyVirtualNpuCudaKernels(*registry));
#endif

    return registry;
  }();
  return kernel_registry;
}

std::unique_ptr<IDataTransfer> MyVirtualNpuExecutionProvider::GetDataTransfer() const {
#ifdef USE_CUDA
  if (info_.use_cuda) {
    return nullptr;  // Use default CUDA data transfer
  }
#endif
  return nullptr;  // Use default CPU data transfer
}

}  // namespace onnxruntime
