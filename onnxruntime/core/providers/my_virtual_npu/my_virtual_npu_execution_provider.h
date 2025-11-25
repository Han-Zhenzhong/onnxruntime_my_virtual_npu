// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/execution_provider.h"

namespace onnxruntime {

// Information needed to construct MyVirtualNPU execution provider.
struct MyVirtualNpuExecutionProviderInfo {
  bool create_arena{true};
  bool use_cuda{false};  // Enable CUDA support
  int device_id{0};      // CUDA device ID

  explicit MyVirtualNpuExecutionProviderInfo(bool use_arena, bool cuda = false, int dev_id = 0)
      : create_arena(use_arena), use_cuda(cuda), device_id(dev_id) {}

  MyVirtualNpuExecutionProviderInfo() = default;
};

// MyVirtualNPU Execution Provider
class MyVirtualNpuExecutionProvider : public IExecutionProvider {
 public:
  explicit MyVirtualNpuExecutionProvider(const MyVirtualNpuExecutionProviderInfo& info);
  virtual ~MyVirtualNpuExecutionProvider();

  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;
  std::unique_ptr<IDataTransfer> GetDataTransfer() const override;

 private:
  MyVirtualNpuExecutionProviderInfo info_;
};

}  // namespace onnxruntime
