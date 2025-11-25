// Example: How to use MyVirtualNPU Execution Provider

#include <onnxruntime_cxx_api.h>
#include "core/providers/my_virtual_npu/my_virtual_npu_execution_provider.h"

int main() {
    // Method 1: Using Session Options (Recommended)
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;

    // Create and append MyVirtualNPU provider
    onnxruntime::MyVirtualNpuExecutionProviderInfo info;
    info.create_arena = true;  // Enable memory arena for better performance

    auto my_virtual_npu_provider =
        std::make_unique<onnxruntime::MyVirtualNpuExecutionProvider>(info);

    session_options.AppendExecutionProvider_MyVirtualNpu(info);

    // Create session with the model
    Ort::Session session(env, "model.onnx", session_options);

    // Run inference as usual
    // ...

    return 0;
}

// Method 2: Programmatic Registration (Advanced)
/*
#include "core/session/onnxruntime_cxx_api.h"
#include "core/providers/my_virtual_npu/my_virtual_npu_execution_provider.h"

void RegisterMyVirtualNpuProvider(Ort::SessionOptions& options) {
    onnxruntime::MyVirtualNpuExecutionProviderInfo info;
    info.create_arena = true;

    auto provider = std::make_unique<onnxruntime::MyVirtualNpuExecutionProvider>(info);
    options.AppendExecutionProvider(std::move(provider));
}
*/
