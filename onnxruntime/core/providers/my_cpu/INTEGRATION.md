# Integration Guide for my_cpu Custom Operators

This document describes how to integrate the my_cpu custom operators into ONNX Runtime build system.

## Quick Start

### Step 1: Verify Directory Structure

Ensure the following structure exists:

```
onnxruntime/
├── my_cpu/
│   ├── bert/
│   │   ├── fast_gelu.h
│   │   └── fast_gelu.cc
│   ├── my_cpu_kernels.h
│   ├── my_cpu_kernels.cc
│   ├── CMakeLists.txt
│   ├── README.md
│   └── generate_test_data.py
└── test/my_cpu/
    ├── fast_gelu_op_test.cc
    └── CMakeLists.txt
```

### Step 2: Integrate with Main Build System

#### Option A: Add to Main CMakeLists.txt (Recommended)

Add the following to `onnxruntime/CMakeLists.txt`:

```cmake
# Add custom CPU operators (my_cpu)
if(EXISTS ${ONNXRUNTIME_ROOT}/my_cpu/CMakeLists.txt)
  message(STATUS "Building custom CPU operators (my_cpu)")
  add_subdirectory(my_cpu)
  list(APPEND onnxruntime_EXTERNAL_LIBRARIES onnxruntime_my_cpu)
endif()
```

#### Option B: Standalone Build (For Testing)

```bash
cd my_cpu
mkdir build && cd build
cmake .. -DONNXRUNTIME_ROOT=..
make -j$(nproc)
```

### Step 3: Register Operators at Runtime

Add to your initialization code (or modify ONNX Runtime provider registration):

```cpp
#include "my_cpu/my_cpu_kernels.h"

// In your initialization function
onnxruntime::KernelRegistry* registry = /* get kernel registry */;
onnxruntime::my_cpu::RegisterMyCpuKernels(*registry);
```

### Step 4: Build ONNX Runtime

```bash
# From onnxruntime root
./build.sh --config Release --parallel --build_shared_lib

# On Windows
.\build.bat --config Release --parallel --build_shared_lib
```

### Step 5: Run Tests

```bash
cd build/Release
./onnxruntime_test_all --gtest_filter="*FastGelu*"
```

## Detailed Integration Steps

### 1. Modify Core CMakeLists.txt

Edit `onnxruntime/CMakeLists.txt` and add:

```cmake
# Around line where other subdirectories are added
add_subdirectory(core)
add_subdirectory(contrib_ops)

# ADD THIS LINE:
add_subdirectory(my_cpu)
```

### 2. Link to Main Library

In the same file, find where `onnxruntime` library is defined and add:

```cmake
target_link_libraries(onnxruntime PRIVATE
  onnxruntime_common
  onnxruntime_framework
  # ... other libraries ...

  # ADD THIS LINE:
  onnxruntime_my_cpu
)
```

### 3. Integrate Tests

Edit `onnxruntime/test/CMakeLists.txt`:

```cmake
# Add test sources
set(onnxruntime_test_framework_src_patterns
  # ... existing patterns ...
)

# ADD THIS:
file(GLOB onnxruntime_test_my_cpu_src
  "${ONNXRUNTIME_ROOT}/test/my_cpu/*.cc"
)

list(APPEND onnxruntime_test_all_srcs ${onnxruntime_test_my_cpu_src})
```

### 4. Operator Schema Registration (If Needed)

If you need to register custom operator schemas (not just kernels), add to:

`onnxruntime/core/graph/contrib_ops/contrib_defs.cc`

```cpp
#include "my_cpu/bert/fast_gelu.h"

void RegisterMyCpuSchemas() {
  ONNX_CONTRIB_OPERATOR_SCHEMA(FastGelu)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetDoc("Fast GELU activation")
      .Input(0, "X", "Input tensor", "T")
      .Output(0, "Y", "Output tensor", "T")
      .TypeConstraint("T", {"tensor(float)"}, "Constrain to float");
}
```

## Verification

### Check Build

```bash
# Verify my_cpu library is built
ls build/Release/my_cpu/
# Should see: libonnxruntime_my_cpu.a (or .lib on Windows)

# Verify test binary includes my_cpu tests
./build/Release/onnxruntime_test_all --gtest_list_tests | grep FastGelu
```

### Run Tests

```bash
# Run all my_cpu tests
./build/Release/onnxruntime_test_all --gtest_filter="*FastGelu*" -v

# Expected output:
# [==========] Running X tests from 1 test suite.
# [ RUN      ] FastGeluTest.BasicFloat32
# [       OK ] FastGeluTest.BasicFloat32
# ...
```

### Test with Python

```python
import onnxruntime as ort
import numpy as np

# Check available providers
print("Available providers:", ort.get_available_providers())

# Create a simple model with FastGelu (requires ONNX model with FastGelu op)
# session = ort.InferenceSession("model_with_fastgelu.onnx")
```

## Troubleshooting

### Issue: "Cannot find my_cpu/my_cpu_kernels.h"

**Solution:** Ensure `${ONNXRUNTIME_ROOT}` is in include path:

```cmake
target_include_directories(your_target PRIVATE ${ONNXRUNTIME_ROOT})
```

### Issue: "Undefined reference to RegisterMyCpuKernels"

**Solution:** Link against `onnxruntime_my_cpu`:

```cmake
target_link_libraries(your_target PRIVATE onnxruntime_my_cpu)
```

### Issue: "Operator FastGelu not found"

**Solution:** Verify operator is registered:
1. Check `my_cpu_kernels.cc` includes operator
2. Ensure `RegisterMyCpuKernels()` is called at runtime
3. Check operator domain matches (kMSDomain = "com.microsoft")

### Issue: Build fails with "Status not found"

**Solution:** Add ONNX Runtime headers to include path:

```cmake
target_include_directories(onnxruntime_my_cpu PRIVATE
  ${ONNXRUNTIME_ROOT}
  ${ONNXRUNTIME_ROOT}/core
)
```

## Testing with Tiny-GPT2 Model

### 1. Prepare Model

```python
# Export Tiny-GPT2 to ONNX with custom ops
from transformers import GPT2LMHeadModel
import torch

model = GPT2LMHeadModel.from_pretrained("sshleifer/tiny-gpt2")
model.eval()

dummy_input = torch.randint(0, 50000, (1, 8))

torch.onnx.export(
    model,
    dummy_input,
    "tiny_gpt2.onnx",
    input_names=["input_ids"],
    output_names=["logits"],
    dynamic_axes={"input_ids": {0: "batch", 1: "sequence"}},
    opset_version=14
)
```

### 2. Optimize Model

```python
from onnxruntime.transformers import optimizer

optimized = optimizer.optimize_model(
    "tiny_gpt2.onnx",
    model_type="gpt2",
    num_heads=12,
    hidden_size=768
)

optimized.save_model_to_file("tiny_gpt2_optimized.onnx")
```

### 3. Run Inference

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession(
    "tiny_gpt2_optimized.onnx",
    providers=["CPUExecutionProvider"]
)

input_ids = np.array([[1, 2, 3, 4]], dtype=np.int64)
outputs = session.run(None, {"input_ids": input_ids})

print("Output shape:", outputs[0].shape)
print("Success!")
```

## Next Steps

1. **Add More Operators**: Follow the same pattern for LayerNorm, Attention
2. **Optimize**: Implement TODO-OPTIMIZE suggestions (AVX2, OpenMP)
3. **Benchmark**: Measure performance vs baseline
4. **Deploy**: Package as custom op library

## References

- [ONNX Runtime Build Docs](https://onnxruntime.ai/docs/build/)
- [Custom Operators Guide](https://onnxruntime.ai/docs/reference/operators/add-custom-op.html)
- [Implementation Plan](../docs/my_operators/operator_implementation_plan.md)
