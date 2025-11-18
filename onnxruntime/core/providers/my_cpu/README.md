# My CPU Custom Operators for Tiny-GPT2

This directory contains custom CPU operator implementations for running Tiny-GPT2 model on ONNX Runtime.

## Overview

This is a standalone implementation independent of `contrib_ops/`, designed for:
- Learning and experimentation
- Basic working version first, optimization later
- Clear documentation with TODO-OPTIMIZE markers for future improvements

## Directory Structure

```
my_cpu/
├── bert/
│   ├── fast_gelu.h          # FastGELU operator header
│   └── fast_gelu.cc         # FastGELU implementation (basic)
├── my_cpu_kernels.h         # Kernel registration header
├── my_cpu_kernels.cc        # Kernel registration implementation
├── CMakeLists.txt           # Build configuration
└── README.md                # This file
```

## Implemented Operators

### 1. FastGelu (✅ Implemented)

Fast GELU activation function using tanh approximation.

**Formula:** `GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`

**Current Implementation:**
- Basic scalar loop (correctness-focused)
- Support for optional bias input (for BiasGelu fusion)
- Tolerance: < 1e-3 error compared to reference

**Optimization Opportunities (TODO-OPTIMIZE):**
- `[SIMD]` AVX2 vectorization - Expected 4-8x speedup
- `[Parallel]` OpenMP parallelization for large tensors
- `[Cache]` Memory layout optimization

## Building

### Integration with ONNX Runtime

Add to main `CMakeLists.txt`:

```cmake
# Add custom CPU operators
add_subdirectory(my_cpu)

# Link to main library
target_link_libraries(onnxruntime PRIVATE onnxruntime_my_cpu)
```

### Build Commands

```bash
# From onnxruntime root directory
./build.sh --config Release --parallel

# Or on Windows
.\build.bat --config Release --parallel
```

## Testing

### Unit Tests

Tests are located in `test/my_cpu/`:

```bash
# Run all tests
cd build/Release
./onnxruntime_test_all

# Run only FastGelu tests
./onnxruntime_test_all --gtest_filter="*FastGelu*"
```

### Test Coverage

- ✅ Basic functionality (different shapes)
- ✅ Edge cases (large/small values, zero)
- ✅ Single element and large tensors
- ⏭️ Performance benchmarks (TODO)

## Usage Example

### Python

```python
import onnxruntime as ort

# Create session with custom ops
session = ort.InferenceSession(
    "tiny_gpt2_optimized.onnx",
    providers=['CPUExecutionProvider']
)

# Run inference
import numpy as np
input_ids = np.array([[1, 2, 3, 4]], dtype=np.int64)
outputs = session.run(None, {"input_ids": input_ids})
```

### C++

```cpp
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include "my_cpu/my_cpu_kernels.h"

Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "TinyGPT2");
Ort::SessionOptions session_options;

// Register custom operators
// (Automatically registered if linked with onnxruntime_my_cpu)

Ort::Session session(env, "tiny_gpt2_optimized.onnx", session_options);
// Run inference...
```

## Development Guide

### Adding New Operators

1. Create header and implementation files in `bert/` subdirectory
2. Add to `my_cpu_kernels.cc` registration
3. Update `CMakeLists.txt` to include new sources
4. Write unit tests in `test/my_cpu/`
5. Update this README

### Coding Conventions

- **Correctness First**: Basic implementation before optimization
- **TODO-OPTIMIZE Comments**: Mark all optimization opportunities
  - Format: `// TODO-OPTIMIZE: [Type] Description`
  - Types: `[SIMD]`, `[Parallel]`, `[Cache]`, `[Fusion]`, `[Memory]`
- **Namespace**: All code in `onnxruntime::my_cpu`
- **Testing**: Every operator needs comprehensive unit tests

### Optimization Guide

When ready to optimize, look for TODO-OPTIMIZE comments:

```cpp
// TODO-OPTIMIZE: [SIMD] AVX2 vectorization possible here
// Expected speedup: 4-8x for float32
// Implementation reference: contrib_ops/cpu/bert/...
```

Priority order:
1. **SIMD** - Biggest impact (4-8x speedup)
2. **Parallel** - Good for batch processing (2-4x speedup)
3. **Cache** - Memory optimization (1.5-2x speedup)
4. **Fusion** - Reduce kernel launches

## Performance Targets (Future)

### Current Status (Basic Implementation)
- ✅ Correctness: < 1e-3 error
- ⏭️ Performance: Not optimized yet

### Future Targets (After Optimization)
- First token latency: < 30ms
- Subsequent tokens: < 20ms/token
- Throughput: > 50 tokens/sec
- Speedup vs PyTorch: 1.5-2.0x

## References

- [Implementation Plan](../docs/my_operators/operator_implementation_plan.md)
- [ONNX Runtime Custom Ops](https://onnxruntime.ai/docs/reference/operators/add-custom-op.html)
- [Tiny-GPT2 Model](https://huggingface.co/sshleifer/tiny-gpt2)

## Contributing

This is a learning/experimental project. Feel free to:
- Add new operators following the same pattern
- Implement TODO-OPTIMIZE suggestions
- Improve tests and documentation
- Share performance optimization techniques

## License

Copyright (c) Microsoft Corporation. Licensed under the MIT License.
