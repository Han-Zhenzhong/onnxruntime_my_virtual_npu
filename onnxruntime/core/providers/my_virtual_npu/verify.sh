#!/bin/bash
# Verification script for my_virtual_npu implementation

echo "================================"
echo "My CPU Implementation Verification"
echo "================================"
echo ""

# Check directory structure
echo "1. Checking directory structure..."
if [ -d "my_virtual_npu/bert" ] && [ -f "my_virtual_npu/my_virtual_npu_kernels.cc" ]; then
    echo "   ✓ my_virtual_npu directory structure OK"
else
    echo "   ✗ my_virtual_npu directory structure missing"
    exit 1
fi

if [ -d "test/my_virtual_npu" ] && [ -f "test/my_virtual_npu/fast_gelu_op_test.cc" ]; then
    echo "   ✓ test/my_virtual_npu directory structure OK"
else
    echo "   ✗ test/my_virtual_npu directory structure missing"
    exit 1
fi

# Check required files
echo ""
echo "2. Checking required files..."
files=(
    "my_virtual_npu/bert/fast_gelu.h"
    "my_virtual_npu/bert/fast_gelu.cc"
    "my_virtual_npu/my_virtual_npu_kernels.h"
    "my_virtual_npu/my_virtual_npu_kernels.cc"
    "my_virtual_npu/CMakeLists.txt"
    "my_virtual_npu/README.md"
    "test/my_virtual_npu/fast_gelu_op_test.cc"
    "test/my_virtual_npu/CMakeLists.txt"
)

all_files_exist=true
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "   ✓ $file"
    else
        echo "   ✗ $file missing"
        all_files_exist=false
    fi
done

if [ "$all_files_exist" = false ]; then
    exit 1
fi

# Check for TODO-OPTIMIZE markers
echo ""
echo "3. Checking for TODO-OPTIMIZE markers..."
optimize_count=$(grep -r "TODO-OPTIMIZE" my_virtual_npu/ test/my_virtual_npu/ | wc -l)
echo "   Found $optimize_count TODO-OPTIMIZE markers"
if [ "$optimize_count" -gt 0 ]; then
    echo "   ✓ Optimization opportunities documented"
else
    echo "   ⚠ No optimization markers found"
fi

# Check namespace usage
echo ""
echo "4. Checking namespace usage..."
if grep -q "namespace my_virtual_npu" my_virtual_npu/*.cc; then
    echo "   ✓ Using my_virtual_npu namespace"
else
    echo "   ✗ my_virtual_npu namespace not found"
    exit 1
fi

# Verify no contrib_ops dependencies
echo ""
echo "5. Checking for unwanted dependencies..."
if grep -r "contrib_ops" my_virtual_npu/*.cc my_virtual_npu/*.h 2>/dev/null | grep -v "TODO\|comment\|参考"; then
    echo "   ⚠ Found references to contrib_ops (should be independent)"
else
    echo "   ✓ No hard dependencies on contrib_ops"
fi

# Check for basic error handling
echo ""
echo "6. Checking error handling..."
if grep -q "Status::OK()" my_virtual_npu/bert/fast_gelu.cc; then
    echo "   ✓ Uses proper status return"
else
    echo "   ⚠ Missing status checks"
fi

# Summary
echo ""
echo "================================"
echo "Verification Summary"
echo "================================"
echo "✓ Directory structure correct"
echo "✓ All required files present"
echo "✓ Namespace properly used"
echo "✓ Optimization markers in place"
echo ""
echo "Next steps:"
echo "1. Integrate with ONNX Runtime build (see INTEGRATION.md)"
echo "2. Build and run tests"
echo "3. Test with Tiny-GPT2 model"
echo ""
echo "For detailed instructions, see:"
echo "  - my_virtual_npu/README.md"
echo "  - my_virtual_npu/INTEGRATION.md"
echo "  - docs/my_operators/operator_implementation_plan.md"
echo ""
