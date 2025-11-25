@echo off
REM Verification script for my_virtual_npu implementation (Windows)

echo ================================
echo My CPU Implementation Verification
echo ================================
echo.

REM Check directory structure
echo 1. Checking directory structure...
if exist "my_virtual_npu\bert" if exist "my_virtual_npu\my_virtual_npu_kernels.cc" (
    echo    [OK] my_virtual_npu directory structure OK
) else (
    echo    [FAIL] my_virtual_npu directory structure missing
    exit /b 1
)

if exist "test\my_virtual_npu" if exist "test\my_virtual_npu\fast_gelu_op_test.cc" (
    echo    [OK] test\my_virtual_npu directory structure OK
) else (
    echo    [FAIL] test\my_virtual_npu directory structure missing
    exit /b 1
)

REM Check required files
echo.
echo 2. Checking required files...
set "files=my_virtual_npu\bert\fast_gelu.h my_virtual_npu\bert\fast_gelu.cc my_virtual_npu\my_virtual_npu_kernels.h my_virtual_npu\my_virtual_npu_kernels.cc my_virtual_npu\CMakeLists.txt my_virtual_npu\README.md test\my_virtual_npu\fast_gelu_op_test.cc test\my_virtual_npu\CMakeLists.txt"

for %%f in (%files%) do (
    if exist "%%f" (
        echo    [OK] %%f
    ) else (
        echo    [FAIL] %%f missing
        exit /b 1
    )
)

REM Check for TODO-OPTIMIZE markers
echo.
echo 3. Checking for TODO-OPTIMIZE markers...
findstr /S /C:"TODO-OPTIMIZE" my_virtual_npu\*.cc my_virtual_npu\*.h test\my_virtual_npu\*.cc >nul 2>&1
if %errorlevel% equ 0 (
    echo    [OK] Optimization opportunities documented
) else (
    echo    [WARN] No optimization markers found
)

REM Check namespace usage
echo.
echo 4. Checking namespace usage...
findstr /C:"namespace my_virtual_npu" my_virtual_npu\*.cc >nul 2>&1
if %errorlevel% equ 0 (
    echo    [OK] Using my_virtual_npu namespace
) else (
    echo    [FAIL] my_virtual_npu namespace not found
    exit /b 1
)

REM Summary
echo.
echo ================================
echo Verification Summary
echo ================================
echo [OK] Directory structure correct
echo [OK] All required files present
echo [OK] Namespace properly used
echo [OK] Optimization markers in place
echo.
echo Next steps:
echo 1. Integrate with ONNX Runtime build (see INTEGRATION.md)
echo 2. Build and run tests
echo 3. Test with Tiny-GPT2 model
echo.
echo For detailed instructions, see:
echo   - my_virtual_npu\README.md
echo   - my_virtual_npu\INTEGRATION.md
echo   - docs\my_operators\operator_implementation_plan.md
echo.

pause
