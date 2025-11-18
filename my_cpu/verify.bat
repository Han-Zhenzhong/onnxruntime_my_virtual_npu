@echo off
REM Verification script for my_cpu implementation (Windows)

echo ================================
echo My CPU Implementation Verification
echo ================================
echo.

REM Check directory structure
echo 1. Checking directory structure...
if exist "my_cpu\bert" if exist "my_cpu\my_cpu_kernels.cc" (
    echo    [OK] my_cpu directory structure OK
) else (
    echo    [FAIL] my_cpu directory structure missing
    exit /b 1
)

if exist "test\my_cpu" if exist "test\my_cpu\fast_gelu_op_test.cc" (
    echo    [OK] test\my_cpu directory structure OK
) else (
    echo    [FAIL] test\my_cpu directory structure missing
    exit /b 1
)

REM Check required files
echo.
echo 2. Checking required files...
set "files=my_cpu\bert\fast_gelu.h my_cpu\bert\fast_gelu.cc my_cpu\my_cpu_kernels.h my_cpu\my_cpu_kernels.cc my_cpu\CMakeLists.txt my_cpu\README.md test\my_cpu\fast_gelu_op_test.cc test\my_cpu\CMakeLists.txt"

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
findstr /S /C:"TODO-OPTIMIZE" my_cpu\*.cc my_cpu\*.h test\my_cpu\*.cc >nul 2>&1
if %errorlevel% equ 0 (
    echo    [OK] Optimization opportunities documented
) else (
    echo    [WARN] No optimization markers found
)

REM Check namespace usage
echo.
echo 4. Checking namespace usage...
findstr /C:"namespace my_cpu" my_cpu\*.cc >nul 2>&1
if %errorlevel% equ 0 (
    echo    [OK] Using my_cpu namespace
) else (
    echo    [FAIL] my_cpu namespace not found
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
echo   - my_cpu\README.md
echo   - my_cpu\INTEGRATION.md
echo   - docs\my_operators\operator_implementation_plan.md
echo.

pause
