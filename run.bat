@echo off
setlocal

cd /d "%~dp0"
set BRIAN2_NUMPY_FALLBACK=0

REM ── Try to locate a Visual Studio C++ compiler ────────────────────────────────
REM   Check common install paths in order: VS 2022 Community, Professional,
REM   Enterprise, then standalone Build Tools (x86 and x64 program dirs).

for %%P in (
    "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat"
    "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat"
    "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall.bat"
    "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
    "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
) do (
    if exist %%P (
        echo [run.bat] Found compiler: %%P
        call %%P x64
        if %errorlevel% equ 0 (
            echo [run.bat] C++ backend ENABLED
            goto :launch
        )
    )
)

REM ── No compiler found — fall back to numpy backend ────────────────────────────
echo [run.bat] No Visual Studio C++ compiler found.
echo [run.bat] Falling back to numpy backend (~10x slower but no compiler needed).
echo [run.bat] To enable C++ backend, install Visual Studio 2022 with
echo [run.bat] "Desktop development with C++" workload.
set BRIAN2_NUMPY_FALLBACK=1

:launch
echo.
wenv310\Scripts\python.exe fly_brain_body_simulation.py
endlocal
