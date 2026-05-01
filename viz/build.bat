@echo off
REM Build script for Mamba Visualizer
REM Requires: Rust toolchain + Delphi 12 Athens (dcc64)

echo === Building mamba_rs.dll (Rust FFI) ===
cd ..
cargo build --release --features ffi
if errorlevel 1 exit /b 1

copy /Y target\release\mamba_rs.dll viz\
if errorlevel 1 exit /b 1

echo === Building Delphi VCL Application ===
cd viz
"C:\Program Files (x86)\Embarcadero\Studio\23.0\bin\msbuild.exe" MambaVisualizer.dproj /p:Config=Release /p:Platform=Win64
if errorlevel 1 exit /b 1

echo === Build complete ===
echo Output: viz\x64\Release\MambaVisualizer.exe
echo DLL: viz\mamba_rs.dll
pause
