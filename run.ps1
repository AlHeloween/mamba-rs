# Mamba-RS Centralized Test & Benchmark Runner
# Usage: .\run.ps1 [--all] [--lib] [--examples] [--csv] [--complexity] [--training] [--clean]

param(
    [switch]$All,
    [switch]$Lib,
    [switch]$Examples,
    [switch]$CSV,
    [switch]$Complexity,
    [switch]$Training,
    [switch]$Clean,
    [switch]$Cuda,
    [string]$OutputDir = "results"
)

$ErrorActionPreference = "Stop"
$ResultsDir = Join-Path $PSScriptRoot $OutputDir

if (-not (Test-Path $ResultsDir)) {
    New-Item -ItemType Directory -Path $ResultsDir | Out-Null
    Write-Host "Created results directory: $ResultsDir" -ForegroundColor Green
}

function Run-Command {
    param([string]$Cmd, [string]$Description)
    Write-Host ""
    Write-Host "=== $Description ===" -ForegroundColor Cyan
    Write-Host "Running: $Cmd" -ForegroundColor Gray
    
    $timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
    $outputFile = Join-Path $ResultsDir "${timestamp}_${Description.Replace(' ', '_').ToLower()}.log"
    
    Invoke-Expression $Cmd 2>&1 | Tee-Object -FilePath $outputFile
}

if ($Clean) {
    Write-Host "Cleaning build artifacts..." -ForegroundColor Yellow
    cargo clean
    Get-ChildItem $ResultsDir -File | Remove-Item -Force
    Write-Host "Cleaned." -ForegroundColor Green
    exit 0
}

# Default: run all if no specific flag
if (-not ($Lib -or $Examples -or $CSV -or $Complexity -or $Training -or $All)) {
    $All = $true
}

$CudaFlag = if ($Cuda) { "--features cuda" } else { "" }

if ($Lib -or $All) {
    Run-Command "cargo test --lib $CudaFlag" "Lib Tests"
    Run-Command "cargo clippy --workspace --all-targets $CudaFlag -- -D warnings" "Clippy Check"
    Run-Command "cargo fmt --all -- --check" "Format Check"
}

if ($Examples -or $All) {
    Run-Command "cargo run --example ode_function_approx $CudaFlag" "ODE Function Approx Example"
}

if ($CSV -or $All) {
    $csvFile = Join-Path $ResultsDir "pointwise_$(Get-Date -Format 'yyyyMMdd_HHmmss').csv"
    Write-Host ""
    Write-Host "=== Pointwise CSV Output ===" -ForegroundColor Cyan
    $env:CSV_OUTPUT_PATH = $csvFile
    cargo test --test m_pointwise_csv -- --nocapture 2>&1 | Out-Default
    Remove-Item Env:\CSV_OUTPUT_PATH -ErrorAction SilentlyContinue
    if (Test-Path $csvFile) {
        $lines = (Get-Content $csvFile | Measure-Object -Line).Lines
        Write-Host "CSV saved to: $csvFile ($lines lines)" -ForegroundColor Green
    }
}

if ($Complexity -or $All) {
    Run-Command "cargo test --test m_complexity_ladder -- --nocapture" "Complexity Ladder"
}

if ($Training -or $All) {
    $csvFile = Join-Path $ResultsDir "benchmark_$(Get-Date -Format 'yyyyMMdd_HHmmss').csv"
    Write-Host ""
    Write-Host "=== Training Optimization Benchmark ===" -ForegroundColor Cyan
    $env:CSV_OUTPUT_PATH = $csvFile
    cargo test --test m_training_optimization -- --nocapture 2>&1 | Out-Default
    Remove-Item Env:\CSV_OUTPUT_PATH -ErrorAction SilentlyContinue
    if (Test-Path $csvFile) {
        $lines = (Get-Content $csvFile | Measure-Object -Line).Lines
        Write-Host "CSV saved to: $csvFile ($lines lines)" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "=== Summary ===" -ForegroundColor Green
Write-Host "Results saved to: $ResultsDir"
Get-ChildItem $ResultsDir -File | Sort-Object LastWriteTime -Descending | Select-Object -First 10 Name, Length, LastWriteTime | Format-Table
