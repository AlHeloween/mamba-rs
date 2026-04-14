# AGENTS.md — mamba-rs

## Project

Rust implementation of Mamba SSM (Selective State Space Model) with CPU + CUDA GPU inference and training. Two architectures: Mamba-1 (`mamba_ssm`) and Mamba-3 SISO (`mamba3_siso`). Standalone library — no PyTorch/Burn/Candle dependency.

## Commands

```bash
cargo test                          # all tests
cargo test --features cuda          # includes GPU tests (requires NVIDIA GPU)
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
cargo doc --no-deps --workspace     # RUSTDOCFLAGS=-Dwarnings in CI
cargo publish --dry-run
```

## Tooling

Use `cmd_runner` for long-running tasks (compilation, training, benchmarks):

```bash
cmd_runner start -- <command ...>    # start interactive session in separate terminal
cmd_runner tail <run_id> --follow    # follow output
cmd_runner wait <run_id>             # wait for completion
```

For quick commands (fmt, clippy, short tests), use direct bash execution.

## Architecture

- `src/mamba_ssm/` — Mamba-1: CPU (`cpu/`) + GPU (`gpu/`) inference & training
- `src/mamba3_siso/` — Mamba-3 SISO: CPU (`cpu/`) + GPU (`gpu/`) inference & training
- `src/ops/` — shared: BLAS, norms (RMSNorm, BCNorm, RMSNormGated), SIMD fast_math, dims
- `src/module/` — high-level `MambaBackbone` API
- `kernels/` — CUDA .cu source files (compiled at runtime via NVRTC)
- `tests/` — correctness + benchmark tests (m1 = Mamba-1, m3 = Mamba-3)

## Key Constraints

- **Mamba-1**: `d_state <= 256`, `d_conv <= 8`, `d_inner` divisible by 4 (CUDA float4 kernels)
- **Mamba-3**: `headdim <= 32` and power-of-2, `d_state <= 64`, `headdim * d_state <= 1024`, `rope_fraction` is 0.5 or 1.0
- Default config: `d_model=128, d_state=16, d_conv=4, expand=2, n_layers=3`
- MSRV: 1.94, edition: 2024

## Features

- `cuda` — NVIDIA GPU support (cudarc 0.19.4 with driver, cublas, nvrtc)
- `accelerate` — Apple Accelerate framework BLAS
- `gemm-blas` — `gemm` crate BLAS (AVX2/AVX-512/NEON microkernels)

## GPU Notes

- Kernels compiled at runtime via NVRTC; supports SM 60–120 (Pascal → Blackwell)
- TF32 Tensor Cores on Ampere/Hopper; GPU/CPU parity tolerance: `atol=0.5, rtol=0.10` (TF32 compounds ~5e-3 per SGEMM across ~50 ops)
- CUDA Graph capture: ~1.6x inference speedup
- GPU tests gated behind `#[cfg(feature = "cuda")]` — skip without NVIDIA GPU

## Training

- Full BPTT through recurrent SSM state
- Loss function: `sum(temporal)` (NOT `sum(temporal^2)` — RMSNorm causes near-cancellation)
- Parallel batch training via Rayon with thread-local scratch + epoch-based gradient zeroing
- Finite-diff gradient checks use `eps=1e-3`, threshold 5% for well-conditioned params, 10% for saturated params (e.g. dt_proj_b)

## CI

Gate job requires: lockfile, fmt, clippy, docs (nightly), msrv (1.94), deny. Tests run on ubuntu/macos/windows × stable/nightly. Release: push `v*` tag after CI passes.
