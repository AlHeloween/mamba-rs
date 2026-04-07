# mamba-rs

Mamba SSM implementation in Rust with optional CUDA GPU acceleration. Supports Mamba-1 and Mamba-3 SISO.

Full inference and training pipelines with BPTT through recurrent SSM state. Custom CUDA kernels with CUDA Graph capture for minimal-latency GPU inference.

## Features

- **Two architectures** — Mamba SSM (Gu & Dao, 2023) and Mamba-3 SISO (Lahoti et al., ICLR 2026)
- **CPU inference** — zero-allocation single-step recurrent forward pass with SIMD + BLAS
- **GPU inference** — CUDA kernels with optional CUDA Graph capture (~1.6x speedup)
- **CPU training** — full backward pass with BPTT, parallel batch training via Rayon
- **GPU training** — custom CUDA forward + backward kernels (47 for M3, 12 for M1)
- **Serialization** — safetensors format (HuggingFace compatible)
- **Standalone** — no framework dependency (no PyTorch, no Burn, no Candle)
- **f32** — native single precision, TF32 Tensor Cores on Ampere/Hopper

## Quick Start — Mamba SSM

```rust
use mamba_rs::{MambaConfig, MambaState, MambaStepScratch, MambaWeights, mamba_step};

let cfg = MambaConfig::default(); // d_model=128, 3 layers
let weights = MambaWeights::init(&cfg, input_dim, 42);
let mut state = MambaState::zeros(cfg.n_layers, cfg.d_inner(), cfg.d_state, cfg.d_conv);
let mut scratch = MambaStepScratch::new(&cfg);
let mut output = vec![0.0f32; cfg.d_model];

mamba_step(&input, &mut output, &weights, &mut state.layers, &mut scratch, &cfg, input_dim);
state.reset(); // episode boundary
```

## Quick Start — Mamba-3 SISO

```rust
use mamba_rs::mamba3_siso::config::Mamba3Config;
use mamba_rs::mamba3_siso::cpu::inference::{Mamba3StepScratch, mamba3_step};
use mamba_rs::mamba3_siso::state::Mamba3State;
use mamba_rs::mamba3_siso::weights::Mamba3Weights;

let cfg = Mamba3Config::default();
let weights = Mamba3Weights::init(&cfg, input_dim, 42);
let mut state = Mamba3State::zeros(&cfg);
let mut scratch = Mamba3StepScratch::new(&cfg);
let mut output = vec![0.0f32; cfg.d_model];

mamba3_step(&mut output, &input, &mut scratch, &weights, &mut state.layers, &cfg);
```

## GPU Inference (CUDA)

```toml
[dependencies]
mamba-rs = { version = "0.2", features = ["cuda"] }
```

### Mamba SSM

```rust
use mamba_rs::gpu::inference::GpuMambaBackbone;

let mut gpu = GpuMambaBackbone::new(0, &weights, cfg, input_dim, batch)?;
gpu.capture_graph()?; // optional ~2x speedup
gpu.step(&input, &mut output)?;
gpu.reset()?;
```

### Mamba-3

```rust
use mamba_rs::mamba3_siso::gpu::inference::GpuMamba3Backbone;

let mut gpu = GpuMamba3Backbone::new(0, &weights, cfg, input_dim, batch)?;
gpu.capture_graph()?; // optional ~1.6x speedup
gpu.step(&input, &mut output)?;
gpu.reset()?;
```

Requires NVIDIA GPU + CUDA toolkit. Kernels compiled at runtime via NVRTC.

## Weight Serialization

```rust
use mamba_rs::serialize;

// Mamba-1
serialize::save(Path::new("model.safetensors"), backbone.weights(), cfg, input_dim)?;
let (weights, cfg, input_dim) = serialize::load(Path::new("model.safetensors"))?;

// Mamba-3
use mamba_rs::mamba3_siso::serialize::{save_mamba3, load_mamba3};
save_mamba3(Path::new("m3.safetensors"), &weights, &cfg, input_dim)?;
let (weights, input_dim) = load_mamba3(Path::new("m3.safetensors"), &cfg)?;
```

## Performance (RTX 6000 Ada)

| | Mamba SSM | Mamba-3 SISO |
|---|---|---|
| GPU Inference B=1 (CUDA Graph) | **79 us** | **86 us** |
| GPU Training Fwd+Bwd (T=32) | 1,640 us | 2,169 us |
| CPU Inference B=1 | 84 us | **65 us** |
| CPU Training Fwd+Bwd (T=32) | 15,874 us | **3,609 us** |

Zero heap allocations per inference step. See detailed results:
- [Mamba SSM benchmarks](docs/mamba1-benchmarks.md)
- [Mamba-3 benchmarks](docs/mamba3-benchmarks.md)

## Documentation

- [Mamba SSM architecture](docs/mamba1-architecture.md) — pipeline, modular API, weight layout
- [Mamba-3 architecture](docs/mamba3-architecture.md) — trapezoidal SSM, RoPE, BCNorm, CUDA kernels
- [Mamba SSM benchmarks](docs/mamba1-benchmarks.md) — GPU/CPU inference + training numbers
- [Mamba-3 benchmarks](docs/mamba3-benchmarks.md) — GPU/CPU inference + training numbers

## Citation

```bibtex
@inproceedings{mamba,
  title={Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
  author={Gu, Albert and Dao, Tri},
  booktitle={International Conference on Learning Representations},
  year={2024}
}

@inproceedings{mamba3,
  title={Mamba-3: Improved Sequence Modeling using State Space Principles},
  author={Lahoti, Aakash and Li, Kevin Y. and Chen, Berlin and Wang, Caitlin and Bick, Aviv and Kolter, J. Zico and Dao, Tri and Gu, Albert},
  booktitle={International Conference on Learning Representations},
  year={2026}
}
```

## License

Dual-licensed under MIT or Apache-2.0.
