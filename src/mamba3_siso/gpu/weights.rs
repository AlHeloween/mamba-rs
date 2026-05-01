//! GPU weight storage for Mamba-3 SISO.
//!
//! Same dual pattern as Mamba-1:
//! - **Inference**: flat buffer + WeightSlice views (CUDA Graph safe, read-only)
//! - **Training**: per-tensor GpuBuffer (optimizer compatible)
//! - **Gradients**: flat buffer + GradSlice views (single zero() clears all)

use crate::mamba3_siso::config::Mamba3Config;
use crate::mamba_ssm::gpu::buffers::{GpuBuffer, GradSlice, WeightSlice};
use std::sync::Arc;

type Stream = Arc<cudarc::driver::CudaStream>;

// ═══ Inference weights — flat buffer + WeightSlice ═══

/// GPU inference weights for a single Mamba-3 layer.
pub struct GpuMamba3LayerWeightsInf {
    pub norm_weight: WeightSlice,      // [d_model]
    pub in_proj_w: WeightSlice,        // [d_model * in_proj_dim]
    pub dt_bias: WeightSlice,          // [nheads]
    pub b_norm_weight: WeightSlice,    // [d_state]
    pub c_norm_weight: WeightSlice,    // [d_state]
    pub b_bias: WeightSlice,           // [nheads * d_state]
    pub c_bias: WeightSlice,           // [nheads * d_state]
    pub d_param: WeightSlice,          // [nheads]
    pub norm_gate_weight: WeightSlice, // [d_inner]
    pub out_proj_w: WeightSlice,       // [d_inner * d_model]
}

/// GPU inference weights for the full Mamba-3 backbone (flat buffer).
pub struct GpuMamba3WeightsInf {
    pub input_proj_w: WeightSlice,
    pub input_proj_b: WeightSlice,
    pub layers: Vec<GpuMamba3LayerWeightsInf>,
    pub norm_f_weight: WeightSlice,
    pub flat: GpuBuffer,
}

impl GpuMamba3WeightsInf {
    /// Upload CPU weights into a single flat GPU buffer with WeightSlice views.
    pub fn from_cpu(
        stream: &Stream,
        cpu: &crate::mamba3_siso::weights::Mamba3Weights,
        _input_dim: usize,
    ) -> Result<Self, String> {
        // Concatenate all weights into a flat Vec
        let mut flat_data = Vec::new();
        flat_data.extend_from_slice(&cpu.input_proj_w);
        flat_data.extend_from_slice(&cpu.input_proj_b);
        for lw in &cpu.layers {
            flat_data.extend_from_slice(&lw.norm_weight);
            flat_data.extend_from_slice(&lw.in_proj_w);
            flat_data.extend_from_slice(&lw.dt_bias);
            flat_data.extend_from_slice(&lw.b_norm_weight);
            flat_data.extend_from_slice(&lw.c_norm_weight);
            flat_data.extend_from_slice(&lw.b_bias);
            flat_data.extend_from_slice(&lw.c_bias);
            flat_data.extend_from_slice(&lw.d_param);
            flat_data.extend_from_slice(&lw.norm_gate_weight);
            flat_data.extend_from_slice(&lw.out_proj_w);
        }
        flat_data.extend_from_slice(&cpu.norm_f_weight);

        let flat = GpuBuffer::from_cpu(stream, &flat_data)?;
        let base = flat.cached_ptr();

        let mut off = 0usize;
        let mut slice = |len: usize| -> WeightSlice {
            let s = WeightSlice::from_offset(base, off, len);
            off += len;
            s
        };

        let input_proj_w = slice(cpu.input_proj_w.len());
        let input_proj_b = slice(cpu.input_proj_b.len());

        let mut layers = Vec::new();
        for lw in &cpu.layers {
            layers.push(GpuMamba3LayerWeightsInf {
                norm_weight: slice(lw.norm_weight.len()),
                in_proj_w: slice(lw.in_proj_w.len()),
                dt_bias: slice(lw.dt_bias.len()),
                b_norm_weight: slice(lw.b_norm_weight.len()),
                c_norm_weight: slice(lw.c_norm_weight.len()),
                b_bias: slice(lw.b_bias.len()),
                c_bias: slice(lw.c_bias.len()),
                d_param: slice(lw.d_param.len()),
                norm_gate_weight: slice(lw.norm_gate_weight.len()),
                out_proj_w: slice(lw.out_proj_w.len()),
            });
        }
        let norm_f_weight = slice(cpu.norm_f_weight.len());

        Ok(Self {
            input_proj_w,
            input_proj_b,
            layers,
            norm_f_weight,
            flat,
        })
    }
}

// ═══ Training weights — per-tensor GpuBuffer ═══

/// GPU training weights for a single Mamba-3 layer.
pub struct GpuMamba3LayerWeights {
    pub norm_weight: GpuBuffer,
    pub in_proj_w: GpuBuffer,
    pub dt_bias: GpuBuffer,
    pub b_norm_weight: GpuBuffer,
    pub c_norm_weight: GpuBuffer,
    pub b_bias: GpuBuffer,
    pub c_bias: GpuBuffer,
    pub d_param: GpuBuffer,
    pub norm_gate_weight: GpuBuffer,
    pub out_proj_w: GpuBuffer,
}

/// GPU training weights for the full Mamba-3 backbone.
pub struct GpuMamba3Weights {
    pub input_proj_w: GpuBuffer,
    pub input_proj_b: GpuBuffer,
    pub layers: Vec<GpuMamba3LayerWeights>,
    pub norm_f_weight: GpuBuffer,
}

impl GpuMamba3LayerWeights {
    pub fn from_cpu(
        stream: &Stream,
        lw: &crate::mamba3_siso::weights::Mamba3LayerWeights,
        _cfg: &Mamba3Config,
    ) -> Result<Self, String> {
        Ok(Self {
            norm_weight: GpuBuffer::from_cpu(stream, &lw.norm_weight)?,
            in_proj_w: GpuBuffer::from_cpu(stream, &lw.in_proj_w)?,
            dt_bias: GpuBuffer::from_cpu(stream, &lw.dt_bias)?,
            b_norm_weight: GpuBuffer::from_cpu(stream, &lw.b_norm_weight)?,
            c_norm_weight: GpuBuffer::from_cpu(stream, &lw.c_norm_weight)?,
            b_bias: GpuBuffer::from_cpu(stream, &lw.b_bias)?,
            c_bias: GpuBuffer::from_cpu(stream, &lw.c_bias)?,
            d_param: GpuBuffer::from_cpu(stream, &lw.d_param)?,
            norm_gate_weight: GpuBuffer::from_cpu(stream, &lw.norm_gate_weight)?,
            out_proj_w: GpuBuffer::from_cpu(stream, &lw.out_proj_w)?,
        })
    }

    pub fn zeros(stream: &Stream, cfg: &Mamba3Config) -> Result<Self, String> {
        let dm = cfg.d_model;
        let di = cfg.d_inner();
        let ds = cfg.d_state;
        let nh = cfg.nheads();
        let ip = cfg.in_proj_out_dim();
        Ok(Self {
            norm_weight: GpuBuffer::zeros(stream, dm)?,
            in_proj_w: GpuBuffer::zeros(stream, dm * ip)?,
            dt_bias: GpuBuffer::zeros(stream, nh)?,
            b_norm_weight: GpuBuffer::zeros(stream, ds)?,
            c_norm_weight: GpuBuffer::zeros(stream, ds)?,
            b_bias: GpuBuffer::zeros(stream, nh * ds)?,
            c_bias: GpuBuffer::zeros(stream, nh * ds)?,
            d_param: GpuBuffer::zeros(stream, nh)?,
            norm_gate_weight: GpuBuffer::zeros(stream, di)?,
            out_proj_w: GpuBuffer::zeros(stream, di * dm)?,
        })
    }
}

impl GpuMamba3Weights {
    pub fn from_cpu(
        stream: &Stream,
        cpu: &crate::mamba3_siso::weights::Mamba3Weights,
        cfg: &Mamba3Config,
        _input_dim: usize,
    ) -> Result<Self, String> {
        Ok(Self {
            input_proj_w: GpuBuffer::from_cpu(stream, &cpu.input_proj_w)?,
            input_proj_b: GpuBuffer::from_cpu(stream, &cpu.input_proj_b)?,
            layers: cpu
                .layers
                .iter()
                .map(|lw| GpuMamba3LayerWeights::from_cpu(stream, lw, cfg))
                .collect::<Result<Vec<_>, _>>()?,
            norm_f_weight: GpuBuffer::from_cpu(stream, &cpu.norm_f_weight)?,
        })
    }

    pub fn zeros(stream: &Stream, cfg: &Mamba3Config, input_dim: usize) -> Result<Self, String> {
        let dm = cfg.d_model;
        Ok(Self {
            input_proj_w: GpuBuffer::zeros(stream, input_dim * dm)?,
            input_proj_b: GpuBuffer::zeros(stream, dm)?,
            layers: (0..cfg.n_layers)
                .map(|_| GpuMamba3LayerWeights::zeros(stream, cfg))
                .collect::<Result<Vec<_>, _>>()?,
            norm_f_weight: GpuBuffer::zeros(stream, dm)?,
        })
    }
}

// ═══ Gradients — flat buffer + GradSlice ═══

/// GPU gradients for a single Mamba-3 layer.
pub struct GpuMamba3LayerGrads {
    pub norm_weight: GradSlice,
    pub in_proj_w: GradSlice,
    pub dt_bias: GradSlice,
    pub b_norm_weight: GradSlice,
    pub c_norm_weight: GradSlice,
    pub b_bias: GradSlice,
    pub c_bias: GradSlice,
    pub d_param: GradSlice,
    pub norm_gate_weight: GradSlice,
    pub out_proj_w: GradSlice,
}

/// GPU gradients for the full Mamba-3 backbone.
pub struct GpuMamba3Grads {
    pub input_proj_w: GradSlice,
    pub input_proj_b: GradSlice,
    pub layers: Vec<GpuMamba3LayerGrads>,
    pub norm_f_weight: GradSlice,
    pub flat: GpuBuffer,
}

impl GpuMamba3Grads {
    /// Allocate gradient buffer for full Mamba-3 backbone.
    pub fn new(stream: &Stream, cfg: &Mamba3Config, input_dim: usize) -> Result<Self, String> {
        let dm = cfg.d_model;
        let di = cfg.d_inner();
        let ds = cfg.d_state;
        let nh = cfg.nheads();
        let ip = cfg.in_proj_out_dim();
        let per_layer = dm + dm * ip + nh + ds + ds + nh * ds + nh * ds + nh + di + di * dm;
        let total = input_dim * dm + dm + cfg.n_layers * per_layer + dm;
        let flat = GpuBuffer::zeros(stream, total)?;
        let base = flat.cached_ptr();

        let mut off = 0usize;
        let mut slice = |len: usize| -> GradSlice {
            let s = GradSlice::from_offset(base, off, len);
            off += len;
            s
        };

        let input_proj_w = slice(input_dim * dm);
        let input_proj_b = slice(dm);

        let mut layers = Vec::new();
        for _ in 0..cfg.n_layers {
            layers.push(GpuMamba3LayerGrads {
                norm_weight: slice(dm),
                in_proj_w: slice(dm * ip),
                dt_bias: slice(nh),
                b_norm_weight: slice(ds),
                c_norm_weight: slice(ds),
                b_bias: slice(nh * ds),
                c_bias: slice(nh * ds),
                d_param: slice(nh),
                norm_gate_weight: slice(di),
                out_proj_w: slice(di * dm),
            });
        }
        let norm_f_weight = slice(dm);

        Ok(Self {
            input_proj_w,
            input_proj_b,
            layers,
            norm_f_weight,
            flat,
        })
    }

    /// Zero all gradients (single memset on flat buffer).
    pub fn zero(&mut self, stream: &Stream) -> Result<(), String> {
        self.flat.zero(stream)
    }
}
