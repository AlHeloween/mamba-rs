//! Compile and register Mamba-3 SISO CUDA kernels.
//!
//! 47 kernels across 5 .cu files, compiled via NVRTC at runtime.
//! Separate from Mamba-1's `MambaKernels` — different pipeline, no conv1d.

use cudarc::driver::{CudaContext, CudaFunction, CudaModule};
use std::sync::Arc;

/// All compiled Mamba-3 SISO CUDA kernels.
pub struct Mamba3Kernels {
    _module: Arc<CudaModule>,

    // ── Sequential SSM (mamba3_ssd.cu) ──
    pub m3_step_fwd: CudaFunction,
    pub m3_burnin_fwd: CudaFunction,
    pub m3_burnin_fwd_nosave: CudaFunction,
    pub m3_backward_seq: CudaFunction,
    pub m3_reduce_d_d: CudaFunction,

    // ── Shared ops (mamba3_ops.cu) ──
    pub m3_split: CudaFunction,
    pub m3_split_bwd: CudaFunction,
    pub bcnorm_fwd: CudaFunction,
    pub bcnorm_bwd: CudaFunction,
    pub bc_bias_add: CudaFunction,
    pub bc_bias_add_bwd: CudaFunction,
    pub angle_dt_fwd: CudaFunction,
    pub m3_angle_dt_fwd_batch: CudaFunction,
    pub m3_angle_dt_fwd_seq: CudaFunction,
    pub angle_dt_bwd: CudaFunction,
    pub m3_angle_dt_bwd_seq: CudaFunction,
    pub rope_fwd: CudaFunction,
    pub rope_bwd: CudaFunction,
    pub m3_compute_abg: CudaFunction,
    pub m3_abg_bwd: CudaFunction,
    pub silu_gate_fwd: CudaFunction,
    pub silu_gate_bwd: CudaFunction,
    pub rmsnorm_gated_fwd: CudaFunction,
    pub rmsnorm_gated_bwd: CudaFunction,

    // ── Shared kernels from norms.cu + elementwise.cu (used by training pipeline) ──
    pub rmsnorm_fwd: CudaFunction,
    pub rmsnorm_bwd: CudaFunction,
    pub colsum_accumulate: CudaFunction,
    pub vec_add_inplace: CudaFunction,
    pub elementwise_mul: CudaFunction,
    pub fill_scalar: CudaFunction,
    pub residual_add: CudaFunction,
    pub gather_last_timestep: CudaFunction,

    // ── Chunked parallel scan (mamba3_chunked.cu) ──
    pub m3_preprocess_chunks: CudaFunction,
    pub m3_da_cumsum: CudaFunction,
    pub m3_chunk_state_fwd: CudaFunction,
    pub m3_state_passing_fwd: CudaFunction,
    pub m3_writeback_parallel_states: CudaFunction,
    pub m3_chunk_scan_fwd: CudaFunction,
    pub m3_chunk_scan_bwd: CudaFunction,
    pub m3_state_passing_bwd: CudaFunction,
    pub m3_chunk_state_bwd: CudaFunction,
    pub m3_cumsum_bwd: CudaFunction,
    pub m3_extract_da_cs_sum: CudaFunction,
    pub m3_dqkv: CudaFunction,
    pub m3_dqktheta: CudaFunction,
    pub m3_ddt_dtrap: CudaFunction,
    pub m3_final_grads: CudaFunction,
}

impl Mamba3Kernels {
    /// Compile all 47 Mamba-3 CUDA kernels from source. Takes ~100-200ms.
    pub fn compile(ctx: &Arc<CudaContext>, arch: &'static str) -> Result<Self, String> {
        let sources = [
            include_str!("../../../kernels/mamba3_ssd.cu"),
            include_str!("../../../kernels/mamba3_ops.cu"),
            include_str!("../../../kernels/mamba3_chunked.cu"),
            // Shared kernels needed by training pipeline
            include_str!("../../../kernels/norms.cu"),
            include_str!("../../../kernels/elementwise.cu"),
        ];

        let combined = sources.join("\n");
        let opts = cudarc::nvrtc::CompileOptions {
            arch: Some(arch),
            options: vec![
                "--fmad=true".to_string(),
                "--extra-device-vectorization".to_string(),
            ],
            ..Default::default()
        };

        let ptx = cudarc::nvrtc::compile_ptx_with_opts(combined, opts)
            .map_err(|e| format!("NVRTC M3 compile failed: {e:?}"))?;

        let module = ctx
            .load_module(ptx)
            .map_err(|e| format!("M3 module load failed: {e:?}"))?;

        let get = |name: &str| -> Result<CudaFunction, String> {
            module
                .load_function(name)
                .map_err(|e| format!("M3 kernel '{name}' not found: {e:?}"))
        };

        Ok(Self {
            // Sequential SSM
            m3_step_fwd: get("m3_step_fwd")?,
            m3_burnin_fwd: get("m3_burnin_fwd")?,
            m3_burnin_fwd_nosave: get("m3_burnin_fwd_nosave")?,
            m3_backward_seq: get("m3_backward_seq")?,
            m3_reduce_d_d: get("m3_reduce_d_D")?,

            // Shared ops
            m3_split: get("m3_split")?,
            m3_split_bwd: get("m3_split_bwd")?,
            bcnorm_fwd: get("bcnorm_fwd")?,
            bcnorm_bwd: get("bcnorm_bwd")?,
            bc_bias_add: get("bc_bias_add")?,
            bc_bias_add_bwd: get("bc_bias_add_bwd")?,
            angle_dt_fwd: get("angle_dt_fwd")?,
            m3_angle_dt_fwd_batch: get("m3_angle_dt_fwd_batch")?,
            m3_angle_dt_fwd_seq: get("m3_angle_dt_fwd_seq")?,
            angle_dt_bwd: get("angle_dt_bwd")?,
            m3_angle_dt_bwd_seq: get("m3_angle_dt_bwd_seq")?,
            rope_fwd: get("rope_fwd")?,
            rope_bwd: get("rope_bwd")?,
            m3_compute_abg: get("m3_compute_abg")?,
            m3_abg_bwd: get("m3_abg_bwd")?,
            silu_gate_fwd: get("silu_gate_fwd")?,
            silu_gate_bwd: get("silu_gate_bwd")?,
            rmsnorm_gated_fwd: get("rmsnorm_gated_forward")?,
            rmsnorm_gated_bwd: get("rmsnorm_gated_backward")?,

            // Shared (norms.cu + elementwise.cu)
            rmsnorm_fwd: get("rmsnorm_forward")?,
            rmsnorm_bwd: get("rmsnorm_backward")?,
            colsum_accumulate: get("colsum_accumulate")?,
            vec_add_inplace: get("vec_add_inplace")?,
            elementwise_mul: get("elementwise_mul")?,
            fill_scalar: get("fill_scalar")?,
            residual_add: get("residual_add")?,
            gather_last_timestep: get("gather_last_timestep")?,

            // Chunked parallel scan
            m3_preprocess_chunks: get("m3_preprocess_chunks")?,
            m3_da_cumsum: get("m3_dA_cumsum")?,
            m3_chunk_state_fwd: get("m3_chunk_state_fwd")?,
            m3_state_passing_fwd: get("m3_state_passing_fwd")?,
            m3_writeback_parallel_states: get("m3_writeback_parallel_states")?,
            m3_chunk_scan_fwd: get("m3_chunk_scan_fwd")?,
            m3_chunk_scan_bwd: get("m3_chunk_scan_bwd")?,
            m3_state_passing_bwd: get("m3_state_passing_bwd")?,
            m3_chunk_state_bwd: get("m3_chunk_state_bwd")?,
            m3_cumsum_bwd: get("m3_cumsum_bwd")?,
            m3_extract_da_cs_sum: get("m3_extract_da_cs_sum")?,
            m3_dqkv: get("m3_dqkv")?,
            m3_dqktheta: get("m3_dqktheta")?,
            m3_ddt_dtrap: get("m3_ddt_dtrap")?,
            m3_final_grads: get("m3_final_grads")?,

            _module: module,
        })
    }
}
