//! Reusable scratch buffers for Mamba-3 SISO batched forward/backward.
//!
//! Allocated once per training session, reused every gradient step.
//! 25 buffers for batched SGEMM phases and gradient propagation.

use super::dims::Mamba3Dims;

/// Reusable scratch buffers for Mamba-3 batched forward and backward.
pub struct Mamba3Scratch {
    // ── Forward ──
    pub post_norm_flat: Vec<f32>, // [T * d_model]
    pub proj_flat: Vec<f32>,      // [T * in_proj_dim]
    pub gated_flat: Vec<f32>,     // [T * d_inner]
    pub out_flat: Vec<f32>,       // [T * d_model]

    // ── Backward B1: out_proj ──
    pub d_gated_flat: Vec<f32>, // [T * d_inner]
    pub gated_buf: Vec<f32>,    // [T * d_inner]

    // ── Backward B2: output gating ──
    pub d_y_flat: Vec<f32>, // [T * d_inner]
    pub d_z_flat: Vec<f32>, // [T * d_inner]

    // ── Backward B3: BPTT ──
    pub d_x_flat: Vec<f32>,     // [T * d_inner]
    pub d_b_flat: Vec<f32>,     // [T * ngroups * d_state]
    pub d_c_flat: Vec<f32>,     // [T * ngroups * d_state]
    pub d_h: Vec<f32>,          // [nheads * headdim * d_state]
    pub d_alpha_flat: Vec<f32>, // [T * nheads]
    pub d_beta_flat: Vec<f32>,  // [T * nheads]
    pub d_gamma_flat: Vec<f32>, // [T * nheads]

    // ── Backward B4: RoPE ──
    pub d_b_pre_rope_flat: Vec<f32>,   // [T * ngroups * d_state]
    pub d_c_pre_rope_flat: Vec<f32>,   // [T * ngroups * d_state]
    pub d_angle_cumsum_flat: Vec<f32>, // [T * nheads * n_angles]

    // ── Backward B5: BCNorm ──
    pub d_b_raw_flat: Vec<f32>, // [T * ngroups * d_state]
    pub d_c_raw_flat: Vec<f32>, // [T * ngroups * d_state]

    // ── Backward B6: discretization ──
    pub d_dd_dt_flat: Vec<f32>,  // [T * nheads]
    pub d_dd_a_flat: Vec<f32>,   // [T * nheads]
    pub d_trap_flat: Vec<f32>,   // [T * nheads]
    pub d_angles_flat: Vec<f32>, // [T * n_angles]

    // ── Backward B7: in_proj ──
    pub d_proj_flat: Vec<f32>,      // [T * in_proj_dim]
    pub d_post_norm_flat: Vec<f32>, // [T * d_model]
    pub post_norm_buf: Vec<f32>,    // [T * d_model]

    // ── Backward B8: residual ──
    pub d_residual_buf: Vec<f32>, // [T * d_model]
}

impl Mamba3Scratch {
    /// Allocate zero-filled scratch buffers.
    pub fn zeros(dims: &Mamba3Dims) -> Self {
        let t = dims.seq_len;
        let dm = dims.d_model;
        let di = dims.d_inner;
        let ds = dims.d_state;
        let nh = dims.nheads;
        let hd = dims.headdim;
        let ng = dims.ngroups;
        let ip = dims.in_proj_dim;
        let na = dims.num_rope_angles.max(1);
        Self {
            post_norm_flat: vec![0.0; t * dm],
            proj_flat: vec![0.0; t * ip],
            gated_flat: vec![0.0; t * di],
            out_flat: vec![0.0; t * dm],
            d_gated_flat: vec![0.0; t * di],
            gated_buf: vec![0.0; t * di],
            d_y_flat: vec![0.0; t * di],
            d_z_flat: vec![0.0; t * di],
            d_x_flat: vec![0.0; t * di],
            d_b_flat: vec![0.0; t * ng * ds],
            d_c_flat: vec![0.0; t * ng * ds],
            d_h: vec![0.0; nh * hd * ds],
            d_alpha_flat: vec![0.0; t * nh],
            d_beta_flat: vec![0.0; t * nh],
            d_gamma_flat: vec![0.0; t * nh],
            d_b_pre_rope_flat: vec![0.0; t * ng * ds],
            d_c_pre_rope_flat: vec![0.0; t * ng * ds],
            d_angle_cumsum_flat: vec![0.0; t * nh * na],
            d_b_raw_flat: vec![0.0; t * ng * ds],
            d_c_raw_flat: vec![0.0; t * ng * ds],
            d_dd_dt_flat: vec![0.0; t * nh],
            d_dd_a_flat: vec![0.0; t * nh],
            d_trap_flat: vec![0.0; t * nh],
            d_angles_flat: vec![0.0; t * na],
            d_proj_flat: vec![0.0; t * ip],
            d_post_norm_flat: vec![0.0; t * dm],
            post_norm_buf: vec![0.0; t * dm],
            d_residual_buf: vec![0.0; t * dm],
        }
    }
}
