//! Gradient checkpointing for Mamba: trade compute for memory.
//!
//! Standard BPTT saves all 17 activation fields per timestep (~16KB per timestep
//! for default dims). For long sequences, activation memory dominates.
//!
//! Gradient checkpointing saves only essential state at every timestep and full
//! activations only at checkpoint intervals. During backward, the full activations
//! between checkpoints are recomputed on-the-fly.
//!
//! ## Memory vs. compute tradeoff
//!
//! | Checkpoint interval | Activation memory | Recompute overhead |
//! |---------------------|-------------------|--------------------|
//! | 1 (no checkpointing) | 100% | 0% |
//! | 4 | ~30% | ~25% |
//! | 8 | ~18% | ~40% |
//! | 16 | ~12% | ~55% |
//!
//! ## Usage
//!
//! See `CheckpointState` for the main API.

use super::flat::MambaLayerFlat;

/// Minimal checkpoint state saved at every timestep.
/// Much smaller than full activation (saves ~80% memory).
pub struct CheckpointState {
    /// SSM hidden state at each timestep: [T * d_inner * d_state]
    pub ssm_state: Vec<f32>,
    /// Conv state at each timestep: [T * d_inner * d_conv]
    pub conv_state: Vec<f32>,
    /// Residual at each timestep: [T * d_model]
    pub residual: Vec<f32>,
    /// Full activations only at checkpoint timesteps
    pub checkpoint_acts: Vec<MambaLayerFlat>,
    /// Which timesteps are checkpoints
    pub checkpoint_indices: Vec<usize>,
    pub seq_len: usize,
    pub checkpoint_interval: usize,
}

impl CheckpointState {
    /// Allocate checkpoint state for a sequence.
    pub fn new(
        dims: &super::super::super::ops::dims::MambaDims,
        checkpoint_interval: usize,
    ) -> Self {
        let di = dims.d_inner;
        let ds = dims.d_state;
        let dc = dims.d_conv;
        let dm = dims.d_model;
        let seq_len = dims.seq_len;

        let mut checkpoint_indices = Vec::new();
        let mut i = 0usize;
        while i < seq_len {
            checkpoint_indices.push(i);
            i += checkpoint_interval;
        }

        let n_checkpoints = checkpoint_indices.len();

        Self {
            ssm_state: vec![0.0; seq_len * di * ds],
            conv_state: vec![0.0; seq_len * di * dc],
            residual: vec![0.0; seq_len * dm],
            checkpoint_acts: (0..n_checkpoints)
                .map(|_| MambaLayerFlat::zeros(*dims))
                .collect(),
            checkpoint_indices,
            seq_len,
            checkpoint_interval,
        }
    }

    /// Check if timestep t is a checkpoint.
    pub fn is_checkpoint(&self, t: usize) -> bool {
        self.checkpoint_indices.binary_search(&t).is_ok()
    }

    /// Get the index of the checkpoint that contains timestep t's data.
    pub fn checkpoint_index(&self, t: usize) -> Option<usize> {
        self.checkpoint_indices.iter().position(|&c| c == t)
    }

    /// Get the most recent checkpoint at or before timestep t.
    pub fn prev_checkpoint(&self, t: usize) -> (usize, usize) {
        let mut best_t = 0;
        let mut best_idx = 0;
        for (i, &c) in self.checkpoint_indices.iter().enumerate() {
            if c <= t {
                best_t = c;
                best_idx = i;
            } else {
                break;
            }
        }
        (best_t, best_idx)
    }
}

/// Extract checkpoint data from a layer's forward pass.
///
/// Called after `forward_mamba_layer_batched` to save the essential state.
pub fn save_checkpoint_state(
    ckpt_state: &mut CheckpointState,
    acts: &MambaLayerFlat,
    ssm_state: &[f32],
    conv_state: &[f32],
    dims: &super::super::super::ops::dims::MambaDims,
) {
    let di = dims.d_inner;
    let ds = dims.d_state;
    let dc = dims.d_conv;
    let dm = dims.d_model;
    let seq_len = dims.seq_len;

    for t in 0..seq_len {
        ckpt_state.residual[t * dm..(t + 1) * dm].copy_from_slice(acts.residual(t));
        ckpt_state.conv_state[t * di * dc..(t + 1) * di * dc].copy_from_slice(conv_state);
        ckpt_state.ssm_state[t * di * ds..(t + 1) * di * ds].copy_from_slice(ssm_state);

        if let Some(ckpt_idx) = ckpt_state.checkpoint_index(t) {
            ckpt_state.checkpoint_acts[ckpt_idx]
                .data
                .copy_from_slice(&acts.data);
        }
    }
}
