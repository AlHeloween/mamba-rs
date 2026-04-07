//! Target network forward for Mamba-3 SISO.
//!
//! Reuses `forward_mamba3_layer_batched` with a throwaway activation buffer.
//! No backward needed — just state evolution for target Q-network.

use super::dims::Mamba3Dims;
use super::flat::Mamba3LayerFlat;
use super::forward::forward_mamba3_layer_batched;
use super::scratch::Mamba3Scratch;
use super::weights::TrainMamba3LayerWeights;

/// Target network forward — single layer. Mutates state, discards activations.
pub fn target_mamba3_layer_forward(
    temporal_flat: &mut [f32],
    layer_w: &TrainMamba3LayerWeights,
    ssm_state: &mut [f32],
    k_state: &mut [f32],
    v_state: &mut [f32],
    angle_state: &mut [f32],
    scratch: &mut Mamba3Scratch,
    acts: &mut Mamba3LayerFlat,
    dims: &Mamba3Dims,
) {
    // Reuse full forward — activations saved but not used for backward.
    // For production, a no-save variant would save ~50% bandwidth.
    forward_mamba3_layer_batched(
        temporal_flat,
        acts,
        layer_w,
        ssm_state,
        k_state,
        v_state,
        angle_state,
        scratch,
        dims,
    );
}
