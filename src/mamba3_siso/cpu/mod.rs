//! Mamba-3 SISO CPU backend.
//!
//! - `dims` — dimension calculator
//! - `inference` — T=1 recurrent step
//! - `forward` — training forward (batched, 7 phases)
//! - `backward` — training backward (BPTT, 8 phases)

pub mod backward;
pub mod dims;
pub mod flat;
pub mod forward;
pub mod inference;
pub mod parallel;
pub mod scratch;
pub mod target;
pub mod weights;

pub use dims::Mamba3Dims;
pub use flat::{Mamba3FieldOffsets, Mamba3LayerFlat};
pub use inference::{Mamba3StepScratch, mamba3_layer_step, mamba3_step};
pub use scratch::Mamba3Scratch;
pub use weights::{TrainMamba3LayerWeights, TrainMamba3Weights};
