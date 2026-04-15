//! Mamba-3 SISO CPU backend.
//!
//! - `dims` — dimension calculator
//! - `inference` — T=1 recurrent step (TODO)
//! - `forward` — training forward (TODO)
//! - `backward` — training backward (TODO)

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
pub use inference::{mamba3_layer_step, mamba3_step, Mamba3StepScratch};
pub use scratch::Mamba3Scratch;
pub use weights::{TrainMamba3LayerWeights, TrainMamba3Weights};
