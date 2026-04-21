//! High-level Mamba wrappers.
//!
//! [`MambaBackbone`] is the primary Mamba-1 user-facing API.
//! [`Mamba3Backbone`] is the Mamba-3 SISO equivalent.
//! Both own all weights and provide single-step inference
//! and access to raw weights for training integration.

mod backbone;
mod mamba3_backbone;

pub use backbone::MambaBackbone;
pub use mamba3_backbone::Mamba3Backbone;
