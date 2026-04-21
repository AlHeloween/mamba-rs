use crate::mamba3_siso::config::Mamba3Config;
use crate::mamba3_siso::cpu::inference::{Mamba3StepScratch, mamba3_step};
use crate::mamba3_siso::state::Mamba3State;
use crate::mamba3_siso::weights::{Mamba3LayerWeights, Mamba3Weights};

/// Complete Mamba-3 SISO backbone: input_proj → N layers → norm_f.
///
/// Owns all weights. Provides single-step recurrent inference
/// and access to raw weights for training integration.
///
/// ```rust
/// use mamba_rs::module::Mamba3Backbone;
/// use mamba_rs::Mamba3Config;
///
/// let cfg = Mamba3Config::default();
/// let backbone = Mamba3Backbone::init(cfg, 128, 42);
///
/// let mut state = backbone.alloc_state();
/// let mut scratch = backbone.alloc_scratch();
/// let mut output = vec![0.0f32; backbone.config().d_model];
///
/// let input = vec![0.1f32; 128];
/// backbone.forward_step(&input, &mut output, &mut state, &mut scratch);
/// ```
pub struct Mamba3Backbone {
    weights: Mamba3Weights,
    cfg: Mamba3Config,
    input_dim: usize,
}

impl Mamba3Backbone {
    /// Create a backbone with Mamba-3-specific weight initialization.
    ///
    /// Uses Kaiming uniform for projections, inverse-softplus init
    /// for dt_bias (Lahoti et al., Section 3.3).
    pub fn init(cfg: Mamba3Config, input_dim: usize, seed: u64) -> Self {
        let weights = Mamba3Weights::init(&cfg, input_dim, seed);
        Self {
            weights,
            cfg,
            input_dim,
        }
    }

    /// Create a backbone from pre-loaded weights.
    ///
    /// Validates dimensions against config. Returns `Err` on mismatch.
    pub fn from_weights(cfg: Mamba3Config, weights: Mamba3Weights) -> Result<Self, String> {
        weights.validate(&cfg, cfg.d_model)?;
        let input_dim = weights.input_proj_w.len() / cfg.d_model;
        Ok(Self {
            weights,
            cfg,
            input_dim,
        })
    }

    /// Extract owned weights (consuming self).
    pub fn into_weights(self) -> Mamba3Weights {
        self.weights
    }

    /// Read-only weight access.
    pub fn weights(&self) -> &Mamba3Weights {
        &self.weights
    }

    /// Mutable weight access (for optimizer updates).
    pub fn weights_mut(&mut self) -> &mut Mamba3Weights {
        &mut self.weights
    }

    /// Read-only access to a specific layer's weights.
    pub fn layer(&self, index: usize) -> &Mamba3LayerWeights {
        &self.weights.layers[index]
    }

    /// Mutable access to a specific layer's weights.
    pub fn layer_mut(&mut self, index: usize) -> &mut Mamba3LayerWeights {
        &mut self.weights.layers[index]
    }

    /// Number of layers.
    pub fn n_layers(&self) -> usize {
        self.cfg.n_layers
    }

    /// Total parameter count.
    pub fn param_count(&self) -> usize {
        self.weights.param_count(self.input_dim, &self.cfg)
    }

    /// The config this backbone was built with.
    pub fn config(&self) -> &Mamba3Config {
        &self.cfg
    }

    /// External input dimension.
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Single-step recurrent forward through the full backbone.
    ///
    /// `input_proj(input) → N × layer_step → norm_f → output`
    ///
    /// Zero allocations per call. Delegates to [`mamba3_step`].
    pub fn forward_step(
        &self,
        input: &[f32],
        output: &mut [f32],
        state: &mut Mamba3State,
        scratch: &mut Mamba3StepScratch,
    ) {
        mamba3_step(
            output,
            input,
            scratch,
            &self.weights,
            &mut state.layers,
            &self.cfg,
        );
    }

    /// Run T inference steps sequentially, collecting all outputs.
    ///
    /// `inputs`: `[T * input_dim]` — T sequential inputs.
    /// `outputs`: `[T * d_model]` — T sequential outputs (written in-place).
    /// State carries across all T steps.
    pub fn forward_sequence(
        &self,
        inputs: &[f32],
        outputs: &mut [f32],
        state: &mut Mamba3State,
        scratch: &mut Mamba3StepScratch,
        seq_len: usize,
    ) {
        let dm = self.cfg.d_model;
        debug_assert_eq!(inputs.len(), seq_len * self.input_dim);
        debug_assert_eq!(outputs.len(), seq_len * dm);
        for t in 0..seq_len {
            let inp = &inputs[t * self.input_dim..(t + 1) * self.input_dim];
            let out = &mut outputs[t * dm..(t + 1) * dm];
            self.forward_step(inp, out, state, scratch);
        }
    }

    /// Allocate zeroed recurrent state matching this backbone.
    pub fn alloc_state(&self) -> Mamba3State {
        Mamba3State::zeros(&self.cfg)
    }

    /// Allocate inference scratch buffers matching this backbone.
    pub fn alloc_scratch(&self) -> Mamba3StepScratch {
        Mamba3StepScratch::new(&self.cfg)
    }
}
