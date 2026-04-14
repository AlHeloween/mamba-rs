//! ODE state → function approximation with mamba-rs.
//!
//! Demonstrates using Mamba-1 to learn an unknown function f(y(t)) from
//! 4D ODE state trajectories. A simple linear ODE system generates training
//! data; Mamba is trained on CPU to approximate f and evaluated on GPU.
//!
//! ```bash
//! cargo run --example ode_function_approx --features cuda
//! ```

use mamba_rs::ops::dims::{MambaDims, MambaRecurrentState};
use mamba_rs::train::backward::backward_mamba_backbone_batched;
use mamba_rs::train::flat::MambaBackboneFlat;
use mamba_rs::train::forward::forward_mamba_backbone_batched;
use mamba_rs::train::scratch::{BackwardPhaseScratch, PhaseScratch};
use mamba_rs::train::weights::{TrainMambaLayerWeights, TrainMambaWeights};
use mamba_rs::{MambaBackbone, MambaConfig, MambaWeights};

#[cfg(feature = "cuda")]
use mamba_rs::gpu::inference::GpuMambaBackbone;

const ODE_DIM: usize = 4;
const INPUT_DIM: usize = ODE_DIM + 1;
const SEQ_LEN: usize = 360;
const EPOCHS: usize = 50;
const LR: f32 = 1e-3;
const SEED: u64 = 42;

fn main() {
    let cfg = MambaConfig {
        d_model: 64,
        d_state: 16,
        d_conv: 4,
        expand: 2,
        n_layers: 2,
        scan_mode: Default::default(),
    };

    println!("ODE State → Function Approximation with mamba-rs");
    println!("=================================================");
    println!(
        "Config: d_model={}, layers={}, d_inner={}, params={}",
        cfg.d_model,
        cfg.n_layers,
        cfg.d_inner(),
        MambaBackbone::init(cfg, INPUT_DIM, SEED).param_count()
    );
    println!(
        "Input: [t_scaled, y1, y2, y3, y4] ({} dims), seq_len={}",
        INPUT_DIM, SEQ_LEN
    );
    println!();

    let inf_weights = MambaWeights::init(&cfg, INPUT_DIM, SEED);
    let (mut tw, dims) = build_train_scaffolding(&inf_weights, &cfg, INPUT_DIM, SEQ_LEN);

    let (train_inputs, train_targets, test_inputs, test_targets) =
        generate_ode_dataset(SEQ_LEN, 3600);

    println!(
        "Training data: {} sequences × {} steps",
        train_inputs.len() / (SEQ_LEN * INPUT_DIM),
        SEQ_LEN
    );
    println!(
        "Test data: {} sequences × {} steps",
        test_inputs.len() / (SEQ_LEN * INPUT_DIM),
        SEQ_LEN
    );
    println!();

    println!("Training ({EPOCHS} epochs, lr={LR}):");
    for epoch in 0..EPOCHS {
        let mut epoch_loss = 0.0f32;
        let n_seq = train_inputs.len() / (SEQ_LEN * INPUT_DIM);

        for seq_idx in 0..n_seq {
            let inp_start = seq_idx * SEQ_LEN * INPUT_DIM;
            let inp = &train_inputs[inp_start..inp_start + SEQ_LEN * INPUT_DIM];
            let tgt_start = seq_idx * SEQ_LEN;
            let tgt = &train_targets[tgt_start..tgt_start + SEQ_LEN];

            let (temporal, loss, acts_local) = forward_and_loss(&tw, &dims, inp, tgt);
            let grads = backward_and_get_grads(&tw, &acts_local, &temporal, tgt, &dims);
            sgd_step(&mut tw, &grads, LR);
            epoch_loss += loss;
        }

        epoch_loss /= n_seq as f32;
        if (epoch + 1) % 10 == 0 || epoch == 0 {
            println!("  Epoch {:3}: loss = {:.6}", epoch + 1, epoch_loss);
        }
    }

    let trained_weights = train_to_inference(&tw);
    let bb = MambaBackbone::from_weights(cfg, trained_weights).unwrap();

    #[cfg(feature = "cuda")]
    {
        println!("\nEvaluating on GPU...");
        let metrics = eval_gpu(&test_inputs, &test_targets, &bb, INPUT_DIM);
        println!(
            "  MSE={:.6}  MAE={:.6}  MaxErr={:.6}  R²={:.6}",
            metrics.mse, metrics.mae, metrics.max_error, metrics.r_squared
        );

        if metrics.r_squared > 0.5 {
            println!("\n✓ Mamba successfully learned the function (R² > 0.5)");
        } else {
            println!("\n✗ Function approximation needs more training (R² < 0.5)");
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        println!("\nEvaluating on CPU...");
        let metrics = eval_cpu(&test_inputs, &test_targets, &bb, INPUT_DIM);
        println!(
            "  MSE={:.6}  MAE={:.6}  MaxErr={:.6}  R²={:.6}",
            metrics.mse, metrics.mae, metrics.max_error, metrics.r_squared
        );

        if metrics.r_squared > 0.5 {
            println!("\n✓ Mamba successfully learned the function (R² > 0.5)");
        } else {
            println!("\n✗ Function approximation needs more training (R² < 0.5)");
        }
    }
}

fn generate_ode_dataset(
    seq_len: usize,
    total_steps: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let a: [[f32; 4]; 4] = [
        [-0.10, 0.02, 0.00, 0.00],
        [0.00, -0.20, 0.01, 0.00],
        [0.00, 0.00, -0.05, 0.03],
        [0.00, 0.00, 0.00, -0.08],
    ];

    let mut all_states: Vec<[f32; 4]> = Vec::with_capacity(total_steps);
    let mut y = [1.0f32, 0.5, -0.25, 0.1];
    let dt = 0.1f32;

    for _ in 0..total_steps {
        all_states.push(y);
        let dy = [
            a[0][0] * y[0] + a[0][1] * y[1] + a[0][2] * y[2] + a[0][3] * y[3],
            a[1][0] * y[0] + a[1][1] * y[1] + a[1][2] * y[2] + a[1][3] * y[3],
            a[2][0] * y[0] + a[2][1] * y[1] + a[2][2] * y[2] + a[2][3] * y[3],
            a[3][0] * y[0] + a[3][1] * y[1] + a[3][2] * y[2] + a[3][3] * y[3],
        ];
        for i in 0..4 {
            y[i] += dt * dy[i];
        }
    }

    let mut inputs = vec![0.0f32; total_steps * INPUT_DIM];
    let mut targets = vec![0.0f32; total_steps];

    for t in 0..total_steps {
        let t_scaled = t as f32 / total_steps as f32;
        let inp_start = t * INPUT_DIM;
        inputs[inp_start] = t_scaled;
        inputs[inp_start + 1..inp_start + 1 + ODE_DIM].copy_from_slice(&all_states[t]);

        let y = all_states[t];
        targets[t] = y[0] * y[0] + y[1] * y[1] - y[2] + 0.5 * y[3].sin();
    }

    let train_end = seq_len;
    let train_inp = inputs[..train_end * INPUT_DIM].to_vec();
    let train_tgt = targets[..train_end].to_vec();
    let test_inp = inputs[train_end * INPUT_DIM..].to_vec();
    let test_tgt = targets[train_end..].to_vec();

    (train_inp, train_tgt, test_inp, test_tgt)
}

fn build_train_scaffolding(
    weights: &mamba_rs::MambaWeights,
    cfg: &MambaConfig,
    input_dim: usize,
    seq_len: usize,
) -> (TrainMambaWeights, MambaDims) {
    let dims = MambaDims::from_config(cfg, seq_len, input_dim);
    let tw = TrainMambaWeights {
        input_proj_w: weights.input_proj_w.clone(),
        input_proj_b: weights.input_proj_b.clone(),
        layers: weights
            .layers
            .iter()
            .map(|lw| TrainMambaLayerWeights {
                norm_weight: lw.norm_weight.clone(),
                in_proj_w: lw.in_proj_w.clone(),
                conv1d_weight: lw.conv1d_weight.clone(),
                conv1d_bias: lw.conv1d_bias.clone(),
                x_proj_w: lw.x_proj_w.clone(),
                dt_proj_w: lw.dt_proj_w.clone(),
                dt_proj_b: lw.dt_proj_b.clone(),
                a_log: lw.a_log.clone(),
                d_param: lw.d_param.clone(),
                out_proj_w: lw.out_proj_w.clone(),
            })
            .collect(),
        norm_f_weight: weights.norm_f_weight.clone(),
    };
    (tw, dims)
}

fn compute_a_neg(tw: &TrainMambaWeights, n_layers: usize, di: usize, ds: usize) -> Vec<f32> {
    let mut a_neg = vec![0.0f32; n_layers * di * ds];
    for (l, lw) in tw.layers.iter().enumerate() {
        for i in 0..di * ds {
            a_neg[l * di * ds + i] = -lw.a_log[i].exp();
        }
    }
    a_neg
}

fn forward_and_loss(
    tw: &TrainMambaWeights,
    dims: &MambaDims,
    input: &[f32],
    target: &[f32],
) -> (Vec<f32>, f32, MambaBackboneFlat) {
    let di = dims.d_inner;
    let ds = dims.d_state;
    let dc = dims.d_conv;
    let nl = dims.n_layers;
    let t = dims.seq_len;
    let dm = dims.d_model;

    let mut acts = MambaBackboneFlat::zeros(*dims);
    let mut scratch = PhaseScratch::zeros(dims);
    let mut conv = vec![0.0f32; nl * di * dc];
    let mut ssm = vec![0.0f32; nl * di * ds];
    let a_neg = compute_a_neg(tw, nl, di, ds);

    let mut state = MambaRecurrentState {
        conv: &mut conv,
        ssm: &mut ssm,
        a_neg: &a_neg,
    };

    let mut temporal = vec![0.0f32; t * dm];
    forward_mamba_backbone_batched(
        &mut temporal,
        &mut acts,
        tw,
        input,
        &mut state,
        &mut scratch,
        dims,
    );

    let mut loss = 0.0f32;
    for i in 0..t {
        let pred = temporal[i * dm];
        let tgt = target[i];
        loss += 0.5 * (pred - tgt) * (pred - tgt);
    }
    loss /= t as f32;

    (temporal, loss, acts)
}

fn backward_and_get_grads(
    tw: &TrainMambaWeights,
    acts: &MambaBackboneFlat,
    temporal: &[f32],
    target: &[f32],
    dims: &MambaDims,
) -> TrainMambaWeights {
    let di = dims.d_inner;
    let ds = dims.d_state;
    let nl = dims.n_layers;
    let t = dims.seq_len;
    let dm = dims.d_model;

    let mut d_temporal = vec![0.0f32; t * dm];
    for i in 0..t {
        let pred = temporal[i * dm];
        let tgt = target[i];
        d_temporal[i * dm] = (pred - tgt) / t as f32;
    }

    let mut grads = TrainMambaWeights::zeros_from_dims(dims);
    let mut bwd_scratch = BackwardPhaseScratch::zeros(dims);
    let a_neg = compute_a_neg(tw, nl, di, ds);

    backward_mamba_backbone_batched(
        &mut d_temporal,
        &mut grads,
        acts,
        tw,
        &a_neg,
        &mut bwd_scratch,
        dims,
    );

    grads
}

fn sgd_step(tw: &mut TrainMambaWeights, grads: &TrainMambaWeights, lr: f32) {
    fn apply_slice(w: &mut [f32], g: &[f32], lr: f32) {
        for (wi, gi) in w.iter_mut().zip(g.iter()) {
            *wi -= lr * gi;
        }
    }
    apply_slice(&mut tw.input_proj_w, &grads.input_proj_w, lr);
    apply_slice(&mut tw.input_proj_b, &grads.input_proj_b, lr);
    apply_slice(&mut tw.norm_f_weight, &grads.norm_f_weight, lr);
    for (lw, lg) in tw.layers.iter_mut().zip(grads.layers.iter()) {
        apply_slice(&mut lw.norm_weight, &lg.norm_weight, lr);
        apply_slice(&mut lw.in_proj_w, &lg.in_proj_w, lr);
        apply_slice(&mut lw.conv1d_weight, &lg.conv1d_weight, lr);
        apply_slice(&mut lw.conv1d_bias, &lg.conv1d_bias, lr);
        apply_slice(&mut lw.x_proj_w, &lg.x_proj_w, lr);
        apply_slice(&mut lw.dt_proj_w, &lg.dt_proj_w, lr);
        apply_slice(&mut lw.dt_proj_b, &lg.dt_proj_b, lr);
        apply_slice(&mut lw.a_log, &lg.a_log, lr);
        apply_slice(&mut lw.d_param, &lg.d_param, lr);
        apply_slice(&mut lw.out_proj_w, &lg.out_proj_w, lr);
    }
}

fn train_to_inference(tw: &TrainMambaWeights) -> mamba_rs::MambaWeights {
    mamba_rs::MambaWeights {
        input_proj_w: tw.input_proj_w.clone(),
        input_proj_b: tw.input_proj_b.clone(),
        layers: tw
            .layers
            .iter()
            .map(|tl| mamba_rs::MambaLayerWeights {
                norm_weight: tl.norm_weight.clone(),
                in_proj_w: tl.in_proj_w.clone(),
                conv1d_weight: tl.conv1d_weight.clone(),
                conv1d_bias: tl.conv1d_bias.clone(),
                x_proj_w: tl.x_proj_w.clone(),
                dt_proj_w: tl.dt_proj_w.clone(),
                dt_proj_b: tl.dt_proj_b.clone(),
                a_log: tl.a_log.clone(),
                a_neg: tl.a_log.iter().map(|v| -v.exp()).collect(),
                d_param: tl.d_param.clone(),
                out_proj_w: tl.out_proj_w.clone(),
            })
            .collect(),
        norm_f_weight: tw.norm_f_weight.clone(),
    }
}

struct Metrics {
    mse: f64,
    mae: f64,
    max_error: f64,
    r_squared: f64,
}

#[cfg(feature = "cuda")]
fn eval_gpu(
    test_inputs: &[f32],
    test_targets: &[f32],
    bb: &MambaBackbone,
    input_dim: usize,
) -> Metrics {
    let _gpu_bb = GpuMambaBackbone::new(0, bb.weights(), *bb.config(), input_dim, 1).unwrap();

    let cfg = bb.config();
    let dm = cfg.d_model;
    let total_steps = test_targets.len();
    let mut predictions = vec![0.0f32; total_steps];
    let mut out = vec![0.0f32; dm];
    let mut state = bb.alloc_state();
    let mut scratch = bb.alloc_scratch();

    for (t, pred) in predictions.iter_mut().enumerate().take(total_steps) {
        let inp_start = t * input_dim;
        let inp = &test_inputs[inp_start..inp_start + input_dim];
        state.reset();
        bb.forward_step(inp, &mut out, &mut state, &mut scratch);
        *pred = out[0];
    }

    compute_metrics(&predictions, test_targets)
}

#[cfg(not(feature = "cuda"))]
fn eval_cpu(
    test_inputs: &[f32],
    test_targets: &[f32],
    bb: &MambaBackbone,
    input_dim: usize,
) -> Metrics {
    let cfg = bb.config();
    let dm = cfg.d_model;
    let total_steps = test_targets.len();
    let mut predictions = vec![0.0f32; total_steps];
    let mut out = vec![0.0f32; dm];
    let mut state = bb.alloc_state();
    let mut scratch = bb.alloc_scratch();

    for (t, pred) in predictions.iter_mut().enumerate().take(total_steps) {
        let inp_start = t * input_dim;
        let inp = &test_inputs[inp_start..inp_start + input_dim];
        state.reset();
        bb.forward_step(inp, &mut out, &mut state, &mut scratch);
        *pred = out[0];
    }

    compute_metrics(&predictions, test_targets)
}

fn compute_metrics(predictions: &[f32], targets: &[f32]) -> Metrics {
    let n = predictions.len();
    let mut sum_sq_err = 0.0;
    let mut sum_abs_err = 0.0;
    let mut max_err = 0.0f64;
    let mut sum_target = 0.0;

    for (pred, target) in predictions.iter().zip(targets.iter()) {
        let err = (*pred - *target) as f64;
        sum_sq_err += err * err;
        sum_abs_err += err.abs();
        max_err = max_err.max(err);
        sum_target += *target as f64;
    }

    let mean_target = sum_target / n as f64;
    let mut sum_sq_total = 0.0;
    for t in targets.iter() {
        let diff = *t as f64 - mean_target;
        sum_sq_total += diff * diff;
    }

    let mse = sum_sq_err / n as f64;
    let r_squared = if sum_sq_total > 0.0 {
        1.0 - sum_sq_err / sum_sq_total
    } else {
        1.0
    };

    Metrics {
        mse,
        mae: sum_abs_err / n as f64,
        max_error: max_err,
        r_squared,
    }
}
