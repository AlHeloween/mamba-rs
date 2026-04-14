//! Function class discovery test: Can Mamba learn periodicity and continuity?
//!
//! Tests Mamba-1 on 4 function classes:
//! 1. Periodic + continuous: sin(πt/180) + cos(πt/180)
//! 2. Periodic + discontinuous: square wave, sawtooth
//! 3. Non-periodic + continuous: linear ramp, quadratic
//! 4. Non-periodic + discontinuous: step function, random walk
//!
//! Each trained on 1 period (or equivalent), evaluated on held-out range.
//! Hypothesis: Mamba should generalize best to periodic+continuous functions.

#![cfg(feature = "cuda")]

use mamba_rs::gpu::inference::GpuMambaBackbone;
use mamba_rs::ops::dims::{MambaDims, MambaRecurrentState};
use mamba_rs::train::backward::backward_mamba_backbone_batched;
use mamba_rs::train::flat::MambaBackboneFlat;
use mamba_rs::train::forward::forward_mamba_backbone_batched;
use mamba_rs::train::scratch::{BackwardPhaseScratch, PhaseScratch};
use mamba_rs::train::weights::{TrainMambaLayerWeights, TrainMambaWeights};
use mamba_rs::{MambaBackbone, MambaConfig};
use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Signal generators
// ---------------------------------------------------------------------------

fn signal_sin_cos(t: usize) -> f32 {
    (PI * t as f32 / 180.0).sin() + (PI * t as f32 / 180.0).cos()
}

fn signal_square(t: usize) -> f32 {
    let period = 360;
    if (t % period) < period / 2 { 1.0 } else { -1.0 }
}

fn signal_sawtooth(t: usize) -> f32 {
    let period = 360;
    2.0 * (t % period) as f32 / period as f32 - 1.0
}

fn signal_linear(t: usize) -> f32 {
    t as f32 * 0.001
}

fn signal_quadratic(t: usize) -> f32 {
    (t as f32 * 0.01).powi(2) * 0.0001
}

fn signal_step(t: usize) -> f32 {
    if t < 18000 { 0.0 } else { 1.0 }
}

fn signal_random_walk(t: usize, seed: u64) -> f32 {
    let mut x = 0.0f32;
    for i in 0..t {
        let h = ((seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407u64.wrapping_mul(i as u64)))
            >> 33) as f32;
        x += (h - 0.5) * 0.02;
        x = x.clamp(-2.0, 2.0);
    }
    x
}

// ---------------------------------------------------------------------------
// Function class definitions
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
struct FunctionSpec {
    name: &'static str,
    periodic: bool,
    continuous: bool,
    fn_ptr: fn(usize) -> f32,
    period: usize,
}

fn function_specs() -> Vec<FunctionSpec> {
    vec![
        FunctionSpec {
            name: "sin+cos",
            periodic: true,
            continuous: true,
            fn_ptr: signal_sin_cos,
            period: 360,
        },
        FunctionSpec {
            name: "square",
            periodic: true,
            continuous: false,
            fn_ptr: signal_square,
            period: 360,
        },
        FunctionSpec {
            name: "sawtooth",
            periodic: true,
            continuous: false,
            fn_ptr: signal_sawtooth,
            period: 360,
        },
        FunctionSpec {
            name: "linear",
            periodic: false,
            continuous: true,
            fn_ptr: signal_linear,
            period: 36000,
        },
        FunctionSpec {
            name: "quadratic",
            periodic: false,
            continuous: true,
            fn_ptr: signal_quadratic,
            period: 36000,
        },
        FunctionSpec {
            name: "step",
            periodic: false,
            continuous: false,
            fn_ptr: signal_step,
            period: 36000,
        },
    ]
}

// ---------------------------------------------------------------------------
// Dataset building
// ---------------------------------------------------------------------------

fn generate_signal(fn_ptr: fn(usize) -> f32, total_steps: usize) -> Vec<f32> {
    (0..total_steps).map(fn_ptr).collect()
}

fn build_lookback_dataset(signal: &[f32], lookback: usize) -> (Vec<f32>, Vec<f32>) {
    let n_samples = signal.len() - lookback;
    let mut inputs = vec![0.0f32; n_samples * lookback];
    let mut targets = vec![0.0f32; n_samples];
    for i in 0..n_samples {
        inputs[i * lookback..(i + 1) * lookback].copy_from_slice(&signal[i..i + lookback]);
        targets[i] = signal[i + lookback];
    }
    (inputs, targets)
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
struct Metrics {
    mse: f64,
    mae: f64,
    max_error: f64,
    r_squared: f64,
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

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const LOOKBACK: usize = 90;
const TOTAL_STEPS: usize = 36000;
const TRAIN_PERIODS: usize = 1;
const EPOCHS: usize = 20;
const LR: f32 = 1e-3;
const SEED: u64 = 42;

// ---------------------------------------------------------------------------
// Training scaffolding
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Training
// ---------------------------------------------------------------------------

fn train_m1_cpu(
    train_inputs: &[f32],
    train_targets: &[f32],
    n_train: usize,
    input_dim: usize,
) -> mamba_rs::MambaWeights {
    let cfg = MambaConfig::default();
    let seq_len = n_train;

    let weights = mamba_rs::MambaWeights::init(&cfg, input_dim, SEED);
    let (mut tw, dims) = build_train_scaffolding(&weights, &cfg, input_dim, seq_len);

    for _epoch in 0..EPOCHS {
        let (temporal, _loss, acts) = forward_and_loss(&tw, &dims, train_inputs, train_targets);
        let grads = backward_and_get_grads(&tw, &acts, &temporal, train_targets, &dims);
        sgd_step(&mut tw, &grads, LR);
    }

    mamba_rs::MambaWeights {
        input_proj_w: tw.input_proj_w,
        input_proj_b: tw.input_proj_b,
        layers: tw
            .layers
            .into_iter()
            .map(|tl| {
                let a_log_len = tl.a_log.len();
                mamba_rs::MambaLayerWeights {
                    norm_weight: tl.norm_weight,
                    in_proj_w: tl.in_proj_w,
                    conv1d_weight: tl.conv1d_weight,
                    conv1d_bias: tl.conv1d_bias,
                    x_proj_w: tl.x_proj_w,
                    dt_proj_w: tl.dt_proj_w,
                    dt_proj_b: tl.dt_proj_b,
                    a_log: tl.a_log,
                    a_neg: vec![0.0f32; a_log_len],
                    d_param: tl.d_param,
                    out_proj_w: tl.out_proj_w,
                }
            })
            .collect(),
        norm_f_weight: tw.norm_f_weight,
    }
}

// ---------------------------------------------------------------------------
// GPU evaluation
// ---------------------------------------------------------------------------

fn eval_m1_gpu(
    test_inputs: &[f32],
    test_targets: &[f32],
    weights: mamba_rs::MambaWeights,
    input_dim: usize,
) -> Metrics {
    let cfg = MambaConfig::default();
    let bb = MambaBackbone::from_weights(cfg, weights).unwrap();
    let mut gpu_bb =
        GpuMambaBackbone::new(0, bb.weights(), bb.config().clone(), input_dim, 1).unwrap();

    let sample_step = 50;
    let total_test_samples = test_targets.len();
    let n_sample = total_test_samples / sample_step;
    let mut predictions = vec![0.0f32; n_sample];
    let mut out = vec![0.0f32; cfg.d_model];
    let mut sample_targets = vec![0.0f32; n_sample];

    for i in 0..n_sample {
        let idx = i * sample_step;
        let inp_start = idx * LOOKBACK;
        let inp_end = inp_start + LOOKBACK;
        if inp_end > test_inputs.len() {
            break;
        }
        let inp = &test_inputs[inp_start..inp_end];
        gpu_bb.reset().unwrap();
        gpu_bb.step(inp, &mut out).unwrap();
        predictions[i] = out[0];
        sample_targets[i] = test_targets[idx];
    }

    compute_metrics(&predictions, &sample_targets)
}

// ---------------------------------------------------------------------------
// Main test
// ---------------------------------------------------------------------------

#[test]
fn test_function_class_discovery() {
    let specs = function_specs();
    let input_dim = LOOKBACK;
    let cfg = MambaConfig::default();

    println!("\n=== Function Class Discovery: Periodicity & Continuity ===");
    println!(
        "Lookback: {} | Train: {} step | Test: {} step | Epochs: {}",
        LOOKBACK,
        TRAIN_PERIODS,
        TOTAL_STEPS / 360 - TRAIN_PERIODS,
        EPOCHS
    );
    println!(
        "{:<14} {:>10} {:>10} {:>8} {:>8} {:>10} {:>10} {:>8} {:>8}",
        "Function", "Periodic", "Contin.", "Period", "TrainN", "MSE", "MAE", "MaxErr", "R²"
    );
    println!("{}", "-".repeat(100));

    for spec in specs {
        let signal = generate_signal(spec.fn_ptr, TOTAL_STEPS);
        let (all_inputs, all_targets) = build_lookback_dataset(&signal, LOOKBACK);

        let train_n = spec.period.min(TOTAL_STEPS / 10).min(720);
        let train_samples = train_n;
        let train_inputs = &all_inputs[..train_samples * LOOKBACK];
        let train_targets = &all_targets[..train_samples];
        let test_inputs = &all_inputs[train_samples * LOOKBACK..];
        let test_targets = &all_targets[train_samples..];

        let trained_weights = train_m1_cpu(train_inputs, train_targets, train_samples, input_dim);
        let metrics = eval_m1_gpu(test_inputs, test_targets, trained_weights, input_dim);

        println!(
            "{:<14} {:>10} {:>10} {:>8} {:>8} {:>10.4} {:>10.4} {:>8.4} {:>8.4}",
            spec.name,
            if spec.periodic { "yes" } else { "no" },
            if spec.continuous { "yes" } else { "no" },
            spec.period,
            train_samples,
            metrics.mse,
            metrics.mae,
            metrics.max_error,
            metrics.r_squared
        );
    }

    println!("\nHypothesis: Periodic+continuous functions should have highest R²");
    println!("(Mamba's SSM recurrence is well-suited for smooth periodic patterns)");
}
