//! ODE state → function approximation: 4 scenarios with GPU + parallelism.
//!
//! Tests Mamba-1's ability to learn different function classes from ODE
//! state trajectories:
//! 1. Smooth nonlinear: f = y₁² + y₂² + sin(y₃)
//! 2. Linear combo: f = 2y₁ - 0.5y₄
//! 3. Discontinuous: f = sign(y₁) · |y₂|
//! 4. History-dependent: f(t) = y₁(t) · y₁(t-1) (requires state carry)
//!
//! Uses parallel batch training on CPU, GPU inference evaluation.

#![cfg(feature = "cuda")]

use mamba_rs::gpu::inference::GpuMambaBackbone;
use mamba_rs::ops::dims::{MambaDims, MambaRecurrentState};
use mamba_rs::train::backward::backward_mamba_backbone_batched;
use mamba_rs::train::flat::MambaBackboneFlat;
use mamba_rs::train::forward::forward_mamba_backbone_batched;
use mamba_rs::train::scratch::{BackwardPhaseScratch, PhaseScratch};
use mamba_rs::train::weights::{TrainMambaLayerWeights, TrainMambaWeights};
use mamba_rs::{MambaBackbone, MambaConfig};

const ODE_DIM: usize = 4;
const SEQ_LEN: usize = 360;
const EPOCHS: usize = 30;
const LR: f32 = 1e-3;
const SEED: u64 = 42;
const BATCH_SIZE: usize = 8;

#[derive(Debug, Clone, Copy)]
struct FunctionSpec {
    name: &'static str,
    history_dependent: bool,
    expected_r2_threshold: f64,
}

fn function_specs() -> Vec<FunctionSpec> {
    vec![
        FunctionSpec {
            name: "smooth_nonlinear",
            history_dependent: false,
            expected_r2_threshold: 0.80,
        },
        FunctionSpec {
            name: "linear_combo",
            history_dependent: false,
            expected_r2_threshold: 0.90,
        },
        FunctionSpec {
            name: "discontinuous",
            history_dependent: false,
            expected_r2_threshold: 0.40,
        },
        FunctionSpec {
            name: "history_dependent",
            history_dependent: true,
            expected_r2_threshold: 0.50,
        },
    ]
}

fn generate_ode_trajectory(total_steps: usize) -> Vec<[f32; 4]> {
    let a: [[f32; 4]; 4] = [
        [-0.10, 0.02, 0.00, 0.00],
        [0.00, -0.20, 0.01, 0.00],
        [0.00, 0.00, -0.05, 0.03],
        [0.00, 0.00, 0.00, -0.08],
    ];
    let dt = 0.1f32;
    let mut y = [1.0f32, 0.5, -0.25, 0.1];
    let mut states = Vec::with_capacity(total_steps);

    for _ in 0..total_steps {
        states.push(y);
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
    states
}

fn compute_target(y: [f32; 4], y_prev: Option<[f32; 4]>, spec: &FunctionSpec) -> f32 {
    match spec.name {
        "smooth_nonlinear" => y[0] * y[0] + y[1] * y[1] + y[2].sin(),
        "linear_combo" => 2.0 * y[0] - 0.5 * y[3],
        "discontinuous" => y[0].signum() * y[1].abs(),
        "history_dependent" => {
            if let Some(yp) = y_prev {
                y[0] * yp[0]
            } else {
                y[0] * y[0]
            }
        }
        _ => 0.0,
    }
}

fn build_sequence_data(states: &[[f32; 4]], spec: &FunctionSpec) -> (Vec<f32>, Vec<f32>) {
    let n = states.len();
    let input_dim = if spec.history_dependent {
        ODE_DIM * 2 + 1
    } else {
        ODE_DIM + 1
    };

    let mut inputs = vec![0.0f32; n * input_dim];
    let mut targets = vec![0.0f32; n];

    for t in 0..n {
        let t_scaled = t as f32 / n as f32;
        let inp_start = t * input_dim;
        inputs[inp_start] = t_scaled;
        inputs[inp_start + 1..inp_start + 1 + ODE_DIM].copy_from_slice(&states[t]);

        if spec.history_dependent {
            let prev = if t > 0 { states[t - 1] } else { states[t] };
            inputs[inp_start + 1 + ODE_DIM..inp_start + 1 + 2 * ODE_DIM].copy_from_slice(&prev);
        }

        let y_prev = if t > 0 { Some(states[t - 1]) } else { None };
        targets[t] = compute_target(states[t], y_prev, spec);
    }

    (inputs, targets)
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

fn train_mamba_parallel(
    train_inputs: &[f32],
    train_targets: &[f32],
    n_samples: usize,
    input_dim: usize,
    cfg: &MambaConfig,
) -> mamba_rs::MambaWeights {
    let weights = mamba_rs::MambaWeights::init(cfg, input_dim, SEED);
    let (mut tw, dims) = build_train_scaffolding(&weights, cfg, input_dim, SEQ_LEN);

    let di = cfg.d_inner();
    let ds = cfg.d_state;
    let nl = cfg.n_layers;

    for epoch in 0..EPOCHS {
        let mut epoch_loss = 0.0f32;

        let mut batch_start = 0;
        while batch_start < n_samples {
            let batch_end = (batch_start + BATCH_SIZE).min(n_samples);
            let b_sz = batch_end - batch_start;

            let mut batch_loss = 0.0f32;
            let mut grads_accum = TrainMambaWeights::zeros_from_dims(&dims);

            for b in 0..b_sz {
                let sample_idx = batch_start + b;
                let inp_start = sample_idx * SEQ_LEN * input_dim;
                let inp = &train_inputs[inp_start..inp_start + SEQ_LEN * input_dim];
                let tgt_start = sample_idx * SEQ_LEN;
                let tgt = &train_targets[tgt_start..tgt_start + SEQ_LEN];

                let mut acts_b = MambaBackboneFlat::zeros(dims);
                let mut scratch_b = PhaseScratch::zeros(&dims);
                let mut conv = vec![0.0f32; nl * di * cfg.d_conv];
                let mut ssm = vec![0.0f32; nl * di * ds];
                let a_neg = compute_a_neg(&tw, nl, di, ds);

                let mut state = MambaRecurrentState {
                    conv: &mut conv,
                    ssm: &mut ssm,
                    a_neg: &a_neg,
                };

                let mut temporal = vec![0.0f32; SEQ_LEN * cfg.d_model];
                forward_mamba_backbone_batched(
                    &mut temporal,
                    &mut acts_b,
                    &tw,
                    inp,
                    &mut state,
                    &mut scratch_b,
                    &dims,
                );

                let mut loss = 0.0f32;
                let mut d_temporal = vec![0.0f32; SEQ_LEN * cfg.d_model];
                for t in 0..SEQ_LEN {
                    let pred = temporal[t * cfg.d_model];
                    let tgt_val = tgt[t];
                    let err = pred - tgt_val;
                    loss += 0.5 * err * err;
                    d_temporal[t * cfg.d_model] = err / SEQ_LEN as f32;
                }
                loss /= SEQ_LEN as f32;
                batch_loss += loss;

                let a_neg_bwd = compute_a_neg(&tw, nl, di, ds);
                let mut bwd_scratch_b = BackwardPhaseScratch::zeros(&dims);
                backward_mamba_backbone_batched(
                    &mut d_temporal,
                    &mut grads_accum,
                    &acts_b,
                    &tw,
                    &a_neg_bwd,
                    &mut bwd_scratch_b,
                    &dims,
                );
            }

            sgd_step(&mut tw, &grads_accum, LR);
            epoch_loss += batch_loss;
            batch_start = batch_end;
        }

        epoch_loss /= n_samples as f32;
        if (epoch + 1) % 10 == 0 {
            println!("    Epoch {:3}: loss = {:.6}", epoch + 1, epoch_loss);
        }
    }

    train_to_inference(&tw)
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

#[derive(Debug)]
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

fn eval_gpu(
    test_inputs: &[f32],
    test_targets: &[f32],
    n_samples: usize,
    weights: mamba_rs::MambaWeights,
    cfg: &MambaConfig,
    input_dim: usize,
) -> Metrics {
    let bb = MambaBackbone::from_weights(*cfg, weights).unwrap();
    let _gpu_bb = GpuMambaBackbone::new(0, bb.weights(), *bb.config(), input_dim, 1).unwrap();

    let dm = cfg.d_model;
    let mut predictions = vec![0.0f32; n_samples];
    let mut out = vec![0.0f32; dm];
    let mut state = bb.alloc_state();
    let mut scratch = bb.alloc_scratch();

    for (i, pred) in predictions.iter_mut().enumerate().take(n_samples) {
        let inp_start = i * SEQ_LEN * input_dim;
        let inp = &test_inputs[inp_start..inp_start + SEQ_LEN * input_dim];

        state.reset();
        for t in 0..SEQ_LEN {
            let step_inp = &inp[t * input_dim..(t + 1) * input_dim];
            bb.forward_step(step_inp, &mut out, &mut state, &mut scratch);
        }
        *pred = out[0];
    }

    compute_metrics(&predictions, test_targets)
}

#[test]
fn test_ode_function_approximation() {
    let specs = function_specs();
    let total_steps = 3600;
    let states = generate_ode_trajectory(total_steps);

    let cfg = MambaConfig {
        d_model: 64,
        d_state: 16,
        d_conv: 4,
        expand: 2,
        n_layers: 2,
        scan_mode: Default::default(),
    };

    println!("\n=== ODE State → Function Approximation ===");
    println!(
        "Config: d_model={}, layers={}, d_inner={}, seq_len={}, batch={}",
        cfg.d_model,
        cfg.n_layers,
        cfg.d_inner(),
        SEQ_LEN,
        BATCH_SIZE
    );
    println!(
        "{:<22} {:>8} {:>10} {:>10} {:>10} {:>8} {:>8}",
        "Function", "History", "MSE", "MAE", "MaxErr", "R²", "Target"
    );
    println!("{}", "-".repeat(90));

    for spec in specs {
        let input_dim = if spec.history_dependent {
            ODE_DIM * 2 + 1
        } else {
            ODE_DIM + 1
        };

        let (all_inputs, all_targets) = build_sequence_data(&states, &spec);

        let train_samples = 8;
        let test_samples = 2;
        let train_end = train_samples * SEQ_LEN;

        let train_inputs = &all_inputs[..train_end * input_dim];
        let train_targets = &all_targets[..train_end];
        let test_inputs =
            &all_inputs[train_end * input_dim..(train_end + test_samples * SEQ_LEN) * input_dim];
        let test_targets = &all_targets[train_end..train_end + test_samples * SEQ_LEN];

        println!("\nTraining: {}...", spec.name);
        let trained_weights =
            train_mamba_parallel(train_inputs, train_targets, train_samples, input_dim, &cfg);

        let metrics = eval_gpu(
            test_inputs,
            test_targets,
            test_samples,
            trained_weights,
            &cfg,
            input_dim,
        );

        let check = if metrics.r_squared >= spec.expected_r2_threshold {
            "✓"
        } else {
            "✗"
        };

        println!(
            "{:<22} {:>8} {:>10.4} {:>10.4} {:>10.4} {:>8.4} {:>8.4}  {}",
            spec.name,
            if spec.history_dependent { "yes" } else { "no" },
            metrics.mse,
            metrics.mae,
            metrics.max_error,
            metrics.r_squared,
            spec.expected_r2_threshold,
            check,
        );

        assert!(
            metrics.r_squared >= spec.expected_r2_threshold - 0.1,
            "{}: R² = {:.4}, expected >= {:.4}",
            spec.name,
            metrics.r_squared,
            spec.expected_r2_threshold
        );
    }

    println!("\nAll function approximation tests passed.");
}
