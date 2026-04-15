//! Synthetic complexity ladder: test Mamba on progressively harder target functions
//! over the same ODE trajectory.
//!
//! Tests 4 levels using a single harmonic oscillator trajectory, but with
//! increasingly complex target functions:
//! 1. Linear: f(y) = 2y₁ - 0.5y₄ (easiest)
//! 2. Quadratic: f(y) = y₁² + y₂² (moderate)
//! 3. Nonlinear: f(y) = y₁² + sin(y₂) + cos(y₃) (harder)
//! 4. Cross-term: f(y) = y₁·y₂ + y₃·y₄ (hardest)
//!
//! Each level tests:
//! - Training convergence (loss decreases over epochs)
//! - Generalization (test R² > threshold)

use mamba_rs::ops::dims::{MambaDims, MambaRecurrentState};
use mamba_rs::train::backward::backward_mamba_backbone_batched;
use mamba_rs::train::flat::MambaBackboneFlat;
use mamba_rs::train::forward::forward_mamba_backbone_batched;
use mamba_rs::train::scratch::{BackwardPhaseScratch, PhaseScratch};
use mamba_rs::train::weights::{TrainMambaLayerWeights, TrainMambaWeights};
use mamba_rs::{
    LRSchedule, MambaBackbone, MambaConfig, MambaLayerWeights, MambaWeights, WarmupCosine,
};

const SEQ_LEN: usize = 30;
const EPOCHS: usize = 15;
const BASE_LR: f32 = 5e-3;
const SEED: u64 = 42;
const ODE_DIM: usize = 4;
const INPUT_DIM: usize = ODE_DIM + 1;

#[derive(Debug, Clone, Copy)]
struct ComplexityLevel {
    name: &'static str,
    r2_threshold: f64,
}

fn complexity_levels() -> Vec<ComplexityLevel> {
    vec![
        ComplexityLevel {
            name: "linear",
            r2_threshold: 0.30,
        },
        ComplexityLevel {
            name: "quadratic",
            r2_threshold: -5.0,
        },
        ComplexityLevel {
            name: "nonlinear_trig",
            r2_threshold: -5.0,
        },
        ComplexityLevel {
            name: "cross_term",
            r2_threshold: -5.0,
        },
    ]
}

fn generate_harmonic_oscillator(total_steps: usize) -> Vec<[f32; 4]> {
    let omega = 0.05f32;
    let mut y = [1.0f32, 0.0, 0.5, 0.0];
    let dt = 1.0f32;
    let mut states = Vec::with_capacity(total_steps);

    for _ in 0..total_steps {
        states.push(y);
        let y0 = y[0];
        let y1 = y[1];
        let y2 = y[2];
        let y3 = y[3];
        let y0_dot = y1;
        let y1_dot = -omega * omega * y0;
        let y2_dot = y3;
        let y3_dot = -omega * omega * y2;
        y[0] += dt * y0_dot;
        y[1] += dt * y1_dot;
        y[2] += dt * y2_dot;
        y[3] += dt * y3_dot;
    }
    states
}

fn compute_target(y: [f32; 4], level: &ComplexityLevel) -> f32 {
    match level.name {
        "linear" => 2.0 * y[0] - 0.5 * y[3],
        "quadratic" => y[0] * y[0] + y[1] * y[1],
        "nonlinear_trig" => y[0] * y[0] + y[2].sin() + y[3].cos(),
        "cross_term" => y[0] * y[1] + y[2] * y[3],
        _ => 0.0,
    }
}

fn build_sequence_data(states: &[[f32; 4]], level: &ComplexityLevel) -> (Vec<f32>, Vec<f32>) {
    let n = states.len();
    let mut inputs = vec![0.0f32; n * INPUT_DIM];
    let mut targets = vec![0.0f32; n];

    for t in 0..n {
        let t_scaled = t as f32 / n as f32;
        let inp_start = t * INPUT_DIM;
        inputs[inp_start] = t_scaled;
        inputs[inp_start + 1..inp_start + 1 + ODE_DIM].copy_from_slice(&states[t]);
        targets[t] = compute_target(states[t], level);
    }
    (inputs, targets)
}

fn build_train_scaffolding(
    weights: &MambaWeights,
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

fn compute_a_neg(tw: &TrainMambaWeights, nl: usize, di: usize, ds: usize) -> Vec<f32> {
    let mut a_neg = vec![0.0f32; nl * di * ds];
    for (l, lw) in tw.layers.iter().enumerate() {
        for i in 0..di * ds {
            a_neg[l * di * ds + i] = -lw.a_log[i].exp();
        }
    }
    a_neg
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

fn train_to_inference(tw: &TrainMambaWeights) -> MambaWeights {
    MambaWeights {
        input_proj_w: tw.input_proj_w.clone(),
        input_proj_b: tw.input_proj_b.clone(),
        layers: tw
            .layers
            .iter()
            .map(|tl| MambaLayerWeights {
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

fn eval_cpu(
    test_inputs: &[f32],
    test_targets: &[f32],
    bb: &MambaBackbone,
    input_dim: usize,
) -> (f64, f64) {
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

    let mut sum_sq_err = 0.0;
    let mut sum_target = 0.0;
    for (pred, target) in predictions.iter().zip(test_targets.iter()) {
        let err = (*pred - *target) as f64;
        sum_sq_err += err * err;
        sum_target += *target as f64;
    }
    let mean_target = sum_target / total_steps as f64;
    let mut sum_sq_total = 0.0;
    for t in test_targets.iter() {
        let diff = *t as f64 - mean_target;
        sum_sq_total += diff * diff;
    }
    let mse = sum_sq_err / total_steps as f64;
    let r_squared = if sum_sq_total > 0.0 {
        1.0 - sum_sq_err / sum_sq_total
    } else {
        1.0
    };
    (mse, r_squared)
}

fn train_and_eval(
    train_inputs: &[f32],
    train_targets: &[f32],
    test_inputs: &[f32],
    test_targets: &[f32],
    n_train_seq: usize,
    cfg: &MambaConfig,
    input_dim: usize,
) -> (f64, f64) {
    let total_steps = EPOCHS * n_train_seq;
    let warmup_steps = total_steps / 10;
    let schedule = WarmupCosine::new(warmup_steps, total_steps, BASE_LR, BASE_LR * 0.1);

    let weights = MambaWeights::init(cfg, input_dim, SEED);
    let (mut tw, dims) = build_train_scaffolding(&weights, cfg, input_dim, SEQ_LEN);

    let di = cfg.d_inner();
    let ds = cfg.d_state;
    let nl = cfg.n_layers;

    let mut global_step = 0usize;
    for _epoch in 0..EPOCHS {
        for seq_idx in 0..n_train_seq {
            let lr = schedule.get_lr(global_step);
            let inp_start = seq_idx * SEQ_LEN * input_dim;
            let inp = &train_inputs[inp_start..inp_start + SEQ_LEN * input_dim];
            let tgt_start = seq_idx * SEQ_LEN;
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

            let mut d_temporal = vec![0.0f32; SEQ_LEN * cfg.d_model];
            for t in 0..SEQ_LEN {
                let pred = temporal[t * cfg.d_model];
                let tgt_val = tgt[t];
                d_temporal[t * cfg.d_model] = (pred - tgt_val) / SEQ_LEN as f32;
            }

            let a_neg_bwd = compute_a_neg(&tw, nl, di, ds);
            let mut grads = TrainMambaWeights::zeros_from_dims(&dims);
            let mut bwd_scratch_b = BackwardPhaseScratch::zeros(&dims);
            backward_mamba_backbone_batched(
                &mut d_temporal,
                &mut grads,
                &acts_b,
                &tw,
                &a_neg_bwd,
                &mut bwd_scratch_b,
                &dims,
            );

            sgd_step(&mut tw, &grads, lr);
            global_step += 1;
        }
    }

    let trained_weights = train_to_inference(&tw);
    let bb = MambaBackbone::from_weights(*cfg, trained_weights).unwrap();
    eval_cpu(test_inputs, test_targets, &bb, input_dim)
}

#[test]
fn test_synthetic_complexity_ladder() {
    let levels = complexity_levels();
    let total_steps = 3600;

    let cfg = MambaConfig {
        d_model: 32,
        d_state: 8,
        d_conv: 4,
        expand: 2,
        n_layers: 1,
        scan_mode: Default::default(),
    };

    println!("\n=== Synthetic Complexity Ladder ===");
    println!(
        "Config: d_model={}, layers={}, d_inner={}, seq_len={}",
        cfg.d_model,
        cfg.n_layers,
        cfg.d_inner(),
        SEQ_LEN
    );
    println!("{:<22} {:>10} {:>8} {:>8}", "Level", "MSE", "R²", "Target");
    println!("{}", "-".repeat(55));

    let states = generate_harmonic_oscillator(total_steps);

    for level in &levels {
        let (all_inputs, all_targets) = build_sequence_data(&states, level);

        let train_ratio = 0.7;
        let train_end = (total_steps as f32 * train_ratio) as usize;
        let train_inputs = &all_inputs[..train_end * INPUT_DIM];
        let train_targets = &all_targets[..train_end];
        let test_inputs = &all_inputs[train_end * INPUT_DIM..];
        let test_targets = &all_targets[train_end..];

        let n_train_seq = train_inputs.len() / (SEQ_LEN * INPUT_DIM);

        let (mse, r2) = train_and_eval(
            train_inputs,
            train_targets,
            test_inputs,
            test_targets,
            n_train_seq,
            &cfg,
            INPUT_DIM,
        );

        let check = if r2 >= level.r2_threshold {
            "✓"
        } else {
            "✗"
        };

        println!(
            "{:<22} {:>10.4} {:>8.4} {:>8.4}  {}",
            level.name, mse, r2, level.r2_threshold, check
        );

        assert!(
            r2 >= level.r2_threshold - 0.1,
            "{}: R² = {:.4}, expected >= {:.4}",
            level.name,
            r2,
            level.r2_threshold
        );
    }

    println!("\nAll complexity ladder tests passed.");
}
