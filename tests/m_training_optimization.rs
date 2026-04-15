//! Benchmark comparison: constant LR vs warmup+cosine, with/without gradient accumulation.
//!
//! Outputs CSV results to stdout, or to file via CSV_OUTPUT_PATH env var.
//!
//! ```bash
//! cargo test --test m_training_optimization -- --nocapture
//! CSV_OUTPUT_PATH=results.csv cargo test --test m_training_optimization -- --nocapture
//! ```

use mamba_rs::ops::dims::{MambaDims, MambaRecurrentState};
use mamba_rs::train::backward::backward_mamba_backbone_batched;
use mamba_rs::train::flat::MambaBackboneFlat;
use mamba_rs::train::forward::forward_mamba_backbone_batched;
use mamba_rs::train::scratch::{BackwardPhaseScratch, PhaseScratch};
use mamba_rs::train::weights::{TrainMambaLayerWeights, TrainMambaWeights};
use mamba_rs::{
    LRSchedule, MambaBackbone, MambaConfig, MambaLayerWeights, MambaWeights, WarmupCosine,
};
use std::time::Instant;

const ODE_DIM: usize = 4;
const SEQ_LEN: usize = 30;
const EPOCHS: usize = 15;
const BASE_LR: f32 = 2e-3;
const SEED: u64 = 42;

fn generate_ode_data(total_steps: usize) -> (Vec<f32>, Vec<f32>) {
    let omega = 0.05f32;
    let mut y = [1.0f32, 0.0, 0.5, 0.0];
    let dt = 1.0f32;
    let mut states = Vec::with_capacity(total_steps);

    for _ in 0..total_steps {
        states.push(y);
        let y1_new = y[0] + dt * y[1];
        let y2_new = y[2] + dt * y[3];
        let y1_dot = -omega * omega * y[0];
        let y2_dot = -omega * omega * y[2];
        y[0] = y1_new + dt * y1_dot;
        y[1] += dt * y1_dot;
        y[2] = y2_new + dt * y2_dot;
        y[3] += dt * y2_dot;
    }

    let input_dim = ODE_DIM + 1;
    let mut inputs = vec![0.0f32; total_steps * input_dim];
    let mut targets = vec![0.0f32; total_steps];

    for t in 0..total_steps {
        let t_scaled = t as f32 / total_steps as f32;
        let inp_start = t * input_dim;
        inputs[inp_start] = t_scaled;
        inputs[inp_start + 1..inp_start + 1 + ODE_DIM].copy_from_slice(&states[t]);
        let y = states[t];
        targets[t] = 2.0 * y[0] - 0.5 * y[3];
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

struct BenchmarkConfig {
    name: &'static str,
    use_schedule: bool,
    accum_steps: usize,
}

fn run_benchmark(
    train_inputs: &[f32],
    train_targets: &[f32],
    test_inputs: &[f32],
    test_targets: &[f32],
    bc: &BenchmarkConfig,
) -> (f64, f64, f64) {
    let input_dim = ODE_DIM + 1;
    let cfg = MambaConfig {
        d_model: 32,
        d_state: 8,
        d_conv: 4,
        expand: 2,
        n_layers: 1,
        scan_mode: Default::default(),
    };
    let n_train_seq = train_inputs.len() / (SEQ_LEN * input_dim);
    let total_steps = EPOCHS * n_train_seq;
    let warmup_steps = total_steps / 10;

    let schedule = if bc.use_schedule {
        WarmupCosine::new(warmup_steps, total_steps, BASE_LR, BASE_LR * 0.1)
    } else {
        WarmupCosine::new(1, total_steps, BASE_LR, BASE_LR)
    };

    let weights = MambaWeights::init(&cfg, input_dim, SEED);
    let (mut tw, dims) = build_train_scaffolding(&weights, &cfg, input_dim, SEQ_LEN);

    let di = cfg.d_inner();
    let ds = cfg.d_state;
    let nl = cfg.n_layers;

    let mut global_step = 0usize;
    let start = Instant::now();

    for _epoch in 0..EPOCHS {
        for seq_idx in 0..n_train_seq {
            let mut grads_accum = TrainMambaWeights::zeros_from_dims(&dims);

            for accum_i in 0..bc.accum_steps {
                let sample_idx = (seq_idx * bc.accum_steps + accum_i) % n_train_seq;
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

                let mut d_temporal = vec![0.0f32; SEQ_LEN * cfg.d_model];
                for t in 0..SEQ_LEN {
                    let pred = temporal[t * cfg.d_model];
                    let tgt_val = tgt[t];
                    d_temporal[t * cfg.d_model] = (pred - tgt_val) / SEQ_LEN as f32;
                }

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

            let lr = schedule.get_lr(global_step);
            sgd_step(&mut tw, &grads_accum, lr / bc.accum_steps as f32);
            global_step += 1;
        }
    }

    let elapsed = start.elapsed().as_secs_f64();

    let trained_weights = train_to_inference(&tw);
    let bb = MambaBackbone::from_weights(cfg, trained_weights).unwrap();
    let (_mse, r2) = eval_cpu(test_inputs, test_targets, &bb, input_dim);

    (elapsed, r2, global_step as f64)
}

#[test]
fn test_training_optimization_benchmark() {
    let input_dim = ODE_DIM + 1;
    let (all_inputs, all_targets) = generate_ode_data(3600);

    let train_end = (3600.0 * 0.7) as usize;
    let train_inputs = &all_inputs[..train_end * input_dim];
    let train_targets = &all_targets[..train_end];
    let test_inputs = &all_inputs[train_end * input_dim..];
    let test_targets = &all_targets[train_end..];

    let configs = vec![
        BenchmarkConfig {
            name: "constant_lr",
            use_schedule: false,
            accum_steps: 1,
        },
        BenchmarkConfig {
            name: "warmup_cosine",
            use_schedule: true,
            accum_steps: 1,
        },
        BenchmarkConfig {
            name: "warmup_cosine_accum4",
            use_schedule: true,
            accum_steps: 4,
        },
    ];

    let csv_path = std::env::var("CSV_OUTPUT_PATH").ok();
    let mut writer: Box<dyn std::io::Write> = if let Some(ref path) = csv_path {
        Box::new(std::fs::File::create(path).expect("Failed to create CSV file"))
    } else {
        Box::new(std::io::stdout())
    };

    writeln!(writer, "config,elapsed_secs,r_squared,steps_per_sec").unwrap();

    for bc in &configs {
        let (elapsed, r2, total_steps) =
            run_benchmark(train_inputs, train_targets, test_inputs, test_targets, bc);
        let steps_per_sec = total_steps / elapsed;
        writeln!(
            writer,
            "{},{:.2},{:.4},{:.1}",
            bc.name, elapsed, r2, steps_per_sec
        )
        .unwrap();
    }

    if let Some(ref path) = csv_path {
        eprintln!("CSV written to: {}", path);
    }
}
