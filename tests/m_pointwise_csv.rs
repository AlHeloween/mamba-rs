//! Point-by-point CSV output: actual vs predicted for every timestep.
//!
//! Output format:
//! ```csv
//! timestep,actual,predicted,error,abs_error
//! 0,1.234,1.100,0.134,0.134
//! 1,1.456,1.400,0.056,0.056
//! ...
//! ```
//!
//! Run with: `cargo test --test m_pointwise_csv -- --nocapture` (outputs to stdout)
//! Or set CSV_OUTPUT_PATH env var to write directly to a file.

use mamba_rs::ops::dims::{MambaDims, MambaRecurrentState};
use mamba_rs::train::backward::backward_mamba_backbone_batched;
use mamba_rs::train::flat::MambaBackboneFlat;
use mamba_rs::train::forward::forward_mamba_backbone_batched;
use mamba_rs::train::scratch::{BackwardPhaseScratch, PhaseScratch};
use mamba_rs::train::weights::{TrainMambaLayerWeights, TrainMambaWeights};
use mamba_rs::{
    LRSchedule, MambaBackbone, MambaConfig, MambaLayerWeights, MambaWeights, WarmupCosine,
};

const SEQ_LEN: usize = 60;
const EPOCHS: usize = 50;
const BASE_LR: f32 = 5e-3;
const SEED: u64 = 42;
const ODE_DIM: usize = 4;
const INPUT_DIM: usize = ODE_DIM + 1;

fn quat_multiply(a: &[f32; 4], b: &[f32; 4]) -> [f32; 4] {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]
}

fn normalize(v: &[f32; 3]) -> [f32; 3] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if n < 1e-8 {
        [1.0, 0.0, 0.0]
    } else {
        [v[0] / n, v[1] / n, v[2] / n]
    }
}

fn generate_quaternion_oscillator(total_steps: usize) -> Vec<[f32; 4]> {
    let omega = [0.0f32, 0.03, 0.04, 0.05];
    let dt = 1.0f32;
    let mut q = [1.0f32, 0.0, 0.0, 0.0];
    let mut states = Vec::with_capacity(total_steps);

    let omega_norm = (omega[1] * omega[1] + omega[2] * omega[2] + omega[3] * omega[3]).sqrt();
    let half_angle = 0.5 * omega_norm * dt;
    let sin_a = half_angle.sin();
    let cos_a = half_angle.cos();
    let axis = normalize(&omega[1..].try_into().unwrap());

    for _ in 0..total_steps {
        states.push(q);
        let dq = [cos_a, axis[0] * sin_a, axis[1] * sin_a, axis[2] * sin_a];
        q = quat_multiply(&dq, &q);
    }
    states
}

fn compute_target(y: [f32; 4]) -> f32 {
    2.0 * y[0] - 0.5 * y[3]
}

fn build_sequence_data(states: &[[f32; 4]]) -> (Vec<f32>, Vec<f32>) {
    let n = states.len();
    let mut inputs = vec![0.0f32; n * INPUT_DIM];
    let mut targets = vec![0.0f32; n];

    for t in 0..n {
        let t_scaled = t as f32 / n as f32;
        let inp_start = t * INPUT_DIM;
        inputs[inp_start] = t_scaled;
        inputs[inp_start + 1..inp_start + 1 + ODE_DIM].copy_from_slice(&states[t]);
        targets[t] = compute_target(states[t]);
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

fn train_model(
    train_inputs: &[f32],
    train_targets: &[f32],
    n_train_seq: usize,
    cfg: &MambaConfig,
    input_dim: usize,
) -> MambaWeights {
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

    train_to_inference(&tw)
}

fn predict(
    inputs: &[f32],
    n_steps: usize,
    weights: MambaWeights,
    cfg: &MambaConfig,
    input_dim: usize,
) -> Vec<f32> {
    let bb = MambaBackbone::from_weights(*cfg, weights).unwrap();
    let dm = cfg.d_model;
    let mut predictions = vec![0.0f32; n_steps];
    let mut out = vec![0.0f32; dm];
    let mut state = bb.alloc_state();
    let mut scratch = bb.alloc_scratch();

    for (t, pred) in predictions.iter_mut().enumerate().take(n_steps) {
        let inp_start = t * input_dim;
        let inp = &inputs[inp_start..inp_start + input_dim];
        state.reset();
        bb.forward_step(inp, &mut out, &mut state, &mut scratch);
        *pred = out[0];
    }
    predictions
}

#[test]
fn test_pointwise_csv_output() {
    let total_steps = 3600;
    let states = generate_quaternion_oscillator(total_steps);
    let (all_inputs, all_targets) = build_sequence_data(&states);

    let train_ratio = 0.7;
    let train_end = (total_steps as f32 * train_ratio) as usize;
    let train_inputs = &all_inputs[..train_end * INPUT_DIM];
    let train_targets = &all_targets[..train_end];
    let test_inputs = &all_inputs[train_end * INPUT_DIM..];
    let test_targets = &all_targets[train_end..];

    let n_train_seq = train_inputs.len() / (SEQ_LEN * INPUT_DIM);

    let cfg = MambaConfig {
        d_model: 64,
        d_state: 16,
        d_conv: 4,
        expand: 2,
        n_layers: 2,
        scan_mode: Default::default(),
    };

    eprintln!("Training model...");
    let trained_weights = train_model(train_inputs, train_targets, n_train_seq, &cfg, INPUT_DIM);

    eprintln!("Predicting on test set...");
    let predictions = predict(
        test_inputs,
        test_targets.len(),
        trained_weights,
        &cfg,
        INPUT_DIM,
    );

    let csv_path = std::env::var("CSV_OUTPUT_PATH").ok();
    let mut writer: Box<dyn std::io::Write> = if let Some(ref path) = csv_path {
        Box::new(std::fs::File::create(path).expect("Failed to create CSV file"))
    } else {
        Box::new(std::io::stdout())
    };

    writeln!(writer, "timestep,actual,predicted,error,abs_error").unwrap();
    for (t, (actual, predicted)) in test_targets.iter().zip(predictions.iter()).enumerate() {
        let error = *predicted - *actual;
        let abs_error = error.abs();
        writeln!(
            writer,
            "{},{:.6},{:.6},{:.6},{:.6}",
            train_end + t,
            actual,
            predicted,
            error,
            abs_error
        )
        .unwrap();
    }

    if let Some(ref path) = csv_path {
        eprintln!("CSV written to: {}", path);
    }
}
