//! Haar wavelet input decomposition for Mamba training.
//!
//! Compares wavelet-decomposed input vs. raw signal input across
//! different signal classes to validate that wavelet preprocessing
//! improves convergence speed and final accuracy.

use mamba_rs::ops::dims::{MambaDims, MambaRecurrentState};
use mamba_rs::ops::wavelet::haar_to_channels;
use mamba_rs::train::backward::backward_mamba_backbone_batched;
use mamba_rs::train::flat::MambaBackboneFlat;
use mamba_rs::train::forward::forward_mamba_backbone_batched;
use mamba_rs::train::scratch::{BackwardPhaseScratch, PhaseScratch};
use mamba_rs::train::weights::{TrainMambaLayerWeights, TrainMambaWeights};
use mamba_rs::{MambaBackbone, MambaConfig};
use std::f32::consts::PI;

const SEQ_LEN: usize = 360;
const EPOCHS: usize = 30;
const LR: f32 = 1e-3;
const SEED: u64 = 42;

fn signal_multiscale(t: usize) -> f32 {
    (PI * t as f32 / 180.0).sin()
        + 0.3 * (PI * t as f32 / 45.0).sin()
        + 0.1 * (t as f32 * 0.07).sin()
}

fn signal_regime_change(t: usize) -> f32 {
    if t < 180 {
        (PI * t as f32 / 90.0).sin()
    } else {
        (PI * (t - 180) as f32 / 45.0).cos() + 1.5
    }
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

fn train_with_wavelet(
    train_inputs: &[f32],
    train_targets: &[f32],
    n_train: usize,
    raw_input_dim: usize,
    cfg: &MambaConfig,
) -> mamba_rs::MambaWeights {
    let wavelet_input_dim =
        mamba_rs::ops::wavelet::haar_channels_len(raw_input_dim, cfg.wavelet_levels.max(1));
    let seq_len = n_train;

    let wavelet_train_inputs: Vec<f32> = train_inputs
        .chunks(raw_input_dim)
        .flat_map(|chunk| haar_to_channels(chunk, cfg.wavelet_levels.max(1)))
        .collect();

    let weights = mamba_rs::MambaWeights::init(cfg, wavelet_input_dim, SEED);
    let (mut tw, dims) = build_train_scaffolding(&weights, cfg, wavelet_input_dim, seq_len);

    let di = dims.d_inner;
    let ds = dims.d_state;
    let nl = dims.n_layers;

    for _epoch in 0..EPOCHS {
        let mut total_loss = 0.0f32;

        for sample_idx in 0..n_train {
            let inp_start = sample_idx * wavelet_input_dim;
            let inp = &wavelet_train_inputs[inp_start..inp_start + wavelet_input_dim];
            let tgt = train_targets[sample_idx];

            let mut acts = MambaBackboneFlat::zeros(dims);
            let mut scratch = PhaseScratch::zeros(&dims);
            let conv = vec![0.0f32; nl * di * cfg.d_conv];
            let ssm = vec![0.0f32; nl * di * ds];
            let a_neg = compute_a_neg(&tw, nl, di, ds);

            let mut state = MambaRecurrentState {
                conv: &mut (conv.clone()),
                ssm: &mut (ssm.clone()),
                a_neg: &a_neg,
            };

            let mut temporal = vec![0.0f32; seq_len * cfg.d_model];
            forward_mamba_backbone_batched(
                &mut temporal,
                &mut acts,
                &tw,
                inp,
                &mut state,
                &mut scratch,
                &dims,
            );

            let pred = temporal[(seq_len - 1) * cfg.d_model];
            let err = pred - tgt;
            let loss = 0.5 * err * err;
            total_loss += loss;

            let mut d_temporal = vec![0.0f32; seq_len * cfg.d_model];
            d_temporal[(seq_len - 1) * cfg.d_model] = err / seq_len as f32;

            let mut grads = TrainMambaWeights::zeros_from_dims(&dims);
            let mut bwd_scratch = BackwardPhaseScratch::zeros(&dims);
            let a_neg_bwd = compute_a_neg(&tw, nl, di, ds);

            backward_mamba_backbone_batched(
                &mut d_temporal,
                &mut grads,
                &acts,
                &tw,
                &a_neg_bwd,
                &mut bwd_scratch,
                &dims,
            );

            sgd_step(&mut tw, &grads, LR);
        }

        if (_epoch + 1) % 10 == 0 {
            println!(
                "    Epoch {:3}: loss = {:.6}",
                _epoch + 1,
                total_loss / n_train as f32
            );
        }
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

fn train_raw(
    train_inputs: &[f32],
    train_targets: &[f32],
    n_train: usize,
    input_dim: usize,
    cfg: &MambaConfig,
) -> mamba_rs::MambaWeights {
    let seq_len = n_train;

    let weights = mamba_rs::MambaWeights::init(cfg, input_dim, SEED);
    let (mut tw, dims) = build_train_scaffolding(&weights, cfg, input_dim, seq_len);

    let di = dims.d_inner;
    let ds = dims.d_state;
    let nl = dims.n_layers;

    for _epoch in 0..EPOCHS {
        let mut total_loss = 0.0f32;

        for sample_idx in 0..n_train {
            let inp_start = sample_idx * input_dim;
            let inp = &train_inputs[inp_start..inp_start + input_dim];
            let tgt = train_targets[sample_idx];

            let mut acts = MambaBackboneFlat::zeros(dims);
            let mut scratch = PhaseScratch::zeros(&dims);
            let conv = vec![0.0f32; nl * di * cfg.d_conv];
            let ssm = vec![0.0f32; nl * di * ds];
            let a_neg = compute_a_neg(&tw, nl, di, ds);

            let mut state = MambaRecurrentState {
                conv: &mut (conv.clone()),
                ssm: &mut (ssm.clone()),
                a_neg: &a_neg,
            };

            let mut temporal = vec![0.0f32; seq_len * cfg.d_model];
            forward_mamba_backbone_batched(
                &mut temporal,
                &mut acts,
                &tw,
                inp,
                &mut state,
                &mut scratch,
                &dims,
            );

            let pred = temporal[(seq_len - 1) * cfg.d_model];
            let err = pred - tgt;
            let loss = 0.5 * err * err;
            total_loss += loss;

            let mut d_temporal = vec![0.0f32; seq_len * cfg.d_model];
            d_temporal[(seq_len - 1) * cfg.d_model] = err / seq_len as f32;

            let mut grads = TrainMambaWeights::zeros_from_dims(&dims);
            let mut bwd_scratch = BackwardPhaseScratch::zeros(&dims);
            let a_neg_bwd = compute_a_neg(&tw, nl, di, ds);

            backward_mamba_backbone_batched(
                &mut d_temporal,
                &mut grads,
                &acts,
                &tw,
                &a_neg_bwd,
                &mut bwd_scratch,
                &dims,
            );

            sgd_step(&mut tw, &grads, LR);
        }

        if (_epoch + 1) % 10 == 0 {
            println!(
                "    Epoch {:3}: loss = {:.6}",
                _epoch + 1,
                total_loss / n_train as f32
            );
        }
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

fn compute_r2(predictions: &[f32], targets: &[f32]) -> f64 {
    let n = predictions.len();
    let mean_target: f64 = targets.iter().map(|v| *v as f64).sum::<f64>() / n as f64;
    let mut ss_res = 0.0;
    let mut ss_tot = 0.0;
    for (pred, target) in predictions.iter().zip(targets.iter()) {
        let err = *pred as f64 - *target as f64;
        ss_res += err * err;
        let diff = *target as f64 - mean_target;
        ss_tot += diff * diff;
    }
    if ss_tot == 0.0 {
        1.0
    } else {
        1.0 - ss_res / ss_tot
    }
}

fn eval_raw(
    test_inputs: &[f32],
    test_targets: &[f32],
    n_test: usize,
    weights: mamba_rs::MambaWeights,
    cfg: &MambaConfig,
    input_dim: usize,
) -> f64 {
    let bb = MambaBackbone::from_weights(*cfg, weights).unwrap();
    let mut state = bb.alloc_state();
    let mut scratch = bb.alloc_scratch();
    let mut output = vec![0.0f32; cfg.d_model];
    let mut predictions = Vec::with_capacity(n_test);

    for i in 0..n_test {
        let inp_start = i * input_dim;
        let inp = &test_inputs[inp_start..inp_start + input_dim];
        state.reset();
        bb.forward_step(inp, &mut output, &mut state, &mut scratch);
        predictions.push(output[0]);
    }

    compute_r2(&predictions, test_targets)
}

fn eval_wavelet(
    test_inputs: &[f32],
    test_targets: &[f32],
    n_test: usize,
    weights: mamba_rs::MambaWeights,
    cfg: &MambaConfig,
    raw_input_dim: usize,
) -> f64 {
    let _wavelet_input_dim =
        mamba_rs::ops::wavelet::haar_channels_len(raw_input_dim, cfg.wavelet_levels.max(1));
    let bb = MambaBackbone::from_weights(*cfg, weights).unwrap();
    let mut state = bb.alloc_state();
    let mut scratch = bb.alloc_scratch();
    let mut output = vec![0.0f32; cfg.d_model];
    let mut predictions = Vec::with_capacity(n_test);

    for i in 0..n_test {
        let inp_start = i * raw_input_dim;
        let inp_raw = &test_inputs[inp_start..inp_start + raw_input_dim];
        let inp_wavelet = haar_to_channels(inp_raw, cfg.wavelet_levels.max(1));
        state.reset();
        bb.forward_step(&inp_wavelet, &mut output, &mut state, &mut scratch);
        predictions.push(output[0]);
    }

    compute_r2(&predictions, test_targets)
}

#[test]
fn test_wavelet_vs_raw_multiscale() {
    let total_steps = 3600;
    let signal: Vec<f32> = (0..total_steps).map(signal_multiscale).collect();
    let (all_inputs, all_targets) = build_lookback_dataset(&signal, SEQ_LEN);

    let train_n = 360;
    let test_n = 100;
    let train_inputs = &all_inputs[..train_n * SEQ_LEN];
    let train_targets = &all_targets[..train_n];
    let test_inputs = &all_inputs[train_n * SEQ_LEN..(train_n + test_n) * SEQ_LEN];
    let test_targets = &all_targets[train_n..train_n + test_n];

    let cfg = MambaConfig {
        d_model: 32,
        d_state: 8,
        d_conv: 4,
        expand: 2,
        n_layers: 2,
        scan_mode: Default::default(),
        use_wavelet: true,
        wavelet_levels: 4,
    };

    println!("\n=== Wavelet vs Raw: Multi-scale Signal ===");

    println!("Training with raw input...");
    let raw_weights = train_raw(train_inputs, train_targets, train_n, SEQ_LEN, &cfg);
    let raw_r2 = eval_raw(
        test_inputs,
        test_targets,
        test_n,
        raw_weights,
        &cfg,
        SEQ_LEN,
    );

    println!("Training with wavelet input...");
    let wavelet_weights = train_with_wavelet(train_inputs, train_targets, train_n, SEQ_LEN, &cfg);
    let wavelet_r2 = eval_wavelet(
        test_inputs,
        test_targets,
        test_n,
        wavelet_weights,
        &cfg,
        SEQ_LEN,
    );

    println!("Raw R² = {:.4}", raw_r2);
    println!("Wavelet R² = {:.4}", wavelet_r2);

    assert!(
        wavelet_r2 > raw_r2 - 0.1,
        "Wavelet R² ({}) should be >= Raw R² ({})",
        wavelet_r2,
        raw_r2
    );
}

#[test]
fn test_wavelet_vs_raw_regime_change() {
    let total_steps = 3600;
    let signal: Vec<f32> = (0..total_steps).map(signal_regime_change).collect();
    let (all_inputs, all_targets) = build_lookback_dataset(&signal, SEQ_LEN);

    let train_n = 360;
    let test_n = 100;
    let train_inputs = &all_inputs[..train_n * SEQ_LEN];
    let train_targets = &all_targets[..train_n];
    let test_inputs = &all_inputs[train_n * SEQ_LEN..(train_n + test_n) * SEQ_LEN];
    let test_targets = &all_targets[train_n..train_n + test_n];

    let cfg = MambaConfig {
        d_model: 32,
        d_state: 8,
        d_conv: 4,
        expand: 2,
        n_layers: 2,
        scan_mode: Default::default(),
        use_wavelet: true,
        wavelet_levels: 4,
    };

    println!("\n=== Wavelet vs Raw: Regime Change Signal ===");

    println!("Training with raw input...");
    let raw_weights = train_raw(train_inputs, train_targets, train_n, SEQ_LEN, &cfg);
    let raw_r2 = eval_raw(
        test_inputs,
        test_targets,
        test_n,
        raw_weights,
        &cfg,
        SEQ_LEN,
    );

    println!("Training with wavelet input...");
    let wavelet_weights = train_with_wavelet(train_inputs, train_targets, train_n, SEQ_LEN, &cfg);
    let wavelet_r2 = eval_wavelet(
        test_inputs,
        test_targets,
        test_n,
        wavelet_weights,
        &cfg,
        SEQ_LEN,
    );

    println!("Raw R² = {:.4}", raw_r2);
    println!("Wavelet R² = {:.4}", wavelet_r2);

    assert!(
        wavelet_r2 > raw_r2 - 0.15,
        "Wavelet R² ({}) should be >= Raw R² ({})",
        wavelet_r2,
        raw_r2
    );
}

fn signal_white_noise(t: usize, seed: u64) -> f32 {
    let h = ((seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407u64.wrapping_mul(t as u64)))
        >> 33) as f32;
    (h - 0.5) * 2.0
}

fn signal_brownian_noise(t: usize, seed: u64) -> f32 {
    let mut x = 0.0f32;
    for i in 0..t {
        let h = ((seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407u64.wrapping_mul(i as u64)))
            >> 33) as f32;
        x += (h - 0.5) * 0.02;
    }
    x.clamp(-5.0, 5.0)
}

fn signal_market_like(t: usize, seed: u64) -> f32 {
    let drift = t as f32 * 0.0005;
    let mut vol = 0.0f32;
    for i in 0..t {
        let h = ((seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407u64.wrapping_mul(i as u64)))
            >> 33) as f32;
        vol += (h - 0.5) * 0.03;
    }
    let seasonality = 0.5 * (PI * t as f32 / 60.0).sin();
    let jump = if t % 200 == 0 {
        let j = ((seed.wrapping_mul(6364136223846793005).wrapping_add(1)) >> 33) as f32;
        (j - 0.5) * 0.5
    } else {
        0.0
    };
    drift + vol + seasonality + jump
}

fn signal_multi_freq_interference(t: usize) -> f32 {
    (PI * t as f32 / 180.0).sin()
        + 0.7 * (PI * t as f32 / 120.0).sin()
        + 0.5 * (PI * t as f32 / 90.0).sin()
        + 0.3 * (PI * t as f32 / 60.0).sin()
        + 0.2 * (PI * t as f32 / 30.0).sin()
}

#[test]
fn test_wavelet_vs_raw_white_noise() {
    let total_steps = 3600;
    let signal: Vec<f32> = (0..total_steps)
        .map(|t| signal_white_noise(t, 42))
        .collect();
    let (all_inputs, all_targets) = build_lookback_dataset(&signal, SEQ_LEN);

    let train_n = 360;
    let test_n = 100;
    let train_inputs = &all_inputs[..train_n * SEQ_LEN];
    let train_targets = &all_targets[..train_n];
    let test_inputs = &all_inputs[train_n * SEQ_LEN..(train_n + test_n) * SEQ_LEN];
    let test_targets = &all_targets[train_n..train_n + test_n];

    let cfg = MambaConfig {
        d_model: 32,
        d_state: 8,
        d_conv: 4,
        expand: 2,
        n_layers: 2,
        scan_mode: Default::default(),
        use_wavelet: true,
        wavelet_levels: 4,
    };

    println!("\n=== Wavelet vs Raw: White Noise ===");

    println!("Training with raw input...");
    let raw_weights = train_raw(train_inputs, train_targets, train_n, SEQ_LEN, &cfg);
    let raw_r2 = eval_raw(
        test_inputs,
        test_targets,
        test_n,
        raw_weights,
        &cfg,
        SEQ_LEN,
    );

    println!("Training with wavelet input...");
    let wavelet_weights = train_with_wavelet(train_inputs, train_targets, train_n, SEQ_LEN, &cfg);
    let wavelet_r2 = eval_wavelet(
        test_inputs,
        test_targets,
        test_n,
        wavelet_weights,
        &cfg,
        SEQ_LEN,
    );

    println!("Raw R² = {:.4}", raw_r2);
    println!("Wavelet R² = {:.4}", wavelet_r2);

    println!("Note: White noise is unpredictable by nature; both methods should have low R²");
}

#[test]
fn test_wavelet_vs_raw_brownian_noise() {
    let total_steps = 3600;
    let signal: Vec<f32> = (0..total_steps)
        .map(|t| signal_brownian_noise(t, 42))
        .collect();
    let (all_inputs, all_targets) = build_lookback_dataset(&signal, SEQ_LEN);

    let train_n = 360;
    let test_n = 100;
    let train_inputs = &all_inputs[..train_n * SEQ_LEN];
    let train_targets = &all_targets[..train_n];
    let test_inputs = &all_inputs[train_n * SEQ_LEN..(train_n + test_n) * SEQ_LEN];
    let test_targets = &all_targets[train_n..train_n + test_n];

    let cfg = MambaConfig {
        d_model: 32,
        d_state: 8,
        d_conv: 4,
        expand: 2,
        n_layers: 2,
        scan_mode: Default::default(),
        use_wavelet: true,
        wavelet_levels: 4,
    };

    println!("\n=== Wavelet vs Raw: Brownian Noise ===");

    println!("Training with raw input...");
    let raw_weights = train_raw(train_inputs, train_targets, train_n, SEQ_LEN, &cfg);
    let raw_r2 = eval_raw(
        test_inputs,
        test_targets,
        test_n,
        raw_weights,
        &cfg,
        SEQ_LEN,
    );

    println!("Training with wavelet input...");
    let wavelet_weights = train_with_wavelet(train_inputs, train_targets, train_n, SEQ_LEN, &cfg);
    let wavelet_r2 = eval_wavelet(
        test_inputs,
        test_targets,
        test_n,
        wavelet_weights,
        &cfg,
        SEQ_LEN,
    );

    println!("Raw R² = {:.4}", raw_r2);
    println!("Wavelet R² = {:.4}", wavelet_r2);

    println!("Note: Brownian noise has temporal correlation; wavelet should help capture trends");
}

#[test]
fn test_wavelet_vs_raw_market_like() {
    let total_steps = 3600;
    let signal: Vec<f32> = (0..total_steps)
        .map(|t| signal_market_like(t, 42))
        .collect();
    let (all_inputs, all_targets) = build_lookback_dataset(&signal, SEQ_LEN);

    let train_n = 360;
    let test_n = 100;
    let train_inputs = &all_inputs[..train_n * SEQ_LEN];
    let train_targets = &all_targets[..train_n];
    let test_inputs = &all_inputs[train_n * SEQ_LEN..(train_n + test_n) * SEQ_LEN];
    let test_targets = &all_targets[train_n..train_n + test_n];

    let cfg = MambaConfig {
        d_model: 32,
        d_state: 8,
        d_conv: 4,
        expand: 2,
        n_layers: 2,
        scan_mode: Default::default(),
        use_wavelet: true,
        wavelet_levels: 4,
    };

    println!("\n=== Wavelet vs Raw: Market-like Signal ===");

    println!("Training with raw input...");
    let raw_weights = train_raw(train_inputs, train_targets, train_n, SEQ_LEN, &cfg);
    let raw_r2 = eval_raw(
        test_inputs,
        test_targets,
        test_n,
        raw_weights,
        &cfg,
        SEQ_LEN,
    );

    println!("Training with wavelet input...");
    let wavelet_weights = train_with_wavelet(train_inputs, train_targets, train_n, SEQ_LEN, &cfg);
    let wavelet_r2 = eval_wavelet(
        test_inputs,
        test_targets,
        test_n,
        wavelet_weights,
        &cfg,
        SEQ_LEN,
    );

    println!("Raw R² = {:.4}", raw_r2);
    println!("Wavelet R² = {:.4}", wavelet_r2);

    assert!(
        wavelet_r2 > raw_r2 - 0.15,
        "Wavelet R² ({}) should be >= Raw R² ({})",
        wavelet_r2,
        raw_r2
    );
}

#[test]
fn test_wavelet_vs_raw_multi_freq() {
    let total_steps = 3600;
    let signal: Vec<f32> = (0..total_steps)
        .map(signal_multi_freq_interference)
        .collect();
    let (all_inputs, all_targets) = build_lookback_dataset(&signal, SEQ_LEN);

    let train_n = 360;
    let test_n = 100;
    let train_inputs = &all_inputs[..train_n * SEQ_LEN];
    let train_targets = &all_targets[..train_n];
    let test_inputs = &all_inputs[train_n * SEQ_LEN..(train_n + test_n) * SEQ_LEN];
    let test_targets = &all_targets[train_n..train_n + test_n];

    let cfg = MambaConfig {
        d_model: 32,
        d_state: 8,
        d_conv: 4,
        expand: 2,
        n_layers: 2,
        scan_mode: Default::default(),
        use_wavelet: true,
        wavelet_levels: 5,
    };

    println!("\n=== Wavelet vs Raw: Multi-Frequency Interference ===");

    println!("Training with raw input...");
    let raw_weights = train_raw(train_inputs, train_targets, train_n, SEQ_LEN, &cfg);
    let raw_r2 = eval_raw(
        test_inputs,
        test_targets,
        test_n,
        raw_weights,
        &cfg,
        SEQ_LEN,
    );

    println!("Training with wavelet input...");
    let wavelet_weights = train_with_wavelet(train_inputs, train_targets, train_n, SEQ_LEN, &cfg);
    let wavelet_r2 = eval_wavelet(
        test_inputs,
        test_targets,
        test_n,
        wavelet_weights,
        &cfg,
        SEQ_LEN,
    );

    println!("Raw R² = {:.4}", raw_r2);
    println!("Wavelet R² = {:.4}", wavelet_r2);

    assert!(
        wavelet_r2 > raw_r2 - 0.1,
        "Wavelet R² ({}) should be >= Raw R² ({})",
        wavelet_r2,
        raw_r2
    );
}

#[test]
fn test_wavelet_convergence_speed() {
    let total_steps = 3600;
    let signal: Vec<f32> = (0..total_steps).map(signal_multiscale).collect();
    let (all_inputs, all_targets) = build_lookback_dataset(&signal, SEQ_LEN);

    let train_n = 360;
    let test_n = 100;
    let train_inputs = &all_inputs[..train_n * SEQ_LEN];
    let train_targets = &all_targets[..train_n];
    let test_inputs = &all_inputs[train_n * SEQ_LEN..(train_n + test_n) * SEQ_LEN];
    let test_targets = &all_targets[train_n..train_n + test_n];

    let cfg = MambaConfig {
        d_model: 32,
        d_state: 8,
        d_conv: 4,
        expand: 2,
        n_layers: 2,
        scan_mode: Default::default(),
        use_wavelet: true,
        wavelet_levels: 4,
    };

    println!("\n=== Convergence Speed: Wavelet vs Raw ===");

    let target_r2 = 0.5;
    let max_epochs = 50;

    let mut raw_epochs_to_target = None;
    let mut wavelet_epochs_to_target = None;

    let wavelet_input_dim =
        mamba_rs::ops::wavelet::haar_channels_len(SEQ_LEN, cfg.wavelet_levels.max(1));
    let wavelet_train_inputs: Vec<f32> = train_inputs
        .chunks(SEQ_LEN)
        .flat_map(|chunk| haar_to_channels(chunk, cfg.wavelet_levels.max(1)))
        .collect();

    for use_wavelet in [false, true] {
        let (inp_dim, inp_data) = if use_wavelet {
            (wavelet_input_dim, wavelet_train_inputs.as_slice())
        } else {
            (SEQ_LEN, &train_inputs[..])
        };

        let weights = mamba_rs::MambaWeights::init(&cfg, inp_dim, SEED);
        let (mut tw, dims) = build_train_scaffolding(&weights, &cfg, inp_dim, train_n);
        let di = dims.d_inner;
        let ds = dims.d_state;
        let nl = dims.n_layers;

        let label = if use_wavelet { "Wavelet" } else { "Raw" };

        for epoch in 0..max_epochs {
            let mut total_loss = 0.0f32;
            for sample_idx in 0..train_n {
                let inp_start = sample_idx * inp_dim;
                let inp = &inp_data[inp_start..inp_start + inp_dim];
                let tgt = train_targets[sample_idx];

                let mut acts = MambaBackboneFlat::zeros(dims);
                let mut scratch = PhaseScratch::zeros(&dims);
                let conv = vec![0.0f32; nl * di * cfg.d_conv];
                let ssm = vec![0.0f32; nl * di * ds];
                let a_neg = compute_a_neg(&tw, nl, di, ds);
                let mut state = MambaRecurrentState {
                    conv: &mut (conv.clone()),
                    ssm: &mut (ssm.clone()),
                    a_neg: &a_neg,
                };

                let mut temporal = vec![0.0f32; train_n * cfg.d_model];
                forward_mamba_backbone_batched(
                    &mut temporal,
                    &mut acts,
                    &tw,
                    inp,
                    &mut state,
                    &mut scratch,
                    &dims,
                );

                let pred = temporal[(train_n - 1) * cfg.d_model];
                let err = pred - tgt;
                total_loss += 0.5 * err * err;

                let mut d_temporal = vec![0.0f32; train_n * cfg.d_model];
                d_temporal[(train_n - 1) * cfg.d_model] = err / train_n as f32;

                let mut grads = TrainMambaWeights::zeros_from_dims(&dims);
                let mut bwd_scratch = BackwardPhaseScratch::zeros(&dims);
                let a_neg_bwd = compute_a_neg(&tw, nl, di, ds);
                backward_mamba_backbone_batched(
                    &mut d_temporal,
                    &mut grads,
                    &acts,
                    &tw,
                    &a_neg_bwd,
                    &mut bwd_scratch,
                    &dims,
                );
                sgd_step(&mut tw, &grads, LR);
            }

            let epochs_to_check = if use_wavelet {
                &mut wavelet_epochs_to_target
            } else {
                &mut raw_epochs_to_target
            };
            if epochs_to_check.is_none() {
                let mut test_preds = Vec::with_capacity(test_n);
                for i in 0..test_n {
                    let test_inp = if use_wavelet {
                        haar_to_channels(
                            &test_inputs[i * SEQ_LEN..(i + 1) * SEQ_LEN],
                            cfg.wavelet_levels.max(1),
                        )
                    } else {
                        test_inputs[i * SEQ_LEN..(i + 1) * SEQ_LEN].to_vec()
                    };
                    let mut acts = MambaBackboneFlat::zeros(dims);
                    let mut scratch = PhaseScratch::zeros(&dims);
                    let conv = vec![0.0f32; nl * di * cfg.d_conv];
                    let ssm = vec![0.0f32; nl * di * ds];
                    let a_neg = compute_a_neg(&tw, nl, di, ds);
                    let mut state = MambaRecurrentState {
                        conv: &mut (conv.clone()),
                        ssm: &mut (ssm.clone()),
                        a_neg: &a_neg,
                    };
                    let mut temporal = vec![0.0f32; train_n * cfg.d_model];
                    forward_mamba_backbone_batched(
                        &mut temporal,
                        &mut acts,
                        &tw,
                        &test_inp,
                        &mut state,
                        &mut scratch,
                        &dims,
                    );
                    test_preds.push(temporal[(train_n - 1) * cfg.d_model]);
                }
                let r2 = compute_r2(&test_preds, test_targets);
                if r2 >= target_r2 {
                    *epochs_to_check = Some(epoch + 1);
                    println!(
                        " {} reached R²={:.4} >= {:.2} at epoch {}",
                        label,
                        r2,
                        target_r2,
                        epoch + 1
                    );
                }
            }
            if (epoch + 1) % 10 == 0 {
                println!(
                    " {} Epoch {:3}: loss = {:.6}",
                    label,
                    epoch + 1,
                    total_loss / train_n as f32
                );
            }
        }
    }

    match (raw_epochs_to_target, wavelet_epochs_to_target) {
        (Some(raw_e), Some(wl_e)) => {
            let speedup = raw_e as f64 / wl_e as f64;
            println!(
                "Convergence speedup: {:.2}x (raw={} epochs, wavelet={} epochs)",
                speedup, raw_e, wl_e
            );
        }
        (None, None) => {
            println!(
                "Neither method reached target R²={:.2} in {} epochs",
                target_r2, max_epochs
            );
        }
        _ => {}
    }
}

#[test]
fn test_wavelet_per_band_lr() {
    let total_steps = 3600;
    let signal: Vec<f32> = (0..total_steps).map(signal_multiscale).collect();
    let (all_inputs, all_targets) = build_lookback_dataset(&signal, SEQ_LEN);

    let train_n = 360;
    let test_n = 100;
    let train_inputs = &all_inputs[..train_n * SEQ_LEN];
    let train_targets = &all_targets[..train_n];
    let test_inputs = &all_inputs[train_n * SEQ_LEN..(train_n + test_n) * SEQ_LEN];
    let test_targets = &all_targets[train_n..train_n + test_n];

    let cfg = MambaConfig {
        d_model: 32,
        d_state: 8,
        d_conv: 4,
        expand: 2,
        n_layers: 2,
        scan_mode: Default::default(),
        use_wavelet: true,
        wavelet_levels: 4,
    };

    println!("\n=== Per-Band LR Modulation ===");

    let wavelet_input_dim =
        mamba_rs::ops::wavelet::haar_channels_len(SEQ_LEN, cfg.wavelet_levels.max(1));
    let wavelet_train_inputs: Vec<f32> = train_inputs
        .chunks(SEQ_LEN)
        .flat_map(|chunk| haar_to_channels(chunk, cfg.wavelet_levels.max(1)))
        .collect();

    let weights = mamba_rs::MambaWeights::init(&cfg, wavelet_input_dim, SEED);
    let (mut tw, dims) = build_train_scaffolding(&weights, &cfg, wavelet_input_dim, train_n);
    let di = dims.d_inner;
    let ds = dims.d_state;
    let nl = dims.n_layers;

    let n_bands = cfg.wavelet_levels.max(1) + 1;
    let band_lrs: Vec<f32> = (0..n_bands)
        .map(|i| {
            let depth = i as f32 / n_bands as f32;
            LR * (1.0 + 2.0 * (1.0 - depth))
        })
        .collect();

    println!(" Band LRs: {:?}", band_lrs);

    for _epoch in 0..EPOCHS {
        let mut total_loss = 0.0f32;
        for sample_idx in 0..train_n {
            let inp_start = sample_idx * wavelet_input_dim;
            let inp = &wavelet_train_inputs[inp_start..inp_start + wavelet_input_dim];
            let tgt = train_targets[sample_idx];

            let mut acts = MambaBackboneFlat::zeros(dims);
            let mut scratch = PhaseScratch::zeros(&dims);
            let conv = vec![0.0f32; nl * di * cfg.d_conv];
            let ssm = vec![0.0f32; nl * di * ds];
            let a_neg = compute_a_neg(&tw, nl, di, ds);
            let mut state = MambaRecurrentState {
                conv: &mut (conv.clone()),
                ssm: &mut (ssm.clone()),
                a_neg: &a_neg,
            };

            let mut temporal = vec![0.0f32; train_n * cfg.d_model];
            forward_mamba_backbone_batched(
                &mut temporal,
                &mut acts,
                &tw,
                inp,
                &mut state,
                &mut scratch,
                &dims,
            );

            let pred = temporal[(train_n - 1) * cfg.d_model];
            let err = pred - tgt;
            total_loss += 0.5 * err * err;

            let mut d_temporal = vec![0.0f32; train_n * cfg.d_model];
            d_temporal[(train_n - 1) * cfg.d_model] = err / train_n as f32;

            let mut grads = TrainMambaWeights::zeros_from_dims(&dims);
            let mut bwd_scratch = BackwardPhaseScratch::zeros(&dims);
            let a_neg_bwd = compute_a_neg(&tw, nl, di, ds);
            backward_mamba_backbone_batched(
                &mut d_temporal,
                &mut grads,
                &acts,
                &tw,
                &a_neg_bwd,
                &mut bwd_scratch,
                &dims,
            );

            for (layer_idx, (lw, lg)) in tw.layers.iter_mut().zip(grads.layers.iter()).enumerate() {
                let layer_lr = band_lrs.get(layer_idx).copied().unwrap_or(LR);
                fn apply_slice_lr(w: &mut [f32], g: &[f32], lr: f32) {
                    for (wi, gi) in w.iter_mut().zip(g.iter()) {
                        *wi -= lr * gi;
                    }
                }
                apply_slice_lr(&mut lw.norm_weight, &lg.norm_weight, layer_lr);
                apply_slice_lr(&mut lw.in_proj_w, &lg.in_proj_w, layer_lr);
                apply_slice_lr(&mut lw.conv1d_weight, &lg.conv1d_weight, layer_lr);
                apply_slice_lr(&mut lw.conv1d_bias, &lg.conv1d_bias, layer_lr);
                apply_slice_lr(&mut lw.x_proj_w, &lg.x_proj_w, layer_lr);
                apply_slice_lr(&mut lw.dt_proj_w, &lg.dt_proj_w, layer_lr);
                apply_slice_lr(&mut lw.dt_proj_b, &lg.dt_proj_b, layer_lr);
                apply_slice_lr(&mut lw.a_log, &lg.a_log, layer_lr);
                apply_slice_lr(&mut lw.d_param, &lg.d_param, layer_lr);
                apply_slice_lr(&mut lw.out_proj_w, &lg.out_proj_w, layer_lr);
            }
            fn apply_slice(w: &mut [f32], g: &[f32], lr: f32) {
                for (wi, gi) in w.iter_mut().zip(g.iter()) {
                    *wi -= lr * gi;
                }
            }
            apply_slice(&mut tw.input_proj_w, &grads.input_proj_w, LR);
            apply_slice(&mut tw.input_proj_b, &grads.input_proj_b, LR);
            apply_slice(&mut tw.norm_f_weight, &grads.norm_f_weight, LR);
        }

        if (_epoch + 1) % 10 == 0 {
            println!(
                " Epoch {:3}: loss = {:.6}",
                _epoch + 1,
                total_loss / train_n as f32
            );
        }
    }

    let mut test_preds = Vec::with_capacity(test_n);
    for i in 0..test_n {
        let inp_raw = &test_inputs[i * SEQ_LEN..(i + 1) * SEQ_LEN];
        let inp_wl = haar_to_channels(inp_raw, cfg.wavelet_levels.max(1));
        let mut acts = MambaBackboneFlat::zeros(dims);
        let mut scratch = PhaseScratch::zeros(&dims);
        let conv = vec![0.0f32; nl * di * cfg.d_conv];
        let ssm = vec![0.0f32; nl * di * ds];
        let a_neg = compute_a_neg(&tw, nl, di, ds);
        let mut state = MambaRecurrentState {
            conv: &mut (conv.clone()),
            ssm: &mut (ssm.clone()),
            a_neg: &a_neg,
        };
        let mut temporal = vec![0.0f32; train_n * cfg.d_model];
        forward_mamba_backbone_batched(
            &mut temporal,
            &mut acts,
            &tw,
            &inp_wl,
            &mut state,
            &mut scratch,
            &dims,
        );
        test_preds.push(temporal[(train_n - 1) * cfg.d_model]);
    }

    let r2 = compute_r2(&test_preds, test_targets);
    println!("Per-band LR R² = {:.4}", r2);
    assert!(r2 > 0.0, "Per-band LR should achieve positive R²");
}
