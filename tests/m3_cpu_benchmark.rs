use std::time::Instant;

use mamba_rs::mamba3_siso::config::Mamba3Config;
use mamba_rs::mamba3_siso::cpu::backward::backward_mamba3_layer_batched;
use mamba_rs::mamba3_siso::cpu::dims::Mamba3Dims;
use mamba_rs::mamba3_siso::cpu::flat::Mamba3LayerFlat;
use mamba_rs::mamba3_siso::cpu::forward::forward_mamba3_layer_batched;
use mamba_rs::mamba3_siso::cpu::inference::{Mamba3StepScratch, mamba3_step};
use mamba_rs::mamba3_siso::cpu::scratch::Mamba3Scratch;
use mamba_rs::mamba3_siso::cpu::weights::TrainMamba3LayerWeights;
use mamba_rs::mamba3_siso::state::Mamba3State;
use mamba_rs::mamba3_siso::weights::Mamba3Weights;

fn configs() -> [(&'static str, Mamba3Config); 4] {
    [
        (
            "small",
            Mamba3Config {
                d_model: 64,
                d_state: 8,
                expand: 2,
                headdim: 8,
                ngroups: 1,
                n_layers: 2,
                rope_fraction: 0.5,
                a_floor: 0.0625,
                is_outproj_norm: false,
            },
        ),
        ("default", Mamba3Config::default()),
        (
            "medium",
            Mamba3Config {
                d_model: 256,
                d_state: 16,
                expand: 2,
                headdim: 16,
                ngroups: 1,
                n_layers: 4,
                rope_fraction: 0.5,
                a_floor: 0.0625,
                is_outproj_norm: false,
            },
        ),
        (
            "large",
            Mamba3Config {
                d_model: 512,
                d_state: 16,
                expand: 2,
                headdim: 16,
                ngroups: 1,
                n_layers: 6,
                rope_fraction: 0.5,
                a_floor: 0.0625,
                is_outproj_norm: false,
            },
        ),
    ]
}

fn init_layer_w(dims: &Mamba3Dims) -> TrainMamba3LayerWeights {
    let mut w = TrainMamba3LayerWeights::zeros(dims);
    for v in &mut w.norm_weight {
        *v = 1.0;
    }
    for v in &mut w.d_param {
        *v = 1.0;
    }
    for v in &mut w.b_norm_weight {
        *v = 1.0;
    }
    for v in &mut w.c_norm_weight {
        *v = 1.0;
    }
    for (i, v) in w.in_proj_w.iter_mut().enumerate() {
        *v = ((i % 7) as f32 - 3.0) * 0.01;
    }
    for (i, v) in w.out_proj_w.iter_mut().enumerate() {
        *v = ((i % 5) as f32 - 2.0) * 0.01;
    }
    w
}

/// Full Mamba-3 CPU benchmark: inference + training.
///
/// Run: `cargo test --release --test m3_cpu_benchmark -- --ignored --nocapture`
#[test]
#[ignore]
fn m3_cpu_benchmark() {
    // ===================================================================
    // Part 1: Inference (T=1 step)
    // ===================================================================
    println!("mamba-3 SISO inference benchmark (T=1)");
    println!("=======================================");
    println!();

    for (name, cfg) in &configs() {
        cfg.validate();
        let input_dim = cfg.d_model;
        let w = Mamba3Weights::init(cfg, input_dim, 42);
        let mut state = Mamba3State::zeros(cfg);
        let mut scratch = Mamba3StepScratch::new(cfg);
        let mut output = vec![0.0f32; cfg.d_model];
        let input = vec![0.1f32; input_dim];

        // warmup
        for _ in 0..100 {
            mamba3_step(
                &mut output,
                &input,
                &mut scratch,
                &w,
                &mut state.layers,
                cfg,
            );
        }
        state.reset();

        // bench
        let iterations = 10_000;
        let t0 = Instant::now();
        for _ in 0..iterations {
            mamba3_step(
                &mut output,
                &input,
                &mut scratch,
                &w,
                &mut state.layers,
                cfg,
            );
        }
        let us_per_step = t0.elapsed().as_micros() as f64 / iterations as f64;

        let nheads = cfg.nheads();
        let d_inner = cfg.d_inner();
        println!(
            "{name:>8}: d_model={:>3}, layers={}, nheads={:>2}, d_inner={:>4} | {:.1} us/step",
            cfg.d_model, cfg.n_layers, nheads, d_inner, us_per_step,
        );
    }

    println!();

    // ===================================================================
    // Part 2: Training forward + backward (B=1, T=32, single layer)
    // ===================================================================
    let seq_len = 32;
    println!("mamba-3 SISO CPU training benchmark (B=1, T={seq_len}, single layer)");
    println!("=====================================================================");
    println!();

    for (name, cfg) in &configs() {
        cfg.validate();
        let dims = Mamba3Dims::from_config(cfg, seq_len);
        let w = init_layer_w(&dims);

        let nh = dims.nheads;
        let hd = dims.headdim;
        let ds = dims.d_state;
        let na = dims.num_rope_angles.max(1);

        // Pre-allocate (reused every iteration)
        let mut acts = Mamba3LayerFlat::zeros(dims);
        let mut scratch = Mamba3Scratch::zeros(&dims);
        let mut temporal = vec![0.5f32; seq_len * dims.d_model];
        let mut d_temporal = vec![1.0f32; seq_len * dims.d_model];
        let mut d_w = TrainMamba3LayerWeights::zeros(&dims);

        // --- Forward only ---
        let iters = if cfg.d_model >= 512 { 100 } else { 500 };
        for _ in 0..20 {
            temporal.fill(0.5);
            let mut ssm = vec![0.0; nh * hd * ds];
            let mut k = vec![0.0; nh * ds];
            let mut v = vec![0.0; nh * hd];
            let mut a = vec![0.0; nh * na];
            forward_mamba3_layer_batched(
                &mut temporal,
                &mut acts,
                &w,
                &mut ssm,
                &mut k,
                &mut v,
                &mut a,
                &mut scratch,
                &dims,
            );
        }

        let t0 = Instant::now();
        for _ in 0..iters {
            temporal.fill(0.5);
            let mut ssm = vec![0.0; nh * hd * ds];
            let mut k = vec![0.0; nh * ds];
            let mut v = vec![0.0; nh * hd];
            let mut a = vec![0.0; nh * na];
            forward_mamba3_layer_batched(
                &mut temporal,
                &mut acts,
                &w,
                &mut ssm,
                &mut k,
                &mut v,
                &mut a,
                &mut scratch,
                &dims,
            );
        }
        let fwd_us = t0.elapsed().as_micros() as f64 / iters as f64;

        // --- Forward + Backward ---
        let iters = if cfg.d_model >= 512 { 50 } else { 200 };
        let t0 = Instant::now();
        for _ in 0..iters {
            temporal.fill(0.5);
            let mut ssm = vec![0.0; nh * hd * ds];
            let mut k = vec![0.0; nh * ds];
            let mut v = vec![0.0; nh * hd];
            let mut a = vec![0.0; nh * na];
            forward_mamba3_layer_batched(
                &mut temporal,
                &mut acts,
                &w,
                &mut ssm,
                &mut k,
                &mut v,
                &mut a,
                &mut scratch,
                &dims,
            );

            d_temporal.fill(1.0);
            d_w.zero();
            backward_mamba3_layer_batched(
                &mut d_temporal,
                &acts,
                &w,
                &mut d_w,
                &mut scratch,
                &dims,
                None,
            );
        }
        let fwdbwd_us = t0.elapsed().as_micros() as f64 / iters as f64;
        let bwd_us = fwdbwd_us - fwd_us;

        println!(
            "{name:>8}: d_model={:>3}, layers={} | fwd {fwd_us:>8.1} us | bwd {bwd_us:>8.1} us | total {fwdbwd_us:>8.1} us",
            cfg.d_model, cfg.n_layers,
        );
    }
    println!();
}
