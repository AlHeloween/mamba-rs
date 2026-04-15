use mamba_rs::ops::dims::{MambaDims, MambaRecurrentState};
use mamba_rs::train::flat::MambaBackboneFlat;
use mamba_rs::train::forward::forward_mamba_backbone_batched;
use mamba_rs::train::scratch::PhaseScratch;
use mamba_rs::train::truncated::{chunk_bounds, num_chunks, truncated_forward};
use mamba_rs::train::weights::TrainMambaWeights;
use mamba_rs::{MambaConfig, MambaWeights};

fn train_weights_from_inference(w: &MambaWeights) -> TrainMambaWeights {
    TrainMambaWeights {
        input_proj_w: w.input_proj_w.clone(),
        input_proj_b: w.input_proj_b.clone(),
        layers: w
            .layers
            .iter()
            .map(|lw| mamba_rs::train::weights::TrainMambaLayerWeights {
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
        norm_f_weight: w.norm_f_weight.clone(),
    }
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

#[test]
fn test_truncated_forward_matches_standard() {
    let cfg = MambaConfig::default();
    let input_dim = cfg.d_model;
    let seq_len = 64;
    let chunk_size = 16;

    let inf_w = MambaWeights::init(&cfg, input_dim, 42);
    let tw = train_weights_from_inference(&inf_w);
    let dims = MambaDims::from_config(&cfg, seq_len, input_dim);

    let input: Vec<f32> = (0..seq_len * input_dim)
        .map(|i| ((i * 7 + 13) % 100) as f32 * 0.01 - 0.5)
        .collect();

    let mut temporal_std = vec![0.0f32; seq_len * cfg.d_model];
    let mut acts_std = MambaBackboneFlat::zeros(dims);
    let mut scratch_std = PhaseScratch::zeros(&dims);
    let di = cfg.d_inner();
    let ds = cfg.d_state;
    let dc = cfg.d_conv;
    let mut conv_std = vec![0.0f32; cfg.n_layers * di * dc];
    let mut ssm_std = vec![0.0f32; cfg.n_layers * di * ds];
    let a_neg_std = compute_a_neg(&tw, cfg.n_layers, di, ds);
    let mut state_std = MambaRecurrentState {
        conv: &mut conv_std,
        ssm: &mut ssm_std,
        a_neg: &a_neg_std,
    };

    forward_mamba_backbone_batched(
        &mut temporal_std,
        &mut acts_std,
        &tw,
        &input,
        &mut state_std,
        &mut scratch_std,
        &dims,
    );

    let mut temporal_trunc = vec![0.0f32; seq_len * cfg.d_model];
    let mut acts_trunc = MambaBackboneFlat::zeros(dims);
    let mut scratch_trunc = PhaseScratch::zeros(&dims);
    let mut conv_trunc = vec![0.0f32; cfg.n_layers * di * dc];
    let mut ssm_trunc = vec![0.0f32; cfg.n_layers * di * ds];
    let a_neg_trunc = compute_a_neg(&tw, cfg.n_layers, di, ds);
    let mut state_trunc = MambaRecurrentState {
        conv: &mut conv_trunc,
        ssm: &mut ssm_trunc,
        a_neg: &a_neg_trunc,
    };

    truncated_forward(
        &mut temporal_trunc,
        &mut acts_trunc,
        &tw,
        &input,
        &mut state_trunc,
        &mut scratch_trunc,
        &dims,
        chunk_size,
    );

    let tol = 1e-5;
    for (i, (&s, &t)) in temporal_std.iter().zip(temporal_trunc.iter()).enumerate() {
        let diff = (s - t).abs();
        assert!(
            diff <= tol,
            "temporal mismatch at idx {i}: std={s}, trunc={t}, diff={diff}",
        );
    }
}

#[test]
fn test_chunk_bounds_correct() {
    let seq_len = 100;
    let chunk_size = 32;
    let n = num_chunks(seq_len, chunk_size);
    assert_eq!(n, 4);

    assert_eq!(chunk_bounds(0, 32, 100), (0, 32));
    assert_eq!(chunk_bounds(1, 32, 100), (32, 64));
    assert_eq!(chunk_bounds(2, 32, 100), (64, 96));
    assert_eq!(chunk_bounds(3, 32, 100), (96, 100));
}
