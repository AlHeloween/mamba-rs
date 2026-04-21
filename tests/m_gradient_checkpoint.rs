use mamba_rs::MambaConfig;
use mamba_rs::ops::dims::MambaDims;
use mamba_rs::train::checkpoint::CheckpointState;

#[test]
fn test_checkpoint_state_allocation() {
    let cfg = MambaConfig::default();
    let dims = MambaDims::from_config(&cfg, 64, cfg.d_model);
    let ckpt = CheckpointState::new(&dims, 16);

    assert_eq!(ckpt.seq_len, 64);
    assert_eq!(ckpt.checkpoint_interval, 16);
    assert_eq!(ckpt.checkpoint_indices, vec![0, 16, 32, 48]);
    assert_eq!(ckpt.checkpoint_acts.len(), 4);

    // Check memory sizes
    let di = cfg.d_inner();
    let ds = cfg.d_state;
    let dc = cfg.d_conv;
    let dm = cfg.d_model;
    assert_eq!(ckpt.ssm_state.len(), 64 * di * ds);
    assert_eq!(ckpt.conv_state.len(), 64 * di * dc);
    assert_eq!(ckpt.residual.len(), 64 * dm);
}

#[test]
fn test_checkpoint_predicates() {
    let cfg = MambaConfig::default();
    let dims = MambaDims::from_config(&cfg, 64, cfg.d_model);
    let ckpt = CheckpointState::new(&dims, 16);

    assert!(ckpt.is_checkpoint(0));
    assert!(!ckpt.is_checkpoint(1));
    assert!(ckpt.is_checkpoint(16));
    assert!(!ckpt.is_checkpoint(17));
    assert!(ckpt.is_checkpoint(32));
    assert!(ckpt.is_checkpoint(48));
    assert!(!ckpt.is_checkpoint(63));

    assert_eq!(ckpt.checkpoint_index(0), Some(0));
    assert_eq!(ckpt.checkpoint_index(16), Some(1));
    assert_eq!(ckpt.checkpoint_index(32), Some(2));
    assert_eq!(ckpt.checkpoint_index(48), Some(3));
    assert_eq!(ckpt.checkpoint_index(1), None);

    assert_eq!(ckpt.prev_checkpoint(0), (0, 0));
    assert_eq!(ckpt.prev_checkpoint(1), (0, 0));
    assert_eq!(ckpt.prev_checkpoint(15), (0, 0));
    assert_eq!(ckpt.prev_checkpoint(16), (16, 1));
    assert_eq!(ckpt.prev_checkpoint(17), (16, 1));
    assert_eq!(ckpt.prev_checkpoint(63), (48, 3));
}
