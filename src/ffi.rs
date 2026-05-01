//! C FFI interface for mamba-rs.
//!
//! Provides a C-compatible API for use from Delphi, C++, Python (ctypes), etc.
//! All functions use `extern "C"` calling convention with flat `f32` buffers.
//!
//! Enable with `--features ffi`. Build as cdylib:
//! ```toml
//! [lib]
//! crate-type = ["cdylib", "rlib"]
//! ```

use crate::config::MambaConfig;
use crate::module::MambaBackbone;
use crate::ops::wavelet;
use std::ffi::c_void;
use std::slice;

use std::f32::consts::PI;

fn lcg64(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn lcg_f32(state: &mut u64) -> f32 {
    (lcg64(state) >> 33) as f32
}

#[unsafe(no_mangle)]
pub extern "C" fn signal_generate_multiscale(out: *mut f32, len: usize) {
    assert!(!out.is_null());
    let buf = unsafe { slice::from_raw_parts_mut(out, len) };
    for (t, v) in buf.iter_mut().enumerate() {
        *v = (PI * t as f32 / 180.0).sin()
            + 0.3 * (PI * t as f32 / 45.0).sin()
            + 0.1 * (t as f32 * 0.07).sin();
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn signal_generate_regime_change(out: *mut f32, len: usize) {
    assert!(!out.is_null());
    let buf = unsafe { slice::from_raw_parts_mut(out, len) };
    for (t, v) in buf.iter_mut().enumerate() {
        *v = if t < 180 {
            (PI * t as f32 / 90.0).sin()
        } else {
            (PI * (t - 180) as f32 / 45.0).cos() + 1.5
        };
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn signal_generate_market_like(out: *mut f32, len: usize, seed: u64) {
    assert!(!out.is_null());
    let buf = unsafe { slice::from_raw_parts_mut(out, len) };
    let mut rng = seed;
    let mut vol = 0.0f32;
    for (t, v) in buf.iter_mut().enumerate() {
        let drift = t as f32 * 0.0005;
        vol += (lcg_f32(&mut rng) - 0.5) * 0.03;
        let seasonality = 0.5 * (PI * t as f32 / 60.0).sin();
        let jump = if t % 200 == 0 {
            let j = lcg_f32(&mut rng);
            (j - 0.5) * 0.5
        } else {
            0.0
        };
        *v = drift + vol + seasonality + jump;
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn signal_generate_white_noise(out: *mut f32, len: usize, seed: u64) {
    assert!(!out.is_null());
    let buf = unsafe { slice::from_raw_parts_mut(out, len) };
    let mut rng = seed;
    for v in buf.iter_mut() {
        *v = (lcg_f32(&mut rng) - 0.5) * 2.0;
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn signal_generate_brownian_noise(out: *mut f32, len: usize, seed: u64) {
    assert!(!out.is_null());
    let buf = unsafe { slice::from_raw_parts_mut(out, len) };
    let mut rng = seed;
    let mut x = 0.0f32;
    for v in buf.iter_mut() {
        x += (lcg_f32(&mut rng) - 0.5) * 0.02;
        x = x.clamp(-5.0, 5.0);
        *v = x;
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn signal_generate_multi_freq(out: *mut f32, len: usize) {
    assert!(!out.is_null());
    let buf = unsafe { slice::from_raw_parts_mut(out, len) };
    for (t, v) in buf.iter_mut().enumerate() {
        *v = (PI * t as f32 / 180.0).sin()
            + 0.7 * (PI * t as f32 / 120.0).sin()
            + 0.5 * (PI * t as f32 / 90.0).sin()
            + 0.3 * (PI * t as f32 / 60.0).sin()
            + 0.2 * (PI * t as f32 / 30.0).sin();
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn haar_auto_levels(seq_len: usize) -> usize {
    wavelet::haar_levels(seq_len)
}

#[unsafe(no_mangle)]
pub extern "C" fn haar_channels_len(seq_len: usize, levels: usize) -> usize {
    wavelet::haar_channels_len(seq_len, levels)
}

#[unsafe(no_mangle)]
pub extern "C" fn haar_decompose_to_channels(
    signal: *const f32,
    seq_len: usize,
    levels: usize,
    out: *mut f32,
) {
    assert!(!signal.is_null());
    assert!(!out.is_null());
    let sig = unsafe { slice::from_raw_parts(signal, seq_len) };
    let buf =
        unsafe { slice::from_raw_parts_mut(out, wavelet::haar_channels_len(seq_len, levels)) };
    buf.copy_from_slice(&wavelet::haar_to_channels(sig, levels));
}

#[unsafe(no_mangle)]
pub extern "C" fn haar_reconstruct_from_channels(
    channels: *const f32,
    channels_len: usize,
    original_len: usize,
    levels: usize,
    out: *mut f32,
) {
    assert!(!channels.is_null());
    assert!(!out.is_null());
    let ch = unsafe { slice::from_raw_parts(channels, channels_len) };
    let buf = unsafe { slice::from_raw_parts_mut(out, original_len) };
    buf.copy_from_slice(&wavelet::haar_from_channels(ch, original_len, levels));
}

pub struct FfiBackbone {
    backbone: MambaBackbone,
}

#[repr(C)]
pub struct FfiMambaConfig {
    pub d_model: usize,
    pub d_state: usize,
    pub d_conv: usize,
    pub expand: usize,
    pub n_layers: usize,
    pub use_wavelet: bool,
    pub wavelet_levels: usize,
}

impl From<&FfiMambaConfig> for MambaConfig {
    fn from(c: &FfiMambaConfig) -> Self {
        MambaConfig {
            d_model: c.d_model,
            d_state: c.d_state,
            d_conv: c.d_conv,
            expand: c.expand,
            n_layers: c.n_layers,
            use_wavelet: c.use_wavelet,
            wavelet_levels: c.wavelet_levels,
            scan_mode: Default::default(),
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn mamba_backbone_new(
    cfg: *const FfiMambaConfig,
    input_dim: usize,
    _seq_len: usize,
    seed: u64,
) -> *mut c_void {
    assert!(!cfg.is_null());
    let config = unsafe { &*cfg };
    let mamba_cfg = MambaConfig::from(config);
    let backbone = MambaBackbone::init(mamba_cfg, input_dim, seed);
    Box::into_raw(Box::new(FfiBackbone { backbone })) as *mut c_void
}

#[unsafe(no_mangle)]
pub extern "C" fn mamba_backbone_free(handle: *mut c_void) {
    if !handle.is_null() {
        let _ = unsafe { Box::from_raw(handle as *mut FfiBackbone) };
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn mamba_backbone_forward_step(
    handle: *mut c_void,
    input: *const f32,
    input_len: usize,
    output: *mut f32,
) -> bool {
    assert!(!handle.is_null());
    assert!(!input.is_null());
    assert!(!output.is_null());
    let ffi = unsafe { &*(handle as *const FfiBackbone) };
    let inp = unsafe { slice::from_raw_parts(input, input_len) };
    let out = unsafe { slice::from_raw_parts_mut(output, ffi.backbone.config().d_model) };
    let mut state = ffi.backbone.alloc_state();
    let mut scratch = ffi.backbone.alloc_scratch();
    ffi.backbone
        .forward_step(inp, out, &mut state, &mut scratch);
    out.iter().all(|v| v.is_finite())
}

#[unsafe(no_mangle)]
pub extern "C" fn mamba_backbone_forward_sequence(
    handle: *mut c_void,
    input: *const f32,
    seq_len: usize,
    input_dim: usize,
    output: *mut f32,
) -> bool {
    assert!(!handle.is_null());
    assert!(!input.is_null());
    assert!(!output.is_null());
    let ffi = unsafe { &*(handle as *const FfiBackbone) };
    let cfg = ffi.backbone.config();
    let dm = cfg.d_model;
    let inp = unsafe { slice::from_raw_parts(input, seq_len * input_dim) };
    let out = unsafe { slice::from_raw_parts_mut(output, seq_len * dm) };
    let mut state = ffi.backbone.alloc_state();
    let mut scratch = ffi.backbone.alloc_scratch();
    ffi.backbone
        .forward_sequence(inp, out, &mut state, &mut scratch, seq_len);
    out.iter().all(|v| v.is_finite())
}

#[unsafe(no_mangle)]
pub extern "C" fn mamba_backbone_d_model(handle: *mut c_void) -> usize {
    assert!(!handle.is_null());
    let ffi = unsafe { &*(handle as *const FfiBackbone) };
    ffi.backbone.config().d_model
}

#[unsafe(no_mangle)]
pub extern "C" fn mamba_backbone_input_dim(handle: *mut c_void) -> usize {
    assert!(!handle.is_null());
    let ffi = unsafe { &*(handle as *const FfiBackbone) };
    ffi.backbone.input_dim()
}
