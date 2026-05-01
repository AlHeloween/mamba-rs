/*
 * mamba_rs.h — C FFI header for mamba-rs
 *
 * Generated from src/ffi.rs
 *
 * Usage:
 *   1. Compile Rust: cargo build --release --features ffi
 *   2. Link against target/release/mamba_rs.dll (Windows) or .so (Linux)
 *
 * All functions use C calling convention (cdecl).
 * All f32 buffers are in row-major (C) order.
 */

#ifndef MAMBA_RS_H
#define MAMBA_RS_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef _WIN32
#define MAMBA_API __declspec(dllimport)
#else
#define MAMBA_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ── Signal generators ────────────────────────────────────────────── */

MAMBA_API void signal_generate_multiscale(float *out, size_t len);
MAMBA_API void signal_generate_regime_change(float *out, size_t len);
MAMBA_API void signal_generate_market_like(float *out, size_t len, uint64_t seed);
MAMBA_API void signal_generate_white_noise(float *out, size_t len, uint64_t seed);
MAMBA_API void signal_generate_brownian_noise(float *out, size_t len, uint64_t seed);
MAMBA_API void signal_generate_multi_freq(float *out, size_t len);

/* ── Wavelet FFI ──────────────────────────────────────────────────── */

MAMBA_API size_t haar_auto_levels(size_t seq_len);
MAMBA_API size_t haar_channels_len(size_t seq_len, size_t levels);

MAMBA_API void haar_decompose_to_channels(
    const float *signal, size_t seq_len, size_t levels,
    float *out);

MAMBA_API void haar_reconstruct_from_channels(
    const float *channels, size_t channels_len,
    size_t original_len, size_t levels,
    float *out);

/* ── Mamba backbone FFI ───────────────────────────────────────────── */

typedef struct FfiMambaConfig {
    size_t d_model;
    size_t d_state;
    size_t d_conv;
    size_t expand;
    size_t n_layers;
    int use_wavelet;      /* 0 = false, 1 = true */
    size_t wavelet_levels;
} FfiMambaConfig;

MAMBA_API void *mamba_backbone_new(
    const FfiMambaConfig *cfg,
    size_t input_dim,
    size_t seq_len,
    uint64_t seed);

MAMBA_API void mamba_backbone_free(void *handle);

MAMBA_API bool mamba_backbone_forward_step(
    void *handle,
    const float *input, size_t input_len,
    float *output);

MAMBA_API bool mamba_backbone_forward_sequence(
    void *handle,
    const float *input, size_t seq_len, size_t input_dim,
    float *output);

MAMBA_API size_t mamba_backbone_d_model(void *handle);
MAMBA_API size_t mamba_backbone_input_dim(void *handle);

#ifdef __cplusplus
}
#endif

#endif /* MAMBA_RS_H */
