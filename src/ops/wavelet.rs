//! Haar wavelet transform for multi-scale signal decomposition.
//!
//! Decomposes input signals into approximation (low-frequency) and detail
//! (high-frequency) coefficients at multiple dyadic scales. The transform
//! is orthogonal, so reconstruction is perfect and gradients flow through
//! without any Jacobian correction.
//!
//! Used as input preprocessing for Mamba training: the raw signal is
//! decomposed into wavelet bands, each band becomes an input channel,
//! and the SSM naturally learns different decay timescales per band.
//!
//! ## Layout
//!
//! `haar_to_channels` returns a flat buffer of length `padded_len * (levels+1)`
//! where `padded_len = next_power_of_two(signal.len())`. Each band occupies
//! `padded_len` elements (bands are naturally shorter but stored at full
//! padded length for simple indexing).
//!
//! `haar_from_channels` reverses this, reconstructing the original signal.

fn next_power_of_two(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    let mut v = n - 1;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v + 1
}

pub fn haar_levels(seq_len: usize) -> usize {
    if seq_len <= 1 {
        return 0;
    }
    let max_levels = (seq_len as f32).log2().floor() as usize;
    max_levels.min(6)
}

pub fn haar_decompose_full(signal: &[f32], levels: usize) -> Vec<Vec<f32>> {
    if signal.is_empty() || levels == 0 {
        return vec![signal.to_vec()];
    }

    let padded_len = next_power_of_two(signal.len());
    let mut buf = vec![0.0f32; padded_len];
    buf[..signal.len()].copy_from_slice(signal);

    let scale = (2.0f32).sqrt();
    let mut details: Vec<Vec<f32>> = Vec::with_capacity(levels + 1);
    let mut current_len = padded_len;

    for _ in 0..levels {
        if current_len < 2 {
            break;
        }
        let half = current_len / 2;
        let mut temp = vec![0.0f32; current_len];
        for i in 0..half {
            temp[i] = (buf[2 * i] + buf[2 * i + 1]) / scale;
            temp[half + i] = (buf[2 * i] - buf[2 * i + 1]) / scale;
        }
        buf[..current_len].copy_from_slice(&temp);
        details.push(buf[half..current_len].to_vec());
        current_len = half;
    }

    details.push(buf[..current_len].to_vec());
    details.reverse();
    details
}

pub fn haar_reconstruct_full(level_buffers: &[Vec<f32>], original_len: usize) -> Vec<f32> {
    if level_buffers.is_empty() {
        return vec![];
    }

    let padded_len = next_power_of_two(original_len);
    let mut buf = vec![0.0f32; padded_len];

    let scale = (2.0f32).sqrt();
    let approx = &level_buffers[0];
    buf[..approx.len()].copy_from_slice(approx);

    let mut current_len = approx.len();
    for detail_buf in &level_buffers[1..] {
        if current_len * 2 > padded_len {
            break;
        }
        let half = current_len;
        let detail_len = detail_buf.len().min(half);
        for i in 0..detail_len {
            buf[half + i] = detail_buf[i];
        }
        let new_len = half * 2;
        let mut temp = vec![0.0f32; new_len];
        for i in 0..half {
            temp[2 * i] = (buf[i] + buf[half + i]) / scale;
            temp[2 * i + 1] = (buf[i] - buf[half + i]) / scale;
        }
        buf[..new_len].copy_from_slice(&temp);
        current_len = new_len;
    }

    buf[..original_len].to_vec()
}

pub fn haar_to_channels(signal: &[f32], levels: usize) -> Vec<f32> {
    if signal.is_empty() {
        return vec![];
    }

    let actual_levels = levels.min(haar_levels(signal.len()));
    if actual_levels == 0 {
        return signal.to_vec();
    }

    let padded_len = next_power_of_two(signal.len());
    let decomposed = haar_decompose_full(signal, actual_levels);
    let n_channels = decomposed.len();

    let mut channels = vec![0.0f32; padded_len * n_channels];
    for (ch, band) in decomposed.iter().enumerate() {
        channels[ch * padded_len..ch * padded_len + band.len()].copy_from_slice(band);
    }

    channels
}

pub fn haar_from_channels(channels: &[f32], original_len: usize, levels: usize) -> Vec<f32> {
    if channels.is_empty() || original_len == 0 {
        return vec![];
    }

    let padded_len = next_power_of_two(original_len);
    let actual_levels = levels.min(haar_levels(original_len));
    if actual_levels == 0 {
        return channels[..original_len].to_vec();
    }

    let n_channels = channels.len() / padded_len;
    if n_channels == 0 {
        return vec![];
    }

    let mut level_buffers = Vec::with_capacity(n_channels);
    let mut current_len = padded_len >> actual_levels;

    for ch in 0..n_channels {
        let start = ch * padded_len;
        if ch == 0 {
            level_buffers.push(channels[start..start + current_len].to_vec());
        } else {
            level_buffers.push(channels[start..start + current_len].to_vec());
            current_len *= 2;
        }
    }

    haar_reconstruct_full(&level_buffers, original_len)
}

pub fn haar_channels_len(seq_len: usize, levels: usize) -> usize {
    if seq_len == 0 {
        return 0;
    }
    let actual_levels = levels.min(haar_levels(seq_len));
    if actual_levels == 0 {
        return seq_len;
    }
    let padded_len = next_power_of_two(seq_len);
    padded_len * (actual_levels + 1)
}

pub fn haar_per_channel_rmsnorm(channels: &mut [f32], padded_len: usize, _levels: usize) {
    const EPS: f32 = 1e-5;
    if channels.is_empty() || padded_len == 0 {
        return;
    }

    let n_channels = channels.len() / padded_len;
    for ch in 0..n_channels {
        let start = ch * padded_len;
        let end = (ch + 1) * padded_len;
        let end = end.min(channels.len());
        let band: Vec<f32> = channels[start..end].to_vec();
        let mean_sq: f32 = band.iter().map(|v| v * v).sum::<f32>() / band.len() as f32;
        let rms = (mean_sq + EPS).sqrt();
        let inv_rms = 1.0 / rms;
        for v in &mut channels[start..end] {
            *v *= inv_rms;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_haar_levels() {
        assert_eq!(haar_levels(1), 0);
        assert_eq!(haar_levels(2), 1);
        assert_eq!(haar_levels(4), 2);
        assert_eq!(haar_levels(8), 3);
        assert_eq!(haar_levels(360), 6);
        assert_eq!(haar_levels(1000), 6);
    }

    #[test]
    fn test_haar_roundtrip_power_of_two() {
        let signal: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
        let levels = haar_levels(signal.len());
        let decomposed = haar_decompose_full(&signal, levels);
        let reconstructed = haar_reconstruct_full(&decomposed, signal.len());
        for (a, b) in signal.iter().zip(reconstructed.iter()) {
            assert!((a - b).abs() < 1e-5, "a={a}, b={b}");
        }
    }

    #[test]
    fn test_haar_roundtrip_non_power_of_two() {
        let signal: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
        let levels = haar_levels(signal.len());
        let decomposed = haar_decompose_full(&signal, levels);
        let reconstructed = haar_reconstruct_full(&decomposed, signal.len());
        assert_eq!(reconstructed.len(), signal.len());
        for (a, b) in signal.iter().zip(reconstructed.iter()) {
            assert!((a - b).abs() < 1e-4, "a={a}, b={b}");
        }
    }

    #[test]
    fn test_haar_to_channels_roundtrip_pow2() {
        let signal: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();
        let levels = 3;
        let channels = haar_to_channels(&signal, levels);
        let reconstructed = haar_from_channels(&channels, signal.len(), levels);
        assert_eq!(reconstructed.len(), signal.len());
        for (a, b) in signal.iter().zip(reconstructed.iter()) {
            assert!((a - b).abs() < 1e-4, "a={a}, b={b}");
        }
    }

    #[test]
    fn test_haar_to_channels_roundtrip_non_pow2() {
        let signal: Vec<f32> = (0..100).map(|i| (i as f32 * 0.01).sin()).collect();
        let levels = 3;
        let channels = haar_to_channels(&signal, levels);
        let reconstructed = haar_from_channels(&channels, signal.len(), levels);
        assert_eq!(reconstructed.len(), signal.len());
        for (a, b) in signal.iter().zip(reconstructed.iter()) {
            assert!((a - b).abs() < 1e-4, "a={a}, b={b}");
        }
    }

    #[test]
    fn test_haar_to_channels_shape() {
        let signal: Vec<f32> = (0..360).map(|i| (i as f32 * 0.01).sin()).collect();
        let levels = 5;
        let channels = haar_to_channels(&signal, levels);
        let padded = next_power_of_two(360);
        assert_eq!(channels.len(), padded * (levels + 1));
    }

    #[test]
    fn test_haar_per_channel_rmsnorm() {
        let signal: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();
        let levels = 3;
        let padded = next_power_of_two(64);
        let mut channels = haar_to_channels(&signal, levels);
        haar_per_channel_rmsnorm(&mut channels, padded, levels);

        let n_ch = channels.len() / padded;
        for ch in 0..n_ch {
            let start = ch * padded;
            let band = &channels[start..start + padded];
            let mean_sq: f32 = band.iter().map(|v| v * v).sum::<f32>() / band.len() as f32;
            let rms = mean_sq.sqrt();
            assert!(rms > 0.0 && rms <= 1.0 + 1e-3, "channel {ch} rms={rms}");
        }
    }
}
