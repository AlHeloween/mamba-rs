//! Learning rate schedules for training.
//!
//! Provides common LR scheduling strategies:
//! - [`ConstantLR`]: fixed learning rate
//! - [`LinearWarmup`]: linear warmup from 0 to base LR
//! - [`WarmupCosine`]: linear warmup + cosine decay
//! - [`WarmupLinear`]: linear warmup + linear decay
//! - [`StepDecay`]: step-wise decay (multiply by gamma every K steps)

use std::f32::consts::PI;

/// Learning rate schedule trait.
///
/// All implementations must be `Send + Sync` to support parallel training.
pub trait LRSchedule: Send + Sync + std::fmt::Debug {
    /// Get learning rate for a given training step (0-indexed).
    fn get_lr(&self, step: usize) -> f32;

    /// Get the base (maximum) learning rate.
    fn base_lr(&self) -> f32;
}

/// Constant learning rate (no scheduling).
#[derive(Debug, Clone)]
pub struct ConstantLR {
    pub lr: f32,
}

impl ConstantLR {
    pub fn new(lr: f32) -> Self {
        Self { lr }
    }
}

impl LRSchedule for ConstantLR {
    fn get_lr(&self, _step: usize) -> f32 {
        self.lr
    }

    fn base_lr(&self) -> f32 {
        self.lr
    }
}

/// Linear warmup from 0 to base LR over `warmup_steps`.
#[derive(Debug, Clone)]
pub struct LinearWarmup {
    pub warmup_steps: usize,
    pub base_lr: f32,
}

impl LinearWarmup {
    pub fn new(warmup_steps: usize, base_lr: f32) -> Self {
        Self {
            warmup_steps,
            base_lr,
        }
    }
}

impl LRSchedule for LinearWarmup {
    fn get_lr(&self, step: usize) -> f32 {
        if step < self.warmup_steps {
            self.base_lr * (step + 1) as f32 / self.warmup_steps as f32
        } else {
            self.base_lr
        }
    }

    fn base_lr(&self) -> f32 {
        self.base_lr
    }
}

/// Linear warmup + cosine decay.
///
/// Warmup linearly from 0 to `base_lr` over `warmup_steps`,
/// then decay following a cosine curve to `min_lr` over the
/// remaining `total_steps - warmup_steps`.
#[derive(Debug, Clone)]
pub struct WarmupCosine {
    pub warmup_steps: usize,
    pub total_steps: usize,
    pub base_lr: f32,
    pub min_lr: f32,
}

impl WarmupCosine {
    pub fn new(warmup_steps: usize, total_steps: usize, base_lr: f32, min_lr: f32) -> Self {
        Self {
            warmup_steps,
            total_steps,
            base_lr,
            min_lr,
        }
    }
}

impl LRSchedule for WarmupCosine {
    fn get_lr(&self, step: usize) -> f32 {
        if step < self.warmup_steps {
            self.base_lr * (step + 1) as f32 / self.warmup_steps as f32
        } else {
            let progress = (step - self.warmup_steps) as f32
                / (self.total_steps - self.warmup_steps).max(1) as f32;
            let progress = progress.min(1.0);
            self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + (PI * progress).cos())
        }
    }

    fn base_lr(&self) -> f32 {
        self.base_lr
    }
}

/// Linear warmup + linear decay.
///
/// Warmup linearly from 0 to `base_lr` over `warmup_steps`,
/// then decay linearly to `min_lr` over the remaining steps.
#[derive(Debug, Clone)]
pub struct WarmupLinear {
    pub warmup_steps: usize,
    pub total_steps: usize,
    pub base_lr: f32,
    pub min_lr: f32,
}

impl WarmupLinear {
    pub fn new(warmup_steps: usize, total_steps: usize, base_lr: f32, min_lr: f32) -> Self {
        Self {
            warmup_steps,
            total_steps,
            base_lr,
            min_lr,
        }
    }
}

impl LRSchedule for WarmupLinear {
    fn get_lr(&self, step: usize) -> f32 {
        if step < self.warmup_steps {
            self.base_lr * (step + 1) as f32 / self.warmup_steps as f32
        } else {
            let progress = (step - self.warmup_steps) as f32
                / (self.total_steps - self.warmup_steps).max(1) as f32;
            let progress = progress.min(1.0);
            self.base_lr - progress * (self.base_lr - self.min_lr)
        }
    }

    fn base_lr(&self) -> f32 {
        self.base_lr
    }
}

/// Step-wise decay: multiply LR by `gamma` every `step_size` steps.
#[derive(Debug, Clone)]
pub struct StepDecay {
    pub base_lr: f32,
    pub gamma: f32,
    pub step_size: usize,
}

impl StepDecay {
    pub fn new(base_lr: f32, gamma: f32, step_size: usize) -> Self {
        Self {
            base_lr,
            gamma,
            step_size,
        }
    }
}

impl LRSchedule for StepDecay {
    fn get_lr(&self, step: usize) -> f32 {
        let num_decays = step / self.step_size;
        self.base_lr * self.gamma.powi(num_decays as i32)
    }

    fn base_lr(&self) -> f32 {
        self.base_lr
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, tol: f32) {
        assert!(
            (a - b).abs() < tol,
            "Expected {} ≈ {} (tol={}), got diff={}",
            a,
            b,
            tol,
            (a - b).abs()
        );
    }

    #[test]
    fn test_constant_lr() {
        let sched = ConstantLR::new(1e-3);
        assert_eq!(sched.get_lr(0), 1e-3);
        assert_eq!(sched.get_lr(100), 1e-3);
    }

    #[test]
    fn test_linear_warmup() {
        let sched = LinearWarmup::new(100, 1e-3);
        approx_eq(sched.get_lr(0), 1e-3 * 1.0 / 100.0, 1e-6);
        approx_eq(sched.get_lr(50), 1e-3 * 51.0 / 100.0, 1e-6);
        approx_eq(sched.get_lr(99), 1e-3, 1e-6);
        assert_eq!(sched.get_lr(100), 1e-3);
        assert_eq!(sched.get_lr(200), 1e-3);
    }

    #[test]
    fn test_warmup_cosine() {
        let sched = WarmupCosine::new(100, 1000, 1e-3, 1e-5);
        approx_eq(sched.get_lr(0), 1e-3 * 1.0 / 100.0, 1e-6);
        assert_eq!(sched.get_lr(99), 1e-3);
        assert_eq!(sched.get_lr(100), 1e-3);

        let mid = sched.get_lr(550);
        let expected_mid = 1e-5 + 0.5 * (1e-3 - 1e-5) * (1.0 + (PI * 0.5).cos());
        approx_eq(mid, expected_mid, 1e-6);

        let last = sched.get_lr(999);
        approx_eq(last, 1e-5, 1e-5);
    }

    #[test]
    fn test_warmup_linear() {
        let sched = WarmupLinear::new(100, 1000, 1e-3, 1e-5);
        approx_eq(sched.get_lr(0), 1e-3 * 1.0 / 100.0, 1e-6);
        assert_eq!(sched.get_lr(100), 1e-3);
        approx_eq(sched.get_lr(550), 1e-3 - 0.5 * (1e-3 - 1e-5), 1e-6);
        approx_eq(sched.get_lr(999), 1e-5, 1e-5);
    }

    #[test]
    fn test_step_decay() {
        let sched = StepDecay::new(1e-3, 0.1, 100);
        assert_eq!(sched.get_lr(0), 1e-3);
        assert_eq!(sched.get_lr(99), 1e-3);
        approx_eq(sched.get_lr(100), 1e-4, 1e-8);
        approx_eq(sched.get_lr(199), 1e-4, 1e-8);
        approx_eq(sched.get_lr(200), 1e-5, 1e-9);
    }
}
