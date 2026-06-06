//! # ternary-accumulator
//!
//! Ternary gradient accumulation for sign-based training.
//!
//! In ternary neural networks, gradients are reduced to their sign {-1, 0, +1}
//! before accumulation. This crate implements accumulation strategies that
//! preserve the ternary constraint while tracking gradient statistics.

/// A single ternary value in {-1, 0, +1}.
pub type Trit = i8;

/// Accumulator for ternary gradients.
#[derive(Debug, Clone)]
pub struct TernaryAccumulator {
    /// Accumulated counts: [neg_count, zero_count, pos_count]
    counts: [usize; 3],
    /// Momentum buffer (integer accumulation before ternarization)
    momentum: i64,
    /// Momentum coefficient (scaled by 1000, e.g. 0.9 = 900)
    momentum_coeff: i64,
    /// Total trits accumulated
    total: usize,
}

impl TernaryAccumulator {
    pub fn new() -> Self {
        Self {
            counts: [0, 0, 0],
            momentum: 0,
            momentum_coeff: 900, // 0.9
            total: 0,
        }
    }

    pub fn with_momentum(momentum: f64) -> Self {
        Self {
            counts: [0, 0, 0],
            momentum: 0,
            momentum_coeff: (momentum * 1000.0) as i64,
            total: 0,
        }
    }

    /// Accumulate a single ternary gradient.
    pub fn accumulate(&mut self, trit: Trit) {
        match trit {
            -1 => self.counts[0] += 1,
            0 => self.counts[1] += 1,
            1 => self.counts[2] += 1,
            _ => panic!("Invalid trit: must be -1, 0, or +1"),
        }
        self.momentum = self.momentum * self.momentum_coeff / 1000 + (trit as i64) * 1000;
        self.total += 1;
    }

    /// Accumulate a batch of ternary gradients.
    pub fn accumulate_batch(&mut self, trits: &[Trit]) {
        for &t in trits {
            self.accumulate(t);
        }
    }

    /// Get the majority-vote ternary value.
    pub fn majority(&self) -> Trit {
        if self.counts[0] > self.counts[1] && self.counts[0] > self.counts[2] {
            -1
        } else if self.counts[2] > self.counts[1] && self.counts[2] > self.counts[0] {
            1
        } else {
            0
        }
    }

    /// Get the momentum-ternarized value.
    pub fn momentum_trit(&self) -> Trit {
        if self.momentum > 500 {
            1
        } else if self.momentum < -500 {
            -1
        } else {
            0
        }
    }

    /// Get the weighted sum (net ternary charge).
    pub fn net_charge(&self) -> i64 {
        (self.counts[2] as i64) - (self.counts[0] as i64)
    }

    /// Get distribution as (neg_frac, zero_frac, pos_frac).
    pub fn distribution(&self) -> (f64, f64, f64) {
        if self.total == 0 {
            return (0.0, 1.0, 0.0);
        }
        (
            self.counts[0] as f64 / self.total as f64,
            self.counts[1] as f64 / self.total as f64,
            self.counts[2] as f64 / self.total as f64,
        )
    }

    /// Reset accumulator state.
    pub fn reset(&mut self) {
        self.counts = [0, 0, 0];
        self.momentum = 0;
        self.total = 0;
    }

    /// Total accumulated trits.
    pub fn total(&self) -> usize {
        self.total
    }

    /// Shannon entropy of the trit distribution.
    pub fn entropy(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        let (p_neg, p_zero, p_pos) = self.distribution();
        let mut h = 0.0;
        for p in &[p_neg, p_zero, p_pos] {
            if *p > 0.0 {
                h -= p * p.log2();
            }
        }
        h
    }
}

impl Default for TernaryAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

/// Multi-parameter gradient accumulator for training ternary networks.
#[derive(Debug, Clone)]
pub struct GradientAccumulator {
    /// One accumulator per parameter
    accumulators: Vec<TernaryAccumulator>,
    /// Accumulation steps taken
    steps: usize,
    /// Target accumulation steps before apply
    target_steps: usize,
}

impl GradientAccumulator {
    pub fn new(num_params: usize, target_steps: usize) -> Self {
        Self {
            accumulators: (0..num_params).map(|_| TernaryAccumulator::new()).collect(),
            steps: 0,
            target_steps,
        }
    }

    /// Accumulate gradients for one micro-batch.
    pub fn step(&mut self, gradients: &[Trit]) {
        assert_eq!(gradients.len(), self.accumulators.len(),
            "Gradient length must match number of parameters");
        for (acc, &g) in self.accumulators.iter_mut().zip(gradients.iter()) {
            acc.accumulate(g);
        }
        self.steps += 1;
    }

    /// Check if we've accumulated enough steps.
    pub fn ready(&self) -> bool {
        self.steps >= self.target_steps
    }

    /// Get the aggregated ternary gradient (majority vote per parameter).
    pub fn aggregated(&self) -> Vec<Trit> {
        self.accumulators.iter().map(|a| a.majority()).collect()
    }

    /// Get the aggregated gradient using momentum.
    pub fn aggregated_momentum(&self) -> Vec<Trit> {
        self.accumulators.iter().map(|a| a.momentum_trit()).collect()
    }

    /// Reset after applying gradients.
    pub fn reset(&mut self) {
        for acc in &mut self.accumulators {
            acc.reset();
        }
        self.steps = 0;
    }

    /// Get per-parameter entropy (measures agreement).
    pub fn entropies(&self) -> Vec<f64> {
        self.accumulators.iter().map(|a| a.entropy()).collect()
    }

    /// Mean entropy across all parameters.
    pub fn mean_entropy(&self) -> f64 {
        let entropies = self.entropies();
        if entropies.is_empty() { 0.0 } else { entropies.iter().sum::<f64>() / entropies.len() as f64 }
    }

    pub fn steps(&self) -> usize { self.steps }
    pub fn num_params(&self) -> usize { self.accumulators.len() }
}

/// Checkpoint for gradient accumulator state.
#[derive(Debug, Clone)]
pub struct AccumulatorCheckpoint {
    pub counts: Vec<[usize; 3]>,
    pub momenta: Vec<i64>,
    pub steps: usize,
    pub target_steps: usize,
}

impl GradientAccumulator {
    pub fn checkpoint(&self) -> AccumulatorCheckpoint {
        AccumulatorCheckpoint {
            counts: self.accumulators.iter().map(|a| a.counts).collect(),
            momenta: self.accumulators.iter().map(|a| a.momentum).collect(),
            steps: self.steps,
            target_steps: self.target_steps,
        }
    }

    pub fn restore(cp: &AccumulatorCheckpoint) -> Self {
        let mut accs: Vec<TernaryAccumulator> = cp.counts.iter().zip(cp.momenta.iter())
            .map(|(&counts, &mom)| TernaryAccumulator {
                counts,
                momentum: mom,
                momentum_coeff: 900,
                total: counts[0] + counts[1] + counts[2],
            })
            .collect();
        Self {
            accumulators: accs,
            steps: cp.steps,
            target_steps: cp.target_steps,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accumulate_majority_negative() {
        let mut acc = TernaryAccumulator::new();
        acc.accumulate_batch(&[-1, -1, -1, 0, 1]);
        assert_eq!(acc.majority(), -1);
    }

    #[test]
    fn test_accumulate_majority_positive() {
        let mut acc = TernaryAccumulator::new();
        acc.accumulate_batch(&[1, 1, 1, 0, -1]);
        assert_eq!(acc.majority(), 1);
    }

    #[test]
    fn test_accumulate_majority_tie() {
        let mut acc = TernaryAccumulator::new();
        acc.accumulate_batch(&[-1, 0, 1]);
        assert_eq!(acc.majority(), 0); // tie → 0
    }

    #[test]
    fn test_net_charge() {
        let mut acc = TernaryAccumulator::new();
        acc.accumulate_batch(&[-1, -1, 0, 1, 1, 1]);
        assert_eq!(acc.net_charge(), 1); // 3 pos - 2 neg
    }

    #[test]
    fn test_distribution() {
        let mut acc = TernaryAccumulator::new();
        acc.accumulate_batch(&[-1, 0, 1]);
        let (neg, zero, pos) = acc.distribution();
        assert!((neg - 1.0/3.0).abs() < 1e-10);
        assert!((zero - 1.0/3.0).abs() < 1e-10);
        assert!((pos - 1.0/3.0).abs() < 1e-10);
    }

    #[test]
    fn test_entropy_uniform() {
        let mut acc = TernaryAccumulator::new();
        acc.accumulate_batch(&[-1, 0, 1]);
        // Maximum entropy for 3 outcomes: log2(3) ≈ 1.585
        assert!((acc.entropy() - 3.0_f64.log2()).abs() < 1e-10);
    }

    #[test]
    fn test_entropy_zero() {
        let mut acc = TernaryAccumulator::new();
        acc.accumulate_batch(&[1, 1, 1]);
        assert_eq!(acc.entropy(), 0.0);
    }

    #[test]
    fn test_momentum_accumulation() {
        let mut acc = TernaryAccumulator::new();
        acc.accumulate(1);
        acc.accumulate(1);
        acc.accumulate(1);
        // After 3 positive gradients with momentum 0.9, momentum should be positive
        assert_eq!(acc.momentum_trit(), 1);
    }

    #[test]
    fn test_momentum_decay() {
        let mut acc = TernaryAccumulator::new();
        acc.accumulate(-1);
        acc.accumulate(-1);
        acc.accumulate(1);
        acc.accumulate(1);
        acc.accumulate(1);
        // Net positive after momentum
        assert_eq!(acc.momentum_trit(), 1);
    }

    #[test]
    fn test_reset() {
        let mut acc = TernaryAccumulator::new();
        acc.accumulate_batch(&[-1, 0, 1]);
        acc.reset();
        assert_eq!(acc.total(), 0);
        assert_eq!(acc.majority(), 0);
    }

    #[test]
    fn test_gradient_accumulator_majority() {
        let mut ga = GradientAccumulator::new(3, 3);
        ga.step(&[1, -1, 0]);
        ga.step(&[1, -1, 0]);
        ga.step(&[1, -1, 0]);
        assert!(ga.ready());
        let agg = ga.aggregated();
        assert_eq!(agg, vec![1, -1, 0]);
    }

    #[test]
    fn test_gradient_accumulator_not_ready() {
        let ga = GradientAccumulator::new(3, 5);
        assert!(!ga.ready());
    }

    #[test]
    fn test_gradient_entropy_measures_agreement() {
        let mut ga = GradientAccumulator::new(2, 2);
        ga.step(&[1, 1]);
        ga.step(&[1, 1]);
        // Perfect agreement → low entropy
        assert!(ga.mean_entropy() < 0.01);
    }

    #[test]
    fn test_checkpoint_restore() {
        let mut ga = GradientAccumulator::new(3, 5);
        ga.step(&[1, -1, 0]);
        ga.step(&[0, 1, -1]);
        let cp = ga.checkpoint();
        let restored = GradientAccumulator::restore(&cp);
        assert_eq!(restored.steps(), 2);
        assert_eq!(restored.aggregated(), ga.aggregated());
    }
}
