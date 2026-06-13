# Ternary Accumulator

Ternary gradient accumulation for **sign-based neural network training**. Implements accumulation strategies that preserve the ternary constraint {-1, 0, +1} while tracking gradient statistics including majority vote, momentum, Shannon entropy, and checkpoint-based state management.

## Why It Matters

Modern quantized neural networks (QNNs) often represent weights and gradients as ternary values {-1, 0, +1} — pioneered by TWN (Ternary Weight Networks) and Bireal-Net. The challenge is gradient accumulation: you can't simply average ternary gradients because averaging destroys the sign information that ternary networks rely on.

This crate solves three problems:

1. **Aggregation**: How do you combine N ternary gradients into one decision? (Answer: majority vote or momentum-ternarization)
2. **Statistics**: How do you measure gradient quality? (Answer: Shannon entropy of the trit distribution)
3. **Scaling**: How do you accumulate across thousands of parameters efficiently? (Answer: vectorized accumulators with O(1) per-parameter state)

## How It Works

### Majority Vote Aggregation

Given N ternary gradients $g_1, \ldots, g_N \in \{-1, 0, +1\}$, the majority vote is:

$$\text{maj}(\vec{g}) = \begin{cases} +1 & \text{if } |\{i : g_i = +1\}| > |\{i : g_i = -1\}| \text{ and } |\{i : g_i = +1\}| > |\{i : g_i = 0\}| \\ -1 & \text{if } |\{i : g_i = -1\}| > |\{i : g_i = +1\}| \text{ and } |\{i : g_i = -1\}| > |\{i : g_i = 0\}| \\ 0 & \text{otherwise (tie)} \end{cases}$$

### Momentum-Ternarized Accumulation

Momentum is maintained as an integer buffer (scaled by 1000 to avoid floats):

$$m_t = \beta \cdot m_{t-1} + g_t \cdot 1000$$
$$\text{trit}(m_t) = \begin{cases} +1 & m_t > 500 \\ -1 & m_t < -500 \\ 0 & \text{otherwise} \end{cases}$$

The threshold of 500 (half of 1000) provides a **deadband** that prevents oscillation from small momentum values.

### Shannon Entropy of Gradient Distribution

Gradient quality is measured by the Shannon entropy of the ternary distribution $(p_{-1}, p_0, p_{+1})$:

$$H = -\sum_{i} p_i \log_2 p_i$$

Maximum entropy $H_{\max} = \log_2 3 \approx 1.585$ bits indicates uniform disagreement (poor signal). $H = 0$ indicates perfect consensus (strong signal). Mean entropy across all parameters serves as a training health metric.

### Complexity

| Operation | Time | Space |
|-----------|------|-------|
| `accumulate(trit)` | O(1) | O(1) |
| `accumulate_batch(&[trit])` | O(N) | O(1) |
| `majority()` | O(1) | — |
| `momentum_trit()` | O(1) | — |
| `entropy()` | O(1) | — |
| `GradientAccumulator::step(&[trit])` | O(P) | O(P) total |
| `checkpoint()` / `restore()` | O(P) | O(P) |

Where *N* = batch size, *P* = number of parameters.

## Quick Start

```rust
use ternary_accumulator::{TernaryAccumulator, GradientAccumulator};

// Single-parameter accumulation
let mut acc = TernaryAccumulator::new();
acc.accumulate_batch(&[1, 1, 1, 0, -1]);
assert_eq!(acc.majority(), 1);
assert_eq!(acc.net_charge(), 2); // 3 positive - 1 negative
let h = acc.entropy(); // Shannon entropy of distribution

// Multi-parameter gradient accumulation
let mut ga = GradientAccumulator::new(3, 4); // 3 params, 4 micro-batches
ga.step(&[1, -1, 0]);
ga.step(&[1, -1, 0]);
ga.step(&[1, -1, 0]);
ga.step(&[1, -1, 0]);
assert!(ga.ready());
assert_eq!(ga.aggregated(), vec![1, -1, 0]);

// Checkpoint and restore
let cp = ga.checkpoint();
let restored = GradientAccumulator::restore(&cp);
```

## API

### `TernaryAccumulator`

| Method | Description |
|--------|-------------|
| `new()` | Default accumulator (momentum = 0.9) |
| `with_momentum(f64)` | Custom momentum coefficient |
| `accumulate(Trit)` | Add one gradient |
| `accumulate_batch(&[Trit])` | Add multiple gradients |
| `majority() → Trit` | Majority-vote ternary value |
| `momentum_trit() → Trit` | Momentum-ternarized value |
| `net_charge() → i64` | $\sum_{+1} - \sum_{-1}$ |
| `distribution() → (f64, f64, f64)` | (p_neg, p_zero, p_pos) |
| `entropy() → f64` | Shannon entropy in bits |
| `reset()` | Clear all state |

### `GradientAccumulator`

| Method | Description |
|--------|-------------|
| `new(num_params, target_steps)` | Initialize for P params, N steps |
| `step(&[Trit])` | Accumulate one micro-batch |
| `ready() → bool` | Enough steps accumulated? |
| `aggregated() → Vec<Trit>` | Majority-vote gradient |
| `aggregated_momentum() → Vec<Trit>` | Momentum-based gradient |
| `mean_entropy() → f64` | Average per-parameter entropy |
| `checkpoint()` / `restore()` | Save/restore state |

## Architecture Notes

The accumulator maintains the **γ + η = C** conservation link:

- **γ (structure)**: the fixed parameter dimensionality — P accumulators, one per weight
- **η (dynamics)**: the incoming gradient stream — the perturbation signal driving updates
- **C (conservation)**: the net charge invariant — $\sum \text{trits} = \text{counts}[+1] - \text{counts}[-1]$, always recoverable from internal state

The Shannon entropy measurement directly quantifies η: high entropy means the perturbation is uninformative (random), low entropy means it carries a strong directional signal. Training health is monotonic in decreasing entropy.

## References

- Li, F. et al. (2016). *Ternary Weight Networks*. arXiv:1605.04711. — Original TWN paper.
- Liu, Z. et al. (2018). *Bi-Real Net: Enhancing the Performance of 1-bit CNNs*. arXiv:1808.00278.
- Hubara, I. et al. (2017). *Quantized Neural Networks: Training Neural Networks with Low Precision Weights and Activations*. JMLR.
- Rastegari, M. et al. (2016). *XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks*. ECCV.

## License: MIT
