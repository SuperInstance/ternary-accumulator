# ternary-accumulator

Ternary gradient accumulation for sign-based neural network training.

## Why This Exists

In ternary neural networks, gradients are reduced to their sign {-1, 0, +1} before being applied. You can't average ternary gradients the way you average floating-point gradients — the sign constraint must be preserved. This crate implements accumulation strategies that count how many times each sign appears and produce a ternary output via majority vote. It also supports momentum (exponential moving average of ternary distributions) and tracks statistics like entropy and net charge across parameters.

## Architecture

### Core Types

- **`TernaryAccumulator`** — Single-parameter accumulator tracking positive/zero/negative counts, net charge, and optional momentum.
- **`GradientAccumulator`** — Multi-parameter accumulator for full model gradients, with configurable gradient accumulation steps.
- **`AccumulatorCheckpoint`** — Snapshot of accumulator state for fault-tolerant training.

### Accumulation Strategy

1. Each gradient arrives as a Trit (-1, 0, +1).
2. Counts are maintained: `n_pos`, `n_zero`, `n_neg`.
3. `majority()` returns the most common sign.
4. `momentum_trit()` applies exponential decay to the count distribution before taking majority.
5. `entropy()` measures how uncertain the accumulation is (high entropy = disagreement among gradients).

## Usage

```rust
use ternary_accumulator::{TernaryAccumulator, GradientAccumulator};

// Single parameter
let mut acc = TernaryAccumulator::with_momentum(0.9);
acc.accumulate_batch(&[1, 1, 1, 0, -1]);
assert_eq!(acc.majority(), 1); // mostly positive
println!("Entropy: {:.3}", acc.entropy()); // how certain
println!("Net charge: {}", acc.net_charge());

// Full model with gradient accumulation steps
let mut grad_acc = GradientAccumulator::new(1000, 4); // 1000 params, 4 steps
grad_acc.step(&vec![1; 1000]); // step 1
grad_acc.step(&vec![-1; 1000]); // step 2
grad_acc.step(&vec![1; 1000]); // step 3
grad_acc.step(&vec![1; 1000]); // step 4
assert!(grad_acc.ready()); // all 4 steps done
let aggregated = grad_acc.aggregated(); // majority vote per parameter
```

## API Reference

### TernaryAccumulator

| Method | Returns | Description |
|--------|---------|-------------|
| `new()` | `Self` | No momentum |
| `with_momentum(momentum)` | `Self` | With EMA decay |
| `accumulate(trit)` | `()` | Single gradient |
| `accumulate_batch(trits)` | `()` | Batch of gradients |
| `majority()` | `Trit` | Most common sign |
| `momentum_trit()` | `Trit` | Momentum-weighted majority |
| `net_charge()` | `i64` | Σ positive - Σ negative |
| `distribution()` | `(f64, f64, f64)` | (neg_frac, zero_frac, pos_frac) |
| `entropy()` | `f64` | Uncertainty in bits |
| `reset()` | `()` | Clear accumulator |
| `total()` | `usize` | Total accumulated |

### GradientAccumulator

| Method | Returns | Description |
|--------|---------|-------------|
| `new(num_params, target_steps)` | `Self` | Configure accumulation |
| `step(gradients)` | `()` | Accumulate one step |
| `ready()` | `bool` | All steps complete |
| `aggregated()` | `Vec<Trit>` | Per-parameter majority vote |
| `aggregated_momentum()` | `Vec<Trit>` | Momentum-weighted vote |
| `mean_entropy()` | `f64` | Average parameter entropy |
| `steps()` | `usize` | Steps accumulated |

## The Deeper Idea

Ternary accumulation is **democratic voting for gradients**. Each training step casts a vote (-1, 0, or +1) for each parameter's update direction. The majority vote is the accumulated gradient. This is robust to outliers (a single wildly wrong gradient can't dominate) and naturally handles sparsity (many zero votes = "don't touch this parameter"). The entropy metric tells you which parameters the training process is most uncertain about — useful for active learning and curriculum scheduling.

## Related Crates

- **ternary-gc** — garbage collection with ternary marking
- **ternary-epoch** — epoch detection in ternary training histories
- **ternary-cortex** — neural network layers with ternary processing
