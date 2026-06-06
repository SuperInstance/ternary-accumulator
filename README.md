# ternary-accumulator

*Ternary gradient accumulation for sign-based training. Gradients become {-1, 0, +1} before they accumulate — preserving the ternary constraint while tracking statistics.*

## Why This Exists

Ternary neural networks (BitNet b1.58, Ternary Weight Networks) train with sign-based gradients: instead of accumulating float gradients, you reduce each gradient to its sign and accumulate the ternary values. This is fundamentally different from float accumulation — you're counting consensus, not averaging magnitudes.

This crate implements three accumulation strategies: naive counting, momentum-guided accumulation, and threshold-based accumulation with configurable deadband.

## Architecture

```
Float Gradient: [+0.03, -0.47, +0.12, +0.001, -0.89]
                         ↓ sign()
Ternary Gradient: [+1, -1, +1, 0, -1]  (near-zero → 0)
                         ↓ accumulate
Accumulator: {pos: [+1, 0, +1, 0, 0], neg: [0, -1, 0, 0, -1], count: 1}
                         ↓ after 64 steps
Ternary Update: sign(pos - neg) → [+1, -1, +1, 0, -1]
```

### Key Types

- **`TernaryAccumulator`** — Core accumulator: tracks positive/negative counts per parameter. Produces ternary updates via sign(pos_count - neg_count).
- **`GradientAccumulator`** — Higher-level: wraps TernaryAccumulator with gradient sign reduction, deadband filtering (gradients below threshold → 0), and micro-batch accumulation.
- **`AccumulatorCheckpoint`** — Save/restore accumulator state for long training runs with checkpointing.

## Usage

```rust
use ternary_accumulator::*;

let mut acc = TernaryAccumulator::new(5); // 5 parameters

// Accumulate ternary gradients over micro-batches
acc.accumulate(&[1, -1, 0, 1, -1]);
acc.accumulate(&[1, 0, 1, 1, -1]);
acc.accumulate(&[1, -1, 1, 0, -1]);

// Get the accumulated ternary update
let update = acc.ternary_update();
// update = [1, -1, 1, 1, -1] (majority sign per parameter)

// Reset for next accumulation window
acc.reset();
```

## The Deeper Idea

Ternary accumulation is democracy applied to gradients. Each micro-batch "votes" for each parameter's direction. After N votes, the majority wins. Parameters with no clear majority (equal pos/neg counts) stay at 0 — the ternary "abstain."

This is mathematically equivalent to sign SGD with majority voting, and it connects directly to the consensus mechanisms in `ternary-consensus` and the voting protocols in `ternary-negotiate`.

## Related Crates

- `ternary-optimizer` — Uses accumulator for weight updates
- `ternary-gradient-queue` — Priority scheduling for accumulated gradients
- `ternary-checkpoint` — Save accumulator state
- `ternary-distill` — Distillation training loop
