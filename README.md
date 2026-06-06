# ternary-accumulator

*Gradient accumulation where every bit of direction matters.*

---

## What This Does

In ternary neural networks, gradients are reduced to their sign: {-1, 0, +1}. You can't just add them up like floats. This crate implements accumulation strategies that respect the ternary constraint — majority vote, momentum-buffered accumulation, entropy tracking, and checkpoint/restore.

The core question: after seeing 100 ternary gradients for one parameter — 60 positive, 30 zero, 10 negative — what's the update? This crate answers that question several ways.

## Key Types

- **TernaryAccumulator** — single-parameter accumulator with majority vote, momentum, net charge, distribution, and Shannon entropy
- **GradientAccumulator** — multi-parameter accumulator for micro-batch gradient accumulation (N steps before apply)
- **AccumulatorCheckpoint** — save/restore state mid-training

## Why Entropy Matters

The entropy of accumulated gradients tells you how much *agreement* there is. Low entropy = all gradients point the same way (confident). High entropy = mixed signals (uncertain). You can use this to dynamically adjust learning rates or skip uncertain updates entirely.

14 tests covering majority vote, momentum, net charge, distribution, entropy, gradient accumulation, checkpoint/restore.

## Part of [SuperInstance](https://github.com/SuperInstance/SuperInstance)

License: MIT
