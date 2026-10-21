# MLP-28 Soft-Target Robust Optimization Smoke Report

Protocol: Cora/CiteSeer/CiteSeerFull, uniform 0.6, train-only noise, percent 0.2/0.1/0.7.

## Paired attribution

| Strategy | Dataset | LCSS | Robust-only | Combination | Shuffled combination | Synergy | Real-shuffled |
|---|---|---:|---:|---:|---:|---:|---:|
| logitclip | cora | 38.94 | 32.66 | 39.42 | 34.20 | +0.47pp | +5.22pp |
| logitclip | citeseer | 38.62 | 32.13 | 37.84 | 32.30 | -0.77pp | +5.54pp |
| logitclip | citeseerfull | 39.68 | 36.27 | 42.08 | 33.54 | +2.40pp | +8.54pp |
| soft_small_loss | cora | 38.94 | 37.68 | 44.96 | 34.56 | +6.02pp | +10.40pp |
| soft_small_loss | citeseer | 38.62 | 36.94 | 41.80 | 31.92 | +3.18pp | +9.88pp |
| soft_small_loss | citeseerfull | 39.68 | 38.26 | 49.11 | 30.56 | +9.42pp | +18.54pp |

## Preregistered gate

| Strategy | Pass | Mean synergy | Minimum synergy | Positive datasets | Structure >=1pp datasets | Reasons |
|---|---:|---:|---:|---:|---:|---|
| logitclip | 0 | +0.70pp | -0.77pp | 2/3 | 3/3 | regression_over_0p5pp |
| soft_small_loss | 1 | +6.21pp | +3.18pp | 3/3 | 3/3 | pass |

## Decision

A passing strategy is only a candidate for paired five-seed confirmation. It does not modify the frozen LCSS paper mainline.
