# MLP-34 Strong Baseline, Noise-Rate Estimation, and Selection Mechanism Report

## Phase A: PubMed JoCoR five-seed comparison

| Noise | Combination | JoCoR | Paired delta | Positive seeds | Gate |
|---|---:|---:|---:|---:|---:|
| pair 0.4 | 77.71%±0.73% | 73.93%±1.32% | +3.78pp | 5/5 | 1 |
| instance 0.4 | 80.94%±0.93% | 76.89%±0.72% | +4.05pp | 5/5 | 1 |
| uniform 0.6 | 59.91%±6.21% | 58.12%±5.22% | +1.79pp | 3/5 | 0 |

Strong three-noise boundary elimination: 0.
Conditional boundary elimination: 1.

## Phase B: noisy-only LOO noise-rate estimate

| Dataset | Estimate | Absolute error | Coverage | Effective sample size |
|---|---:|---:|---:|---:|
| citeseer | 0.734 | 0.132 | 0.998 | 526.5 |
| cora | 0.724 | 0.120 | 1.000 | 425.2 |
| corafull | 0.838 | 0.233 | 1.000 | 3227.1 |

B1 gate: 0. B2 was skipped by preregistration.

## Phase C: selection mechanism

Trace regression: 40/45 exact, 45/45 protocol/selection matched within the observed ROCm band; maximum absolute metric delta=2.76pp.

Mechanism gate: {"n_settings": 10, "accuracy_pass_count": 7, "retention_pass_count": 1, "real_vs_shuffled_pass_count": 10, "joint_pass_count": 1, "pass": false}

### Saturation correlations

| Predictor | n | Spearman | LODO range | LODO consistent | Saturation direction |
|---|---:|---:|---:|---:|---:|
| brier_gain | 15 | 0.039 | [-0.210, 0.301] | 0 | 0 |
| ncr | 15 | -0.509 | [-0.692, -0.310] | 1 | 1 |
| cdr | 15 | -0.528 | [-0.743, -0.401] | 1 | 0 |
| mean_margin | 15 | 0.636 | [0.524, 0.769] | 1 | 0 |

Correlations meeting magnitude, LODO, and preregistered semantic direction: 1.
A single supporting diagnostic is insufficient for a general saturation claim when other diagnostics point in the opposite direction.

## Decision

- Phase A determines whether the PubMed claim may be conditional; the strict three-noise claim remains prohibited if any row misses the frozen gate.
- Phase B does not tune a replacement estimator after failure; the known-noise-rate limitation remains.
- Phase C is seed-0, clean-dependent post-hoc diagnosis only and is not five-seed performance evidence.