# E1: Purity-Coverage Adaptive Granular Balls - Seed0 Report

- Date: 2026-06-10
- Python: `.venv-rocm/Scripts/python.exe` (GRACE-master ROCm environment), device `cuda:0`.
- Protocol: NoisyGL percent split 0.2/0.1/0.7, uniform noise 0.4, seed 0.
- Method family: frozen-GRACE balls -> SoftDist targets (multiclass alpha + ball distribution) -> fresh end-to-end GCN. Only the BALL CONSTRUCTION varies.
- Boundary: seed0 smokes only. 5-seed std on this protocol is about 0.01, so single-seed deltas near 1pp are within noise.

## Research Record

- Hypothesis: geometric balls straddle class boundaries, which is where hard/soft majority votes damage clean labels (Cora CDR 22%). Label-aware adaptive granularity (split mixed balls finer, keep clean regions coarse) should reduce CDR without losing NCR, and beat geometric SoftDist.
- Pre-registered constraints (agreed before implementation):
  1. Splitting uses noisy TRAIN label purity only; this is noisy-supervised ball construction, NOT unsupervised GB. Val/test labels never enter (leakage unit test included).
  2. E1 only changes train-node targets. Unlabeled-node supervision expansion (E2) is frozen until E1 passes.
  3. Diagnostics must report noisy purity vs true purity (clean labels, diagnostic only) to detect noise-fitting.
- Pre-registered gate: adaptive must not lose to geometric SoftDist and should clearly reduce CDR. Failure -> adaptive-granularity hypothesis rejected, do NOT proceed to E2, fall back to Route A.

## Implementation

- `utils/ordinary_gb.py`: `build_purity_adaptive_granular_balls` - geometric 2-means split moves (label-blind), label-aware accept/stop. Three acceptance rules were tried:
  - v1 threshold: split while noisy purity < tau_p=0.6, coverage guard min_train_per_child=2, min_train_to_split=6.
  - v2 strict gain: accept split only if child weighted noisy purity >= parent + 0.05; min_train_per_child=4, min_train_to_split=16.
  - v3 multi-attempt gain: 3 kmeans attempts per ball, best valid child purity, gain >= 0.02; min_train_per_child=3, min_train_to_split=12.
- `utils/frozen_grace_targets.py`: `ball_mode` switch + `partition_purity_diagnostics` (noisy vs true purity, train-weighted).
- New methods `frozengrace_gb_gcn_adaptive[_shuffled]`; configs under `config/frozengrace_gb_gcn_adaptive*/`.
- Tests: `tests/test_adaptive_gb.py` (pure-stays-coarse, mixed-splits-finer, coverage guard, non-train-label leakage guard, purity diagnostics). 26/26 tests pass.

## Seed0 Test Accuracy

| Variant | Granularity (Cora/CiteSeer balls) | Cora | Cora shuffled | CiteSeer | CiteSeer shuffled |
|---|---|---:|---:|---:|---:|
| GCN+CE | - | 0.7974 | - | 0.6435 | - |
| SoftDist geometric (baseline) | 52 / 64 | 0.8042 | 0.8047 | 0.6748 | 0.6491 |
| v1 adaptive threshold (fine) | 59 / 139 | 0.8005 | 0.7826 | 0.6770 | 0.6430 |
| v2 adaptive strict gain (coarse) | 6 / 4 | **0.8158** | 0.8058 | 0.6705 | 0.6465 |
| v3 adaptive multi-attempt (mid-coarse) | 9 / 6 | 0.7995 | 0.8063 | 0.6740 | 0.6430 |

## Diagnostics (real balls)

| Dataset | Variant | Balls | Train/ball | NCR | CDR | Noisy purity | True purity (train) |
|---|---|---:|---:|---:|---:|---:|---:|
| Cora | geometric | 52 | ~10.4 | 0.688 | 0.224 | 0.545 | 0.789 |
| Cora | v1 fine | 59 | 9.2 | 0.683 | 0.222 | 0.545 | 0.806 |
| Cora | v2 coarse | 6 | 90.2 | 0.683 | 0.313 | 0.468 | 0.686 |
| Cora | v3 mid | 9 | 60.1 | 0.693 | 0.293 | 0.484 | 0.704 |
| CiteSeer | geometric | 64 | ~10.4 | 0.591 | 0.298 | 0.462 | 0.701 |
| CiteSeer | v1 fine | 139 | 4.8 | 0.357 | 0.333 | 0.534 | 0.752 |
| CiteSeer | v2 coarse | 4 | 166.3 | 0.561 | 0.449 | 0.358 | 0.555 |
| CiteSeer | v3 mid | 6 | 110.8 | 0.658 | 0.369 | 0.395 | 0.642 |

## Findings

1. **The pre-registered gate FAILED.** No adaptive variant beats geometric SoftDist on both datasets, and CDR did not drop in any variant (it rose in the coarse variants). The one Cora win (v2, +1.2pp) is within seed noise and its shuffled control sits at 0.8058; in v3 Cora shuffled even beats real. The label-aware-granularity signal is not robust.
2. **Why it failed - vote-density physics, not implementation.** At 20% train coverage with 40% symmetric noise, the noise-consistent purity ceiling is ~0.66, and per-ball purity estimates carry std ~0.12-0.17 at 5-15 votes per ball. The label signal is too weak to drive granularity decisions: the threshold rule degenerates to "split until guards stop" (over-fine, votes diluted, CiteSeer NCR 0.59->0.36), while gain rules reject most splits (over-coarse, mixed regions stay merged, CDR rises).
3. **Noisy purity underestimates true purity by 0.2-0.3 everywhere** (e.g. geometric Cora 0.545 noisy vs 0.789 true). No noise-fitting was observed (true purity never moved opposite to noisy purity), but the gap quantifies why noisy-label-driven splitting is unreliable at this coverage.
4. **Granularity itself matters more than label-awareness.** Across variants, accuracy varies ~1.5pp with ball count while NCR/CDR trade off mechanically with votes-per-ball. The geometric max_balls=64 setting already sits near the practical optimum.
5. **Real > shuffled in 5/6 comparisons** - ball structure keeps carrying signal, consistent with all previous experiments.

## Decision

- E1 fails per the pre-registered criterion. **Do NOT proceed to E2** (unlabeled-node supervision expansion built on adaptive purity) - its premise (trustworthy per-ball purity) is exactly what failed.
- **Fall back to Route A**: geometric SoftDist remains the main mechanism. Paper positioning: first granular-ball mechanism for noisy-label node classification + coverage sensitivity analysis + soft/hard correction comparison + honest negative result that label-aware granularity cannot be driven by 20%-coverage noisy votes (this E1 result is itself paper material as a mechanism boundary).
- The adaptive builder, purity diagnostics, and leakage tests stay in the codebase as analysis tooling.
