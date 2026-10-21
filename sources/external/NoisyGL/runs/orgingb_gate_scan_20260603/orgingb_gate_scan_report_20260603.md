# OrginGB-GCN v2 Gate Scan And Selected Smoke Report

## Basic Info

- Date: 2026-06-03
- Dataset: Cora
- Noise: uniform 0.4
- Seed: 3000
- Device target: cuda:0
- Main outputs:
  - `gate_scan.csv`
  - `gate_scan_summary.md`
  - `training_smoke_selected_gate.log`
  - `../orgingb_gcn_diagnostics_v3.csv`

## Research Hypothesis

Ordinary granular balls may still be useful for graph label-noise denoising if they are not used as a strong soft-label corrector. The v2 hypothesis is narrower: use ordinary GB only as a high-suspicion gate, then down-weight a small subset of likely noisy training nodes. If GB provides real denoising signal, the gated set should have higher noisy-label precision than no-ball and random controls under the same gate size.

## GB Role

In this experiment, GB is treated as a local confidence and suspicion prior. It is not used as a hard positive relation, hard negative filter, or strong target-correction mechanism.

## Method Motivation

The v1 soft-target design did not produce reliable improvement because it changed many targets while the ordinary GB signal was not strong enough to justify broad label correction. The v2 reweight design therefore keeps each noisy label as the target and only changes the sample weight of high-suspicion nodes.

## Code Changes

- `utils/soft_targets.py`
  - Added reweight-compatible diagnostics through `SoftTargetResult`.
  - Exposed `gate_mask` and `node_ball_confidence` for offline gate analysis.
- `predictor/OrginGB_GCN_Predictor.py`
  - Added/used reweight variants:
    - `orgingb_gcn_reweight`
    - `orgingb_gcn_reweight_random`
    - `orgingb_gcn_reweight_noball`
- `scripts/diagnose_orgingb_gate_scan.py`
  - Added offline scan over `suspicion_top_ratio` and `min_ball_conf`.
  - Added strength-matched no-ball and random controls.
- `tests/test_orgingb_tools.py`
  - Added coverage for reweight target preservation, gate behavior, size-matched random assignment, and no-ball behavior.

## Config Changes

Selected Cora smoke configs were set to the best offline passing threshold:

- `suspicion_top_ratio: 0.1`
- `min_ball_conf: 0.1`
- `beta: 0.5`
- `min_weight: 0.3`
- `warmup_epochs: 30`

The no-ball control uses the same threshold but `suspicion_mode: no_ball`.

## Expected Mechanism

For each train node, the reweight mode keeps:

`target = one_hot(noisy_label)`

and uses a suspicion score derived from warmup confidence and GB label distribution:

`s_i = 1 - sqrt(warmup_prob_i[noisy_y_i] * q_B[noisy_y_i])`

Only high-suspicion nodes above the train-set top-ratio threshold and passing the ball-confidence gate are down-weighted. The intended effect is to reduce the gradient contribution of likely mislabeled train nodes without forcing potentially wrong target correction.

## Success Criteria

Offline gate scan success requires:

- gated noisy precision above 0.55
- real GB precision higher than strength-matched no-ball by at least 0.05
- real GB precision higher than strength-matched random by at least 0.10
- gated ratio between 0.05 and 0.25
- gated clean ratio not above 0.20

Training smoke success requires:

- best validation checkpoint comes from the GB phase, not warmup
- real GB reweight test accuracy is above GCN or v1
- real GB beats random and no-ball controls
- gated precision remains meaningfully above the random noise rate

## Offline Gate Scan Result

The offline scan wrote 126 rows to `gate_scan.csv` and reported PASS in `gate_scan_summary.md`.

Best passing setting:

| top_ratio | min_ball_conf | gated_ratio | precision | recall | clean_ratio | gain_vs_no_ball | gain_vs_random |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.10 | 0.10 | 0.0571 | 0.7500 | 0.1250 | 0.0217 | 0.2500 | 0.2500 |

This result says the offline GB gate can find a small suspicious subset with high precision under this run.

## Selected Training Smoke Result

The selected-threshold training smoke ran:

- `gcn`
- `orgingb_gcn_reweight`
- `orgingb_gcn_reweight_random`
- `orgingb_gcn_reweight_noball`

All four methods reported:

| method | test acc |
| --- | ---: |
| gcn | 70.50 |
| orgingb_gcn_reweight | 70.50 |
| orgingb_gcn_reweight_random | 70.50 |
| orgingb_gcn_reweight_noball | 70.50 |

The training diagnostics show the selected real-GB reweight run did not satisfy the smoke standard:

| method | best_phase | gated_ratio | gated_noisy_precision | gated_noisy_recall | gated_clean_ratio | weight_mean | weight_min |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| orgingb_gcn_reweight | warmup | 0.0571 | 0.5000 | 0.0833 | 0.0435 | 0.9821 | 0.5000 |
| orgingb_gcn_reweight_random | warmup | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| orgingb_gcn_reweight_noball | warmup | 0.1000 | 0.6429 | 0.1875 | 0.0543 | 0.9855 | 0.5000 |

## Key Discrepancy

The offline scan selected `top_ratio=0.1, min_ball_conf=0.1` because real GB reached precision 0.75 and beat both controls. However, the protocol-matched training diagnostics for the selected config showed real GB precision 0.50, while the no-ball control reached 0.6429.

This means the offline diagnostic is useful as a probe, but it is not yet sufficient as a training-selection criterion. The most likely explanation is that the offline script and predictor training path are not perfectly protocol-matched in model initialization, warmup trajectory, ball construction state, or random-control timing. Until this is resolved, the offline PASS should not be treated as evidence that ordinary GB reweight improves denoising.

## Final Decision

Do not expand this OrginGB-GCN v2 reweight setting to Cora/CiteSeer/PubMed 5-seed formal experiments yet.

This smoke fails the stronger training criterion because:

- best phase is still `warmup`
- test accuracy does not improve over GCN
- real GB does not beat no-ball in the protocol-matched diagnostics
- the offline selection result is not reproduced by the training pipeline

Ordinary GB reweight can be retained as a diagnostic feature, but it should not be the next main denoising mechanism unless the offline scan is made protocol-matched and then shows a stable advantage over no-ball/random controls.

## Next Recommendation

Before trying another full training expansion, either:

1. Make the gate scan reuse the exact predictor warmup and GB construction path, then rerun the selected-threshold test.
2. Or pause ordinary GB as the main mechanism and move to a graph-aware GB variant where ball confidence is combined with graph-neighborhood label conflict, instead of relying on ordinary feature-space balls alone.
