# OrginGB-GCN v2 Protocol-Matched Gate Scan Report

## Basic Info

- Date: 2026-06-03
- Dataset: Cora
- Noise: uniform 0.4
- Seed: 3000
- Device: cuda:0
- Output directory: `runs/orgingb_gate_scan_protocol_matched_20260603`
- Selected training config:
  - `suspicion_top_ratio: 0.2`
  - `min_ball_conf: 0.0`
  - `beta: 0.5`
  - `min_weight: 0.3`
  - `warmup_epochs: 30`

## Research Hypothesis

The previous offline gate scan might have overestimated ordinary GB's denoising value because the scan script was not fully protocol-matched to `single_exp.py` and `total_exp.py`. This experiment tests whether a protocol-matched scan still finds an independent GB suspicious-node signal, and whether the selected setting improves training.

## GB Role

GB is still used only as a local suspicion prior for sample reweighting. It does not change the target label and does not use clean labels in training, target construction, weights, gates, or ball construction.

## Method Motivation

The earlier scan selected `top_ratio=0.1, min_ball_conf=0.1` with real-GB precision 0.75, but the training pipeline reported real-GB precision 0.50 under the same threshold. The most likely mismatch was that the scan script did not construct the dataset exactly like the main experiment entry points.

The confirmed issue was that the scan script did not pass `modify.add_self_loop` from `_dataset/cora.yaml`. Cora config sets `add_self_loop: false`, while `Dataset` defaults to `add_self_loop=True`. This changed the warmup graph, hidden embeddings, and ordinary GB partition.

## Code Changes

- `scripts/diagnose_orgingb_gate_scan.py`
  - Added `dataset_kwargs_from_conf` to mirror `single_exp.py` / `total_exp.py` dataset construction.
  - Added `conf_value` helper.
  - Reads GB config values from `orgingb_gcn_reweight_cora.yaml` instead of hardcoding all GB parameters.
- `tests/test_orgingb_tools.py`
  - Added a regression test ensuring the gate scan inherits dataset config fields such as `add_self_loop`, `from_npz`, and `split_type`.

## Config Changes

For the selected smoke, Cora reweight configs were changed from the old threshold:

- old: `top_ratio=0.1`, `min_ball_conf=0.1`

to the protocol-matched best passing threshold:

- new: `top_ratio=0.2`, `min_ball_conf=0.0`

Applied to:

- `orgingb_gcn_reweight`
- `orgingb_gcn_reweight_random`
- `orgingb_gcn_reweight_noball`

## Expected Mechanism

The selected gate should down-weight the top 20% most suspicious training nodes. Under protocol-matched scan, real GB should identify a noisier subset than no-ball and random controls. If this suspicion signal is useful for training, the GB phase should produce the best validation checkpoint and improve test accuracy over GCN.

## Success Criteria

Offline scan success:

- real GB gated precision above 0.55
- real GB precision higher than no-ball and random controls
- gated clean ratio not abnormally high

Training smoke success:

- `best_phase=gb`
- test accuracy above GCN or at least above previous v1/v2 smoke
- real GB beats no-ball and random controls

## Failure Criteria

The experiment fails if:

- best validation checkpoint remains in warmup
- test accuracy is unchanged from GCN
- no-ball/random match or beat real GB
- the selected gate has high precision offline but does not affect training outcome

## Protocol-Matched Offline Gate Scan Result

The scan wrote 126 rows and reported PASS.

Best setting by the current selection rule:

| setting | real precision | no-ball precision | random precision | real gated ratio | real recall | real clean ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| top_ratio=0.2, min_ball_conf=0.0 | 0.6071 | 0.4643 | 0.4643 | 0.2000 | 0.3542 | 0.1196 |

This confirms that, after protocol matching, ordinary GB still has some independent suspicious-node selection value at this seed.

The old selected setting now matches the training diagnostics:

| setting | real precision | no-ball precision | random precision | num_balls | mean_ball_size |
| --- | ---: | ---: | ---: | ---: | ---: |
| top_ratio=0.1, min_ball_conf=0.1 | 0.5000 | 0.6250 | 0.3750 | 28 | 96.7143 |

This resolves the previous discrepancy: the old offline precision 0.75 came from protocol mismatch.

## Selected Training Smoke Result

The selected training smoke ran:

- `gcn`
- `orgingb_gcn_reweight`
- `orgingb_gcn_reweight_random`
- `orgingb_gcn_reweight_noball`

All methods reported the same test accuracy:

| method | test acc |
| --- | ---: |
| gcn | 70.50 |
| orgingb_gcn_reweight | 70.50 |
| orgingb_gcn_reweight_random | 70.50 |
| orgingb_gcn_reweight_noball | 70.50 |

Training diagnostics for the selected threshold:

| method | best_phase | gated_ratio | gated_noisy_precision | gated_noisy_recall | gated_clean_ratio | weight_mean | weight_min |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| orgingb_gcn_reweight | warmup | 0.2000 | 0.6071 | 0.3542 | 0.1196 | 0.9581 | 0.5000 |
| orgingb_gcn_reweight_random | warmup | 0.2000 | 0.4643 | 0.2708 | 0.1630 | 0.9332 | 0.5000 |
| orgingb_gcn_reweight_noball | warmup | 0.2000 | 0.4643 | 0.2708 | 0.1630 | 0.9830 | 0.5000 |

## Interpretation

The corrected scan changes the conclusion in a useful way:

- Ordinary GB does provide a measurable suspicious-node selection signal at this seed.
- The earlier `0.1/0.1` success was invalid because the scan was not protocol-matched.
- Even with a protocol-matched selected gate, the current reweight loss does not improve validation or test performance.

The likely reason is that the gate is too late and too weak to affect the selected model. The best checkpoint occurs at epoch 6, before GB is constructed at epoch 30. After the GB phase starts, validation never surpasses the warmup checkpoint. Therefore, better gate precision alone is not enough under the current training schedule and loss.

## Final Decision

Do not expand ordinary OrginGB-GCN v2 reweight to three datasets or five seeds as a main experiment.

This direction should be marked as:

- useful diagnostic signal: yes
- validated denoising training mechanism: no
- suitable for formal expansion now: no

## Next Recommendation

The next experiment should not merely rescan ordinary GB thresholds. A more promising next step is to keep the useful part, namely suspicious-node diagnostics, but change the mechanism:

- either construct GB earlier or refresh it during training and force evaluation of GB-phase checkpoints only as an ablation;
- or move to graph-aware GB suspicion, combining ordinary GB confidence with graph-neighborhood label conflict;
- or use GB suspicion as an auxiliary feature for selecting nodes to ignore temporarily, rather than as a smooth per-node weight.

For paper direction, ordinary GB reweight should be described as a negative/diagnostic result unless later variants show multi-seed improvement over GCN and no-ball controls.
