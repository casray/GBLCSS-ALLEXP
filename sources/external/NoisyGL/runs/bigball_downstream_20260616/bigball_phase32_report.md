# Phase 32: Big-Ball Scaling Diagnostic and Downstream Check

Date: 2026-06-16  
Protocol: percent split 0.2 / 0.1 / 0.7, uniform noise 0.4, frozen GRACE H, no anchor config changes.

## Phase 32a Geometry Scan

The geometry-only scan rejects the strongest version of the earlier "large graphs cannot be split into meaningful balls" hypothesis. With runtime-only construction parameters, DBLP/PubMed/Amazon-Photo all produce non-collapsed, high-purity balls.

| Dataset | Best params `max / min / split` | num_balls | true_all_purity | mean_train_per_ball | Geometry verdict |
|---|---:|---:|---:|---:|---|
| Cora reference | 64 / 32 / 0.05 | 52.33 | 0.7871 | 10.34 | default small-graph anchor |
| CiteSeer reference | 64 / 32 / 0.05 | 64.00 | 0.6991 | 10.39 | default small-graph anchor |
| DBLP | 256 / 32 / 0.001 | 256.00 | 0.8377 | 13.84 | construction rescued |
| PubMed | 256 / 32 / 0.005 | 256.00 | 0.8290 | 15.40 | construction rescued |
| Amazon-Photo | 256 / 32 / 0.001 | 129.00 | 0.8698 | 11.87 | construction healthy |

Raw scan outputs:

- `runs/ball_scaling_diag_20260616/ball_scaling_raw.csv`
- `runs/ball_scaling_diag_20260616/ball_scaling_summary.csv`
- `runs/ball_scaling_diag_20260616/ball_scaling_report.md`

Interpretation: default `max_balls=64, min_ball_size=32, split_improvement=0.05` was too restrictive for DBLP/PubMed. The original DBLP `num_balls=1` collapse was a construction-parameter mismatch, not proof that the frozen H had no clusterable geometry.

## Phase 32b Downstream Check

Downstream validation does not support adding DBLP/PubMed/Amazon-Photo as successful ball-contribution datasets. High geometric purity did not translate into a robust real-ball advantage over shuffled balls.

| Dataset | Params | Seeds | SoftDist [X\|\|H] | GCN [X\|\|H] | Shuffled | Real - GCN | Real - Shuffled | Verdict |
|---|---|---:|---:|---:|---:|---:|---:|---|
| DBLP | 256 / 32 / 0.001 | 0 | 0.8300 | 0.8300 | 0.8383 | +0.00pp | -0.83pp | fail |
| PubMed | 256 / 32 / 0.005 | 0 | 0.8213 | 0.8208 | 0.8250 | +0.05pp | -0.37pp | fail |
| Amazon-Photo | 256 / 32 / 0.001 | 0-4 | 0.9085 ± 0.0123 | 0.9049 ± 0.0145 | 0.9089 ± 0.0140 | +0.35pp | -0.04pp | fail as independent ball evidence |

Amazon-Photo has a small mean gain over `GCN [X||H]`, but it does not beat shuffled balls. Therefore the safe interpretation is that H and training noise randomness explain most of the gain; the real granular-ball structure is not independently supported on this dataset.

## Decision

Phase 32 refines the boundary:

- Construction-level claim: large-graph ball collapse can be fixed by scaling construction parameters.
- Method-level claim: scaled balls still do not provide a reliable downstream label-correction increment on DBLP/PubMed/Amazon-Photo.
- Paper impact: do not add these datasets as main success cases. Use the result as boundary analysis: `H` can be useful as classifier input and can even form high-purity geometric partitions, but the ball-induced noisy-label target may still be redundant with or weaker than the H-augmented classifier/shuffled control on larger or cross-domain graphs.

No 5-seed DBLP/PubMed expansion is warranted under the pre-registered Phase 32b gate because seed0 real balls did not beat shuffled.
