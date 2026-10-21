# Stage B Formal Gate: Cora Seed 0

## Research Hypothesis

B1 should improve granular-ball quality over B0 when label-free granular-ball compactness is applied after warm-up. B3 and B4 are assignment controls and should not match B1 if true local structure is useful.

## Granular-Ball Role

Granular balls are used as a label-free local structure prior. They are not used as hard labels, hard positive labels, or label-derived training targets.

## Method Motivation

This gate tests whether true granular-ball assignment can provide a useful compactness constraint beyond negative-sample-free SSL alone and beyond shuffled or size-matched random ball assignments.

## Code Changes

No code changes were made during this run.

## Configuration

- Dataset: Cora
- Seed: 0
- Methods: B0, B1, B3, B4
- Config file: `configs/stage_b.yaml`
- Epochs: 1000
- Warm-up epochs: 300
- SSL objective: negative-free
- Ball space: H
- Reliable ball min size: 3
- Reliable ball max size: 256
- Reliable ball AD percentile: 50

The initial Cora download timed out. Missing raw Cora files were copied from the local Stage A cache into `gb_denoise_stage_b_compactness/data/Cora/Cora/raw/`.

## Expected Mechanism

After warm-up, B1 should build reliable balls and pull node embeddings toward detached reliable ball centers. B3/B4 should control whether any observed change comes from true assignment rather than arbitrary grouping.

## Success Standard

B1 should improve ball diagnostics over B0 and beat B3/B4 without worsening collapse diagnostics. Improvements must be interpreted only as preliminary because this run uses one seed.

## Failure Standard

The gate fails if compactness is not activated, if B1 is indistinguishable from B0, or if all assignment controls match B1.

## Results

| method | linear_eval_acc | num_balls | mean_radius | mean_average_angular_distance | effective_rank | mean_pairwise_cosine | label_purity_diagnostic |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | 0.344 | 1 | 0.2063869387 | 0.0289594065 | 2.4701464176 | 0.998569 | 0.3020679468 |
| B1 | 0.344 | 1 | 0.2063869387 | 0.0289594065 | 2.4701464176 | 0.998569 | 0.3020679468 |
| B3 | 0.344 | 1 | 0.2063869387 | 0.0289594065 | 2.4701464176 | 0.998569 | 0.3020679468 |
| B4 | 0.344 | 1 | 0.2063869387 | 0.0289594065 | 2.4701464176 | 0.998569 | 0.3020679468 |

Paired deltas against B0 are all zero for B1, B3, and B4.

History inspection showed:

- B0 compact nonzero epochs: 0
- B1 compact nonzero epochs: 0
- B3 compact nonzero epochs: 0
- B4 compact nonzero epochs: 0
- Maximum `covered_nodes`: 0 for every method

## Interpretation

This run does not test the intended B1 mechanism. The final representation produced one large ball, and the default reliable-ball size gate rejected it because `reliable_ball_max_size=256` is far smaller than the Cora node count. As a result, B1/B3/B4 did not apply compactness and collapsed to the same optimization path as B0.

This is not evidence that granular-ball compactness is ineffective. It is evidence that the current reliable-ball generation/gating protocol can disable the compactness term on Cora.

## Final Decision

Modify before drawing conclusions.

Recommended next experiment:

1. Add a diagnostic precheck that fails a run when `covered_nodes` remains 0 after warm-up.
2. Test a revised ball generation policy that splits over-large balls even when average angular distance is below threshold.
3. Keep B0/B1/B3/B4 paired on the same seeds after the compactness term is confirmed active.
