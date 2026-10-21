# Phase 39 LastFM-Asia Preregistration

Date: 2026-06-18

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Verification Status: UNVERIFIED
- Version Label: phase39_preregistration_v1

## Research Hypothesis

LastFM-Asia may provide a fourth dataset with a controlled granular-ball increment because it combines 18 classes with a medium graph size. This is a conditional hypothesis, not a presumption of success. The decisive variable is whether frozen GRACE `H` forms non-collapsed, semantically coherent balls with noise information orthogonal to an early GCN.

## Granular-Ball Role

Granular balls remain frozen local semantic units used to construct soft label distributions. They are not hard positive labels, hard negative filters, or prototype-pulling constraints.

## Method Motivation

Existing evidence supports ball-clusterability rather than graph size or class count alone as the mediator of downstream ball gains. LastFM-Asia adds a medium-size, 18-class social graph that tests this boundary without changing the SoftDist mechanism.

## Code Changes

- Add PyG `LastFMAsia` routing to NoisyGL and the upstream GRACE pretraining loader.
- Add `lastfmasia` CLI choices and percent-split dataset configuration.
- Add LastFM-specific downstream configuration files.
- Extend the existing orthogonality diagnostic with edge homophily reporting.

## Configuration Changes

- Split: percent `0.2 / 0.1 / 0.7`.
- Noise: uniform `0.4`.
- Diagnostic seeds: `0,1,2`.
- Downstream seeds, only if the diagnostic gate passes: `0,1,2,3,4`.
- GRACE: unchanged PubMed official configuration, matching the preregistered Amazon-Photo fallback tier.
- Ball parameters for diagnostic and downstream consistency: `max_balls=256`, `min_ball_size=32`, `split_improvement=0.001`.
- Clean labels are diagnostic-only. GRACE pretraining uses only `X` and `A`.

## Expected Mechanism

If GRACE `H` separates local music-interest communities into multiple sufficiently pure balls, ball-level noisy-label disagreement should retain detection power inside the GCN-clean50 subset. This conditional orthogonal signal is the prerequisite for SoftDist to add information beyond `GCN [X||H]`.

## Success Criteria

Phase 39a passes only if all three conditions hold on the 3-seed mean:

- `ball_auc_in_gcn_clean50 >= 0.70`
- `num_balls >= 8`
- `true_all_purity_weighted >= 0.65`

Only a pass starts Phase 39b. Phase 39b counts as a controlled success only if SoftDist is non-negative versus `gcn_xh_ce` and real balls beat shuffled balls on a majority of paired seeds. Mean/std and paired per-seed deltas must be reported.

## Failure Criteria

Any Phase 39a gate failure stops downstream training and records LastFM-Asia as a boundary point. A Phase 39b run is not successful if real balls do not beat shuffled balls on a majority of paired seeds, or if the apparent mean gain is within unstable seed variation.

## Planned Outputs

- Raw download/load log.
- GRACE raw pretraining log, `H.pt`, `metadata.json`, and `losses.csv`.
- H-to-graph shape sanity check.
- `orthogonality_raw.csv`, `orthogonality_summary.csv`, and Markdown diagnostic report.
- Final decision: continue to 39b, or stop and record boundary.
