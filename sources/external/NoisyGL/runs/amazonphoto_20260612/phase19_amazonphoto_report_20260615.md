# Phase 19 Amazon-Photo Report

Date: 2026-06-15

## Research Record

- Hypothesis: Amazon-Photo tests cross-domain generalization beyond citation graphs. The preregistered fallback is to reuse PubMed GRACE hyperparameters without tuning.
- Granular-ball role: same frozen-GRACE soft target mechanism as the anchor; no Amazon-specific ball parameter tuning.
- Code changes: added preregistered GRACE fallback handling for Amazon-Photo and fixed percent-split override so fixed val/test sizes do not leak into percent split runs.
- Config: Amazon-Photo, uniform noise 0.4, percent split 0.2/0.1/0.7, seeds `0-4`, GPU `cuda:0`.
- Success standard: third-dataset success requires SoftDist [X||H] >= GCN and real balls > shuffled. Otherwise report as boundary or partial result.
- Preregistration: `phase19_preregistered_protocol.md`.
- Invalid run note: the first XH run used fixed val/test sizes because of split override inheritance and was renamed to `*_INVALID_fixed_split.*`. The valid run has train 1530, val 765, test 5355.
- Raw outputs: `amazonphoto_xh_summary.csv`, `amazonphoto_standard_summary.csv`, `amazonphoto_ball_diagnostics.csv`.
- Aggregates: `phase19_amazonphoto_aggregate.csv`, `phase19_amazonphoto_paired_deltas.csv`, `phase19_amazonphoto_ball_diag_aggregate.csv`.

## Results

| Method | Test acc |
|---|---:|
| GCN | 0.9072±0.0141 |
| GCN [X||H] | 0.9049±0.0145 |
| SoftDist [X||H] | 0.9051±0.0174 |
| SoftDist shuffled | 0.9066±0.0121 |
| NRGNN | 0.8954±0.0153 |

Paired deltas for SoftDist [X||H]:

| Comparison | Mean delta | Positive seeds | Win? |
|---|---:|---:|---:|
| vs GCN | -0.21pp | 2/5 | no |
| vs GCN [X||H] | +0.01pp | 3/5 | no |
| vs shuffled | -0.16pp | 1/5 | no |
| vs NRGNN | +0.97pp | 4/5 | yes |

Ball diagnostics did not show collapse: `num_balls=64`, eligible train per ball `23.91`, weighted purity `0.861`, NCR `0.834`, CDR `0.142`.

## Decision

Amazon-Photo is not a third-dataset success under the preregistered criterion because real balls do not beat shuffled and SoftDist does not beat plain GCN. It is a partial result: SoftDist beats NRGNN, but the ball-specific contribution is not supported on this dataset. Use as a cross-domain boundary case and keep the invalid fixed-split run excluded.
