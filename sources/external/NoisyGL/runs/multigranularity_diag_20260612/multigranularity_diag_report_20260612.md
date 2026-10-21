# Multigranularity Consistency Diagnostic

## Protocol

- Dataset: Cora, CiteSeer
- Split: percent 0.2/0.1/0.7
- Noise: uniform 0.4
- Seeds: 0, 1, 2
- Balls: label-blind geometric GRACE-H balls, fine 256/8, mid 64/32, coarse 16/128
- Clean labels are used only for diagnostics.

## Structural Difference From Closed Directions

- vs E1 adaptive granularity: construction is purely geometric and label-blind; labels only vote after balls are fixed.
- vs binary gate: the gate signal is cross-scale consistency, not same-scale vote count or margin.
- vs dynamic alpha: no model prediction feedback loop; this is an offline diagnostic.
- vs unlabeled expansion: only train-node correction candidates are evaluated; no unlabeled supervision is added.

## Predeclared Risks

- Gate-class mechanisms may repeat the same NCR/CDR tradeoff failure.
- Nested geometric correlation may inflate agreement and provide little protection.
- CiteSeer may degenerate into almost no useful correction.

## Aggregate Results

| Dataset | Rule | Baseline NCR | Rule NCR | NCR loss | Baseline CDR | Rule CDR | CDR reduction | Protection precision | Protected | Execute mass | Coarse-mid agree | Pass |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| citeseer | all_consistent | 0.6029 | 0.3400 | 0.4365 | 0.2916 | 0.0835 | 0.7139 | 0.4358 | 200.3 | 136.3 | 0.7944 | 0 |
| citeseer | coarse_mid_consistent | 0.6029 | 0.5702 | 0.0531 | 0.2916 | 0.2048 | 0.2983 | 0.5235 | 69.0 | 267.7 | 0.7944 | 0 |
| citeseer | weighted_agreement | 0.6029 | 0.4922 | 0.1832 | 0.2916 | 0.1949 | 0.3319 | 0.4474 | 223.3 | 246.0 | 0.7944 | 0 |
| cora | all_consistent | 0.6821 | 0.3715 | 0.4551 | 0.2061 | 0.0425 | 0.7961 | 0.3490 | 159.0 | 98.3 | 0.7190 | 0 |
| cora | coarse_mid_consistent | 0.6821 | 0.6085 | 0.1078 | 0.2061 | 0.1107 | 0.4587 | 0.4442 | 72.0 | 185.3 | 0.7190 | 0 |
| cora | weighted_agreement | 0.6821 | 0.5486 | 0.1956 | 0.2061 | 0.1240 | 0.3989 | 0.3676 | 175.0 | 181.8 | 0.7190 | 0 |

## Coarse-Mid Agreement Summary

| Dataset | Rule | Both-vote count | Agree count | Agreement rate | Coarse abstain | Mid abstain |
|---|---|---:|---:|---:|---:|---:|
| citeseer | all_consistent | 336.7 | 267.7 | 0.7944 | 0.0000 | 0.0000 |
| citeseer | coarse_mid_consistent | 336.7 | 267.7 | 0.7944 | 0.0000 | 0.0000 |
| citeseer | weighted_agreement | 336.7 | 267.7 | 0.7944 | 0.0000 | 0.0000 |
| cora | all_consistent | 257.3 | 185.3 | 0.7190 | 0.0000 | 0.0000 |
| cora | coarse_mid_consistent | 257.3 | 185.3 | 0.7190 | 0.0000 | 0.0000 |
| cora | weighted_agreement | 257.3 | 185.3 | 0.7190 | 0.0000 | 0.0000 |

## Gate Decision

No rule passes all three gates on both datasets. By the preregistered rule, the multigranularity literature-mining line is closed and no training version should be run.

## Required Experiment Record

- Hypothesis: cross-scale agreement protects class-boundary clean labels while preserving stable-region noisy-label correction.
- Granular-ball role: label-blind local semantic neighborhoods and cross-scale confidence signal.
- Code change: added `scripts/diagnose_multigranularity.py` and unit tests for voting/rule diagnostics.
- Config change: none; existing SoftDist XH anchor behavior is untouched.
- Success standard: CDR reduction >= 30%, NCR loss <= 10%, protection precision >= 70% on both datasets for one rule.
- Failure standard: any gate violation blocks training implementation.
