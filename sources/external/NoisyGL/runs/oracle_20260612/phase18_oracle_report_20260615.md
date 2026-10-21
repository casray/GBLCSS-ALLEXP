# Phase 18 Clean-Label Oracle Report

Date: 2026-06-15

## Research Record

- Hypothesis: clean-label GCN [X||H] provides a useful upper bound for how much of the noisy-label gap SoftDist closes.
- Granular-ball role: none in the oracle run. Clean labels are used only for upper-bound diagnosis, never for method training.
- Code changes: no mechanism changes.
- Config: Cora/CiteSeer/PubMed, `gcn_xh_ce`, `noise_type=clean`, `noise_rate=0.0`, percent split 0.2/0.1/0.7, seeds `0-4`, GPU `cuda:0`.
- Success standard: produce gap-closure numbers for paper context, not a method-selection gate.
- Raw outputs: `oracle_gcn_xh_clean_summary.csv`.
- Aggregates: `phase18_oracle_aggregate.csv`, `phase18_gap_closure.csv`.

## Results

Gap closure is `(SoftDist noisy - GCN noisy) / (clean oracle - GCN noisy)`.

| Dataset | Clean GCN [X||H] oracle | SoftDist noisy | GCN noisy | Gap closure |
|---|---:|---:|---:|---:|
| Cora | 0.8623 | 0.8137 | 0.7842 | 37.8% |
| CiteSeer | 0.7291 | 0.6891 | 0.6530 | 47.4% |
| PubMed | 0.8614 | 0.8320 | 0.8392 | -32.6% |

## Decision

Cora and CiteSeer support the claim that SoftDist closes a meaningful fraction of the noisy-label gap. PubMed is a boundary case: SoftDist is below noisy GCN, so gap closure is negative. The paper should report all three and use PubMed to delimit the method, not hide it.
