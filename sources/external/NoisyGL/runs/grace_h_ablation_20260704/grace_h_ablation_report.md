# GRACE-H Direct Classification Ablation Report

Protocol: Cora/CiteSeer/PubMed, uniform 0.4, train-only noise, percent split 0.2/0.1/0.7, seeds 0-4.

## Mean ± Std

| Dataset | Method | Test Acc | n |
|---|---|---:|---:|
| citeseer | SoftDist-MLP beta=0.5 | 0.5383±0.0275 | 5 |
| citeseer | SoftDist-MLP shuffled | 0.4235±0.0274 | 5 |
| citeseer | GRACE-H linear CE | 0.6887±0.0142 | 5 |
| citeseer | GRACE-H MLP CE | 0.6869±0.0298 | 5 |
| citeseer | MLP+CE | 0.4350±0.0302 | 5 |
| citeseer | MLP[X||H]+CE | 0.5166±0.0422 | 5 |
| cora | SoftDist-MLP beta=0.5 | 0.5559±0.0150 | 5 |
| cora | SoftDist-MLP shuffled | 0.4363±0.0162 | 5 |
| cora | GRACE-H linear CE | 0.7861±0.0128 | 5 |
| cora | GRACE-H MLP CE | 0.7309±0.0230 | 5 |
| cora | MLP+CE | 0.4424±0.0262 | 5 |
| cora | MLP[X||H]+CE | 0.4854±0.0417 | 5 |
| pubmed | SoftDist-MLP beta=0.5 | 0.7190±0.0088 | 5 |
| pubmed | SoftDist-MLP shuffled | 0.6272±0.0449 | 5 |
| pubmed | GRACE-H linear CE | 0.8387±0.0077 | 5 |
| pubmed | GRACE-H MLP CE | 0.8405±0.0054 | 5 |
| pubmed | MLP+CE | 0.6377±0.0440 | 5 |
| pubmed | MLP[X||H]+CE | 0.8368±0.0052 | 5 |

## Paired Delta Summary

| Dataset | Comparison | Mean Delta (pp) | Positive Seeds | n |
|---|---|---:|---:|---:|
| citeseer | grace_h_mlp_vs_mlp_ce | +25.19 | 5 | 5 |
| citeseer | softdist_vs_grace_h_linear | -15.04 | 0 | 5 |
| citeseer | softdist_vs_grace_h_mlp | -14.85 | 0 | 5 |
| citeseer | softdist_vs_mlp_xh | +2.17 | 4 | 5 |
| citeseer | softdist_vs_shuffled | +11.48 | 5 | 5 |
| cora | grace_h_mlp_vs_mlp_ce | +28.84 | 5 | 5 |
| cora | softdist_vs_grace_h_linear | -23.02 | 0 | 5 |
| cora | softdist_vs_grace_h_mlp | -17.50 | 0 | 5 |
| cora | softdist_vs_mlp_xh | +7.05 | 5 | 5 |
| cora | softdist_vs_shuffled | +11.96 | 5 | 5 |
| pubmed | grace_h_mlp_vs_mlp_ce | +20.28 | 5 | 5 |
| pubmed | softdist_vs_grace_h_linear | -11.97 | 0 | 5 |
| pubmed | softdist_vs_grace_h_mlp | -12.15 | 0 | 5 |
| pubmed | softdist_vs_mlp_xh | -11.77 | 0 | 5 |
| pubmed | softdist_vs_shuffled | +9.18 | 5 | 5 |

## Completeness

- Observed rows: 90 / expected 90
- Missing rows: 0
- Paired delta rows: 75

## Interpretation Guardrails

- If SoftDist-MLP exceeds GRACE-H MLP on at least 2/3 datasets with >=4/5 positive seeds and >=1pp mean delta, report an attribution gain beyond direct H classification.
- If GRACE-H MLP or MLP[X||H] is stronger, report a profile trade-off: higher accuracy with H-dependent inference vs graph-free SoftDist-MLP inference.
- Keep shuffled as the ball-structure control; do not treat this ablation as a SOTA leaderboard.
