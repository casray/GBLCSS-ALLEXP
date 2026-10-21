# SoftDist-Linear Ablation Report

Protocol: Cora/CiteSeer/PubMed, uniform 0.4, train-only noise, percent split 0.2/0.1/0.7, seeds 0-4.

## Mean ± Std

| Dataset | Method | Test Acc | n |
|---|---|---:|---:|
| amazonpho | SoftDist-Linear beta=0.5 | 0.8646±0.0037 | 5 |
| amazonpho | SoftDist-Linear shuffled | 0.8161±0.0081 | 5 |
| amazonpho | Linear+CE | 0.8276±0.0111 | 5 |
| citeseer | SoftDist-Linear beta=0.5 | 0.5488±0.0336 | 5 |
| citeseer | SoftDist-Linear shuffled | 0.4847±0.0205 | 5 |
| citeseer | SoftDist-MLP beta=0.5 | 0.5383±0.0275 | 5 |
| citeseer | SoftDist-MLP shuffled | 0.4235±0.0274 | 5 |
| citeseer | Linear+CE | 0.5012±0.0149 | 5 |
| citeseer | MLP+CE | 0.4350±0.0302 | 5 |
| coauthorcs | SoftDist-Linear beta=0.5 | 0.9103±0.0030 | 5 |
| coauthorcs | SoftDist-Linear shuffled | 0.8933±0.0011 | 5 |
| coauthorcs | Linear+CE | 0.8959±0.0018 | 5 |
| cora | SoftDist-Linear beta=0.5 | 0.5406±0.0226 | 5 |
| cora | SoftDist-Linear shuffled | 0.4784±0.0159 | 5 |
| cora | SoftDist-MLP beta=0.5 | 0.5559±0.0150 | 5 |
| cora | SoftDist-MLP shuffled | 0.4363±0.0162 | 5 |
| cora | Linear+CE | 0.4786±0.0130 | 5 |
| cora | MLP+CE | 0.4424±0.0262 | 5 |
| pubmed | SoftDist-Linear beta=0.5 | 0.8003±0.0057 | 5 |
| pubmed | SoftDist-Linear shuffled | 0.7815±0.0045 | 5 |
| pubmed | SoftDist-MLP beta=0.5 | 0.7190±0.0088 | 5 |
| pubmed | SoftDist-MLP shuffled | 0.6272±0.0449 | 5 |
| pubmed | Linear+CE | 0.7935±0.0045 | 5 |
| pubmed | MLP+CE | 0.6377±0.0440 | 5 |

## Paired Delta Summary

| Dataset | Comparison | Mean Delta (pp) | Positive Seeds | n |
|---|---|---:|---:|---:|
| amazonpho | softdist_linear_vs_linear_ce | +3.70 | 5 | 5 |
| amazonpho | softdist_linear_vs_linear_shuffled | +4.85 | 5 | 5 |
| citeseer | softdist_linear_vs_linear_ce | +4.76 | 5 | 5 |
| citeseer | softdist_linear_vs_linear_shuffled | +6.41 | 5 | 5 |
| citeseer | softdist_linear_vs_mlp_ce | +11.38 | 5 | 5 |
| citeseer | softdist_mlp_vs_mlp_ce | +10.34 | 5 | 5 |
| citeseer | softdist_mlp_vs_mlp_shuffled | +11.48 | 5 | 5 |
| citeseer | softdist_mlp_vs_softdist_linear | -1.05 | 1 | 5 |
| coauthorcs | softdist_linear_vs_linear_ce | +1.44 | 5 | 5 |
| coauthorcs | softdist_linear_vs_linear_shuffled | +1.70 | 5 | 5 |
| cora | softdist_linear_vs_linear_ce | +6.20 | 5 | 5 |
| cora | softdist_linear_vs_linear_shuffled | +6.22 | 5 | 5 |
| cora | softdist_linear_vs_mlp_ce | +9.82 | 5 | 5 |
| cora | softdist_mlp_vs_mlp_ce | +11.35 | 5 | 5 |
| cora | softdist_mlp_vs_mlp_shuffled | +11.96 | 5 | 5 |
| cora | softdist_mlp_vs_softdist_linear | +1.53 | 4 | 5 |
| pubmed | softdist_linear_vs_linear_ce | +0.68 | 4 | 5 |
| pubmed | softdist_linear_vs_linear_shuffled | +1.88 | 5 | 5 |
| pubmed | softdist_linear_vs_mlp_ce | +16.26 | 5 | 5 |
| pubmed | softdist_mlp_vs_mlp_ce | +8.13 | 5 | 5 |
| pubmed | softdist_mlp_vs_mlp_shuffled | +9.18 | 5 | 5 |
| pubmed | softdist_mlp_vs_softdist_linear | -8.13 | 0 | 5 |

## Completeness

- Observed rows: 120 / expected 150
- Missing rows: 30
- Paired delta rows: 110

## Interpretation Guardrails

- If SoftDist-Linear beats Linear+CE and shuffled, the soft target mechanism works without nonlinear classifier capacity.
- If SoftDist-MLP beats SoftDist-Linear, report that the X-only student needs mild nonlinear capacity to absorb GRACE-derived soft supervision.
- Keep GRACE-H direct classifiers in the separate H-access scope table; this table is an X-only classifier-capacity ablation.
