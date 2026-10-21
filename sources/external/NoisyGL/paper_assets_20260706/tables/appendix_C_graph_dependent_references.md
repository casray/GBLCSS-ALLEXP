# Appendix C. Graph-dependent references

**Inference requires the graph -- different deployment setting; numbers are same-protocol references, not paired comparisons.**

Important protocol audit: the old `phase24_nrgnn_xh_20260616`, `tfr_comparison_20260616`, and `coauthorcs_baseline_20260617` graph-dependent numbers are invalidated. They were generated before train-only `eligible_mask` support and their raw files show full-graph label corruption (`modified_count` is close to 40% of all nodes, while `modified_train_count` is only the train subset). The table below uses clean reruns with `scripts/run_standard_downstream.py --noise_scope train`, and graph-dependent rows pass `modified_val_count=0`, `modified_test_count=0`.

## Clean Table C candidate (uniform 0.4, train-only noise)

| Method | Inference graph? | Cora | CiteSeer | Coauthor-CS | PubMed | Amazon-Photo |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| GCN(X) | yes | 78.97±1.63 | 64.53±1.54 | 92.06±0.34 | 83.90±0.65 | 90.77±0.43 |
| GCN[X‖H] | yes | 80.70±1.54 | 67.48±1.39 | 92.05±0.19 | 83.28±0.52 | 91.87±0.64 |
| NRGNN | yes | 81.76±1.28 | 71.03±1.01 | 93.56±0.11 | 80.69±0.66 | 89.02±1.08 |
| NRGNN[X‖H] | yes | 81.47±1.28 | 70.74±1.04 | 93.72±0.18 | 83.29±0.41 | 89.53±1.53 |
| TFR-GCN (adapted) | yes | 78.77±0.87 | 68.71±1.71 | 77.11±4.36 | 82.24±0.83 | 35.36±9.43 |
| ProCon (adapted) | yes | 76.93±2.20 | 61.44±3.23 | 86.07±6.79 | 48.15±12.42 | 20.74±4.82 |
| RNCGLN | yes | 68.32±2.35 | 58.32±1.98 | 0.00† | 69.84† | 72.11±7.74 |
| RTGNN | yes | 75.43±1.35 | 66.54±1.75 | 93.19±1.09 | 79.89±0.73 | 76.09±28.60 |
| CLNode | yes | 73.32±2.84 | 55.79±2.20 | 91.91±0.20 | 83.91±0.56 | 91.41±0.96 |
| CRGNN | yes | 76.56±1.94 | 63.76±1.26 | 90.87±0.20 | 83.42±0.64 | 54.64±28.34 |
| LCSS-distill X-only student | no | 80.60±0.69 | 69.40±1.08 | 93.75±0.22 | 85.01±0.93 | 90.49±1.05 |

## Feasibility / exclusions

- † marks seed0-only feasibility cells: RNCGLN on PubMed is slow; RNCGLN on Coauthor-CS collapses to 0.00 in seed0.
- TFR-GCN/ProCon rows on PubMed, Coauthor-CS, and Amazon-Photo are adapted template runs, not official per-dataset tuned results.
- Amazon-Photo RTGNN/CRGNN have very large standard deviations because seed0 is near-collapse while later seeds partially recover; these rows are kept to avoid hiding instability.
- MLP-9 distill rows for Cora/CiteSeer/PubMed do not carry noise-count columns in their raw file; MLP-16 distill rows for Coauthor-CS/Amazon-Photo do carry those columns and pass the same `modified_val/test=0` audit.

Raw and aggregate files:

- `runs/mlp18_graphdomain_tablec_20260708/clean_tablec_raw.csv`
- `runs/mlp18_graphdomain_tablec_20260708/clean_tablec_aggregate.csv`
- `runs/mlp18_graphdomain_tablec_20260708/clean_tablec_report.md`
