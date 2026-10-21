# MLP-54 Teacher Noise-Spectrum Report

All students are identical two-layer X-only MLPs. Teacher routes are
retrained within each seed; no setting is selected by outcome.

## Uniform 0.2

| Teacher | Student inference | Cora | CiteSeer | CoraFull |
|---|---|---:|---:|---:|
| GCN(X)+CE (GLNN-style) | X-only | 82.69±1.03 | 70.30±1.04 | 63.36±0.34 |
| Random-group consensus teacher | X-only | 82.50±1.43 | 71.76±0.92 | 63.10±0.87 |
| kNN consensus teacher (exclude self) | X-only | 83.28±2.00 | 71.60±1.62 | 62.69±0.42 |
| Granular-ball consensus teacher | X-only | 83.76±0.96 | 71.41±1.13 | 62.07±0.69 |
| NRGNN teacher | X-only | 83.01±0.80 | 73.04±0.79 | 60.25±0.35 |

## Uniform 0.4

| Teacher | Student inference | Cora | CiteSeer | CoraFull |
|---|---|---:|---:|---:|
| GCN(X)+CE (GLNN-style) | X-only | 77.95±2.62 | 66.09±1.38 | 58.74±0.82 |
| Random-group consensus teacher | X-only | 78.81±1.63 | 68.60±1.14 | 58.98±0.41 |
| kNN consensus teacher (exclude self) | X-only | 81.61±0.74 | 69.65±1.61 | 58.58±0.55 |
| Granular-ball consensus teacher | X-only | 80.60±0.69 | 69.40±1.08 | 58.22±0.69 |
| NRGNN teacher | X-only | 81.07±0.55 | 70.87±1.06 | 57.71±0.58 |

## Uniform 0.6

| Teacher | Student inference | Cora | CiteSeer | CoraFull |
|---|---|---:|---:|---:|
| GCN(X)+CE (GLNN-style) | X-only | 65.50±3.28 | 53.30±4.42 | 51.63±2.02 |
| Random-group consensus teacher | X-only | 68.51±1.82 | 61.07±2.76 | 52.20±0.72 |
| kNN consensus teacher (exclude self) | X-only | 71.80±2.63 | 62.90±2.26 | 52.65±0.93 |
| Granular-ball consensus teacher | X-only | 71.27±2.46 | 62.28±2.30 | 51.82±0.86 |
| NRGNN teacher | X-only | 74.81±1.17 | 66.62±1.27 | 53.55±1.00 |

## Pair 0.4

| Teacher | Student inference | Cora | CiteSeer | CoraFull |
|---|---|---:|---:|---:|
| GCN(X)+CE (GLNN-style) | X-only | 66.48±2.91 | 54.06±3.08 | 49.24±1.69 |
| Random-group consensus teacher | X-only | 64.69±1.40 | 57.52±1.48 | 48.77±1.69 |
| kNN consensus teacher (exclude self) | X-only | 66.12±3.66 | 58.34±2.05 | 50.83±1.92 |
| Granular-ball consensus teacher | X-only | 66.60±0.71 | 58.26±2.11 | 49.78±2.54 |
| NRGNN teacher | X-only | 69.04±1.60 | 60.09±2.97 | 48.19±2.08 |

## Instance 0.4

| Teacher | Student inference | Cora | CiteSeer | CoraFull |
|---|---|---:|---:|---:|
| GCN(X)+CE (GLNN-style) | X-only | 77.03±2.05 | 64.93±2.49 | 59.03±0.84 |
| Random-group consensus teacher | X-only | 77.97±2.41 | 67.32±2.55 | 59.61±0.39 |
| kNN consensus teacher (exclude self) | X-only | 80.64±2.10 | 69.05±1.91 | 59.39±0.81 |
| Granular-ball consensus teacher | X-only | 80.11±1.88 | 67.88±2.18 | 58.86±0.79 |
| NRGNN teacher | X-only | 78.36±2.56 | 70.23±1.71 | 58.08±0.41 |

## Mean-cell wins

| Teacher | Best/tied-best cells (out of 15 cells) |
|---|---:|
| GCN(X)+CE (GLNN-style) | 1/15 |
| Random-group consensus teacher | 2/15 |
| kNN consensus teacher (exclude self) | 3/15 |
| Granular-ball consensus teacher | 1/15 |
| NRGNN teacher | 8/15 |

## Ball-teacher paired deltas

| Setting | Dataset | Comparison | Mean delta (pp) | Positive seeds |
|---|---|---|---:|---:|
| uniform 0.2 | cora | ball_vs_mlp_distill_gcn | +1.07 | 3/5 |
| uniform 0.2 | cora | ball_vs_mlp_distill_knn_xh | +0.47 | 3/5 |
| uniform 0.2 | cora | ball_vs_mlp_distill_nrgnn | +0.75 | 3/5 |
| uniform 0.2 | cora | ball_vs_mlp_distill_softdist_xh_shuffled | +1.26 | 4/5 |
| uniform 0.2 | citeseer | ball_vs_mlp_distill_gcn | +1.11 | 5/5 |
| uniform 0.2 | citeseer | ball_vs_mlp_distill_knn_xh | -0.19 | 2/5 |
| uniform 0.2 | citeseer | ball_vs_mlp_distill_nrgnn | -1.63 | 0/5 |
| uniform 0.2 | citeseer | ball_vs_mlp_distill_softdist_xh_shuffled | -0.35 | 2/5 |
| uniform 0.2 | corafull | ball_vs_mlp_distill_gcn | -1.29 | 0/5 |
| uniform 0.2 | corafull | ball_vs_mlp_distill_knn_xh | -0.63 | 1/5 |
| uniform 0.2 | corafull | ball_vs_mlp_distill_nrgnn | +1.81 | 5/5 |
| uniform 0.2 | corafull | ball_vs_mlp_distill_softdist_xh_shuffled | -1.03 | 1/5 |
| uniform 0.4 | cora | ball_vs_mlp_distill_gcn | +2.65 | 4/5 |
| uniform 0.4 | cora | ball_vs_mlp_distill_knn_xh | -1.01 | 0/5 |
| uniform 0.4 | cora | ball_vs_mlp_distill_nrgnn | -0.46 | 2/5 |
| uniform 0.4 | cora | ball_vs_mlp_distill_softdist_xh_shuffled | +1.79 | 4/5 |
| uniform 0.4 | citeseer | ball_vs_mlp_distill_gcn | +3.31 | 5/5 |
| uniform 0.4 | citeseer | ball_vs_mlp_distill_knn_xh | -0.25 | 2/5 |
| uniform 0.4 | citeseer | ball_vs_mlp_distill_nrgnn | -1.47 | 1/5 |
| uniform 0.4 | citeseer | ball_vs_mlp_distill_softdist_xh_shuffled | +0.80 | 4/5 |
| uniform 0.4 | corafull | ball_vs_mlp_distill_gcn | -0.52 | 1/5 |
| uniform 0.4 | corafull | ball_vs_mlp_distill_knn_xh | -0.36 | 1/5 |
| uniform 0.4 | corafull | ball_vs_mlp_distill_nrgnn | +0.52 | 4/5 |
| uniform 0.4 | corafull | ball_vs_mlp_distill_softdist_xh_shuffled | -0.75 | 1/5 |
| uniform 0.6 | cora | ball_vs_mlp_distill_gcn | +5.77 | 5/5 |
| uniform 0.6 | cora | ball_vs_mlp_distill_knn_xh | -0.53 | 2/5 |
| uniform 0.6 | cora | ball_vs_mlp_distill_nrgnn | -3.54 | 0/5 |
| uniform 0.6 | cora | ball_vs_mlp_distill_softdist_xh_shuffled | +2.77 | 5/5 |
| uniform 0.6 | citeseer | ball_vs_mlp_distill_gcn | +8.98 | 5/5 |
| uniform 0.6 | citeseer | ball_vs_mlp_distill_knn_xh | -0.62 | 1/5 |
| uniform 0.6 | citeseer | ball_vs_mlp_distill_nrgnn | -4.34 | 0/5 |
| uniform 0.6 | citeseer | ball_vs_mlp_distill_softdist_xh_shuffled | +1.21 | 4/5 |
| uniform 0.6 | corafull | ball_vs_mlp_distill_gcn | +0.19 | 2/5 |
| uniform 0.6 | corafull | ball_vs_mlp_distill_knn_xh | -0.83 | 0/5 |
| uniform 0.6 | corafull | ball_vs_mlp_distill_nrgnn | -1.73 | 0/5 |
| uniform 0.6 | corafull | ball_vs_mlp_distill_softdist_xh_shuffled | -0.38 | 1/5 |
| pair 0.4 | cora | ball_vs_mlp_distill_gcn | +0.12 | 2/5 |
| pair 0.4 | cora | ball_vs_mlp_distill_knn_xh | +0.47 | 2/5 |
| pair 0.4 | cora | ball_vs_mlp_distill_nrgnn | -2.45 | 1/5 |
| pair 0.4 | cora | ball_vs_mlp_distill_softdist_xh_shuffled | +1.91 | 4/5 |
| pair 0.4 | citeseer | ball_vs_mlp_distill_gcn | +4.20 | 5/5 |
| pair 0.4 | citeseer | ball_vs_mlp_distill_knn_xh | -0.08 | 2/5 |
| pair 0.4 | citeseer | ball_vs_mlp_distill_nrgnn | -1.83 | 1/5 |
| pair 0.4 | citeseer | ball_vs_mlp_distill_softdist_xh_shuffled | +0.75 | 4/5 |
| pair 0.4 | corafull | ball_vs_mlp_distill_gcn | +0.54 | 4/5 |
| pair 0.4 | corafull | ball_vs_mlp_distill_knn_xh | -1.06 | 2/5 |
| pair 0.4 | corafull | ball_vs_mlp_distill_nrgnn | +1.59 | 4/5 |
| pair 0.4 | corafull | ball_vs_mlp_distill_softdist_xh_shuffled | +1.01 | 4/5 |
| instance 0.4 | cora | ball_vs_mlp_distill_gcn | +3.07 | 5/5 |
| instance 0.4 | cora | ball_vs_mlp_distill_knn_xh | -0.54 | 2/5 |
| instance 0.4 | cora | ball_vs_mlp_distill_nrgnn | +1.74 | 5/5 |
| instance 0.4 | cora | ball_vs_mlp_distill_softdist_xh_shuffled | +2.13 | 5/5 |
| instance 0.4 | citeseer | ball_vs_mlp_distill_gcn | +2.95 | 5/5 |
| instance 0.4 | citeseer | ball_vs_mlp_distill_knn_xh | -1.17 | 1/5 |
| instance 0.4 | citeseer | ball_vs_mlp_distill_nrgnn | -2.35 | 0/5 |
| instance 0.4 | citeseer | ball_vs_mlp_distill_softdist_xh_shuffled | +0.56 | 4/5 |
| instance 0.4 | corafull | ball_vs_mlp_distill_gcn | -0.17 | 1/5 |
| instance 0.4 | corafull | ball_vs_mlp_distill_knn_xh | -0.53 | 2/5 |
| instance 0.4 | corafull | ball_vs_mlp_distill_nrgnn | +0.78 | 5/5 |
| instance 0.4 | corafull | ball_vs_mlp_distill_softdist_xh_shuffled | -0.76 | 1/5 |
