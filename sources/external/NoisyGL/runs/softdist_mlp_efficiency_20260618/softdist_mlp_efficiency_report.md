# SoftDist-MLP Efficiency Report

## Material Passport

- Artifact type: experiment result
- Status: COMPLETED
- Protocol: Cora/CiteSeer, uniform 0.4, seeds 0-4, ROCm `cuda:0`
- GRACE pretraining is excluded and must be reported as reusable preprocessing.

| Dataset | Method | Params | Init(s) | Ball target(s) | Classifier train(s) | Downstream total(s) | Peak alloc(MiB) | Test |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| citeseer | mlp_ce | 237574 | 0.944+/-0.038 | 0.000+/-0.000 | 2.683+/-0.118 | 3.627+/-0.111 | 161.7 | 44.53+/-4.11 |
| citeseer | softdist_mlp | 237574 | 0.035+/-0.004 | 5.739+/-0.364 | 1.521+/-0.068 | 7.296+/-0.395 | 209.0 | 52.30+/-2.84 |
| citeseer | gcn | 237446 | 0.070+/-0.013 | 0.000+/-0.000 | 3.031+/-0.066 | 3.101+/-0.061 | 258.4 | 65.30+/-1.21 |
| citeseer | softdist_gcn | 237446 | 0.032+/-0.001 | 3.765+/-0.386 | 2.442+/-0.178 | 6.239+/-0.364 | 258.5 | 67.37+/-1.13 |
| citeseer | gcn_xh_ce | 253830 | 0.081+/-0.008 | 0.000+/-0.000 | 2.468+/-0.122 | 2.550+/-0.119 | 309.0 | 67.61+/-1.81 |
| citeseer | softdist_xh | 253830 | 0.040+/-0.004 | 3.740+/-0.396 | 2.469+/-0.106 | 6.250+/-0.398 | 309.1 | 68.91+/-1.60 |
| citeseer | nrgnn | 1190988 | 6.412+/-0.082 | 0.000+/-0.000 | 31.660+/-0.440 | 38.073+/-0.455 | 505.5 | 70.27+/-1.08 |
| cora | mlp_ce | 92359 | 1.002+/-0.073 | 0.000+/-0.000 | 2.190+/-0.141 | 3.193+/-0.137 | 126.8 | 45.88+/-3.44 |
| cora | softdist_mlp | 92359 | 0.022+/-0.001 | 4.507+/-0.089 | 1.151+/-0.065 | 5.680+/-0.147 | 140.3 | 54.63+/-1.46 |
| cora | gcn | 92231 | 0.058+/-0.005 | 0.000+/-0.000 | 2.854+/-0.163 | 2.913+/-0.160 | 177.4 | 78.42+/-1.51 |
| cora | softdist_gcn | 92231 | 0.022+/-0.001 | 2.521+/-0.248 | 2.300+/-0.167 | 4.842+/-0.152 | 177.5 | 79.55+/-0.95 |
| cora | gcn_xh_ce | 100423 | 0.072+/-0.012 | 0.000+/-0.000 | 2.131+/-0.099 | 2.203+/-0.092 | 193.7 | 80.27+/-0.69 |
| cora | softdist_xh | 100423 | 0.029+/-0.004 | 2.529+/-0.120 | 2.297+/-0.149 | 4.855+/-0.242 | 193.8 | 81.37+/-0.67 |
| cora | nrgnn | 464846 | 4.425+/-0.111 | 0.000+/-0.000 | 25.041+/-0.579 | 29.466+/-0.520 | 406.4 | 81.07+/-2.30 |

## Claim Boundary

- This table supports downstream parameter/time/memory comparisons only.
- It does not support an end-to-end complexity win because GRACE pretraining is excluded.
- SoftDist-MLP may be called lightweight downstream only when its measured classifier cost is below the GCN and NRGNN controls.
