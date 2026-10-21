# Topology-Aware SoftDist [X||H] Seed0 Smoke

Date: 2026-06-11

## Isolation

- Trial copy: `C:\Users\Admin\Desktop\实验8去噪5.19-\external\NoisyGL_graphaware_softdist_trial_20260611`
- Original code left untouched: `C:\Users\Admin\Desktop\实验8去噪5.19-\external\NoisyGL`
- Runtime: `C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe`
- Device: `cuda:0`, AMD Radeon RX 7700 XT

## Protocol

- Datasets: Cora, CiteSeer
- Split: `split_type=percent`, `train_percent=0.2`, `val_percent=0.1`, `test_percent=0.7`
- Noise: `noise_type=uniform`, `noise_rate=0.4`
- Seed: 0
- New methods:
  - `frozengrace_gb_gcn_softdist_xh_topoalpha`
  - `frozengrace_gb_gcn_softdist_xh_topoalpha_shuffled`

## Method

The trial keeps SoftDist `[X||H]` unchanged except for a static, label-free topology alpha scale:

```text
alpha_final(B) = alpha_softdist(B) * scale_topology(B)
scale_topology(B) = floor + (1 - floor) * normalized_internal_edge_ratio(B)
floor = 0.5
```

This only downweights low-topology-support balls. It does not update balls dynamically, does not supervise unlabeled nodes, and does not add ball consistency loss.

## Seed0 Test Accuracy

| Dataset | Method | Test acc |
|---|---|---:|
| Cora | `gcn_xh_ce` | 0.8106 |
| Cora | `softdist_xh` | 0.8058 |
| Cora | `softdist_xh_topoalpha` | 0.8063 |
| Cora | `softdist_xh_topoalpha_shuffled` | 0.7947 |
| CiteSeer | `gcn_xh_ce` | 0.6890 |
| CiteSeer | `softdist_xh` | 0.6959 |
| CiteSeer | `softdist_xh_topoalpha` | 0.6963 |
| CiteSeer | `softdist_xh_topoalpha_shuffled` | 0.6869 |

## Diagnostics

| Dataset | Method | Mean alpha | Shift mass | NCR | CDR | Clean mass delta | Topology scale mean |
|---|---|---:|---:|---:|---:|---:|---:|
| Cora | `softdist_xh` | 0.4695 | 99.0472 | 0.6878 | 0.2244 | -35.8371 | - |
| Cora | `softdist_xh_topoalpha` | 0.3629 | 77.0953 | 0.6878 | 0.2244 | -27.0540 | 0.7733 |
| Cora | `topoalpha_shuffled` | 0.1392 | 48.3209 | 0.1376 | 0.6648 | -33.7156 | 0.6384 |
| CiteSeer | `softdist_xh` | 0.3540 | 114.0526 | 0.5911 | 0.2980 | -46.4799 | - |
| CiteSeer | `softdist_xh_topoalpha` | 0.2923 | 94.1246 | 0.5911 | 0.2980 | -39.0013 | 0.8180 |
| CiteSeer | `topoalpha_shuffled` | 0.1393 | 57.4991 | 0.1338 | 0.6641 | -35.3759 | 0.6391 |

## Decision

Do not expand this exact `topology_alpha_floor=0.5` setting to 5 seeds.

The mechanism is active and label-free topology clearly distinguishes real balls from shuffled balls, but the accuracy gain over SoftDist `[X||H]` is only about `+0.05pp` on both datasets. This is far below the threshold needed to justify a 5-seed run.

Useful retained observation: topology scaling reduces soft target damage mass while preserving the same pseudo-label NCR/CDR. This may be useful as a diagnostic or as part of a stronger boundary-confidence design, but this direct multiplicative alpha scaling is too weak to become the next main method.

## Artifacts

- Run status: `runs/topoalpha_seed0_smoke_20260611/run_status.csv`
- Parsed diagnostics: `runs/frozengrace_gb_gcn_diagnostics_v2.csv`
- Logs: `runs/topoalpha_seed0_smoke_20260611/logs/`
