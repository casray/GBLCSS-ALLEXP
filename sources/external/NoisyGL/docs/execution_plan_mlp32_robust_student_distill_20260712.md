# MLP-32 Robust-Student Distillation and Graph-Native Comparison

## Research hypothesis

标准 `GCN[X||H]` teacher 提供图知识，LCSS+soft-small-loss 在 X-only student 端修复 noisy supervision。若二者互补，联合 student 应超过 plain distillation，同时 real LCSS 应超过 shuffled control。

## Granular-ball role

粒球只在 student 的 train-only soft target 中提供局部共识。teacher 使用 noisy train CE，不读取球目标；student 推理只输入 X。

## Frozen protocol

- Datasets: Cora, CiteSeer, CoraFull, Amazon-Photo, PubMed, Coauthor-Physics。
- Noise: uniform 0.4, train-only；split: percent 0.2/0.1/0.7。
- Student smoke: seed0；KD/robust loss weights fixed at 1/1；no grid search。
- Graph-native reference: NRGNN and NRGNN[X||H], seeds 0-4。

## Code changes

- Added `predictor/LCSSRobustStudentDistill_Predictor.py`.
- Registered `mlp_distill_gcn_xh_lcss_small_loss` and shuffled control.
- Added 12 configs with separate ball `embedding_path` and `teacher_embedding_path`.
- Added seed0 and cross-resource summarizers plus regression tests.

## Seed0 robust-student result

| Dataset | Direct | Plain distill | Robust student | Shuffled | Teacher | Robust-plain | Real-shuffled |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cora | 60.16 | 79.42 | 79.58 | 78.58 | 79.58 | +0.16pp | +1.00pp |
| CiteSeer | 59.84 | 69.46 | 68.73 | 67.74 | 67.74 | -0.73pp | +0.99pp |
| CoraFull | 41.27 | 58.61 | 58.51 | 58.48 | 61.42 | -0.09pp | +0.03pp |
| Amazon-Photo | 81.20 | 89.92 | 88.81 | 87.64 | 89.43 | -1.10pp | +1.18pp |
| PubMed | 81.24 | 83.81 | 83.44 | 82.91 | 82.06 | -0.36pp | +0.53pp |
| Coauthor-Physics | 92.71 | 95.22 | 94.36 | 94.44 | 95.25 | -0.86pp | -0.08pp |

Gate: **FAIL**。Only 1/6 datasets exceeded plain distillation; mean delta was -0.50pp; only 2/6 real-shuffled deltas reached +1pp; Amazon-Photo regressed by more than 1pp. The branch stops at seed0 with no weight tuning or five-seed expansion.

## Five-seed cross-resource result

| Dataset | Direct robust X-only | Plain distill X-only | NRGNN | NRGNN[X||H] | Distill-NRGNN |
|---|---:|---:|---:|---:|---:|
| Cora | 59.49+/-1.02 | 80.23+/-1.41 | 81.76+/-1.14 | 81.47+/-1.14 | -1.53pp |
| CiteSeer | 58.28+/-2.39 | 68.81+/-1.00 | 71.03+/-0.90 | 70.74+/-0.93 | -2.23pp |
| CoraFull | 40.63+/-0.50 | 59.45+/-0.56 | 62.70+/-0.57 | 62.06+/-0.46 | -3.25pp |
| Amazon-Photo | 82.73+/-0.88 | 90.51+/-0.96 | 89.02+/-0.97 | 89.53+/-1.37 | +1.49pp |
| PubMed | 81.47+/-0.29 | 84.80+/-0.76 | 80.69+/-0.59 | 83.29+/-0.37 | +4.11pp |
| Coauthor-Physics | 93.21+/-0.35 | 95.44+/-0.15 | 95.84+/-0.18 | 95.89+/-0.14 | -0.40pp |

Coauthor-Physics NRGNN/NRGNN[X||H] was newly completed because no prior rows existed. Each graph-native run required about 10.5 minutes on the current GPU.

## Final decision

1. Equal-weight robust-student fusion is rejected; its student-side LCSS signal is largely absorbed by all-node KD.
2. Plain distillation remains the accuracy extension. It is competitive with graph-native NRGNN and wins on Amazon-Photo/PubMed, but loses on four datasets.
3. The paper must not claim graph-native SOTA. The defensible claim is competitive accuracy with X-only inference and a distinct deployment/resource profile.
4. Direct LCSS+soft-small-loss remains the high-noise robust method; distillation remains a separate accuracy extension rather than part of the core algorithm.

## Outputs

- `runs/mlp32_robust_student_distill_20260712/mlp32_report.md`
- `runs/mlp32_robust_student_distill_20260712/resource_table_report.md`
- `runs/mlp32_robust_student_distill_20260712/resource_table_raw.csv`
- `runs/mlp32_robust_student_distill_20260712/resource_table_aggregate.csv`
- `runs/mlp32_robust_student_distill_20260712/resource_table_paired.csv`

