# Frozen-GRACE Ball-Corrected GCN Seed0 Smoke

- Date: 2026-06-10
- Protocol: NoisyGL `split_type=percent`, train/val/test = 0.2/0.1/0.7, `noise_type=uniform`, `noise_rate=0.4`, seed 0.
- Frozen embedding source: `gb_graph_aware_minival_6_1/checkpoints/embeddings/<dataset>/ratio_train0p2_val0p1_split0_pretrain0/H.pt`.
- Method: build ordinary GB on frozen GRACE H, use train-only noisy-label majority vote to hard-correct train labels, then train a fresh NoisyGL GCN end-to-end from scratch.
- Boundary: single seed smoke; clean labels are diagnostics only.

| Dataset | Method | Test Acc | Delta vs GCN+CE | Actual Noise | Log |
|---|---:|---:|---:|---:|---|
| citeseer | frozengrace_gb_gcn | 0.6830 | 0.0503 | 0.39 | `runs\frozengrace_gb_gcn_ratio_c1_20260610\logs\citeseer_frozengrace_gb_gcn_seed0.log` |
| citeseer | frozengrace_gb_gcn_shuffled | 0.3222 | -0.3105 | 0.39 | `runs\frozengrace_gb_gcn_ratio_c1_20260610\logs\citeseer_frozengrace_gb_gcn_shuffled_seed0.log` |
| citeseer | gcn_ce | 0.6327 | 0.0000 | 0.39 | `runs\ratio_c1_baselines_20260610\logs\citeseer_gcn_seed0.log` |
| cora | frozengrace_gb_gcn | 0.7947 | -0.0153 | 0.38 | `runs\frozengrace_gb_gcn_ratio_c1_20260610\logs\cora_frozengrace_gb_gcn_seed0.log` |
| cora | frozengrace_gb_gcn_shuffled | 0.3953 | -0.4147 | 0.38 | `runs\frozengrace_gb_gcn_ratio_c1_20260610\logs\cora_frozengrace_gb_gcn_shuffled_seed0.log` |
| cora | gcn_ce | 0.8100 | 0.0000 | 0.38 | `runs\ratio_c1_baselines_20260610\logs\cora_gcn_seed0.log` |

## Diagnostics

| Dataset | Method | Changed Train | NCR | CDR | Num Balls | Mean Ball Size |
|---|---:|---:|---:|---:|---:|---:|
| cora | frozengrace_gb_gcn | 246 | 0.6878 | 0.2244 | 52 | 52.08 |
| cora | frozengrace_gb_gcn_shuffled | 367 | 0.1376 | 0.6648 | 52 | 52.08 |
| citeseer | frozengrace_gb_gcn | 358 | 0.5911 | 0.2980 | 64 | 51.98 |
| citeseer | frozengrace_gb_gcn_shuffled | 436 | 0.1338 | 0.6641 | 64 | 51.98 |
