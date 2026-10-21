# NoisyGL Ratio-C1 Baseline Smoke (seed 0)

- Date: 2026-06-10
- Command template: `py -3 single_exp.py --method <method> --dataset <dataset> --noise_type uniform --noise_rate 0.4 --seed 0 --device cpu --split_type percent --train_percent 0.2 --val_percent 0.1 --test_percent 0.7`
- Split target: train/val/test = 0.2/0.1/0.7, NoisyGL percent split.
- C1 seed-0 reference from user: Cora 0.744, CiteSeer 0.571.
- Boundary: single-seed smoke only; not mean/std evidence and not a paper claim.

| Dataset | Method | Test Acc | Actual Noise | Split Samples | Delta vs C1 seed0 | Log |
|---|---:|---:|---:|---:|---:|---|
| citeseer | clnode | 0.5279 | 0.39 | 665/332/2328 | -0.0431 | `runs\ratio_c1_baselines_20260610\logs\citeseer_clnode_seed0.log` |
| citeseer | gcn | 0.6327 | 0.39 | 665/332/2328 | 0.0617 | `runs\ratio_c1_baselines_20260610\logs\citeseer_gcn_seed0.log` |
| citeseer | nrgnn | 0.7006 | 0.39 | 665/332/2328 | 0.1296 | `runs\ratio_c1_baselines_20260610\logs\citeseer_nrgnn_seed0.log` |
| citeseer | pignn | 0.6409 | 0.39 | 665/332/2328 | 0.0699 | `runs\ratio_c1_baselines_20260610\logs\citeseer_pignn_seed0.log` |
| citeseer | rtgnn | 0.6645 | 0.39 | 665/332/2328 | 0.0935 | `runs\ratio_c1_baselines_20260610\logs\citeseer_rtgnn_seed0.log` |
| citeseer | sce | 0.5945 | 0.39 | 665/332/2328 | 0.0235 | `runs\ratio_c1_baselines_20260610\logs\citeseer_sce_seed0.log` |
| cora | clnode | 0.7451 | 0.38 | 541/270/1895 | 0.0011 | `runs\ratio_c1_baselines_20260610\logs\cora_clnode_seed0.log` |
| cora | gcn | 0.8100 | 0.38 | 541/270/1895 | 0.0660 | `runs\ratio_c1_baselines_20260610\logs\cora_gcn_seed0.log` |
| cora | nrgnn | 0.8359 | 0.38 | 541/270/1895 | 0.0919 | `runs\ratio_c1_baselines_20260610\logs\cora_nrgnn_seed0.log` |
| cora | pignn | 0.7847 | 0.38 | 541/270/1895 | 0.0407 | `runs\ratio_c1_baselines_20260610\logs\cora_pignn_seed0.log` |
| cora | rtgnn | 0.7855 | 0.38 | 541/270/1895 | 0.0415 | `runs\ratio_c1_baselines_20260610\logs\cora_rtgnn_seed0.log` |
| cora | sce | 0.7921 | 0.38 | 541/270/1895 | 0.0481 | `runs\ratio_c1_baselines_20260610\logs\cora_sce_seed0.log` |

## Notes

- All reported rows are seed 0 only.
- All runs used CPU because current Python reports `torch.cuda.is_available() == False`.
- `single_exp.py` was extended with optional split override arguments; original dataset YAML files were not changed.
- NoisyGL injects label noise before training and evaluates test accuracy with clean labels via predictor `test()`.
