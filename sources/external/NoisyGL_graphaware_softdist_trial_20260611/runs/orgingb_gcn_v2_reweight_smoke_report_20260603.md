# OrginGB-GCN v2 Reweight Smoke Report 2026-06-03

## 研究假设

OrginGB-GCN v1 的普通粒球包含弱噪声排序信号，但 ball-level soft target 纠错能力不足。v2 将 ordinary GB 从标签修正器降级为 high-suspicion train node 的样本降权器，验证该弱排序信号能否转化为 GCN 标签噪声训练收益。

## 粒球扮演的角色

- 局部 embedding-space 分组。
- 噪声样本 suspicion prior。
- sample reweighting gate。

本轮不使用 hard relabel、target replace、KL auxiliary、graph-aware GB 或动态刷新。

## 方法动机

v1 smoke 中 `suspicion_auroc=0.6492` 高于 random control 的 `0.5444`，说明 ordinary GB 有弱噪声排序能力；但 `target_argmax_changed_ratio=0` 且 `best_phase=warmup`，说明 soft target mixing 没有形成有效纠错。因此 v2 先测试更保守的 loss reweighting。

## 代码改动

- `utils/soft_targets.py`: 增加 `target_mode="reweight"`、`suspicion_mode="gb|no_ball"`、top-ratio gating、ball confidence、sample weight 诊断。
- `predictor/OrginGB_GCN_Predictor.py`: 增加 `orgingb_gcn_reweight`、`orgingb_gcn_reweight_random`、`orgingb_gcn_reweight_noball`，诊断写入 `runs/orgingb_gcn_diagnostics_v3.csv`。
- `single_exp.py`, `total_exp.py`: 注册三个 v2 方法。
- `config/orgingb_gcn_reweight*/*`: 增加 Cora/CiteSeer/PubMed 配置。
- `tests/test_orgingb_tools.py`: 增加 reweight/no-ball 单元测试。

## 配置改动

共享 GCN 协议不变：

- `n_hidden=64`
- `n_layer=2`
- `dropout=0.5`
- `lr=1e-2`
- `weight_decay=5e-4`
- `n_epochs=200`
- `warmup_epochs=30`

v2 reweight 参数：

- `target_mode=reweight`
- `suspicion_top_ratio=0.3`
- `min_ball_conf=0.3`
- `beta=0.5`
- `min_weight=0.3`

## 预期机制

1. GCN 在 noisy labels 上 warmup 30 epoch。
2. 用 hidden embedding 构造 ordinary GB。
3. 计算 ball target `q_B` 与 ball confidence。
4. `orgingb_gcn_reweight` 使用 `1 - sqrt(p_i[noisy_y_i] * q_B[noisy_y_i])` 排序可疑节点。
5. `orgingb_gcn_reweight_noball` 只用 `1 - p_i[noisy_y_i]` 排序，作为 no-ball control。
6. gated train nodes 降低 CE 权重，targets 保持 noisy one-hot。

## 成功标准

- `orgingb_gcn_reweight` 的 `best_phase=gb`。
- test acc 高于 `0.7050` 或至少高于 v1 且最好验证点来自 GB 阶段。
- real GB 优于 random 和 no-ball。
- `gated_noisy_precision > 0.5`。
- `gated_clean_ratio` 不异常升高。

## 失败标准

- `best_phase` 仍是 `warmup`。
- no-ball 或 random 持平/超过 real GB。
- gated precision 接近随机噪声率。
- 降权后 test acc 不高于 GCN。

## 实验结果

Smoke setting:

- Dataset: `Cora`
- Noise: `uniform 0.4`
- Seed: `3000`
- Device: `cuda:0`
- Raw log: `runs/orgingb_gcn_v2_reweight_smoke_20260603/total_exp_cora_uniform0.4_seed3000_cuda0_final.log`
- Diagnostics: `runs/orgingb_gcn_diagnostics_v3.csv`

| method | test acc | best_epoch | best_phase | target_mode | suspicion_mode | gated_ratio | gated_noisy_precision | gated_noisy_recall | gated_clean_ratio | suspicion AUROC |
| --- | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| gcn | 0.7050 | 6 | baseline | - | - | - | - | - | - | - |
| orgingb_gcn | 0.7050 | 6 | warmup | mix | gb | 0.9500 | 0.3308 | 0.9167 | 0.9674 | 0.6492 |
| orgingb_gcn_reweight | 0.7050 | 6 | warmup | reweight | gb | 0.1500 | 0.4762 | 0.2083 | 0.1196 | 0.6474 |
| orgingb_gcn_reweight_random | 0.7050 | 6 | warmup | reweight | gb | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5444 |
| orgingb_gcn_reweight_noball | 0.7050 | 6 | warmup | reweight | no_ball | 0.3000 | 0.4762 | 0.4167 | 0.2391 | 0.6259 |

## 最终决策

本轮 v2 reweight smoke 未通过进入三数据集 5-seed 的条件。

- `orgingb_gcn_reweight` 的最好 checkpoint 仍来自 warmup，不是 GB 阶段。
- test acc 与 GCN、v1、random、no-ball 全部持平为 `0.7050`。
- real GB 的 gated precision 为 `0.4762`，没有达到 `>0.5` 成功标准。
- no-ball control 的 gated precision 同样为 `0.4762`，说明当前收益不能归因于 ordinary GB membership。
- random control 被 `min_ball_conf=0.3` 完全过滤，说明该 control 需要后续改成 intervention-strength-matched 才能更公平。

结论：ordinary GB v2 reweight 已实现并可运行，但当前 smoke 不支持扩展到 Cora/CiteSeer/PubMed 正式 5-seed。下一步不应继续直接扩展正式实验，应先选择以下之一：

1. 调低 `min_ball_conf` 或改 random 为 strength-matched control，确认 real GB 与 random 的公平差距。
2. 只做离线 gate 诊断，扫描 `suspicion_top_ratio` 与 `min_ball_conf`，不进入训练。
3. 暂停 ordinary GB 作为主去噪机制，转向 graph-aware GB 或 teacher/conflict-gated target repair。
