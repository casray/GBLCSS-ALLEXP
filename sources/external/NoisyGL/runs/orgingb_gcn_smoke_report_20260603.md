# OrginGB-GCN Smoke Report 2026-06-03

## 研究假设

普通粒球可以作为 GCN hidden embedding 空间中的局部语义单元，提供弱去噪先验。若同一 ordinary granular ball 内的 warmup 预测分布和 noisy train label 分布形成稳定共识，则可用 ball-level soft target 对疑似噪声样本进行轻量软修正。

## 粒球扮演的角色

- 局部语义关系单元
- 置信度信号
- soft target teacher
- 样本降权依据

本版本不把 same-ball membership 当作 hard label，也不做 hard relabel、prototype pulling、图结构修改或动态刷新。

## 方法动机

前期 GAGB 经验显示，强制同球同类、hard positive 或过强 target 更新容易造成分类边界间接移动。第一版 OrginGB-GCN 因此只在普通 GCN warmup 后构造静态 embedding-space granular balls，并通过 soft target 和 sample weight 进行弱干预。

## 代码改动

- `predictor/OrginGB_GCN_Predictor.py`: 新增 `orgingb_gcn` 和 `orgingb_gcn_random`。
- `utils/ordinary_gb.py`: 新增普通粒球构造与 random-size-matched assignment。
- `utils/soft_targets.py`: 新增 ball-level soft target、sample weight 和 NCR/CDR/AUROC 诊断。
- `predictor/module/GNNs.py`: 为 GCN 增加 `forward_with_hidden()`，原 `forward()` 保持 logits 输出。
- `single_exp.py`, `total_exp.py`: 注册两个新方法，并把 seed/noise 信息传给 predictor 便于诊断记录。
- `config/orgingb_gcn/*`, `config/orgingb_gcn_random/*`: 新增 Cora/CiteSeer/PubMed 配置。
- `tests/test_orgingb_tools.py`: 新增普通粒球和 soft target 单元测试。

## 配置改动

共享 GCN 默认训练协议：

- `n_hidden=64`
- `n_layer=2`
- `dropout=0.5`
- `lr=1e-2`
- `weight_decay=5e-4`
- `n_epochs=200`
- `criterion=metric`

GB 参数：

- `warmup_epochs=30`
- `alpha_max=0.5`
- `gamma=0.5`
- `eta_max=0.4`
- `min_train_per_ball=2`
- `full_train_per_ball=4`
- `split_improvement=0.05`
- `label_prior_smoothing=1.0`

`max_balls` 和 `min_ball_size` 保持代码默认自适应：

- `max_balls=min(64, max(8, train_count//3))`
- `min_ball_size=max(32, ceil(num_nodes/max_balls))`

## 预期机制

1. GCN 在 noisy labels 上 warmup 30 epoch。
2. 从倒数第二层 hidden embedding 构造普通粒球。
3. 对每个 ball 计算 warmup prediction mean 和 noisy train label distribution。
4. 生成 ball-level soft target `q_B`。
5. 对 train node 使用轻量混合 target 和 sample weight：
   - `target_i = (1-alpha_i) * onehot(noisy_y_i) + alpha_i * q_B`
   - `weight_i = 1 - gamma * alpha_i`
6. 全程按验证集 metric 选择 best checkpoint，并记录 `best_phase`，避免把 GB 阶段强行作为最终模型。

## 成功标准

- smoke test 无 NaN/OOM。
- `alpha_mean > 0`，说明 GB 模块实际介入。
- `target_changed_ratio > 0`，但不出现 hard relabel。
- NCR 有正信号，CDR 不明显升高。
- 真实 GB 至少不弱于 GCN，且应优于 random-size-matched control。
- 若进入正式实验，5-seed mean 至少在两个数据集高于 GCN，且 paired delta 稳定。

## 失败标准

- NaN/OOM。
- `orgingb_gcn_random` 持平或超过真实 GB。
- NCR 接近 0 或 CDR 明显高于 NCR。
- best checkpoint 长期停留在 `warmup`，说明 GB 阶段没有提供验证集收益。
- 只有单 seed 或小于 std 的提升，不作为论文结论。

## 实验结果

Smoke setting:

- `Cora uniform 0.4`
- `seed=3000`
- `device=cuda:0`

结果：

| method | test acc | valid acc | best_epoch | best_phase | NCR | CDR | suspicion AUROC |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| gcn | 0.7050 | 0.4420 | 6 | baseline | - | - | - |
| orgingb_gcn | 0.7050 | 0.4420 | 6 | warmup | 0.0208 | 0.0543 | 0.6492 |
| orgingb_gcn_random | 0.7050 | 0.4420 | 6 | warmup | 0.0625 | 0.5870 | 0.5444 |

诊断文件：

- `runs/orgingb_gcn_diagnostics_v2.csv`

## 最终决策

当前实现可以作为第一版 OrginGB-GCN 代码基础，但 smoke test 不支持直接进入三数据集 5-seed 正式实验。主要原因是 best checkpoint 来自 warmup 阶段，GB 阶段没有超过普通 GCN；真实 GB 的 NCR 低于 CDR，random control 的 CDR 很高，说明普通 ball-level soft target 目前还不是可靠的去噪信号。

建议下一步先做方法调参/机制修正，再跑三数据集正式实验。优先方向：

- 降低 `alpha_max` 或按 ball confidence 更严格 gating。
- 对 `q_B` 增加 noisy-label agreement / entropy 过滤。
- 只对高 suspicion 且 ball-level confidence 高的节点启用 soft target。
- 保留 `best_phase` 诊断，正式结果必须区分收益来自 GCN warmup 还是 GB 阶段。
