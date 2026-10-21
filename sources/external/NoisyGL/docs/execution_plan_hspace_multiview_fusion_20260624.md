# 执行记录：Phase 40a · H-space 多视角检测融合诊断

日期：2026-06-24

## 研究假设

H 空间中的 `s_knn`、`s_gcn_conf`、`s_ball_loo`、`s_graphnbr` 可能存在互补性。如果互补性足够强，则无监督 percentile-rank fusion 应稳定超过每个 seed 上的最强单项 detector。

## 粒球扮演的角色

粒球不是主方法，只作为候选视角 `s_ball_loo = 1 - p_{B\i}(noisy_y_i)`，用于检验粗粒度局部一致性是否能补充 kNN / GCN。

## 方法动机

Phase39 显示 `s_ball_loo` 检测力强但整体被 `s_knn` / `s_gcn_conf` 压制；Phase40A 只验证了 `kNN + ball`，未验证 `kNN + GCN + ball` 多视角融合。本轮用最小成本判断是否值得进入 KNN/GCN teacher SoftDist 下游实验。

## 代码改动

- 新增 `scripts/probe_hspace_multiview_fusion.py`
- 新增 `tests/test_hspace_multiview_fusion_probe.py`

## 配置改动

无。沿用 Phase39：

- datasets：`cora,citeseer,coauthorcs`
- seeds：`0,1,2,3,4`
- noise：`uniform` 0.4
- split：percent 0.2/0.1/0.7
- H：frozen GRACE H
- 构球参数：同源 SoftDist `[X‖H]` config

## 检测器

单项：

- `s_knn`
- `s_gcn_conf`
- `s_ball_loo`
- `s_graphnbr`

融合：

- `F1_rank_knn_gcn = rank_avg(s_knn, s_gcn_conf)`
- `F2_rank_knn_ball = rank_avg(s_knn, s_ball_loo)`
- `F3_rank_gcn_ball = rank_avg(s_gcn_conf, s_ball_loo)`
- `F4_rank_knn_gcn_ball = rank_avg(s_knn, s_gcn_conf, s_ball_loo)`
- `F5_rank_knn_gcn_ball_graphnbr = rank_avg(s_knn, s_gcn_conf, s_ball_loo, s_graphnbr)`

Oracle 仅诊断，不参与 go/no-go：

- `O1_logreg_knn_gcn`
- `O2_logreg_knn_gcn_ball`

## 预期机制

- 若 fusion 成功：H-space 多视角噪声检测存在无监督互补价值，可考虑 KNN/GCN teacher SoftDist 下游 sanity check。
- 若 fusion 失败：H-space 噪声检测已基本被最强 kNN/GCN 单项捕获，继续开发 KNN-SoftDist / dual teacher 价值低。

## 成功标准

rank fusion 继续进入下游的最低条件：

- 在 ≥2/3 数据集上超过每 seed 的 `best_single`；
- AUROC 或 AUPRC_lift 平均提升 ≥ 0.01；
- paired seeds 至少 4/5 为正；
- AUPRC / Precision@K 不系统性下降。

关键强标准：

- `F4_rank_knn_gcn_ball` 稳定超过 `max(s_knn, s_gcn_conf)` / per-seed best single。

## 失败标准

满足任一条即停止：

- 所有 rank fusion 都不超过 best single；
- 提升 < 0.005；
- 只有单数据集有效；
- graphnbr 相关增益不稳定或伴随 AUROC/AUPRC/P@K 损失；
- 只有 oracle logistic 提升，rank fusion 不提升。

## 实验结果

输出目录：`runs/hspace_multiview_fusion_20260624/`

核心报告：`runs/hspace_multiview_fusion_20260624/hspace_multiview_fusion_report.md`

关键结果：

| Dataset | best single | 最强/关键 fusion | ΔAUROC vs best single | pos/5 | ΔAUPRC lift | ΔP@K |
|---|---|---|---:|---:|---:|---:|
| Cora | `s_knn` | `F1_rank_knn_gcn` | +0.0076 | 5/5 | +0.0365 | +0.0079 |
| Cora | `s_knn` | `F4_rank_knn_gcn_ball` | +0.0070 | 4/5 | +0.0334 | +0.0009 |
| CiteSeer | `s_knn` | `F4_rank_knn_gcn_ball` | -0.0079 | 1/5 | +0.0145 | -0.0205 |
| Coauthor-CS | `s_gcn_conf` | `F1_rank_knn_gcn` | -0.0023 | 0/5 | -0.0055 | -0.0060 |
| Coauthor-CS | `s_gcn_conf` | `F4_rank_knn_gcn_ball` | -0.0059 | 0/5 | -0.0136 | -0.0179 |

Oracle 上界：

- Cora `O2_logreg_knn_gcn_ball`：ΔAUROC +0.0106，5/5；
- CiteSeer `O2_logreg_knn_gcn_ball`：ΔAUROC +0.0059，5/5；
- Coauthor-CS `O2_logreg_knn_gcn_ball`：ΔAUROC +0.0006，5/5。

Oracle 说明存在少量可拟合互补信息，但它使用 clean/noisy flag 跨 seed 训练，不能作为可部署方法。

## 最终决策

**FAIL / 不进入 Phase40b。**

原因：

- rank fusion 没有在 ≥2/3 数据集上超过 per-seed best single；
- Cora 有小增益，但 < +0.01 AUROC；
- CiteSeer 和 Coauthor-CS 均为负；
- 加入 ball 的 F4 不稳定，不能证明 ball 对 `kNN/GCN` 有可部署互补价值；
- oracle logistic 小幅提升不足以支撑正式方法开发。

结论写法：

> H-space 多视角检测融合没有稳定超过最强单项 kNN/GCN；粒球和无监督 fusion 都不能提供足够稳定的额外收益。KNN-SoftDist / KNN+GCN teacher 下游实验暂不继续。
