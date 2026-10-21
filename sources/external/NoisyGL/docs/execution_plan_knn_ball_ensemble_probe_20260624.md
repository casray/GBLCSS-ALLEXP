# 执行计划：A 实验 · kNN + ball 检测集成探针（Phase 40A）

日期：2026-06-24

## 研究问题

Phase 39 证明 `s_ball_loo` 是强离线噪声检测信号，并具有条件正交性；但其整体 AUROC 被 `kNN-in-H` / `GCN confidence` 压制。因此 A 实验只回答一个更窄的问题：

> `s_ball_loo` 是否能作为 `kNN-in-H` 的辅助残差信号，使 `kNN + ball` 集成稳定超过 `kNN` 单用？

## 研究假设

- H1：如果粒球不仅是粗粒化 kNN，而是包含可利用的残差信息，则 `kNN + s_ball_loo` 的无监督集成应在多数据集上稳定超过 `s_knn`。
- H0：如果粒球基本被 kNN 支配，则 rank/z-score 集成不会稳定超过 `s_knn`，oracle logistic 也只能带来极小或不稳定增益。

## 粒球角色

粒球不再作为独立检测器，也不再作为分类 soft target；这里只作为 `kNN-in-H` 的辅助检测特征。

## 方法动机

Phase 39 中：

- `s_ball_loo` AUROC 强，但低于最强 H/model baseline；
- `s_ball_loo` 与 `s_knn` 高相关；
- 但 `s_ball_loo` 在 `kNN-clean50` 中仍有条件 AUROC，说明可能存在少量 residual 信息。

A 实验用集成直接检验这部分 residual 是否有实际指标价值。

## 代码改动

- 新增 `scripts/probe_knn_ball_ensemble.py`
- 新增 `tests/test_knn_ball_ensemble_probe.py`

不修改 Phase 39 原始结果，不重训模型。

## 配置改动

无。沿用 Phase 39：

- 数据集：`cora,citeseer,coauthorcs`
- seeds：`0,1,2,3,4`
- split：percent 0.2/0.1/0.7
- noise：uniform 0.4
- 构球参数：来自 `frozengrace_gb_gcn_softdist_xh` 的同源 config

## 检测器

Primary deployable probes：

- `rank_mean_knn_ball = mean(percentile_rank(s_knn), percentile_rank(s_ball_loo))`
- `z_mean_knn_ball = mean(z(s_knn), z(s_ball_loo))`

Oracle diagnostic：

- `loo_logreg_knn_ball`：leave-one-seed-out logistic regression，用其他 seeds 的 `[s_knn, s_ball_loo]` 与 noisy flag 训练，在 held-out seed 上评测。它使用 clean flag 训练，因此只能作为上界，不作为可部署方法。

## 指标

每 dataset × seed：

- AUROC
- AUPRC_lift
- Precision@K
- `delta_vs_knn` for AUROC/AUPRC_lift/P@K

每 dataset 聚合：

- mean/std
- paired direction count

## 成功标准

Primary deployable ensemble 成功，当且仅当：

- 在 ≥2/3 数据集上，相对 `s_knn` mean AUROC ≥ +0.01；
- 且这些数据集上 ≥4/5 seeds AUROC 正向；
- AUPRC_lift / Precision@K 不出现系统性下降。

## 失败标准

满足任一情况即失败：

- primary ensemble 平均增益 < +0.005；
- 或正向 seed 不稳定；
- 或 AUROC 有小增益但 AUPRC/P@K 系统性下降；
- 或只有 oracle logistic 有增益，而无监督集成没有增益。

## 预期机制

- 若成功：球虽然整体弱于 kNN，但仍有可利用残差信息；论文可写“粒球作为辅助局部先验有有限互补价值”。
- 若失败：球基本是 H-kNN 的粗粒化近似；Phase 39 的条件正交性不足以转化为实际检测增量。

## 实验结果

结果目录：`runs/knn_ball_ensemble_probe_20260624/`

核心表：

| Dataset | Detector | AUROC | ΔAUROC vs kNN | pos/5 | AUPRC lift | ΔAUPRC lift | P@K | ΔP@K |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Cora | `rank_mean_knn_ball` | 0.9401 | -0.0073 | 1/5 | 0.5170 | +0.0071 | 0.8281 | -0.0280 |
| Cora | `z_mean_knn_ball` | 0.9451 | -0.0022 | 2/5 | 0.5222 | +0.0122 | 0.8439 | -0.0122 |
| Cora | `loo_logreg_knn_ball` | 0.9505 | +0.0031 | 5/5 | 0.5312 | +0.0213 | 0.8587 | +0.0026 |
| CiteSeer | `rank_mean_knn_ball` | 0.8735 | -0.0116 | 0/5 | 0.3876 | +0.0016 | 0.7363 | -0.0271 |
| CiteSeer | `z_mean_knn_ball` | 0.8748 | -0.0102 | 0/5 | 0.3900 | +0.0039 | 0.7355 | -0.0280 |
| CiteSeer | `loo_logreg_knn_ball` | 0.8866 | +0.0016 | 4/5 | 0.4028 | +0.0168 | 0.7608 | -0.0026 |
| Coauthor-CS | `rank_mean_knn_ball` | 0.9768 | -0.0040 | 0/5 | 0.5432 | -0.0036 | 0.9231 | -0.0129 |
| Coauthor-CS | `z_mean_knn_ball` | 0.9810 | +0.0002 | 3/5 | 0.5496 | +0.0029 | 0.9378 | +0.0019 |
| Coauthor-CS | `loo_logreg_knn_ball` | 0.9819 | +0.0011 | 5/5 | 0.5511 | +0.0043 | 0.9412 | +0.0052 |

观察：

- 两个 deployable primary ensemble（rank mean / z mean）均未达到成功标准。
- `rank_mean_knn_ball` 在 3/3 数据集上 AUROC 低于 `s_knn`。
- `z_mean_knn_ball` 只有 Coauthor-CS 出现 +0.0002 AUROC 的极小增益，且只有 3/5 seeds 正向，不满足 4/5 seeds 与 +0.01 mean AUROC 标准。
- oracle `loo_logreg_knn_ball` 在 3 个数据集上均有正 AUROC 增益，但幅度只有 +0.0011 到 +0.0031；这是使用其他 seeds 的 clean/noisy flag 训练得到的上界诊断，不能作为可部署方法。
- AUPRC_lift 在部分集成中略升，但 P@K 常下降；没有形成稳定实用增益。

## 最终决策

**放弃作为下一轮主线。**

结论：`s_ball_loo` 的条件正交性不能转化为对 `s_knn` 的实际无监督集成增益。球相对 kNN 的残差信号存在，但太弱；只有 oracle logistic 能挤出 +0.1pp 到 +0.3pp AUROC 的上界增益。该结果支持 Phase 39 后的机制判断：粒球在当前 GRACE H 空间中主要是 H-kNN 标签一致性的粗粒化近似，不能作为独立检测主方法，也不能作为 kNN 的稳定辅助模块。

后续用途：写入论文/报告的边界分析，而不是继续发展检测任务或球变体。
