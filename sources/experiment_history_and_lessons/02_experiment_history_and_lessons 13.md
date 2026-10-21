# Phase39/40A：检测换题闸门与 kNN-ball 集成探针

日期：2026-06-24  
相关目录：`external/NoisyGL`

本文档记录 Phase39 与 Phase40A。它们不是新的分类主方法，而是对导师建议中"可考虑换任务"的 go/no-go 诊断：粒球离线去噪强，是否足以把论文主任务从"噪声标签下节点分类"换成"图节点标签错误检测与清洗"。

## 1. 总判断

- **结论一：暂不正式换题。** Phase39 显示 `s_ball_loo` 是强检测信号，并具有条件正交性；但整体 AUROC 被 `s_knn` / `s_gcn_conf` 压制，且当前图结构特征 `s_graphdens` 没有增量。证据：`external/NoisyGL/runs/noise_detection_feasibility_20260624/phase39_gate_decision.md`。
- **结论二：`kNN + ball` 辅助路线判负。** Phase40A 显示可部署 rank/z 集成没有稳定超过 `s_knn`，oracle logistic 只有极小上界增益。证据：`external/NoisyGL/runs/knn_ball_ensemble_probe_20260624/knn_ball_ensemble_report.md` 与 `external/NoisyGL/runs/knn_ball_ensemble_probe_20260624/knn_ball_ensemble_summary.csv`。
- **论文定位不变。** 主线仍是 SoftDist `[X‖H]` 分类论文；Phase39/40A 应写成边界/机制分析：粒球能识别噪声，但在当前 GRACE H 空间中被更简单的 H-kNN / GCN 置信度解释并压制。

---

## 2. Phase39：图节点标签错误检测可行性诊断

### 研究假设

- H1：如果粒球不仅是 H 空间 kNN 的粗粒化近似，则节点级球分数应在噪声检测任务中既有较高 AUROC，又能提供独立于 `s_knn` / `s_gcn_conf` 的正交信息。
- H0：如果球信号主要来自 H 空间标签一致性，则即使球检测 AUROC 很高，也会被 `s_knn` / `s_gcn_conf` 压制。

### 粒球角色

粒球不再作为分类 soft target，而是作为节点级噪声检测器：输出 `s_ball_loo`、`s_jsd_loo`、`s_center`、`s_ballnbr`、`s_graphdens` 等分数。其中预注册 primary 是 `s_ball_loo` 与 `s_graphdens`。

### 方法动机

此前离线诊断显示粒球能把训练噪声率显著降低、球纯度较高，但下游分类中球信号被 `[X‖H]` / NRGNN-XH / kNN-in-H 吸收。Phase39 的目标是先判定：把"去噪声"本身作为任务，能否避开分类下游吸收问题。

### 代码改动

- 新增 `external/NoisyGL/scripts/diagnose_label_noise_detection.py`
- 新增 `external/NoisyGL/tests/test_label_noise_detection_diagnostic.py`
- 新增执行计划 `external/NoisyGL/docs/execution_plan_noise_detection_feasibility_20260624.md`
- 输出目录 `external/NoisyGL/runs/noise_detection_feasibility_20260624/`

### 配置改动

无主方法配置改动。沿用：

- 数据集：`cora,citeseer,coauthorcs`
- seeds：`0,1,2,3,4`
- 噪声：`uniform` 0.4
- split：percent 0.2/0.1/0.7
- 构球参数：从 `frozengrace_gb_gcn_softdist_xh` 同源 config 读取，Coauthor-CS 使用 `256/32/0.001`
- clean label 仅用于评测 noisy flag，不参与 score 计算

### 成功标准

预注册闸门：

- G1 正交性：primary 在 `knn_clean50` 与 `gcn_clean50` 中条件 AUROC ≥ 0.6，且 real > shuffled。
- G2 检测质量：primary 不被最强 `s_knn` / `s_gcn_conf` 压制。
- G3 类别覆盖：top-K 标记不塌缩到少数类。
- G4 图增量：`s_graphdens` 要超过纯图邻居 baseline `s_graphnbr`，且 `s_combo_HG - s_combo_H > 0`。

### 实验结果

检测质量主表见 `external/NoisyGL/runs/noise_detection_feasibility_20260624/noise_detection_feasibility_report.md`。

关键数值：

| Dataset | `s_ball_loo` AUROC | 更强 baseline |
|---|---:|---|
| Cora | 0.9016 | `s_knn` 0.9474；`s_gcn_conf` 0.9253 |
| CiteSeer | 0.8257 | `s_knn` 0.8850 |
| Coauthor-CS | 0.9614 | `s_gcn_conf` 0.9938；`s_knn` 0.9808 |

闸门结果：

- G1 正交性：通过。`s_ball_loo` 在 3/3 数据集上均满足条件 AUROC 要求。
- G2 检测质量：严格失败。`s_ball_loo` 绝对 AUROC 强，但整体被 `s_knn` / `s_gcn_conf` 压制。
- G4 图增量：失败。`s_graphdens` 明显弱于 `s_graphnbr`，且 `s_combo_HG - s_combo_H` 为负。
- Phase39b PubMed boundary run 未触发，因为 Phase39 v2 计划要求 G1 与 G2 同时过才补跑。

### 最终决策

**暂不正式换题为"图节点标签错误检测与清洗"。**

可以写入论文/汇报的结论：

1. 粒球 LOO 分数是强离线标签噪声信号；
2. 它具有一定条件正交性；
3. 但整体检测能力仍被简单 H/model baseline 压制；
4. 当前图边密度版本没有提供独立图结构增量；
5. 因此检测方向只保留为边界分析/辅助诊断，不替换主任务。

---

## 3. Phase40A：kNN + ball 检测集成探针

### 研究假设

- H1：如果 `s_ball_loo` 的条件正交性有实际价值，则 `s_knn + s_ball_loo` 的无监督集成应稳定超过 `s_knn`。
- H0：如果球基本是 H-kNN 的粗粒化近似，则 rank/z 集成不会稳定超过 `s_knn`；即使 oracle logistic 有增益，也只会很小。

### 粒球角色

粒球只作为 `s_knn` 的辅助检测特征，不作为独立检测主方法，也不作为分类 soft target。

### 方法动机

Phase39 给出一个细节：`s_ball_loo` 整体弱于 `s_knn`，但在 `kNN-clean50` 子集中仍有条件 AUROC。Phase40A 直接检验这部分 residual 是否能转化为实际集成收益。

### 代码改动

- 新增 `external/NoisyGL/scripts/probe_knn_ball_ensemble.py`
- 新增 `external/NoisyGL/tests/test_knn_ball_ensemble_probe.py`
- 新增执行计划 `external/NoisyGL/docs/execution_plan_knn_ball_ensemble_probe_20260624.md`
- 输出目录 `external/NoisyGL/runs/knn_ball_ensemble_probe_20260624/`

### 配置改动

无。沿用 Phase39 的数据集、seeds、split、噪声与同源构球参数。不重训模型，不修改 Phase39 原始结果。

### 成功标准

可部署 primary ensemble 成功，当且仅当：

- 在 ≥2/3 数据集上相对 `s_knn` mean AUROC ≥ +0.01；
- 且这些数据集上 ≥4/5 seeds AUROC 正向；
- AUPRC lift / Precision@K 不出现系统性下降。

### 实验结果

报告见 `external/NoisyGL/runs/knn_ball_ensemble_probe_20260624/knn_ball_ensemble_report.md`，汇总表见 `external/NoisyGL/runs/knn_ball_ensemble_probe_20260624/knn_ball_ensemble_summary.csv`。

关键结果：

| Dataset | `s_knn` AUROC | 最好 deployable 组合 | ΔAUROC | oracle logistic ΔAUROC |
|---|---:|---:|---:|---:|
| Cora | 0.9474 | `z_mean_knn_ball` 0.9451 | -0.0022 | +0.0031 |
| CiteSeer | 0.8850 | `z_mean_knn_ball` 0.8748 | -0.0102 | +0.0016 |
| Coauthor-CS | 0.9808 | `z_mean_knn_ball` 0.9810 | +0.0002 | +0.0011 |

观察：

- `rank_mean_knn_ball` 在 3/3 数据集上低于 `s_knn`。
- `z_mean_knn_ball` 仅在 Coauthor-CS 有 +0.0002 AUROC 的极小增益，且只有 3/5 seeds 正向，不满足成功标准。
- oracle `loo_logreg_knn_ball` 在三数据集均有正增益，但只有 +0.0011 到 +0.0031 AUROC；它使用其他 seeds 的 clean/noisy flag 训练，只能作为上界诊断，不能作为可部署方法。
- AUPRC lift 有局部微升，但 P@K 常下降；没有形成稳定实用增益。

### 最终决策

**放弃作为下一轮主线。**

结论：`s_ball_loo` 的条件正交性不能转化为对 `s_knn` 的实际无监督集成增益。球相对 kNN 的残差信号存在，但太弱；只有 oracle logistic 能挤出极小上界增益。

---

## 4. 证据分类

### 已被实验支持的结论

- 粒球 LOO 分数能识别训练标签噪声；`s_ball_loo` 在 Cora/CiteSeer/Coauthor-CS 上 AUROC 分别为 0.9016/0.8257/0.9614。
- 粒球检测整体弱于简单 H/model baseline：Cora 与 CiteSeer 被 `s_knn` 压制，Coauthor-CS 被 `s_gcn_conf` 与 `s_knn` 压制。
- 当前 `s_graphdens` 没有提供图结构增量；它弱于 `s_graphnbr`，且 `s_combo_HG - s_combo_H` 为负。
- `kNN + ball` 可部署集成没有稳定超过 `s_knn`。

### 合理猜测

- 在当前 GRACE H 空间中，粒球主要是 H-kNN 标签一致性的粗粒化近似。
- Phase39/40A 与分类任务中的 Phase24/GBSelect 失败共享同一机制：球信号来自 H，因而容易被同空间的更细粒度 kNN 或强 GCN 吸收。

### 尚未验证的假设

- 非 GRACE 表示空间、非 uniform 0.4 噪声、非 Cora/CiteSeer/Coauthor-CS 数据集上，检测任务是否仍被 kNN/GCN 压制。
- 是否存在完全非 H 同源的图结构信号，能与球检测形成真正正交增量。
- 是否可以构造一个不依赖 clean flag 的可部署模型级集成，超过 `s_knn` 且不损失 P@K。

### 已被负结果削弱的方向

- 正式换题为图节点标签错误检测与清洗。
- 继续做球检测变体追赶 `s_knn` / `s_gcn_conf`。
- 用球级图边密度作为图结构增量主卖点。
- 把 ball 作为 kNN 的稳定辅助模块。

---

## 5. 后续建议

- 主线继续写 SoftDist `[X‖H]` 分类论文，不再把检测任务作为替代主线。
- Phase39/40A 应进入论文边界分析：粒球离线去噪强，但其有效信息可被同空间 kNN/GCN 解释并超过。
- 若未来想重启检测方向，必须先预注册一个非 H 同源的正交信号，并把 `s_knn`、`s_gcn_conf`、`s_graphnbr` 作为命门 baseline。

---

## 6. Phase40a-plus：H-space 多视角 rank fusion 诊断（2026-06-24）

### 研究假设

如果 `s_knn`、`s_gcn_conf`、`s_ball_loo`、`s_graphnbr` 之间存在可部署互补性，则无监督 percentile-rank fusion 应稳定超过每个 seed 上的最强单项 detector。

### 粒球角色

粒球只作为候选视角 `s_ball_loo`，用于判断粗粒度局部一致性是否能补充 kNN / GCN；不作为主方法、不训练下游分类器。

### 方法动机

Phase40A 只验证了 `kNN + ball`；本轮最小验证扩展到 `kNN + GCN + ball (+ graphnbr)`，目的是决定是否值得进入 KNN/GCN teacher SoftDist 下游 sanity check。

### 代码改动

- 新增 `external/NoisyGL/scripts/probe_hspace_multiview_fusion.py`
- 新增 `external/NoisyGL/tests/test_hspace_multiview_fusion_probe.py`
- 新增执行记录 `external/NoisyGL/docs/execution_plan_hspace_multiview_fusion_20260624.md`

### 配置改动

无。沿用 Phase39：Cora/CiteSeer/Coauthor-CS，uniform 0.4，seeds 0..4，percent split 0.2/0.1/0.7，frozen GRACE H，同源 SoftDist `[X‖H]` 构球参数。

### 成功标准

rank fusion 继续进入下游的最低条件：

- ≥2/3 数据集超过 per-seed best single；
- mean AUROC 或 AUPRC_lift 提升 ≥0.01；
- paired seeds 至少 4/5 正向；
- AUPRC / Precision@K 不系统性下降。

### 实验结果

结果目录：`external/NoisyGL/runs/hspace_multiview_fusion_20260624/`

核心报告：`external/NoisyGL/runs/hspace_multiview_fusion_20260624/hspace_multiview_fusion_report.md`

关键结果：

| Dataset | best single | Fusion | ΔAUROC vs best single | pos/5 | ΔAUPRC lift | ΔP@K |
|---|---|---|---:|---:|---:|---:|
| Cora | `s_knn` | `F1_rank_knn_gcn` | +0.0076 | 5/5 | +0.0365 | +0.0079 |
| Cora | `s_knn` | `F4_rank_knn_gcn_ball` | +0.0070 | 4/5 | +0.0334 | +0.0009 |
| CiteSeer | `s_knn` | `F4_rank_knn_gcn_ball` | -0.0079 | 1/5 | +0.0145 | -0.0205 |
| Coauthor-CS | `s_gcn_conf` | `F4_rank_knn_gcn_ball` | -0.0059 | 0/5 | -0.0136 | -0.0179 |

Oracle `O2_logreg_knn_gcn_ball` 有小幅上界增益：Cora +0.0106、CiteSeer +0.0059、Coauthor-CS +0.0006 AUROC，但使用 clean/noisy flag 跨 seed 训练，不能作为可部署方法。

### 最终决策

**FAIL / 不进入 Phase40b。**

rank fusion 没有在 ≥2/3 数据集上超过 per-seed best single。Cora 有小正增益但低于 +0.01 AUROC；CiteSeer 和 Coauthor-CS 为负。加入 ball 的 `F4_rank_knn_gcn_ball` 不稳定，不能证明 ball 对 `kNN/GCN` 有可部署互补价值。

新增结论：

> H-space 多视角检测融合没有稳定超过最强单项 kNN/GCN；粒球和无监督 fusion 都不能提供足够稳定的额外收益。KNN-SoftDist / KNN+GCN teacher 下游实验暂不继续。

## 7. 文件索引

- Phase39 执行计划：`external/NoisyGL/docs/execution_plan_noise_detection_feasibility_20260624.md`
- Phase39 决策：`external/NoisyGL/runs/noise_detection_feasibility_20260624/phase39_gate_decision.md`
- Phase39 报告：`external/NoisyGL/runs/noise_detection_feasibility_20260624/noise_detection_feasibility_report.md`
- Phase40A 执行计划：`external/NoisyGL/docs/execution_plan_knn_ball_ensemble_probe_20260624.md`
- Phase40A 报告：`external/NoisyGL/runs/knn_ball_ensemble_probe_20260624/knn_ball_ensemble_report.md`
- Phase40A 汇总：`external/NoisyGL/runs/knn_ball_ensemble_probe_20260624/knn_ball_ensemble_summary.csv`
- Phase40A 脚本：`external/NoisyGL/scripts/probe_knn_ball_ensemble.py`
- Phase40A 测试：`external/NoisyGL/tests/test_knn_ball_ensemble_probe.py`
- Phase40a-plus 执行记录：`external/NoisyGL/docs/execution_plan_hspace_multiview_fusion_20260624.md`
- Phase40a-plus 报告：`external/NoisyGL/runs/hspace_multiview_fusion_20260624/hspace_multiview_fusion_report.md`
- Phase40a-plus 汇总：`external/NoisyGL/runs/hspace_multiview_fusion_20260624/hspace_multiview_fusion_summary.csv`
- Phase40a-plus 节点分数：`external/NoisyGL/runs/hspace_multiview_fusion_20260624/node_scores_raw.csv`
