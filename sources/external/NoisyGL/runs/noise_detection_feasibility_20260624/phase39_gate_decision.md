# Phase 39 Gate Decision

日期：2026-06-24

结论：**暂不正式换题为“图节点标签错误检测与清洗”。**

原因不是粒球没有检测信号，而是按 v2 预注册闸门：

- **G1 正交性：通过（由 `s_ball_loo` 通过）**
  - `s_ball_loo` 在 3/3 数据集上均满足 `knn_clean50` 与 `gcn_clean50` 条件 AUROC ≥ 0.6、5/5 seeds > 0.5、real−shuffled ≥ 0.02。
  - 这说明 LOO 粒球分数确实包含 H-kNN / GCN 认为“干净”的区域中仍能发现真噪声的残差信息。
- **G2 检测质量：严格意义失败**
  - `s_ball_loo` 绝对检测质量强，但整体 AUROC 仍被最强 H/model baseline 压制：
    - Cora: `s_ball_loo` 0.9016±0.0111 < `s_knn` 0.9474±0.0113
    - CiteSeer: `s_ball_loo` 0.8257±0.0111 < `s_knn` 0.8850±0.0106
    - Coauthor-CS: `s_ball_loo` 0.9614±0.0009 < `s_gcn_conf` 0.9938±0.0005
  - 因此不能声称“粒球检测器优于简单 H/model baseline”。
- **G4 图增量：失败**
  - `s_graphdens` 明显弱于纯图邻居 baseline `s_graphnbr`：
    - Cora: −0.1490 AUROC
    - CiteSeer: −0.0777 AUROC
    - Coauthor-CS: −0.2498 AUROC
  - `s_combo_HG − s_combo_H` 也均为负，说明当前图边密度特征没有提供正增量。

## 闸门判定

| Gate | 判定 | 依据 |
|---|---|---|
| G1 正交性 | Pass | `s_ball_loo` 3/3 数据集过线 |
| G2 检测质量 | Fail/Partial | 绝对 AUROC 高，但被 `s_knn` / `s_gcn_conf` 稳定压制 |
| G3 类别覆盖 | 未作为阻断项 | 主结论已由 G2/G4 决定 |
| G4 图增量 | Fail | `s_graphdens < s_graphnbr` 且 `s_combo_HG < s_combo_H` |

## 决策

不进入 Phase 39b PubMed boundary run。按 v2 计划，Phase 39b 仅在 G1 且 G2 过时触发；本轮 G2 未过。

当前最稳妥的论文定位不是“正式换成图节点标签错误检测”，而是：

1. 粒球 LOO 分数是强的离线标签噪声信号；
2. 它确实具有一定条件正交性；
3. 但整体检测能力仍被简单 H/model baseline 压制；
4. 当前图边密度版本没有提供图结构增量；
5. 因此检测方向可作为边界分析/辅助诊断保留，但不足以直接替换主任务。

