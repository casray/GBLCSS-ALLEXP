# 执行计划：2025 外部方法公平对比（Phase 24-27）

日期：2026-06-15
执行者：Codex
前置状态：SoftDist [X||H] 论文实验闭环完整，但外部对比全是 NoisyGL 自带方法（最新 TSS, ICML 2024）。审稿人会质疑「未对标 2025 方法」与「[X||H] 多用了 H 不公平」。本计划补两类关键对比：(1) 2025 代表方法（TFR/ProCon/GD²）在**同协议**下重跑；(2) **公平性对照**——给最强 baseline 也喂 [X||H]，证明球校正有独立于 H 的增量。**不改主方法机制**。

---

## 0. 环境与红线（沿用 + 新增外部仓库注意）

- ROCm Python：`C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe`，`cuda:0`，AMD RX 7700 XT
- NoisyGL 必须显式：`--split_type percent --train_percent 0.2 --val_percent 0.1 --test_percent 0.7`
- GRACE H 锚点：`gb_graph_aware_minival_6_1/checkpoints/embeddings/{Cora,CiteSeer}/ratio_train0p2_val0p1_split0_pretrain0/H.pt`
- 红线：anchor 机制不动；clean label 只进诊断；research-record 报告；测试全绿
- **外部仓库新增红线**：
  1. 任何外部方法必须在**我们的协议**下重跑（相同 split / 噪声率 / seed / train-only 噪声）才算「公平对比」；用原文数字的一律标注「参考结果，非同协议」
  2. ROCm 适配：参考 Phase 23 TSS 经验，外部代码常有 `.cuda()` 硬编码 → 改成用 tensor 的 device，记录每处改动，**不改算法逻辑**
  3. 外部代码克隆到 `external/` 下独立目录，不污染 NoisyGL；依赖装在同一 ROCm venv，冲突则记录

### 当前外部对比锚点（uniform 0.4, 5-seed, percent 0.2/0.1/0.7）

| 方法 | Cora | CiteSeer | 来源 |
|---|---:|---:|---|
| SoftDist [X||H]（本文） | 0.8137±0.0067 | 0.6891±0.0159 | — |
| NRGNN | 0.8107±0.0230 | 0.7027±0.0108 | NoisyGL 自带 |
| 其余 NoisyGL 自带 | 见 baseline_completion | | UnionNET/RTGNN/TSS/CGNN/CRGNN/R2LP |

---

## Phase 24：公平性对照 —— NRGNN + [X||H]（先做，零外部依赖，最关键）

### 目的

**这是全计划最重要的一格，回答审稿人最致命的质疑**：SoftDist 赢是不是只因为多用了冻结表示 H？给最强 baseline NRGNN 同样的 [X||H] 输入，若 SoftDist 仍有独立增量，则证明球校正的价值独立于 H。

### 实现

- NRGNN 的输入特征改为 `[X || L2norm(H)]`（复用 `concat_l2_normalized_embeddings`，与 SoftDist 完全相同的拼接方式）
- 注意 NRGNN 内部可能有基于特征的边预测器，输入维度变化需相应调整其首层；记录改动，不改 NRGNN 算法逻辑
- 新方法名 `nrgnn_xh`，注册进 NoisyGL；config 加 `gb.embedding_path`
- {cora, citeseer} × uniform 0.4 × 5 seeds

### 对照矩阵（报告必须给全）

| 方法 | 用 H? | 用球? | 回答什么 |
|---|---|---|---|
| NRGNN | 否 | 否 | 原始最强 baseline |
| NRGNN + [X||H] | 是 | 否 | H 对 NRGNN 的增量 |
| gcn_xh_ce（已有） | 是 | 否 | H 对 CE 的增量 |
| SoftDist [X||H]（已有） | 是 | 是 | 完整方法 |

### 判定

- **核心问题**：SoftDist [X||H] − NRGNN[X||H] 的 paired delta。若 ≥0（至少 Cora 为正、CiteSeer 不大幅落后）→ **球校正有独立于 H 的增量，公平性质疑被驳倒**，这是论文最强论据之一
- 若 NRGNN[X||H] 反超 SoftDist → 诚实记录：H 是主要增益来源，球的边际价值有限，论文叙事需收缩到「H + 简单软校正的高效组合」
- 单元测试：拼接后维度正确、NRGNN[X||H] 在 λ_ball=0 等价于喂 [X||H] 的原 NRGNN

---

## Phase 25：TFR (CIKM 2025) —— 优先外部方法（同框架，最易集成）

### 为什么先做 TFR

TFR 明确**基于 NoisyGL 框架实现**（https://github.com/eaglelab-zju/TFR），与我们同框架 → split/噪声/seed/protocol 天然对齐，集成成本和翻车风险最低。先用它验证「集成一个 2025 外部方法」这条路是否通畅，再决定是否碰更难的 ProCon/GD²。

### 实现

1. clone TFR 到 `external/TFR/`，比对其与本地 NoisyGL 的版本差异（predictor 接口、config 格式）
2. 若 TFR 的 predictor 能直接注册进本地 NoisyGL：注册 `tfr`，用其官方 Cora/CiteSeer config（不调参，与我们跑 NRGNN 同标准）
3. ROCm 适配：按红线改 `.cuda()` 硬编码
4. {cora, citeseer} × uniform 0.4 × 5 seeds，percent split
5. 输出 `runs/tfr_comparison_20260615/`

### 判定

- 跑通 → 进主表，paired delta vs SoftDist
- 跑不通（依赖/ROCm/接口不兼容）→ 记录失败原因，引用 TFR 原文 Cora/CiteSeer 数字并明确标注「参考结果，非同协议」（注意原文多用标准 split，不可直接比，仅作上下文）

---

## Phase 26：ProCon (IJCAI 2025) —— 机制最近邻（强建议，独立仓库）

### 为什么做 ProCon

ProCon 用 feature similarity + adjacency 找 semantic peers 再做噪声检测（https://www.ijcai.org/proceedings/2025/623），与我们「局部语义单元 + 置信度」叙事最近，正面区分度最高，是 related work 必须正面对比的方法。

### 实现（仅当 Phase 25 证明外部集成可行后再起）

1. clone 到 `external/ProCon/`，独立仓库，集成成本高于 TFR
2. **protocol 对齐是关键风险**：ProCon 官方默认 Planetoid 标准 split。需将其 data loader 改为接受我们的 percent split + train-only 噪声 + 相同 seed。改动必须详细记录，并加一个 sanity check（同 seed 下 split 的 train/val/test 节点集与 NoisyGL 一致）
3. ROCm 适配同前
4. {cora, citeseer} × uniform 0.4 × 5 seeds
5. 输出 `runs/procon_comparison_20260615/`

### 判定

- 同协议跑通 → 进主表（这是最有价值的 2025 对比，机制最近）
- protocol 无法对齐或 ROCm 跑不通 → 引用原文数字标「参考结果，非同协议」，并在 related work 做**机制层面**的定性区分（peer detection vs 粒球软分布校正）

---

## Phase 27（可选）：GD² (NeurIPS 2025)

### 说明

GD² 用 ego-view vs structure-view 预测分歧检测噪声，最新但机制离我们最远、集成最贵。**仅当 24-26 完成且时间富余才做**。跑不通就只在 related work 引用，不强求同协议对比——一篇论文不需要穷尽所有最新方法，2-3 个有代表性的 2025 方法已足够回应「未对标最新」的质疑。

---

## 执行顺序与依赖

```
Phase 24（NRGNN+[X||H]，零外部依赖，公平性核心）─┐ 先做，最关键
Phase 25（TFR，同框架最易集成）─────────────────┤ 与 24 并行
        │
        └─集成可行─► Phase 26（ProCon，独立仓库，protocol 对齐）
                          │
                          └─时间富余─► Phase 27（GD²，可选）
```

- Phase 24 是论文最强论据来源，必须先做且必出结果
- Phase 25 先验证外部集成可行性，再决定 26/27 投入
- 任一外部方法跑不通都有退路（引用原文 + 标注参考结果），不阻塞

## 论文叙事出口

| Phase 24 结果 | 论文定位 |
|---|---|
| SoftDist ≥ NRGNN[X||H] | 「球校正增量独立于 H」——驳倒公平性质疑，最强论据 |
| NRGNN[X||H] 反超 | 收缩为「H + 高效软校正组合」，球作为轻量增强 |

| 2025 方法对比结果 | 论文定位 |
|---|---|
| 同协议下 SoftDist 持平/超过 TFR/ProCon | 「对标 2025 方法仍具竞争力，且非迭代更简单」 |
| 部分输 | regime/复杂度差异化叙事（我们更简单、更稳、表示噪声免疫） |

## 完成后

把 Phase 24-27 结果回填进 `SoftDist_XH_论文初稿雏形.html` §3.1 主表（新增 2025 方法行 + NRGNN[X||H] 公平性行）与 §4 消融（公平性对照矩阵），更新 §9 related work 的机制区分段，更新经验文档与记忆。

---

## 执行记录（2026-06-16）

### Phase 24：NRGNN + [X||H]

已完成本地 NoisyGL 集成，新增 `nrgnn_xh`，只把输入特征替换为与 SoftDist 相同的 `[X || L2norm(H)]`，不改 NRGNN 的训练逻辑、伪标签机制或边机制。

| 数据集 | NRGNN | NRGNN [X||H] | SoftDist [X||H] | 关键 paired delta |
|---|---:|---:|---:|---:|
| Cora | 0.8107 ± 0.0230 | 0.8139 ± 0.0170 | 0.8137 ± 0.0067 | SoftDist - NRGNN[X||H] = -0.02pp, 3/5 |
| CiteSeer | 0.7027 ± 0.0108 | 0.7090 ± 0.0187 | 0.6891 ± 0.0159 | SoftDist - NRGNN[X||H] = -1.99pp, 1/5 |

结论：公平性对照后，不能再声称 SoftDist 全面超过强 H-augmented baseline。更稳妥的表述是：Cora 上与 NRGNN[X||H] 实质持平且方差更低；CiteSeer 上 NRGNN[X||H] 更强。报告见 `runs/phase24_nrgnn_xh_20260616/phase24_fairness_report.md`。

### Phase 25：TFR (CIKM 2025)

已 clone 官方仓库 `https://github.com/eaglelab-zju/TFR.git`，commit `d7e9ff8a58d457b519f068c3d170cd37579c8bd8`。官方仓库没有 Cora/CiteSeer config，因此把官方 `tfr_gcn_dblp.yaml` 模板迁移到 Cora/CiteSeer，不调参。为避免额外依赖，TFR decoder 中的 `torch_scatter.scatter(..., reduce="mean")` 用原生 `index_add_` mean 聚合替代，语义保持为均值聚合。

| 数据集 | TFR 迁移配置 | vs SoftDist [X||H] | vs NRGNN [X||H] |
|---|---:|---:|---:|
| Cora | 0.7758 ± 0.0185 | -3.79pp, 0/5 | -3.81pp, 0/5 |
| CiteSeer | 0.6979 ± 0.0129 | +0.88pp, 3/5 | -1.12pp, 1/5 |

结论：TFR 可同协议运行，但这些数字是“官方 DBLP 模板迁移配置”，不是官方 Cora/CiteSeer tuned 结果。报告见 `runs/tfr_comparison_20260616/tfr_comparison_report.md`。

### Phase 26：ProCon (IJCAI 2025)

已检查 IJCAI 页面与公开搜索结果，未找到官方或作者明确关联的代码仓库。因此不做自实现冒充外部 baseline，不进主表。ProCon 只在 related work 中作为 2025 peer-informed label consistency 方法讨论；若引用原文数字，必须标注为“参考结果，非同协议”。报告见 `runs/procon_comparison_20260616/procon_code_availability_report.md`。

### Phase 27：GD²

本轮未执行。Phase 24 已回答公平性核心问题，Phase 25 已完成一个 2025 同协议外部方法，Phase 26 因代码不可得转为 related-work 处理。GD² 仍保留为可选 related-work 引用。
