# 执行计划：SSL Encoder 升级 + 数据集扩展 + 框架泛化（Phase 8-11）

日期：2026-06-12
执行者：Codex
前置状态：机制探索已收官（见 `execution_plan_xh_expansion_20260610.md` 与经验文档 `02_experiment_history_and_lessons 11.md`）。主方法 **SoftDist [X||H]** 已 5-seed 确认（Cora 0.8137±0.0067 ≥ NRGNN，CiteSeer 0.6891±0.0159 差 NRGNN 1.36pp）。本计划不再改机制，只换/升级管线的**输入资产**（H 的来源、数据集、下游 backbone），目标：CiteSeer 翻盘 NRGNN + 建立"通用去噪框架"的论文 claim。

---

## 0. 环境与红线（必读）

- ROCm Python：`C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe`，device `cuda:0`
- NoisyGL 运行必须显式：`--split_type percent --train_percent 0.2 --val_percent 0.1 --test_percent 0.7`
- 现有 GRACE H：`gb_graph_aware_minival_6_1/checkpoints/embeddings/{Cora,CiteSeer}/ratio_train0p2_val0p1_split0_pretrain0/H.pt`
- GRACE 预训练代码在 `gb_graph_aware_minival_6_1/`（负责产出 H 的上游仓库），新 SSL 方法的预训练脚本也放该仓库或其旁边的新目录，**不要塞进 NoisyGL**
- **红线**：
  1. SoftDist [X||H] anchor（`frozengrace_gb_gcn_softdist_xh`）的行为不得改动；新 H 通过 config 的 `gb.embedding_path` 换入
  2. SSL 预训练永远 label-free：不碰 y、不碰 train_mask，只用 (X, A)。预训练超参不得用下游 test 调
  3. 机制层面冻结：不要动球构造、SoftDist target、alpha 公式（判死清单见旧计划文档）
  4. 每个阶段一份 research-record 格式报告（假设/角色/改动/门槛/结果/决策）
- 单元测试保持全绿：`py -3 -m unittest discover -s tests`

### 当前锚点（uniform 0.4, percent split, 5-seed, ROCm）

| 方法 | Cora | CiteSeer |
|---|---:|---:|
| SoftDist [X||H] (GRACE) | 0.8137±0.0067 | 0.6891±0.0159 |
| GCN+CE [X||H] (GRACE) | 0.8027±0.0069 | 0.6761±0.0181 |
| NRGNN | 0.8107±0.0230 | 0.7027±0.0108 |
| GCN+CE | 0.7842±0.0151 | 0.6530±0.0121 |
| GRACE H linear probe（参考） | ~0.837 | ~0.68（待 Phase 8 统一重测） |

---

## Phase 8：SSL encoder 候选池 + linear probe 筛选（先做，廉价过滤）

### 目的

H 质量是全管线最大杠杆（H 给分类器 +1.9/+2.3pp，给球提供几何，双通道受益）。CiteSeer 的 H 增量更大（+2.3pp），说明其差距很可能就在表示质量。先用 linear probe 筛掉弱候选，省下游算力。

### 候选（4 个新 + 1 个锚点）

| 候选 | 类型 | 来源 | 备注 |
|---|---|---|---|
| GRACE（锚点） | InfoNCE 对比 | 已有 H | 不重训，直接 probe 作为基准 |
| GRACE-learnable-τ | InfoNCE，温度可学习 | 改 GRACE 代码 | τ 初始化 0.5，`log_tau` 作为可训练参数，clamp τ∈[0.05, 2.0]；**预期值低**（预训练侧改动历史上全败），但改动温和值得一测 |
| BGRL | negative-free bootstrap | 官方实现 | 无负样本，超参相对鲁棒 |
| GraphMAE | 生成式 masked autoencoder | 官方实现 | 2022 后节点分类普遍强于 GRACE 的代表 |
| CCA-SSG | 特征去相关 | 官方实现 | 轻量、超参少 |

### 实现要求

1. 每个候选在 Cora/CiteSeer 各预训练（label-free，只用 X 和 A），导出 H 到统一目录：
   `checkpoints/embeddings_v2/{dataset}/{ssl_method}/H.pt`（与现有 GRACE H 同格式：N×d float tensor 或含 "H" 键的 dict）
2. 超参：用各论文官方 Cora/CiteSeer 默认值，**不做网格搜索**（防止用下游信息调预训练）；embedding 维度尽量对齐 GRACE 现有维度，如官方默认不同则记录维度
3. 预训练 seed 固定 0（probe 阶段单 seed 足够；进入 Phase 9 的胜者再补 seed 多样性——见 Phase 9 注 b）
4. **linear probe 协议（统一新写一个脚本 `scripts/linear_probe_embeddings.py`）**：
   - 与下游同一 percent split（0.2/0.1/0.7）、同一 seeds 0-2
   - 用 **clean train 标签**训练逻辑回归/单层线性层 → test acc（这是表示质量上界诊断，clean 标签仅在 probe 内部用，符合惯例且不进任何下游训练）
   - 同时报告 **noisy probe**（用 noisy train 标签训练的 probe test acc），它更贴近下游实际可用性
5. 输出：`runs/ssl_probe_20260612/probe_summary.csv` + 报告 md

### 通过门（进入 Phase 9 的条件）

- clean probe **或** noisy probe 在任一数据集上超过 GRACE 锚点 ≥ 1pp 的候选进入 Phase 9
- 全部候选都不超过 → Phase 9 仅带 GRACE-learnable-τ 中表现最好的一个走完流程作记录，然后跳到 Phase 10（数据集扩展仍用 GRACE）

---

## Phase 9：probe 胜者接入下游管线（seed0 → 5-seed）

### 运行

对每个过门候选 H'：

1. 复制 `config/frozengrace_gb_gcn_softdist_xh/` 与 `gcn_xh_ce` 配置，仅改 `gb.embedding_path` 指向 H'（方法名不变，结果目录区分：`runs/ssl_swap_{method}_20260612/`）
2. seed0 smoke：`softdist_xh` + `gcn_xh_ce` + `softdist_xh_shuffled` × 2 数据集
3. seed0 时 `softdist_xh(H')` ≥ `softdist_xh(GRACE)` − 0.5pp 的候选 → 扩 5-seed（同三方法）

注意：
a. 球建在 H' 上、特征拼的也是 H'，保持"同一表示双通道"的架构语义
b. 若某候选进入 5-seed，预训练仍可用单个 checkpoint（下游 5 seeds 提供方差估计；如果该候选最终成为论文主配置，再补 3 个预训练 seed 验证预训练方差，作为 robustness 附表）

### 判定

- **主成功判据：CiteSeer `softdist_xh(H')` 5-seed mean > 0.7027（NRGNN）**，同时 Cora 不低于 0.8107−std
- 达成 → 新 H' 成为论文主配置，GRACE 降级为消融行（"框架对 SSL 方法即插即用"的证据）
- 未达成但有候选两数据集都 ≥ GRACE 版 → 仍换为主配置（增量照收）
- 全部不如 GRACE → 记录"GRACE 在该协议下已是局部最优 SSL"，主配置不变；probe-vs-下游的相关性数据本身进论文分析

### 必报诊断

- 每个 H' 下的球统计（num_balls、mean_train_per_ball、noisy/true purity——复用 `partition_purity_diagnostics`）与 NCR/CDR：检验"更好的 H 是否同时改善球质量"，这是框架故事的关键证据链
- probe acc vs 下游 acc 的对应表（验证 probe 作为廉价筛选器的有效性）

---

## Phase 10：数据集扩展（论文刚需，与 Phase 9 可部分并行）

### 数据集

| 数据集 | 类型 | 备注 |
|---|---|---|
| PubMed | 引文图，3 类 | NoisyGL 原生支持；3 类下 margin 行为不同（随机 margin=1/3），multiclass alpha 自动适配，无需改码 |
| Amazon-Photo (`amazonpho`) | 共购图，8 类 | NoisyGL 原生支持；非引文域，homophily 性质不同，检验泛化 |

（如时间富余可加 `amazoncom`；roman-empire/amazon-ratings 是异质图，本框架不主张覆盖，不做）

### 运行

1. 用**当前主配置 SSL**（Phase 9 结束时确定；若 Phase 9 未完可先用 GRACE 起跑，胜者出来后只补该胜者）在 PubMed/Amazon-Photo 预训练，导出 H
2. 每个数据集跑 uniform 0.4（percent split 同协议）× 5 seeds：
   - `softdist_xh`、`gcn_xh_ce`、`softdist_xh_shuffled`、`gcn`（CE baseline）、`nrgnn`
3. PubMed 较大（~19.7k 节点），如球构造或 NRGNN 显著变慢，如实记录耗时即可，不为速度改机制
4. 检查各数据集 config 的 `gb:` 参数：`max_balls=64, min_ball_size=32` 起步**不调**；若出现退化球（num_balls < 8 或 singleton 多），报告中记录并允许一次性调整为 num_balls 接近 64 的最小改动，调整必须双数据集同规则

### 判定

- `softdist_xh` 在新数据集上 ≥ GCN+CE 且 real > shuffled → 框架泛化 claim 成立的最低标准
- 对 NRGNN 的胜负如实记录，不强求赢
- 任一数据集上管线失效（输 GCN+CE 或 shuffled 不输 real）→ 该数据集进 limitation，必报球诊断定位原因（H 质量 or 球覆盖 or 类数效应）

---

## Phase 11：下游 backbone 泛化附表（最后做，小成本）

### 目的

支撑"任意 GNN 分类器"的框架 claim。**主表仍是 GCN**（NoisyGL baseline 全是 GCN backbone，换 backbone 比主表不公平）。

### 运行

- backbone：GAT、APPNP（NoisyGL `predictor/module/GNNs.py` 已有则直接用；没有则用 PyG 标准实现加最小 predictor 子类，输入层适配 [X||H] 维度即可）
- 设置：主配置数据集（Cora/CiteSeer + Phase 10 胜出数据集）× uniform 0.4 × seeds 0-2（附表 3 seeds 够）
- 方法：`softdist_xh(backbone)` vs `gcn_xh_ce 同backbone版`（即 backbone+CE+[X||H]，隔离球增量）

### 判定

- 球增量（softdist − CE 同 backbone）在 ≥ 2 个 backbone 上为正 → "分类器无关"claim 成立
- 某 backbone 上球增量消失 → 如实记录为条件性结论

---

## 执行顺序与依赖

```
Phase 8 (probe 筛选) ──► Phase 9 (下游验证, 主战场: CiteSeer 翻盘)
        │                        │
        └──► Phase 10 (数据集扩展) ◄── 主配置确定后补新 SSL 的 H
                       │
                       └──► Phase 11 (backbone 附表)
```

- Phase 8 是全计划成本最低、信息量最大的一步，先做
- Phase 10 的 GRACE 版可与 Phase 9 并行起跑
- 任何阶段不达标都不阻塞后续（数据集扩展和 backbone 附表是论文刚需，与 SSL 升级是否成功无关）

## 论文叙事出口（三种结果都有路）

| Phase 9 结果 | 论文定位 |
|---|---|
| CiteSeer 翻盘 NRGNN | "通用 SSL-GB 去噪框架，超越 2024 SOTA"——主 claim 升级 |
| 换 SSL 有增量但未翻盘 | "框架对 SSL 即插即用，表示质量是清晰的改进轴"——框架 claim + 增量 |
| GRACE 已最优 | 维持现 claim（Cora 持平且稳 + 简单一个量级），probe 分析进附录 |
