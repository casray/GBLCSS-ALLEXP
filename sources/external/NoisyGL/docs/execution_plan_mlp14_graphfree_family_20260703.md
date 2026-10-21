# 执行计划：Graph-free family 定位表补跑（GSDN + Graph-MLP）（Phase MLP-14）

日期：2026-07-03
执行者：Codex
定位：**表 B（T1b，graph-free family 定位表）的 baseline 补跑，不是主表变更、不是新机制。** 把两个 graph-free 竞品放进我们的噪声协议：GSDN（设定最近邻，防御价值最高）与 Graph-MLP（表征侧代表，成本最低）。它们进 **表 B**，与 vanilla MLP / Ours train-only / GLNN-style / Ours-distill / GCN 参照行（全部来自 MLP-9 重组）并列；**绝不进主表 A**（它们使用全体无标签节点/邻接对比 = 蒸馏/表征资源轴，MLP-7 铁律）。NOSMOG（推理需位置嵌入，非严格 X-only）/ VQGraph（codebook 工程量大，FINE 陷阱）维持引用切割不跑。

## 0. 背景与诚实先验

- **为什么跑 GSDN**：arXiv 2210.02097v5（Under review，未正式接收——论文引用时如实标注）。它是设定最近邻：纯 MLP + 训练期邻域自蒸馏 + 推理无图依赖，且自带 asymmetric 噪声曲线。审稿人若熟悉 graph-free 圈，第一个问的就是它。跑了 = "没有回避最相似工作"的最强证明。
- **为什么跑 Graph-MLP**：arXiv 2106.04051。邻域对比损失 + MLP，实现小、是 graph-free 圈的奠基 baseline，作为"表征侧、非噪声中心"的代表进表 B。
- **诚实先验（两个方向都写在前面）**：
  1. **GSDN 在高噪声下预期天然吃亏，这是机制属性不是我们削弱它**：其 label-level 自蒸馏（Eq.6/8）把 train 节点的（噪声）标签直接传播给邻居的预测头监督——train-only 高噪声下等于主动扩散噪声。**预注册这个解释，赢了不吹、输了不藏。**
  2. **反向风险同样预注册**：GSDN 的 feature-level 自蒸馏用了全体节点的邻域一致性（结构正则），在低噪声（u0.2）下可能强于我们的 train-only β0.5——若发生，如实进表，解释挂"资源轴差异"（它用全图边集，我们只用 train 标签），这正是表 B 资源标注列存在的意义。
  3. Graph-MLP 无标签侧机制，预期噪声下接近 CE 的退化速度，但其对比正则可能带来低噪声小优势。
- **表 B 的叙事目的**：不是"我们赢 graph-free 竞品"，而是"在同一噪声协议下，各 graph-free 方法的资源-鲁棒 profile 一目了然"。我们的 train-only β0.5 在绝对精度上可能低于用全体节点的方法——这由 Ours-distill 行回应（资源对齐后我们最强，MLP-9 已证）。

## 0.1 红线（预注册）

1. **进表 B 不进主表 A**；表 B 必须带资源标注列（训练用图？推理用图？用无标签节点？用全部边？）。
2. **忠实适配优先**：两方法的核心机制照原文实现（GSDN：feature-level mixup 插值一致 + 均匀负采样 + label-level 邻居监督，λ 与 batch size 按原文网格用 noisy val 选；Graph-MLP：NContrast loss，r-hop 权重与 τ 按原文默认）。backbone 对齐我们的协议（2 层 MLP、hidden 64——若原文默认宽度差异过大导致明显欠力，允许一次预声明的宽度对齐重跑并记录两版）。标注 "adapted to our protocol"。
3. **公平性双向声明**：不给它们加任何抗噪组件（原文没有的不加）；也不删它们原文有的组件（GSDN 的 label-level 项照跑——它是原设计，哪怕它扩散噪声）。唯一协议性替换 = 数据集/split/噪声注入换成我们的。
4. clean 只评测；噪声只加 train；协议显式 `--split_type percent --train_percent 0.2 --val_percent 0.1 --test_percent 0.7`；ROCm `cuda:0`；paired 同 seed 同 split。
5. seed0 smoke 先行（实现健全门：status=ok、无 NaN、clean 或 u0.2 下 ≥ vanilla MLP——它们都是"增强 MLP"，连 MLP 都不如说明实现坏了），过门才 5-seed。
6. 判读纪律：<1×std 写 comparable；GSDN/Graph-MLP 的任何格照实进表，包括赢我们的格。
7. 本计划不改主表、不改主线 claim；只填表 B 的两行。

## 0.2 复用与新增

- **复用**：表 B 其余行全部现成（MLP-9 T1-T4 + teacher acc 重组；vanilla MLP 来自 MLP-5）；NoisyGL 宿主的噪声注入/split 管线；GSDN 原文文本已在 `文献文件/粒球噪声学习/GSDN.pdf`（实现对照用）。
- **新增**：
  - `predictor/GSDN_Predictor.py`（或并入 MLPNoiseBaselines）：MLP backbone + 双推理头 fθ/gγ + edge 采样 batch + mixup 可学习插值 + 均匀负采样 + label-level 邻居 CE。原文超参网格：λ ∈ {0.5, 0.8, 1.0}、batch ∈ {256, 512, 1024}（noisy val 选，逐数据集一次）。
  - `predictor/GraphMLP_Predictor.py`：MLP + NContrast loss（r=2 默认、τ 原文默认、α 用 noisy val 从原文网格选）。
  - configs：两方法 × {cora, citeseer, pubmed, coauthorcs, amazonpho} 各一份。
  - 单测：(a) GSDN 双头且推理只用 fθ（图不进前向）；(b) mixup 系数在 (0,1)；(c) label-level 项只用 train 标签；(d) Graph-MLP 推理不碰邻接；(e) 噪声只污染 train；(f) backbone 键与 mlp_ce 一致（宽度对齐条款除外，若触发则记录）。
  - 汇总脚本：`summarize_mlp14_graphfree_family.py` → 表 B 完整版（含资源标注列）。

## 1. 阶段 A —— seed0 smoke（20 run）

```
A1: {gsdn, graphmlp} × 5 数据集 × {uniform0.4, uniform0.6} × seed0 = 20 run
门 GA: 全部 status=ok、无 NaN/OOM；
      u0.4 下各自 ≥ vanilla MLP CE −2pp（防实现坏；它们是增强 MLP，不该输裸 MLP 太多）
```

不过门 → 修实现重跑 smoke（一次）；仍不过 → 该方法降级为引用切割，如实记录适配失败（像 FINE 一样标注"可能是我们的适配欠力"）。

## 2. 阶段 B —— 5-seed（过门后）

```
B1: {gsdn, graphmlp} × 5 数据集 × 5 设置(u0.2/0.4/0.6, pair0.4, instance0.4) × seeds 0-4
    = 250 run（seed0 的 u0.4/u0.6 复用，实际新增 230）
```

弱 MLP + edge 采样，约 1-1.5 天（GSDN 的 batch 式训练比我们慢，PubMed/CoauthorCS 注意墙钟）。

## 3. ★预注册判读

| 观察 | 表 B 写法 |
|---|---|
| GSDN/Graph-MLP 高噪声掉得比我们快 | "graph-free 但非噪声中心的方法在高噪声下退化"——预期主结局，挂 label-level 噪声扩散（GSDN）/无标签侧机制（Graph-MLP）解释 |
| 它们在 u0.2/clean 端赢我们 train-only | 如实进表，挂资源轴差异（全图边集 vs 仅 train 标签）；Ours-distill 行回应资源对齐后的对比 |
| 它们某格赢 Ours-distill | 如实进表——这是 kill-risk，若发生则表 B 叙事收窄为"profile 对比"，不写"全面最优" |
| GSDN 适配失败（GA 不过两次） | 降级引用切割，切割句用 related_work_cut_notes §3.2b 现成版本 |

## 4. 产出

- 输出目录：`runs/mlp14_graphfree_family_20260703/`
- 报告 `mlp14_report.md` 必含：
  1. 表 B 完整版：MLP / Graph-MLP / GSDN / Ours β0.5 (train-only) / GLNN-style / Ours-distill / GCN(X)、GCN[X‖H] 参照 × 5 设置 × 5-seed + **资源标注列**；
  2. 超参选择记录（noisy val 网格与选中值，逐数据集）；
  3. 逐方法 vs Ours train-only 与 vs Ours-distill 的 paired delta（≥4/5 才写"稳定"）；
  4. 预注册判读 §3 的对号入座；
  5. 完整性 expected/observed/missing。

## 5. 完成后

- 骨架 v2 T1b 行更新（GSDN/Graph-MLP 从"引用切割"改为"实测行"）；导师初稿表 B 段补两行数字。
- 更新记忆与 `99_session_handoff.md`。
- **表 B 就此关闭**：NOSMOG/VQGraph/TGS 维持引用切割（理由已预注册）；这是发稿前最后一批 baseline 补跑。

---

## 6. 执行结果备案（Codex 跑完后按 MLP-8~13 格式追加于此）

### 6.1 执行状态

- 时间：2026-07-03。
- 输出目录：`runs/mlp14_graphfree_family_20260703/`。
- 实现：新增 adapted graph-free family predictor（`gsdn` / `graphmlp`）、两方法五数据集配置、runner、summarizer 与单测。
- 协议：`split_type=percent, train_percent=0.2, val_percent=0.1, test_percent=0.7`；`noise_scope=train`；ROCm `cuda:0`；5 seeds。
- Smoke：20/20 run 完成，`MLP14_SMOKE=pass`。
- Full：250/250 graph-free run 完成，`MLP14_MISSING_GROUPS=0`。
- 参考行：MLP / Ours train-only 复用 MLP-5/10/12；GLNN-style、GLNN[X||H]、Ours-distill 复用 MLP-9，其中蒸馏参照仅覆盖 Cora/CiteSeer/PubMed 的 uniform0.4/0.6，CoauthorCS/AmazonPhoto 对应格保留 `NA`。
- 重要口径修正：本轮是 **adapted fixed-default** 实现，没有执行 noisy-val 超参网格选择。报告中 `mlp14_hyperparams.csv` 明确标记 `fixed_default_adapted_to_protocol`。论文不得写成“按 noisy val 调参后结果”，只能写“固定适配超参，忠实保留核心机制，adapted to our protocol”。
- 2026-07-03 修正：`summarize_mlp14_graphfree_family.py` 读取 `mlp_maintable_20260701` 参照行时已强制过滤 `train_percent=0.2`，避免把标签率端点（0.05/0.65）混入 Cora/CiteSeer u0.4 的 Ours train-only 参照。

### 6.2 验证

- `py -3 -m pytest tests\test_mlp14_graphfree_family.py -q`：`7 passed, 15 warnings, 12 subtests passed`。
- `py -3 -m py_compile predictor\GraphFreeFamily_Predictor.py scripts\summarize_mlp14_graphfree_family.py`：通过。
- `py -3 scripts\summarize_mlp14_graphfree_family.py --run_dir runs\mlp14_graphfree_family_20260703`：`MLP14_GRAPHFREE_ROWS=250`，`MLP14_REFERENCE_ROWS=340`，`MLP14_SMOKE=pass`，`MLP14_MISSING_GROUPS=0`。

### 6.3 关键结果

u0.4 主协议（mean ± std，%）：

| Dataset | MLP | Graph-MLP | GSDN | Ours train-only | GLNN | GLNN[X\|\|H] | Ours-distill |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cora | 44.24±2.62 | 63.23±3.07 | 55.97±4.29 | 55.59±1.50 | 77.95±2.62 | 80.23±1.57 | 80.99±1.02 |
| CiteSeer | 43.50±3.02 | 57.23±1.85 | 50.36±3.39 | 53.83±2.75 | 66.09±1.38 | 68.81±1.12 | 69.57±1.46 |
| PubMed | 63.77±4.40 | 67.54±3.11 | 71.42±1.15 | 71.90±0.88 | 84.32±0.47 | 84.80±0.85 | 85.23±0.94 |
| CoauthorCS | 88.42±0.62 | 88.95±0.43 | 86.55±0.78 | 90.62±0.57 | NA | NA | NA |
| AmazonPhoto | 73.90±1.80 | 77.05±1.24 | 73.31±2.00 | 81.64±0.93 | NA | NA | NA |

u0.6 高噪声（mean ± std，%）：

| Dataset | MLP | Graph-MLP | GSDN | Ours train-only | Ours-distill |
|---|---:|---:|---:|---:|---:|
| Cora | 32.51±3.38 | 45.66±2.71 | 36.76±3.01 | 41.04±2.37 | 71.07±2.29 |
| CiteSeer | 33.35±1.36 | 44.04±2.15 | 35.85±3.16 | 40.17±2.75 | 61.96±2.98 |
| PubMed | 45.50±2.45 | 47.89±3.86 | 50.83±1.73 | 47.53±2.42 | 75.76±2.70 |
| CoauthorCS | 80.03±1.91 | 81.28±1.32 | 79.32±1.05 | 87.17±0.62 | NA |
| AmazonPhoto | 58.60±3.77 | 65.90±4.53 | 56.57±1.67 | 70.78±3.19 | NA |

跨 5 数据集平均（%）：

| Regime | MLP | Graph-MLP | GSDN | Ours train-only |
|---|---:|---:|---:|---:|
| uniform0.2 | 71.68 | 78.92 | 75.71 | 77.31 |
| uniform0.4 | 62.77 | 70.80 | 67.52 | 70.72 |
| uniform0.6 | 50.00 | 56.95 | 51.87 | 57.34 |
| pair0.4 | 57.19 | 61.59 | 61.00 | 65.47 |
| instance0.4 | 61.78 | 69.81 | 66.69 | 70.48 |

paired delta 终判：

- Graph-MLP vs Ours train-only：25 个 dataset/regime 组中均值 `-0.65pp`，12/25 组均值为正；9/25 稳定正向，13/25 稳定负向。结论仍是 **profile 互有胜负**，但修正后 Ours train-only 的整体均值略高；不能写我们全面赢 Graph-MLP，因为 Graph-MLP 在 Cora/CiteSeer 的多个噪声设置稳定占优。
- GSDN vs Ours train-only：25 组均值 `-3.70pp`，4/25 组均值为正；2/25 稳定正向，20/25 稳定负向。结论是 adapted GSDN 在本 train-only 噪声协议下整体弱于 Ours train-only，尤其高噪声和 Coauthor/Amazon 上退化明显。
- Graph-MLP/GSDN vs Ours-distill：仅 Cora/CiteSeer/PubMed 的 uniform0.4/0.6 可配对；Graph-MLP 平均 `-19.83pp`，GSDN 平均 `-23.90pp`，6/6 组稳定负向。结论是资源轴对齐到蒸馏扩展后，Ours-distill 明显强于这两个 adapted graph-free baseline。

### 6.4 论文判读

1. 表 B 应写成 **resource/profile 对比**，不是“我们全面赢 graph-free family”。Graph-MLP 在 Cora/CiteSeer 和低噪声下很强，必须如实呈现。
2. Ours train-only 的优势集中在更难的高噪声均值、pair/instance 噪声、CoauthorCS/AmazonPhoto，以及不使用全图边/无标签节点的资源约束下。
3. GSDN 的结果支持预注册先验：其训练期 label-level 邻居监督在 train-only 高噪声协议下容易扩散噪声；但这是 adapted 实现下的观察，论文表述应克制。
4. 主表 A 不变；MLP-14 只进入表 B / graph-free family 定位表。NOSMOG/VQGraph/TGS 仍维持引用切割。
