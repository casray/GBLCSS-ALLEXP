# 执行计划：投稿前缺失实验补全（Phase 15-19）

日期：2026-06-12
执行者：Codex
前置状态：SoftDist [X||H] 论文初稿雏形已成（`SoftDist_XH_论文初稿雏形.html`）。主表、消融、噪声谱（uniform/pair）、SSL 表示分析、三类 gate 负结果、backbone 泛化数据齐备。本计划补齐投稿前必需的 P0/P1 缺口，**不改任何机制**（红线同前），全部是现成管线/现成 baseline 的算力消耗。

---

## 0. 环境与红线（沿用，必读）

- ROCm Python：`C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe`，device `cuda:0`
- NoisyGL 必须显式：`--split_type percent --train_percent <r> --val_percent <v> --test_percent <t>`（覆盖率实验 r 会变，见 Phase 17）
- GRACE H 锚点：`gb_graph_aware_minival_6_1/checkpoints/embeddings/{Cora,CiteSeer}/ratio_train0p2_val0p1_split0_pretrain0/H.pt`
- **红线**：
  1. `frozengrace_gb_gcn_softdist_xh` anchor 机制不得改动（球构造/α 公式/target 形式全部冻结）
  2. clean label 只进诊断与 oracle 上界，不进任何去噪训练
  3. SSL 预训练 label-free，超参用官方默认，不用下游 test 调
  4. 每 Phase 一份 research-record 报告（假设/角色/改动/门槛/结果/决策），standalone 可读
- 测试保持全绿：`py -3 -m unittest discover -s tests`

### 现有锚点（uniform 0.4, 5-seed, percent 0.2/0.1/0.7, ROCm）

| 方法 | Cora | CiteSeer | PubMed |
|---|---:|---:|---:|
| SoftDist [X||H] | 0.8137±0.0067 | 0.6891±0.0159 | 0.8320±0.0078 |
| NRGNN | 0.8107±0.0230 | 0.7027±0.0108 | 0.8080±0.0088 |
| GCN+CE | 0.7842±0.0151 | 0.6530±0.0121 | 0.8392±0.0057 |

---

## Phase 15：instance-dependent 噪声（P0，审稿人必问）

### 目的

当前噪声谱只有 uniform + pair。instance-dependent 噪声（特征相关的翻转，更贴近真实标注错误）是噪声标签论文标配。NoisyGL 原生支持 `--noise_type instance`。

### 运行

- 方法（与主表一致，保证可比）：`frozengrace_gb_gcn_softdist_xh`、`frozengrace_gb_gcn_softdist_xh_shuffled`、`gcn_xh_ce`、`gcn`、`nrgnn`
- 数据集 × 噪声率 × seeds：{cora, citeseer} × instance {0.2, 0.4} × {0,1,2,3,4}
- 协议参数照旧（percent 0.2/0.1/0.7）
- 核对每个 run 的 `actual_noise_rate`（instance 模式名义值与实际值常有偏差），写进 raw CSV
- 输出 `runs/instance_noise_20260612/`：raw + aggregate + 报告

### 报告要求

- 与 uniform 0.4 / pair 0.4 并列，扩成完整噪声谱表（论文 Table 3 增列）
- paired delta vs NRGNN 与 vs gcn_xh_ce（球贡献）；胜负判定沿用既有定义（mean>0 且 ≥4/5 正 = win）
- 如实记录：instance 噪声下球贡献是否保持中高噪声更强的趋势

### 判定

- 无硬门槛（这是 robustness 广度补全，不是新机制验证）
- 结果如何都进论文：赢则强化 claim，平/输则进 regime 边界分析

---

## Phase 16：当前管线的覆盖率敏感性曲线（P0，replace 旧 C1 数据）

### 目的

论文初稿引用的 coverage sensitivity 是**旧 C1 管线**（GRACE 微调）的数据，与当前 SoftDist [X||H] 不是同一方法，不能直接进正文。需在**当前管线**上重测。

### 关键前置：每个 train_ratio 需要匹配的 H

> H 是在 `ratio_train0p2_val0p1` 下预训练导出的。但 GRACE 预训练 label-free，**不依赖 train_ratio**——H 只看 (X, A)。因此同一份 0.2-split 的 H 可用于所有 train_ratio（split 只影响下游 train/val/test 划分与噪声注入，不影响 H）。**确认这一点：检查 H.pt 的产出脚本是否真的没用到 train_mask；若用了，需为每个 ratio 重新导出 label-free H。** 预期是没用到（label-free SSL 的定义），但必须核实后写进报告。

### 运行

- train_ratio ∈ {0.05, 0.1, 0.2, 0.4}，对应 val_ratio 固定 0.1、test_ratio = 1 − train − 0.1
- 方法：`frozengrace_gb_gcn_softdist_xh`、`gcn_xh_ce`（球增量 = 二者差）、`softdist_xh_shuffled`、`gcn`（CE 基线）
- {cora, citeseer} × 4 ratio × 5 seeds
- 必报球诊断：每个 ratio 下 mean_train_per_ball、num_balls、NCR、CDR（验证「覆盖率↑→票密度↑→球增量↑」的机制链）
- 输出 `runs/coverage_sensitivity_20260612/`

### 报告要求

- 主曲线：球增量（softdist_xh − gcn_xh_ce）vs train_ratio，双数据集
- 副曲线：mean_train_per_ball vs train_ratio（机制解释）
- 与旧 C1 数据对比一句：当前管线是否复现「覆盖率单调增益」趋势

### 判定

- 无硬门槛。预期复现 C1 阶段趋势（增益随覆盖率单调增）；若不复现则是重要发现，需诊断原因（XH 拼接可能改变了覆盖率依赖性）

---

## Phase 17：效率/复杂度对比表（P1，支撑「非迭代低复杂度」claim）

### 目的

论文反复声称「非迭代、低复杂度、比 NRGNN 简单一个数量级」，必须有数字。

### 运行

- 不新跑训练，**从已有日志汇总**：`runs/xh_5seed_rocm_20260611/logs/`、`runs/baseline_completion_20260612/`、NRGNN 既有 run 的 `total_time`
- 汇总表列：方法 | 是否迭代 | 墙钟时间(s, mean±std over seeds) | 可训练参数量 | 预处理时间（建球+target，仅 SoftDist 有）
- 参数量：从 predictor 的 model 直接 `sum(p.numel())`，写个小脚本 `scripts/count_params.py`，对 GCN([X||H])、NRGNN、UnionNET 各算一次
- 若已有日志的计时不可比（不同批次/负载），补一轮**统一环境单 seed 计时**：SoftDist [X||H] / NRGNN / UnionNET × Cora，同一会话连续跑，记录端到端墙钟
- 输出 `runs/efficiency_20260612/efficiency_table.csv` + 报告

### 报告要求

- 强调三点对比：SoftDist 的迭代次数=0（单遍）vs NRGNN 的每轮图重估；墙钟倍率；参数量级
- 诚实标注：SoftDist 有一次性预训练 H 的成本，但 H 可跨所有噪声/split 复用（摊销后接近零）

---

## Phase 18：clean-label oracle 上界（P1，量化 gap 闭合）

### 目的

给出性能上界，量化「去噪闭合了多少噪声 gap」。

### 运行

- `gcn_xh_ce` 但用 **clean 标签**训练（noise_rate=0 等价，或直接 `--noise_type clean`）× [X||H] 特征
- {cora, citeseer, pubmed} × 5 seeds
- 输出 `runs/oracle_20260612/`

### 报告要求

- 表格：clean oracle / SoftDist(noisy) / GCN+CE(noisy) 三行，算「gap 闭合率」= (SoftDist − GCN+CE) / (oracle − GCN+CE)
- 这个比率是论文里很有力的一句话：「SoftDist 闭合了噪声损失的 X%」

---

## Phase 19（可选，P0 第三数据集）：Amazon-Photo 泛化

### 前置：预注册 fallback（执行前写进报告，不可事后挑）

Amazon-Photo 无官方 GRACE 配置。**预注册规则**：复用 PubMed 的 GRACE 预训练配置（epochs/lr/drop rate 原样），不为 Amazon-Photo 调任何 SSL 超参。这条规则必须在看到结果前写进报告。

### 运行

- label-free GRACE 预训练 Amazon-Photo（`amazonpho`），导出 H 到 `checkpoints/embeddings_v2/AmazonPhoto/grace/H.pt`
- 方法：`softdist_xh`、`gcn_xh_ce`、`softdist_xh_shuffled`、`gcn`、`nrgnn` × uniform 0.4 × 5 seeds
- 球诊断必报（Amazon-Photo 是共购图，homophily 不同，检验球是否退化/塌缩——参考 Phase 9 的 CCA-SSG 塌缩教训）
- 输出 `runs/amazonphoto_20260612/`

### 判定

- `softdist_xh ≥ GCN+CE` 且 real > shuffled → 第三个成功数据集，泛化 claim 立住
- 输 GCN+CE 但赢 NRGNN → 类 PubMed 的 regime 边界
- 球塌缩（num_balls < 8）→ 记录为「共购图 H 几何不适配」，与 §6 可分性≠可聚类性分析呼应

---

## 执行顺序与优先级

```
P0 先行（审稿人必问）：
  Phase 15 (instance 噪声) ─┐
  Phase 16 (覆盖率曲线)   ─┼─ 三者独立可并行
  Phase 19 (Amazon-Photo)─┘   （19 需先预训练 H，最慢，先起跑）
P1 补强：
  Phase 17 (效率表) ── 多为日志汇总，最快
  Phase 18 (oracle 上界) ── 简单
```

- Phase 19 的 H 预训练最耗时，**最先起跑**，跑的同时做 15/16/17/18
- 17 大部分是已有日志汇总，可最先出结果
- 任何 Phase 失败/边界都不阻塞其他，全部结果无论正负都进论文（robustness 广度 + 诚实边界本就是本文优势）

## 报告汇总

全部完成后，把 15/18/19 的新数字回填进 `SoftDist_XH_论文初稿雏形.html` 对应表格（instance 噪声进 Table 3，oracle 进新增上界表，Amazon-Photo 进 Table 6 泛化），并在第 10 节「缺失实验」里勾掉已补项。
