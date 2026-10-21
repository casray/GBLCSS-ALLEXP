# 执行计划：球内逐节点分辨率校正 + NoisyGL baseline 补全（Phase 12-14）

日期：2026-06-12
执行者：Codex
前置状态：机制探索曾收官于 SoftDist [X||H]（见 `execution_plan_xh_expansion_20260610.md`、`execution_plan_ssl_upgrade_20260612.md`）。本计划来自 2026-06-12 文献检索的新发现：**我们所有球级机制（gate/加权/多粒度）都在球级做文章，从未把校正信号做到节点级**。UnionNET (PAKDD 2021) 的 attention 加权 support set 和 TSS (ICML 2024) 的拓扑边界节点不可靠性，指向同一个未测维度：**球内逐节点分辨率**。

---

## 0. 环境与红线（沿用，必读）

- ROCm Python：`C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe`，device `cuda:0`
- NoisyGL 运行必须显式：`--split_type percent --train_percent 0.2 --val_percent 0.1 --test_percent 0.7`
- GRACE H 锚点：`gb_graph_aware_minival_6_1/checkpoints/embeddings/{Cora,CiteSeer}/ratio_train0p2_val0p1_split0_pretrain0/H.pt`
- **红线**：
  1. `frozengrace_gb_gcn_softdist_xh` anchor 行为不得改动（新变体新方法名）
  2. clean label 只进诊断，永不进训练/target/gate
  3. 票源只有 train_mask 内 noisy 标签
  4. 新增涉及标签的函数必须配防泄漏单元测试
  5. 球构造不动（geometric 64 球锚点），本计划只改**投票/alpha 的节点级分辨率**
- 测试保持全绿：`py -3 -m unittest discover -s tests`

### 与判死清单的关系（报告必须写）

| 判死方向 | 本计划为何不同 |
|---|---|
| C1_topk / kNN 投票 | topk 是扔掉球换 kNN 硬投票；想法 A 保留球作为 support 边界（球外 train 不可投票），只在球内做软相似度加权——球结构 × 逐节点分辨率的杂交，未测过 |
| 三种 gate（票数/时间/尺度） | 它们都试图判断"标签是否噪声"（已证不可分）；想法 B 判断"节点是否真属于球的类区域"（即 CDR 发生机制本身），是几何问题不是标签纯度问题 |
| 动态 alpha | 不用模型预测，零反馈回路 |
| 球级无标签扩展 | 只动 train 节点 target |
| E1 自适应粒度 | 球构造完全不动 |

---

## Phase 12：逐节点分辨率离线诊断（先做，零训练成本，几分钟）

### 三个 target 变体

设节点 i 属于球 B，`T_B = B ∩ train_mask`（票源），h 为 L2 归一化 GRACE H。

**变体 A——球内相似度加权投票（UnionNET 启发）**：
```
w_ij = softmax_{j∈T_B}( (h_i·h_j) / tau_sim )        # tau_sim ∈ {0.1, 0.5, 1.0}
p_i  = Σ_{j∈T_B} w_ij · onehot(y_noisy_j)             # 逐节点个性化分布
soft_target_i = (1-alpha_B)·onehot(y_noisy_i) + alpha_B·p_i
```
alpha_B 沿用现行 multiclass margin alpha（margin 仍按均匀计数算，保持单变量改动）。注意 i 自己若在 train 内也是票源之一（与现行行为一致，不特殊处理）。

**变体 B——球心距离调制 alpha（TSS 启发）**：
```
d_i = ||h_i - center_B|| / r_B                        # r_B = 球内最大距离（球已存 radius）
alpha_i = alpha_B · exp(-gamma · d_i)                 # gamma ∈ {0.5, 1.0, 2.0}
soft_target_i = (1-alpha_i)·onehot(y_noisy_i) + alpha_i·ball_dist_B   # 分布仍是均匀票
```

**变体 A+B**：p_i 用 A 的加权分布，alpha_i 用 B 的距离调制（各自网格中诊断最优的参数组合）。

### 实现

新脚本 `scripts/diagnose_nodewise_targets.py`（复用 multigranularity 诊断的骨架）：

1. 统一协议（percent split, uniform 0.4, seeds 0/1/2, Cora/CiteSeer），加载 H，建锚点几何球（max_balls=64, min_ball_size=32, 同 random_state=seed）
2. 对每个变体 × 参数网格，构造全 train 节点的 soft target，计算（clean label 仅诊断）：
   - 预测 NCR / 预测 CDR（pseudo_label = argmax(soft_target)，与现行 `diagnostics_for_soft_targets` 同口径）及相对 SoftDist 基线的变化
   - **按"到球心距离分位"分桶的 CDR/NCR 表**（验证 CDR 是否真集中在球边缘——这是想法 B 的前提，必报）
   - 逐节点分布与球均匀分布的平均 KL（量化 A 的个性化程度；若 ≈0 说明票太少加权无分辨率，提前解释结果）
3. 输出 `runs/nodewise_diag_20260612/`：raw CSV、aggregate CSV、报告 md

### 门槛（预注册，与历次相同）

| 指标 | 门槛 |
|---|---|
| 预测 CDR 相对下降 | ≥ 30%（双数据集，任一变体任一参数点） |
| 预测 NCR 相对损失 | ≤ 10% |

### 判定

- 过门 → 该变体进 Phase 13 训练
- 全部不过门 → 逐节点分辨率线关闭，结论写法："球内 5-10 票的 support 不足以支撑逐节点分辨率"——与无标签扩展的负结果形成互证，进论文分析章节
- 诚实预期：球内 train 票仅 5-10 个，软加权的分辨率可能有限；距离分桶表无论过不过门都是论文素材（CDR 空间分布的首次直接测量）

---

## Phase 13：过门变体训练验证（仅当 Phase 12 过门）

1. `utils/frozen_grace_targets.py` 新增 `apply_nodewise_soft_targets(...)`（参数：`vote_mode ∈ {uniform, sim_weighted}`、`alpha_mode ∈ {ball, distance}`、`tau_sim`、`gamma`），默认参数下必须与现行 `apply_ball_soft_targets` 逐元素一致（anchor 回归测试）
2. 新方法注册：`frozengrace_gb_gcn_softdist_xh_nw` / `_nw_shuffled`（XH 特征 + 节点级 target），config 用 Phase 12 选定参数
3. 单元测试：防泄漏（篡改 val/test 标签 target 不变）、回归（默认参数 = 现行 SoftDist）、加权正确性（手算小例）
4. seed0 smoke：`_nw` + `_nw_shuffled` × 2 数据集；通过门 = 双数据集 `_nw ≥ softdist_xh − 0.5pp` 且至少一个 +1pp 且 real > shuffled
5. 过 smoke → 5-seed（同三方法），主判据：**CiteSeer mean > 0.7027 (NRGNN)** 或双数据集稳定 > softdist_xh

## Phase 14：NoisyGL baseline 补全（与 Phase 12 并行，零实现成本，论文刚需）

检索发现五个已注册却从未跑过的方法，其中 UnionNET 与我们最像（节点级 support set 校正）、TSS 是 ICML 2024（回应"只比 2024 方法"的顾虑）：

1. 方法：`unionnet`、`tss`、`cgnn`、`crgnn`、`r2lp` × {cora, citeseer} × seeds 0-4 × uniform 0.4（统一协议参数）
2. 配置就用 NoisyGL 自带 config，不调参（与我们跑 NRGNN 时同标准）；某方法在本协议下跑挂/异常则记录原因跳过，不修它的代码
3. 汇总进现有 5-seed 主表格式，paired delta vs `softdist_xh`
4. 输出 `runs/baseline_completion_20260612/`
5. 如时间富余：PubMed 也补这五个方法（PubMed 上我们赢 NRGNN +2.4pp，若也赢这五个，PubMed 从 limitation 翻成 selling point 的一部分）

### 执行顺序

```
Phase 12（诊断，几分钟） ──过门──► Phase 13（训练）
Phase 14（baseline 补全）与 Phase 12 并行起跑
```

### 报告要求

每 Phase 一份 research-record 格式报告（假设/角色/改动/门槛/结果/决策），standalone 可读。Phase 12 报告必须包含距离分桶 CDR 表和 KL 个性化度量；Phase 14 报告必须含 UnionNET 与 SoftDist 的机制对比段（它是文献中最近的邻居，论文 related work 直接复用）。
