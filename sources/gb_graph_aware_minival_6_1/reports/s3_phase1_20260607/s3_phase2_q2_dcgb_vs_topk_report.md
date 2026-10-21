# S3 Phase 2-Q2 — DCGB vs TopK 必要性验证（粒球是否比 kNN consensus 多提供了什么）

> 复用 S3 Phase 1 的 5-seed 数据（Cora / symmetric 0.4 / `ratio_train0.2` / frozen GRACE / seeds 0–4，
> 固定 selection schedule）。**NOT planetoid, NOT NoisyGL。** 不重跑，直接 paired 分析。
> 证据 `reports/s3_phase1_20260607/s3_phase1_rows.csv`。本文件回答主线生死问题：
> **“你所谓粒球是不是只是 kNN consensus 的包装？”**

## 对比公平性核验（先排除假象）

| seed | dcgb ball_count | dcgb avg_size | topk k | dcgb cov | topk cov |
|---|---|---|---|---|---|
| 0–4 | 260–267 | ~10.2–10.4 | **5（逐 seed 匹配）** | **0.600** | **0.600** |

selection coverage 两者精确相等（固定 ρ=0.6 保证），topk 的 k 按 dcgb 球大小匹配。**这是同覆盖、同 schedule
的干净对比**，下面的差异不是覆盖造成的假象。

## DCGB − TopK paired delta（每 seed）

| seed | test | sel_precision | noise_excl_recall | coverage |
|---|---|---|---|---|
| 0 | −0.0143 | −0.0400 | −0.0599 | 0 |
| 1 | +0.0063 | −0.0062 | −0.0092 | 0 |
| 2 | +0.0127 | −0.0215 | −0.0323 | 0 |
| 3 | +0.0037 | −0.0123 | −0.0184 | 0 |
| 4 | +0.0079 | −0.0123 | −0.0184 | 0 |
| **mean** | **+0.0033** | **−0.0185** | **−0.0276** | 0 |
| **wins(dcgb>topk)** | **4/5** | **0/5** | **0/5** | – |

- **test**：mean **+0.327pp**，std **1.03pp**，paired **t≈0.71**（df=4，需 \|t\|>2.78 才 p<0.05）→ **不显著**。
- **selection precision**：dcgb **0/5 输**（mean −1.85pp）。
- **noise-excluded recall**：dcgb **0/5 输**（mean −2.76pp）。

## 预注册判定（用户标准，机械）

| 档 | 标准 | 结果 |
|---|---|---|
| 1（建球有独立价值） | dcgb 在 test 或 sel-precision ≥4/5 超 topk **且** mean test Δ≥0.5pp | **不满足**（test mean 仅 +0.327pp；sel-precision 0/5） |
| 2（球是可靠性建模工具） | test 接近但 dcgb 的 sel-precision/class-balance/stability 更好 | **不满足**（sel-precision、noise-recall 反而更差） |
| 3（降级） | dcgb 与 topk 持平或更差 | **命中**（test 不显著 +0.33pp，selection 维度系统性更差） |

## 结论（主线判定）

**命中第三档：“显式建球”相对 kNN consensus 没有可证的独立增量。**

- dcgb 相对 topk 的 test 优势（+0.33pp）**不显著、不稳**（t≈0.71），且 dcgb 在 node-level selection
  precision（0/5）与 noise-recall（0/5）上**系统性更差**——纯 kNN 共识选得更准、排噪更多。
- 真正稳健的是 **commit-style selection + 局部一致性可靠性信号**（S3 Phase 1 的 C 满分 5/5：dcgb/topk
  都远超 shuffled/random/noball）。但“可靠性信号”用**互惠-kNN 球**还是**纯 kNN 共识**实现，下游不可区分。

## 论文叙事调整（建议）

- **主线 claim 从“粒球（granular ball）方法”降级为“局部一致性可靠性信号 + commit-style 选择”**；
  粒球只是该信号的一种实现，**不作为独立创新点/显著增益来源**。这与项目转折点（§12：图感知球→reliable
  球→Phase 1 现象）一脉相承：每一步“球的特异结构”都未转化为可证下游增量。
- 若仍要保留“球”，必须先找到 **球相对 kNN 共识有独立增量的设置**（更高噪声、pair 噪声、更弱表示、或球
  能表达 kNN 共识表达不了的多尺度/非对称邻域）——这是单独 lane，未验证前不能当卖点。

## 下一步（待用户定）

- Q1（解释性，非生死）：为什么 gb 与 dcgb 的 group-purity 相同（0.892 vs 0.893）但 node-level selection
  质量不同——机制分析章节，已有 per-class / loss 诊断在 CSV 里。
- 或：按上面建议**正式把论文主线改写为“局部一致性可靠性信号”**，把“球”收为实现细节。

## 纪律

预注册机械判定；同覆盖同 schedule 公平对比（已核验）；test 差异报显著性（不显著就说不显著）；clean
label 仅诊断；ratio-split symmetric，未与 NoisyGL 混报；未从单 seed/within-std 判定；未硬保“球”卖点。
