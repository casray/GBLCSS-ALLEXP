# S3 Phase 1c — density-guided 球级 selection 诊断（frozen GRACE 上球结构的最后一道检查）

> 多 seed smoke（Cora / symmetric 0.4 / `ratio_train0.2` / seeds 0–4 / frozen GRACE，固定 schedule
> W20/ramp20/ρ=0.6）。**NOT planetoid, NOT NoisyGL。** 灵感来自 GBmooer（超球 + 多尺度球级结构）。
> **不重训表示、不调 schedule、不包装为方法。** clean label 仅 eval/诊断。
> 证据 `reports/s3_phase1c_density_20260607/s3_phase1c_rows.csv`、`.../s3_phase1c_summary.json`、
> `scripts/run_s3_phase1c.py`、`src/noisy_label_gb_grace/evaluation/selection.py:ball_density_reliability`。

## 背景与设计

Q2（§12.7）已证 `dcgb`（互惠-kNN 球共识）≈ `topk`（kNN 共识）。本诊断引入一个**球级密度结构**信号
（kNN 共识无法表达），看它能否在 frozen GRACE 上拉开 kNN 共识：

```
rel_i = density_score(B_i) × compactness(i, B_i) × noisy_consensus(B_i)
```
- `density_score(B)`：球内成员平均 kNN 余弦密度（超球上局部密度，纯几何，无标签）
- `compactness(i,B)`：节点到球中心 `(1+cos)/2`
- `noisy_consensus(B)`：球内 train **noisy** label 最大类占比

距离度量诊断前已验证：L2 归一化表示上 cosine top-k 与角距离 top-k **完全一致（300/300）**，故仅换角
距离是恒等变换、无效实验——本诊断改测"球级密度结构"这一真正新信号。

## 结果（跨 seed 均值）

| method | test | sel_precision | noise_excl_recall |
|---|---|---|---|
| sel_clean_oracle（诊断上限） | 0.8143 | 1.0000 | 1.0000 |
| sel_dcgb_emb | 0.8002 | 0.9040 | 0.8562 |
| sel_topk_matched | 0.7969 | 0.9225 | 0.8839 |
| **sel_density_gb** | **0.7714** | 0.8070 | 0.6654 |
| sel_random_density_matched | 0.7412 | 0.7509 | 0.5088 |
| sel_shuffled_density_matched | 0.7117 | 0.6855 | 0.3032 |
| ce / gb_soft（smoke） | 0.7935 / 0.7974 | – | – |

## 预注册判定（机械）

| 判据 | 结果 | 判定 |
|---|---|---|
| density > topk：test ≥4/5 且 mean Δ≥0.5pp | wins **1/5**，mean **−2.56pp** | **FAIL** |
| sel_precision(density) ≥ topk | 0.807 < 0.923 | **FAIL** |
| noise_excl_recall(density) ≥ topk | 0.665 < 0.884 | **FAIL** |
| （sanity）density > 自身 shuffled / random | +5.97pp / +3.02pp | 通过 |

**→ 触发否定分支：frozen GRACE 上球级密度结构无独立增量。**

## 结论

1. **density 信号甚至不如普通 kNN 共识**（0.7714 < topk 0.7969 < dcgb 0.8002），test 1/5、sel-precision
   与 noise-recall 全面更差。**球结构提供的信息，kNN 共识早已捕捉且捕捉得更好。**
2. density **确实**赢自己的 shuffled/random（+5.97/+3.02pp）→ 它不是纯噪声、捕捉到了某种结构，**但那个
   结构是冗余的**（被 topk 覆盖）。这进一步坐实：在 frozen 表示上，"球级结构 ≈ 局部一致性"，无独立价值。
3. 与 §12.1/§12.5/§12.7 一脉相承：**只要表示是冻结且非为球而生的，任何球结构（图感知 / reliable /
   density）都无法超越 kNN 共识。** frozen-space 下"球有没有救"的检查到此穷尽：**没有。**

## 方向（按预注册）

要保留粒球主线，**唯一剩下的路是让表示为球而生 / 让球反馈进 encoder**：
- CGBC：intra/inter-GB 损失反馈进 encoder（联合训练，打破冻结）+ 球中心/球标签纠正。
- GBmooer：超球对比学习专门训"球可分"的表示，再多尺度球级平滑。
两篇之所以有效，根因正是**表示为球而生 + 球塑造表示**（已读原文核对）；我们此前全程冻结、用通用 GRACE，
所以球退化成 kNN 共识。下一步若继续粒球，应进入 **ball-aware 对比表示 / encoder coupling**（工程更大、
项目历史上联合训练不稳，须严格避开 hard same-ball positive）。

## 纪律

预注册机械判定；clean label 仅诊断（density/consensus 全用 noisy）；配 shuffled+random matched；同
schedule 同覆盖；未从单 seed 判定；density 未被包装为方法；ratio-split symmetric，未与 NoisyGL 混报；
角距离恒等性已先行验证以避免无效实验。
