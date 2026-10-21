# 执行计划：β=0.5 标签噪声主表 + 补最新强 baseline（Phase MLP-5）

日期：2026-07-01
执行者：Codex
定位：**用定型主算法 β=0.5 做完整标签噪声主表；同时补最新强 baseline(FINE / DivideMix，同 backbone)，一次性得到"论文主表 + 相对最新方法的位置"。** 这是论文的核心实验表。

## 0. 背景与目标

- 主算法已定型 = **SoftDist-MLP β=0.5 半硬**(MLP-4)。此前 MLP-1/2 主表用旧 β=0(软)跑，**必须用 β=0.5 重做**（否则低估自己）。
- 用户要"比过最新方法"。**关键纪律（防重蹈错标尺）**：不比最新方法的**绝对精度原版**（图像域/ResNet/数据增强，不可比）；而是把它们的**抗噪核心移植到同一个弱 MLP backbone**，同 Cora/CiteSeer+GRACE+弱MLP，比**抗噪策略**（MgCNL 顶刊范式）。
- **诚实预期**：vs 经典(small-loss/co-teaching/ELR/JoCoR) 多数设定小赢；vs FINE/DivideMix **大概率不是全面赢**（部分赢/平/输）。这不影响论文，因为卖点是**综合定位**(双重鲁棒+轻量graph-less+低标签率优势)，不是单维标签噪声 SOTA。

## 0.1 红线

1. clean 只进评测；噪声只加 train；**所有方法同一 X-only 弱MLP backbone、同 config**（命门公平性）。
2. 多 seed 配对(mean±std over 5 seeds) + 逐 seed 方向计数；多数据集一致性看 ≥2/3。
3. 主算法固定 β=0.5，不再调硬度。Python: ROCm venv；percent split 0.2/0.1/0.7 显式传参。

## 0.2 复用与新增

- **复用**：`frozengrace_gb_mlp_softdist_beta05`(本方法) + `mlp_ce` + `MLPNoiseBaselines`(small_loss/coteaching/jocor/elr/sce) + `beta05_shuffled`。全 X-only 弱MLP，config 已建(cora/citeseer/pubmed)。
- **新增最新 baseline（加进 `MLPNoiseBaselines_Predictor` 的 strategy 分发，天然同backbone）**：
  - **FINE（必做，迁移最自然）**：核心 = 用每类特征的 Gram/协方差主特征向量(eigenvector)算样本"对齐度"，过滤低对齐(噪声)样本再训。**纯特征谱操作，不依赖数据增强，适合弱MLP+GRACE特征**。加 `strategy="fine"`。
  - **DivideMix（可选/简化版）**：核心 = GMM 拟合样本 loss 分 clean/noisy + 半监督(MixMatch)。**MixMatch 依赖数据增强，在图节点/表格特征上别扭**。→ 做**简化版 `strategy="dividemix_lite"`**：GMM 分 clean/noisy + 对 noisy 样本用模型预测做软标签(去掉MixMatch)。**明确标注"MLP-adapted simplified DivideMix，非原版"**，避免 claim 复现原版。若工程超预算，DivideMix 可只做 related-work 讨论、不进表。
- 单测：FINE/dividemix_lite 的 clean 标签不进过滤/分组决策(篡改clean不变)；同backbone(与β0.5的MLP结构一致)。

## 1. 主表（label-noise main table，5-seed）

```
Datasets: Cora, CiteSeer, PubMed  (Coauthor-CS 可选，慢；至少前三个)
Noise: uniform 0.2 / 0.4 / 0.6 + pair 0.4 + instance 0.4   (MLP-1/2 噪声谱)
Seeds: 0-4
Methods (全 X-only 弱MLP, 同config):
  MLP+CE
  small-loss / co-teaching / JoCoR / ELR / SCE   (经典抗噪)
  FINE            (最新, 必做)
  DivideMix-lite  (最新, 可选)
  SoftDist-MLP β=0.5   (本方法主算法)
  β=0.5 shuffled       (真结构对照)
```
指标：test acc mean±std；本方法 vs 每个对照的配对 delta + 逐 seed 方向。

## 2. 并入的两块（省算力，一起跑）

- **低标签率端点 5-seed(承接 MLP-5b)**：Cora/CiteSeer × train_percent{0.05, 0.65} × {β0.5, small-loss, JoCoR, FINE, shuffled} × seeds0-4。确认 MLP-5b 的"低标签率优势"在 5-seed 稳（含 vs FINE）。PubMed 作边界不补。
- **硬度消融**：直接用 MLP-4 已有的 β{0,0.5,1} 5-seed 数据，不重跑，整理进消融表。

## 3. 沿用不重跑

- **结构噪声主表(MLP-2 C2)**：β 只影响训练目标、不影响"MLP推理不用邻接=结构免疫"这个结构事实 → **C2 表原样沿用，不重跑**。
- w/o-GRACE / ellipsoid / gmm 消融：MLP-2/5a 已有，整理即可。

## 4. ★判据（这是主表，不是生死闸；判"论文表长什么样"）

| 观察 | 论文呈现 |
|---|---|
| β0.5 在 ≥2/3 数据集、多数噪声率 ≥ 经典抗噪策略(small-loss/co-teaching/ELR/JoCoR) | 主 claim："在同 backbone 下多数标签噪声设定优于经典抗噪策略"（已知大概率成立） |
| β0.5 vs FINE/DivideMix：部分赢/平/输 | 诚实呈现；**不靠单维赢**，靠综合(双重鲁棒+轻量+低标签率)。若某些设定赢 FINE，重点写；输的诚实列，用适用域解释 |
| PubMed 输强抗噪 baseline（已知） | 作边界数据集，适用域分析 |
| 低标签率 5-seed 仍显示 Cora/CiteSeer 优势 | 低标签率鲁棒主图成立 |
| β0.5 > shuffled 全成立 | 球必要性铁证（已知稳） |

**核心心态**：这一步是"把论文主表做出来 + 看清相对最新方法的真实位置"，不是"必须全面赢最新方法"。诚实的位置图 + 综合卖点 = 普通 SCI 立得住。

## 5. 执行顺序

```
1. 加 FINE(strategy=fine) 到 MLPNoiseBaselines；(可选)dividemix_lite。单测(clean不进决策/同backbone)
2. 跑主表: Cora/CiteSeer/PubMed × {uniform0.2/0.4/0.6, pair0.4, instance0.4} × 全方法 × seeds0-4
3. 跑低标签率端点: Cora/CiteSeer × {0.05,0.65} × {β0.5,small-loss,JoCoR,FINE,shuffled} × seeds0-4
4. 整理: 主表(含FINE/DivideMix位置) + 硬度消融(MLP-4数据) + 结构表(MLP-2沿用) + 低标签率图 + w/o-GRACE/ellipsoid消融
5. 报告 + 明确 β0.5 相对经典/最新方法的位置 + 综合卖点定位
```

- 输出目录：`external/NoisyGL/runs/mlp_maintable_20260701/`
- 报告：完整标签噪声主表(mean±std, 数据集×噪声×方法) + vs经典/vs最新(FINE/DivideMix)配对delta + 低标签率5-seed + 结构表引用 + 消融整合 + 位置诊断("差最新方法多少"的诚实回答)。

## 6. 完成后

- 更新记忆(MLP-5 主表结果 + 相对最新方法真实位置 + 论文表是否齐)。
- **主表 + 相对位置清楚** → 剩下只差：真实噪声数据集(增强说服力，可选) + **进入论文写作**(Claude 起草：双重鲁棒Intro + β0.5半硬方法 + 主表 + 结构表 + 消融 + 低标签率图 + 球必要性 + related-work切割Graph-MLP/MgCNL/FINE + 适用域诚实边界)。
- 至此实验主体基本齐，进收尾。

---

## Phase MLP-5 执行结果备案（待填）
结果文件：`external/NoisyGL/runs/mlp_maintable_20260701/`

完成状态：
- 主表 + 低标签端点全部完成：850/850 rows，missing=0。
- 运行环境：ROCm Python，`cuda:0`，train-only label noise，显式 percent split。
- 新增同 backbone baseline：`mlp_fine`、`mlp_dividemix_lite`。注意：这是 MLP-adapted strategy baseline，不是图像域原版 FINE / DivideMix 复现。
- 汇总文件：
  - `runs/mlp_maintable_20260701/mlp_maintable_report.md`
  - `runs/mlp_maintable_20260701/mlp_maintable_aggregate.csv`
  - `runs/mlp_maintable_20260701/mlp_maintable_paired_deltas.csv`

关键主表结果：
- Cora / CiteSeer 强成立。β=0.5 在两个数据集、5 个噪声设置下，对所有经典 MLP 抗噪 baseline 与 FINE / DivideMix-lite 基本全胜：
  - Cora：vs classic 25/25 setting 胜，平均 delta +6.83pp；vs latest(FINE/DivideMix-lite) 10/10 胜，平均 delta +9.41pp；vs shuffled 5/5 胜，平均 delta +10.62pp。
  - CiteSeer：vs classic 25/25 胜，平均 delta +6.65pp；vs latest 10/10 胜，平均 delta +8.24pp；vs shuffled 5/5 胜，平均 delta +9.22pp。
- PubMed 是明确边界数据集：
  - vs classic 13/25 胜，平均 delta -0.33pp。
  - 主要输给 small-loss / co-teaching / JoCoR，尤其 uniform0.4 / uniform0.6 / pair0.4 / instance0.4。
  - vs FINE 5/5 胜，但 vs DivideMix-lite 多数为负；latest 合计 5/10 胜，平均 delta +3.16pp。
  - vs shuffled 5/5 胜，平均 delta +8.18pp，说明真实球结构仍有信号，但在 PubMed 上不足以压过 small-loss / JoCoR / DivideMix-lite。
- 全主表总体：
  - β=0.5 vs shuffled：15/15 main settings 胜，平均 delta +9.34pp，是球结构必要性的最稳证据。
  - β=0.5 vs CE / ELR / SCE / FINE：15/15 胜。
  - β=0.5 vs small-loss / co-teaching / JoCoR：各 11/15 胜，失败集中在 PubMed。
  - β=0.5 vs DivideMix-lite：10/15 胜，失败也集中在 PubMed。

低标签率端点 5-seed：
- Cora:
  - train=0.05：β=0.5 vs small-loss +5.61pp (5/5)，vs JoCoR +6.68pp (5/5)，vs FINE +8.61pp (5/5)，vs shuffled +6.57pp (5/5)。
  - train=0.65：β=0.5 vs small-loss +1.24pp (5/5)，vs JoCoR +0.18pp (2/5)，vs FINE +12.02pp (5/5)，vs shuffled +11.49pp (5/5)。
- CiteSeer:
  - train=0.05：β=0.5 vs small-loss +5.80pp (5/5)，vs JoCoR +6.11pp (5/5)，vs FINE +7.52pp (5/5)，vs shuffled +8.31pp (5/5)。
  - train=0.65：β=0.5 vs small-loss +1.44pp (4/5)，vs JoCoR +1.35pp (4/5)，vs FINE +5.51pp (5/5)，vs shuffled +7.32pp (5/5)。
- 结论：低标签率优势在 Cora/CiteSeer 5-seed 成立；高标签率下优势缩小，尤其 Cora vs JoCoR 只有 +0.18pp 且 2/5 seeds 正，不能写成高标签率稳定强胜。

论文定位：
- 可以写强主 claim：在 Cora/CiteSeer 型 citation 数据上，SoftDist-MLP β=0.5 在同 X-only weak MLP backbone 下跨 uniform/pair/instance 噪声稳定优于经典抗噪策略、FINE-adapted 和 DivideMix-lite-adapted baseline，并且真实球结构对 shuffled 有稳定增量。
- 必须写边界：PubMed 上真实球结构仍有效（全胜 shuffled），但对 small-loss / JoCoR / DivideMix-lite 不占优；这说明方法不是单维标签噪声 SOTA，而是“citation 图 + GRACE-H 可形成高质量球”时最强。
- FINE 在本节点 MLP 协议下不是强对手；DivideMix-lite 在 PubMed 很强，应作为最新策略位置的诚实对照。
