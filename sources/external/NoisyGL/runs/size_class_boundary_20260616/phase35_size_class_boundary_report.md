# Phase 35 规模 × 类数二维边界诊断报告

日期：2026-06-16  
执行者：Codex  
协议：NoisyGL percent split 0.2/0.1/0.7，uniform noise 0.4，seeds 0-4，ROCm GPU。  
核心问题：SoftDist/GBSelect 在 DBLP、PubMed、Amazon-Photo 上失败，到底是因为图规模大，还是因为类别数/类别复杂度不足？

## 1. 执行摘要

本轮补了两个大规模多类别数据集：

- **Coauthor-CS**：18,333 nodes，15 classes，与 DBLP 规模相近但类数显著更多，是 Phase 35 的 Q2 决胜点。
- **Amazon-Computers**：13,752 nodes，10 classes，作为 35c 扩展确认点。

结论：

- **Coauthor-CS 明确成功**：SoftDist [X||H] 同时超过 GCN [X||H] 和 shuffled，5/5 seeds 为正；GBSelect 也通过命门对照。
- **Amazon-Computers 部分支持但不构成成功**：真实球显著超过 shuffled，GBSelect 也超过 GCN-only / shuffled，但 SoftDist 和 GBSelect 都低于全 train GCN [X||H]。
- 因此，“大图必败”被 Coauthor-CS 直接推翻；但“类数充足必然成功”也被 Amazon-Computers 削弱。更准确的边界是：**规模不是单独死因，类别复杂度/标签噪声可分性是关键调节变量，但仍会被图 regime 和强 GCN [X||H] 基线吸收。**

`cora3c` 暂不执行：它主要用于 Q2 失败时区分“规模主导”与“类数主导”。现在 Q2 已经成功，核心混淆已经被 Coauthor-CS 拆开；继续构造派生数据集的 ROI 低于先写入当前发现。

## 2. 预训练与数据接入

新增上游预训练支持：

- `gb_graph_aware_minival_6_1/src/noisy_label_gb_grace/data.py`
  - 新增 `Coauthor('CS')` 支持。
- `gb_graph_aware_minival_6_1/scripts/pretrain_ssl_candidates.py`
  - `CoauthorCS -> DBLP` GRACE config fallback。
  - `amazoncom -> PubMed` GRACE config fallback，与 Amazon-Photo 保持一致。

导出的 H：

| 数据集 | H path | shape | fallback config |
|---|---|---:|---|
| Coauthor-CS | `checkpoints/embeddings_v2/CoauthorCS/grace/H.pt` | 18333 × 256 | DBLP |
| Amazon-Computers | `checkpoints/embeddings_v2/amazoncom/grace/H.pt` | 13752 × 256 | PubMed |

Sanity check：NoisyGL loader 节点数与 H 第一维一致。

## 3. 诊断结果

大图参数：`max_balls=256, min_ball_size=32, split_improvement=0.001`。

| 数据集 | num_balls | true purity | ball AUC | GCN AUC | ball AUC in GCN-clean50 | 离线判定 |
|---|---:|---:|---:|---:|---:|---|
| Coauthor-CS | 256.0 | 0.9052 | 0.9669 | 0.9942 | 0.8533 | 强阳性 |
| Amazon-Computers | 237.3 | 0.7850 | 0.9273 | 0.7965 | 0.9219 | 强阳性 |

离线 co-teaching：

| 数据集 | train noise | keep=0.6 后噪声率 | retention |
|---|---:|---:|---:|
| Coauthor-CS | 0.4010 | 0.0107 | 0.5610 |
| Amazon-Computers | 0.3976 | 0.0325 | 0.6017 |

诊断解释：

- 两个 Q2 数据集都不是“球塌缩”或“球无信号”。
- Amazon-Computers 的 GCN AUC 远低于 Coauthor-CS，但球 AUC 很高，说明球的噪声判别信号存在；下游失败不是诊断阶段的假阳性，而是训练范式/监督量/强全样本基线的问题。

## 4. Coauthor-CS 下游结果

### SoftDist 线

| 方法 | Coauthor-CS |
|---|---:|
| SoftDist [X||H] | **0.9243 ± 0.0016** |
| GCN [X||H] | 0.9190 ± 0.0009 |
| SoftDist shuffled | 0.9188 ± 0.0012 |

配对差：

- SoftDist - GCN [X||H]：+0.53pp，5/5 seeds。
- SoftDist - shuffled：+0.55pp，5/5 seeds。

### GBSelect 线

| 方法 | Coauthor-CS |
|---|---:|
| GBSelect | **0.9257 ± 0.0011** |
| GCN-only selection | 0.9187 ± 0.0053 |
| Shuffled selection | 0.9042 ± 0.0096 |

配对差：

- GBSelect - GCN-only：+0.71pp，4/5 seeds。
- GBSelect - shuffled：+2.16pp，5/5 seeds。
- GBSelect - GCN [X||H]：+0.67pp，5/5 seeds。

判定：

Coauthor-CS 是 Phase 35 的关键成功点。它与 DBLP 节点规模相近，但类数更多；因此“图规模大导致粒球必败”这个解释不成立。类别复杂度/类数是有效调节变量。

## 5. Amazon-Computers 下游结果

### SoftDist 线

| 方法 | Amazon-Computers |
|---|---:|
| SoftDist [X||H] | 0.8625 ± 0.0056 |
| GCN [X||H] | **0.8695 ± 0.0082** |
| SoftDist shuffled | 0.8443 ± 0.0176 |

配对差：

- SoftDist - GCN [X||H]：-0.70pp，0/5 seeds。
- SoftDist - shuffled：+1.82pp，5/5 seeds。

### GBSelect 线

| 方法 | Amazon-Computers |
|---|---:|
| GBSelect | 0.7178 ± 0.0234 |
| GCN-only selection | 0.6923 ± 0.0213 |
| Shuffled selection | 0.6696 ± 0.0215 |
| GCN [X||H] | **0.8695 ± 0.0082** |

配对差：

- GBSelect - GCN-only：+2.55pp，5/5 seeds。
- GBSelect - shuffled：+4.82pp，5/5 seeds。
- GBSelect - GCN [X||H]：-15.17pp，0/5 seeds。

判定：

Amazon-Computers 支持“真实球结构有信号”（real > shuffled），但不支持“球方法超过全样本 GCN [X||H]”。这和 Amazon-Photo 类似：selection 丢弃监督量的代价过高，SoftDist 的标签侧修正也被强 XH 分类器吸收。

## 6. 二维边界判定

| | 类少 | 类多 |
|---|---|---|
| 小图 | cora3c 暂缓 | Cora / CiteSeer 成功 |
| 大图 | DBLP/PubMed 失败或弱 | Coauthor-CS 成功；Amazon-Computers 部分支持 |

最终判定：

- **规模不是充分解释变量**：Coauthor-CS 是大图，但 SoftDist 和 GBSelect 都成功。
- **类数/类别复杂度是重要变量**：Coauthor-CS 相对 DBLP 的关键差异是 15 类 vs 4 类。
- **但类数不是唯一充分条件**：Amazon-Computers 10 类仍未超过 GCN [X||H]，说明图同质性、特征强度、噪声对 GCN 的实际伤害、以及全样本监督量也会共同决定球增量。

论文建议 claim：

> The effectiveness of ball-based denoising is not bounded by graph size alone. A large multi-class graph (Coauthor-CS) exhibits stable ball gains over both GCN [X||H] and shuffled-ball controls, whereas large low-class graphs fail. However, Amazon-Computers shows that class count is not a sufficient condition: when the full-sample GCN [X||H] baseline is already strong, ball signals remain real but may be absorbed by the classifier or outweighed by sample-selection costs.

## 7. 后续决策

- 不再把论文定位为“只有 Cora/CiteSeer 两个成功数据集”。Coauthor-CS 可以作为第三个真实成功数据集。
- 不声称“大图+多类必然成功”。Amazon-Computers 必须作为边界点写入。
- `cora3c` 可作为附加 ablation，但不是当前必须项；若后续需要完整二维表再做。
- NRGNN on Coauthor-CS/Amazon-Computers 还未跑；这不影响 Phase 35 的边界判定，但若将 Coauthor-CS 放入主表，需要补 NRGNN 或明确为 boundary table 而非 SOTA table。

