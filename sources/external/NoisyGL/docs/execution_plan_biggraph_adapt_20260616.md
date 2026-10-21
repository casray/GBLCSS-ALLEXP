# 执行计划：大图粒球适配可行性诊断（Phase 33）

日期：2026-06-16
执行者：Codex
前置状态：Phase 32 确认大图（DBLP/PubMed/Amazon-Photo）上 SoftDist 球贡献被吸收——构造层面参数调好后球纯度甚至超 Cora/CiteSeer，但下游 `real ≈ shuffled ≈ gcn_xh_ce`，球的 soft-label 校正零独立增量。

**本计划探索"大小图用不同算法"的假设**：小图标签绝对量少（Cora 140-540 train），"造软标签"有意义；大图标签绝对量足（DBLP ~3400 train），瓶颈是噪声比例而非缺监督。文献中大图高噪声 regime 有效的是"**选出干净样本**"（co-teaching / 小损失选择），不是"造更多软标签"。因此大图适配的正确形态可能是 **球作为 label-free、与模型预测正交的噪声判别视角，用于样本选择/加权**，而非 soft-target 校正。

**本计划是纯诊断,回答两个问题,不训练、不写最终算法**：
1. 大图上球的噪声判别信号，是否**正交于** [X‖H]-GCN 自己的噪声判别（即球知道 GCN 不知道的）？
2. 若正交，三种球用法（reweight / 筛除 / co-teaching 第二视角）中哪种的**离线预测效果**最好？

---

## 0. 关键设计原则（必读，这决定诊断对不对）

### 为什么不能只测"球可信度 vs 真噪声相关性"

Phase 32b 的核心信号是 **shuffled 也和 real 一样好**——这说明大图上"球内 train 标签统计"整体没转化成下游差异。陷阱：球可能与真噪声**正相关**（能识别噪声），但仍下游无用，因为 **GCN 从 [X‖H] 也能识别同样的噪声**。相关性存在 ≠ 增量存在。

**所以诊断必须测"条件正交增量"**：在 GCN 认为"干净"的节点里，球还能不能挑出真噪声？这才是球独有的、[X‖H] 给不了的信息。测错（只测边际相关性）会把死路误判成活路，浪费后续算法轮次。

### 红线

- ROCm Python：`C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe`，`cuda:0`
- 已有 H：`gb_graph_aware_minival_6_1/checkpoints/embeddings_v2/{DBLP,PubMed,AmazonPhoto}/grace/H.pt` + Cora/CiteSeer 锚点 H（小图作"已知有效"对照）
- 球构造用 Phase 32a 找到的**大图适配参数**（DBLP/PubMed: max_balls=256, split_improvement≈0.001-0.005），保证球非塌缩，这样诊断的是"球有用没用"而非"球建得起来没"
- clean label **只用于诊断指标计算**，绝不进任何训练/选择/加权的决策逻辑
- 纯诊断，不训练下游（除非 33b 过门才进 Phase 34）
- research-record 报告；测试全绿

---

## Phase 33a：球 vs GCN 噪声判别的条件正交性（核心诊断）

### 构造两个独立的"噪声嫌疑分"

对每个 train 节点 i（数据集 ∈ {DBLP, PubMed, AmazonPhoto}；Cora/CiteSeer 作对照基准），seeds 0,1,2，uniform 0.4：

1. **球嫌疑分 `s_ball(i)`**：i 的噪声标签在所属球内 train 多数票分布中的"反对票占比" = `1 - dist_B[y_noisy_i]`（球内不同意 i 标签的比例）。高 = 球认为 i 可疑。
2. **GCN 嫌疑分 `s_gcn(i)`**：在 `[X‖H]` 上训一个 warmup GCN（noisy CE，~30 epoch，**取早期 epoch 前的预测以避免记忆噪声**，或用 H 空间 kNN 共识作为更便宜的等价物——两者都算，对比），`s_gcn(i) = 1 - p_pred(y_noisy_i | i)`。高 = GCN 认为 i 可疑。

注意：warmup GCN 用 noisy 标签训练（合法，不用 clean），早停在记忆噪声之前（参考 early-learning：best epoch 在 3-6，取 epoch 5-10 窗口预测）。

### 关键诊断指标（用 clean label 仅评估）

设 `noisy(i)` = i 的 noisy 标签 ≠ clean 标签（真噪声,clean 仅诊断用）。

1. **边际判别力**：`s_ball` 和 `s_gcn` 各自的 AUROC（对 `noisy(i)`）——各自能多好地识别真噪声
2. **★条件正交增量（最重要）**：在 `s_gcn` 最低的那 50% 节点中（GCN 觉得"干净"），`s_ball` 对 `noisy(i)` 的 AUROC。若显著 > 0.5 → **球有 GCN 没有的正交信息,大图适配有戏**；若 ≈ 0.5 → 球冗余,大图无解
3. **对称对照**：在 `s_ball` 最低 50% 中，`s_gcn` 的 AUROC（反向，看谁覆盖谁）
4. **两分数相关性**：`corr(s_ball, s_gcn)`——高相关直接预示冗余
5. **小图对照**：Cora/CiteSeer 上同样算指标 2，确认"小图上球的条件正交增量明显 > 大图"（解释为什么小图有用大图没用，本身是论文论点）

### 输出 `runs/biggraph_orthogonality_diag_20260616/`：raw + summary + 报告

### 判定

| 指标 2（GCN-clean 子集内 s_ball 的 AUROC） | 结论 | 后续 |
|---|---|---|
| 大图上显著 > 0.5（如 ≥ 0.6） | 球有正交噪声信息 → **大图适配有戏** | → Phase 33b：离线评估三种用法 |
| 大图上 ≈ 0.5（0.5±0.05） | 球信息被 GCN 覆盖，冗余 | → 大图无解，安心收束 2 数据集，本诊断进论文作为"为什么大图无效"的精确证据 |

---

## Phase 33b：三种球用法的离线效果预测（仅当 33a 判"有戏"）

### 为什么一次测三种而不赌单一算法

同样"球有正交信息"前提下，利用方式不同下游效果可能差很远。离线预测三种，挑最优再真训练，避免赌错：

| 用法 | 操作 | 离线评估指标（clean 仅诊断）|
|---|---|---|
| **A. reweight** | train 节点按 `1 - s_ball` 加权 | 加权后 train 集的有效噪声率（噪声节点权重和 / 总权重）相对均匀加权下降多少 |
| **B. 筛除** | 丢弃 `s_ball` 最高的 top-k% train 节点 | 筛除集里真噪声占比（precision）；剩余 train 集噪声率;丢掉的干净标签量（代价）|
| **C. co-teaching 第二视角** | 只信任 `s_ball` 和 `s_gcn` **都**判干净的节点 | 双判干净集的真噪声率（应远低于单视角）；保留的样本量 |

### 关键：用法 C 是这次新加的、最契合正交性逻辑的

若 33a 证明球与 GCN 正交，则"两个独立噪声判别器取交集"（co-teaching 思想，但用 label-free 几何球当第二视角）应当显著优于任一单视角。这是有 novelty 的形态——经典 co-teaching 用两个 DNN 互教，这里用"球几何 + GCN 预测"两个异质视角。报告要明确对比 C 的双判干净集噪声率 vs A/B。

### 判定

| 离线最优用法的预测增益 | 后续 |
|---|---|
| 某用法（很可能是 C）显著降低有效噪声率且保留足够样本 | → Phase 34（写该用法的大图算法 + 真下游验证,新建执行计划）|
| 三种都不能在保留足够样本下显著降噪 | → 大图无解,收束 |

---

## Phase 34（占位，不在本计划执行）

仅当 33a+33b 都过门：写大图专用算法（最可能是"球作为 co-teaching 第二视角的样本选择"），真训练验证 `softdist-biggraph-variant > gcn_xh_ce` 且优于 shuffled。届时单独写执行计划。这会是"小图 soft-target、大图样本选择，一个 label-free 粒球框架两种用法"的完整故事——若成立是有价值的新贡献。

---

## 执行顺序

```
Phase 33a（条件正交性诊断,核心,几分钟+1个warmup GCN）
   ├─球有正交信息─► Phase 33b（三用法离线评估）
   │                   ├─某用法有效─► Phase 34（写大图算法,另立计划）
   │                   └─都无效─────► 收束 2 数据集
   └─球冗余────────► 收束 2 数据集（本诊断即"为什么大图无效"的精确证据）
```

## 无论结果都有论文价值

- 有戏 → 可能解锁大图 + 双范式框架（小图校正/大图选择），重大升级
- 无解 → 33a 的"条件正交增量：小图高、大图≈0"是论文 §6/§8 **最精确的边界机制证据**，远强于"我们只跑了 2 个数据集"的空白

## 完成后

回填 `SoftDist_XH_论文初稿雏形.html`（§6 边界分析 / §8 泛化）与经验文档、记忆。33a 的小图 vs 大图条件正交增量对比图是关键素材（无论正负）。

---

## 执行记录（2026-06-16）

### 实现与输出

已新增 `scripts/diagnose_biggraph_orthogonality.py`，输出目录：

- `runs/biggraph_orthogonality_diag_20260616/orthogonality_raw.csv`
- `runs/biggraph_orthogonality_diag_20260616/orthogonality_summary.csv`
- `runs/biggraph_orthogonality_diag_20260616/offline_usage_raw.csv`
- `runs/biggraph_orthogonality_diag_20260616/offline_usage_summary.csv`
- `runs/biggraph_orthogonality_diag_20260616/biggraph_orthogonality_report.md`

协议：Cora/CiteSeer/DBLP/PubMed/Amazon-Photo，seeds 0-2，uniform 0.4，percent 0.2/0.1/0.7。warmup GCN 使用 `[X||H]`，10 epochs，取 epochs 5-10 平均 softmax。clean label 只用于诊断指标，不进入任何训练/选择逻辑。

### Phase 33a 结果：条件正交性过门

核心指标是 `ball_auc_in_gcn_clean50`：在 GCN 认为最干净的 50% train 节点内，球嫌疑分对真噪声的 AUROC。预注册门槛为 0.6。

| 数据集 | train 噪声率 | ball AUC | GCN AUC | ball AUC in GCN-clean50 | corr(ball, GCN) | 判定 |
|---|---:|---:|---:|---:|---:|---|
| Cora | 0.3758 | 0.8957 | 0.9296 | 0.8460 | 0.7001 | 小图正对照成立 |
| CiteSeer | 0.3739 | 0.8383 | 0.7651 | 0.8286 | 0.4705 | 小图正对照成立 |
| DBLP | 0.3932 | 0.9059 | 0.9598 | 0.6955 | 0.7797 | 大图有正交信息 |
| PubMed | 0.3975 | 0.8703 | 0.9405 | 0.6633 | 0.7057 | 大图有正交信息 |
| Amazon-Photo | 0.4039 | 0.9467 | 0.7581 | 0.9483 | 0.4095 | 大图有强正交信息 |

结论：Phase 32 的 “soft-label 校正无增量” 不能解释为球信号完全冗余。更准确的解释是：球信号不适合继续做大图 soft-target replacement，但在样本干净性判别上仍有明显条件正交信息。

### Phase 33b 结果：co-teaching 第二视角最优

离线评估显示，`s_ball` 与 `s_gcn` 双判干净的 co-teaching 交集明显优于单独 reweight/filter。以 `keep_0.6_each` 为平衡点：

| 数据集 | 原 train 噪声率 | 双判干净后噪声率 | 噪声率下降 | 保留样本比例 |
|---|---:|---:|---:|---:|
| DBLP | 0.3932 | 0.0572 | 0.3360 | 0.5382 |
| PubMed | 0.3975 | 0.0695 | 0.3280 | 0.5245 |
| Amazon-Photo | 0.4039 | 0.0640 | 0.3399 | 0.4353 |

更激进的 `keep_0.5_each` 可把 DBLP/PubMed/Amazon-Photo 的有效噪声率压到约 1.4%-3.9%，但只保留 35%-44% 样本；`keep_0.7_each` 保留约 54%-62% 样本，噪声率约 10.8%-14.8%。Phase 34 真训练应优先试 `keep_0.6_each`，并把 0.5/0.7 作为敏感性。

### 最终判定

Phase 33 判为 **go to Phase 34**。

推荐的新大图形态不是 SoftDist，而是“球作为 co-teaching 第二视角的样本选择”：

- 小图：标签绝对量少，继续使用 soft distribution target。
- 大图：标签绝对量充足，使用 `s_ball` 与 warmup GCN 的 `s_gcn` 交集选择干净样本。

这给出一个更完整的双范式框架假设：同一个 label-free 粒球几何信号，在小图负责软标签校正，在大图负责干净样本选择。Phase 34 需要单独计划并做真下游验证，成功标准必须仍是 `real > gcn_xh_ce` 且 `real > shuffled/control`。
