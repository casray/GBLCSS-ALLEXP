# 执行计划：跨视角矛盾检测（Phase 40 · 粒球非同源价值的 go/no-go 闸门）

日期：2026-06-29
执行者：Codex
前置状态：Phase 39 已证明在 H 空间，粒球噪声检测 `s_ball_loo` 强（AUROC 0.83–0.96）但被 `s_knn`(kNN-in-H)/`s_gcn_conf` 稳定压制，`corr(ball, kNN-in-H)=0.78–0.90`——**球 = H 空间的粗粒化 kNN**。本周椭球 / GEAF / Mahalanobis 全是"在 H 空间内换球形状"，downstream 大多失败（GEAF 仅 AmazonCom 单 seed 正）。统一根因（"更高维度尺子"）：**粒球价值 = max(0, 球提供的结构 − 下游已能获取的结构)。在"球建在 H + 下游 GCN[X‖H] 也吃 H"的同源设定下，这个差≈0。**

**本计划是粒球这条线的最后一个 go/no-go 闸门：测试唯一一个逻辑上能让差值>0 的方向——让球去捕捉"原始特征 X 与表示 H 之间的跨视角矛盾"，这是任何单空间 kNN / GCN 都看不到、因而不被吸收的信号。** 过→拿到可写论文的突出点（粒球作为跨视角矛盾检测器）；不过→粒球在图节点上彻底关闭，直接转"边界刻画"型论文，不再开新方向。

**这是纯离线诊断，不训练任何 downstream 模型。**

---

## 0. 唯一要回答的问题（判据已锁死）

> 把同一组 train 节点分别在**原始 X 空间**和 **GRACE H 空间**建球，两套球给出的"局部多数标签"发生**冲突**的节点，是不是真噪声？而且——决定性地——这个冲突信号能否在 **"kNN-in-H 认干净" 且 "kNN-in-X 认干净" 的双干净交集子集**里仍抓出真噪声（即两个单视角都漏掉的噪声）？

**第一判据是"双干净交集内的条件 AUROC"，不是绝对 AUROC。** 因为绝对 AUROC 高但被单空间 kNN 压制，正是 Phase 39 的死法；跨视角信号的全部意义在于它**正交于任一单空间**。

---

## 0.1 红线（决定实验对不对）

1. **clean 标签只进评测**：`noisy_flag=(noisy_train!=clean_train)` 只用于 AUROC/AUPRC/P@K/条件 AUROC，**绝不进入任何 score 计算**。
2. **所有 score 只用 train mask 的 noisy 标签**；X 球、H 球、kNN-in-X、kNN-in-H 的邻居/投票池一律限 train。
3. **不改 SoftDist anchor、不动已注册 predictor、不训练 downstream**。本计划只新增一个离线诊断脚本。
4. **seed0 单数据集先 smoke**，通过再 3 数据集 × 5 seeds。
5. 新增涉及标签的函数配防泄漏单测；测试全绿；research-record 报告。

## 0.2 环境与复用（关键：几乎不写新机制，全是组装）

- ROCm Python：`C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe`，`cuda:0`
- 数据集：`cora,citeseer,coauthorcs`；percent split 0.2/0.1/0.7；uniform 0.4；seeds 0–4
- **新脚本 `scripts/diagnose_cross_view_conflict.py`，从 `scripts/diagnose_label_noise_detection.py` 复制扩展**。已确认可直接复用、无需改写：
  - `_load_dataset`（给 `data.feats` / `data.labels` / `data.adj` / `data.n_classes`）
  - `load_softdist_xh_gb_config(dataset_key)` → H 球参数 + `embedding_path`
  - `load_frozen_embeddings` → H
  - **`build_ordinary_granular_balls(features, max_balls, min_ball_size, split_improvement, random_state)`**——内部自带 L2 归一化，**接受任意特征矩阵**，所以把 dense 化的 `data.feats` 传进去即得 X 球（无需新建球代码）
  - `_ball_suspicion_scores`（含 `s_ball_loo` 与 JSD）、`_knn_suspicion_scores`（传 X 即 kNN-in-X、传 H 即 kNN-in-H）
  - `_quantile_mask` / `_safe_auc` / `_safe_corr` / `label_process` / AUPRC、P@K 逻辑
  - X 稀疏处理：`X = data.feats; if X.is_sparse: X = X.to_dense(); X = X.float()`（与 Phase 33 `_prepare_xh_features` 同款）
- **X 空间建球参数**：先用与 H 球**同一组** `max_balls/min_ball_size/split_improvement`（来自 `load_softdist_xh_gb_config`），保证两套球粒度可比、避免引入新自由度；报告输出两套实际参数。**不调参**。
- 固定超参不搜：`knn_k=20`（Phase 39 口径）；双干净交集分位 0.5；`pass_auc=0.6`。

## 0.3 undefined / eligibility（防靠丢难样本刷指标，沿用 Phase 39）

- undefined（球内 train<min_train_per_ball、无 LOO 分布等）score 填中性值 **0.5**。
- primary 指标在 **all-train** 上算；另报 eligible-only + `eligible_rate`。

---

## 1. 跨视角矛盾分数（本计划的唯一新机制）

对每个 train 节点 i，分别在 X 球和 H 球内算 LOO 多数标签与一致性。`maj_X(i)`/`maj_H(i)` = i 所在 X 球 / H 球内（去掉 i 自身）的 train noisy 多数标签；`p_{X\i}` / `p_{H\i}` = 对应 LOO 标签分布。

| score | 定义 | 含义 |
|---|---|---|
| **`s_xview`（primary，硬冲突）** | `1` 若 `maj_X(i) ≠ maj_H(i)` 否则 `0`；两套球任一 undefined 则填 0.5 | 两视角的局部共识互相矛盾——最直接的跨视角信号 |
| **`s_xview_soft`（primary，软冲突）** | `JSD(p_{X\i}, p_{H\i})` | 两视角 LOO 分布的散度，比硬冲突更细 |
| `s_xview_disagree` | `0.5·(1−p_{X\i}[ỹ_i]) + 0.5·(1−p_{H\i}[ỹ_i])` 仅在 `maj_X≠maj_H` 的节点上，否则 0 | 节点标签在"两视角分歧处"的可疑度 |

> 选 `s_xview` + `s_xview_soft` 为预注册 primary。`s_xview_disagree` 与下方 combo 为 exploratory（不触发 go/no-go，防后验挑选，沿用 Phase 39 纪律）。

## 2. Baseline（必须含两个单空间，证明跨视角是增量而非重复）

| baseline | 定义 | 角色 |
|---|---|---|
| `random` | 均匀随机 | 下界 |
| `s_knn_H`（复用） | `1 −` (k=20 H 最近 train 邻居同 noisy 标签比例) | **单空间 H 对照** |
| `s_knn_X`（复用，传 X） | 同上，在原始 X 空间 | **单空间 X 对照** |
| `s_ball_loo_H`（复用） | H 球 LOO 可疑度 | Phase 39 最强球检测器 |
| `s_ball_loo_X`（复用，X 球） | X 球 LOO 可疑度 | 单视角 X 球 |
| `s_combo_2knn`（exploratory） | `max(s_knn_H, s_knn_X)` 与 `mean(...)` 各报一版 | **关键对照：两个单空间 kNN 的朴素组合**——跨视角球若赢不过它，说明矛盾信号 ≤ 两 kNN 取并 |

> `s_combo_2knn` 是本计划的"命门 baseline"：它检验"跨视角矛盾"是否只是"两个单空间 kNN 之一报警"的换皮。primary 必须在双干净交集里超过它，才算真正非同源。

## 3. 评测指标（每 detector × dataset × seed）

- `AUROC`、`AUPRC`、`AUPRC_lift = AUPRC − train_noise_rate`
- `Precision@K`/`Recall@K`/`F1@K`，K=真实噪声数
- eligible-only 版本 + `eligible_rate`
- **两套球一致性诊断**：`maj_X≠maj_H` 的节点占 train 比例（冲突率）；冲突节点里真噪声占比（冲突 precision）；非冲突节点真噪声占比（应显著更低）

## 4. ★第一判据：双干净交集条件 AUROC（决定粒球这条线生死）

对两个 primary（`s_xview`、`s_xview_soft`）与命门 baseline `s_combo_2knn` 各算：

| 指标 | 子集 | 含义 |
|---|---|---|
| `*_auc_in_knnH_clean50` | s_knn_H 最干净 50% | 在"H-kNN 认干净"处还能否抓真噪声 |
| `*_auc_in_knnX_clean50` | s_knn_X 最干净 50% | 在"X-kNN 认干净"处还能否抓真噪声 |
| **`*_auc_in_knnH_AND_knnX_clean50`** | **kNN-H ∩ kNN-X 双干净（决定性）** | **两个单视角都漏掉的噪声，跨视角球能否抓到** |
| `corr(*, knn_H)` / `corr(*, knn_X)` | — | 与两个单空间的相关性（应显著低于 Phase 39 的 0.78–0.90）|

- **小样本护栏（沿用 Phase 39）**：双干净交集子集内真噪声样本 <8 → 该项填 NaN，G1 回退到两个单 clean50 子集同时判定。

## 5. go/no-go 闸门（预注册，按优先级）

均按 seed 平均 + paired 方向计数。

| 闸门 | 判据 | 不过的后果 |
|---|---|---|
| **G1 跨视角非同源（决定性）** | 某个 primary 在 **`knnH_clean50` 与 `knnX_clean50` 上 mean 条件 AUROC 同时 ≥0.6**，**≥4/5 seeds 正向**，**≥2/3 数据集**成立；且 **mean(primary − s_combo_2knn) 在双干净交集上 ≥0.02**（跨视角真正超过两 kNN 取并）。双干净交集子集过线为强加分。 | **粒球线彻底关闭**。直接转止损：写"图神经表示下粒球去噪适用性边界"论文，本结果作为"连跨视角也无非同源增量"的最终封口证据 |
| G2 冲突信号有效 | 冲突节点真噪声占比 > 非冲突节点（≥2/3 数据集，方向一致），primary AUROC 明显 > random | 跨视角冲突与噪声无关，G1 无意义 |
| G3 相关性下降 | `corr(primary, knn_H/X)` 明显低于 Phase 39 球的 0.78–0.90（佐证非同源） | 仅佐证项 |

**决策矩阵**：
- **G1 过** → 粒球作为**跨视角矛盾检测器**成立。这是非同源、不被任一单空间吸收的突出点。下一步规划：(a) 把它做成 detection 主表（vs 所有单空间 baseline + s_combo_2knn）；(b) 试它作为 SoftDist 的 α 门控/样本权重源，看能否转 downstream（**另立 Phase 41 计划，本计划不碰 downstream**）。
- **G1 不过** → **粒球这条线在图节点上关闭**。立即转止损论文，不再开任何新球变体。

## 6. 执行顺序

```
1. 复制 diagnose_label_noise_detection.py → diagnose_cross_view_conflict.py
   - 加载 data.feats（dense+float）作为 X；load_softdist_xh_gb_config 取 H 参数与 H
   - 用同一 builder 建 ball_X（传 X）与 ball_H（传 H），同组参数
   - 实现 §1 三个跨视角分数（primary/exploratory 分离）
   - 复用 §2 全部 baseline（s_knn_X = _knn_suspicion_scores 传 X；s_ball_loo_X = _ball_suspicion_scores 传 ball_X）+ 实现 s_combo_2knn
   - 实现 §4 三个条件 AUROC（含双干净交集 + 小样本护栏）
   - undefined→0.5；all-train 与 eligible-only 双报；输出两套实际构球参数
   单测：篡改 clean 标签 → 所有 score 不变；score 只用 train；
        两套球完全相同（人为令 X=H）时 s_xview 全 0；
        coauthorcs H 球用 256/32/0.001、X 球用同组参数不回退
2. seed0 smoke（cora）：不崩、冲突率合理(非 0 非 100%)、冲突节点真噪声占比 > 非冲突、clean 不进 score
   └─ 异常 → 停，修
3. cora,citeseer,coauthorcs × seeds 0-4 → 按 §5 判 G1-G3
4. 写报告 + go/no-go 结论
```

- 输出目录：`runs/cross_view_conflict_20260629/`
- 报告：`runs/cross_view_conflict_20260629/cross_view_conflict_report.md`（每 detector×dataset 的 AUROC/AUPRC_lift/P@K + 冲突率/冲突 precision + §4 条件 AUROC 表（突出双干净交集列与 primX−combo_2knn 差）+ §5 闸门判定 + 两套实际构球参数 + 明确 go/no-go 结论）

## 7. 完成后

- 更新经验文档 `02_experiment_history_and_lessons 11.md`（新增 Phase 40）、记忆 `gb-direction-turning-point-2026-06.md`、新窗口总结、探索轴总览 HTML。
- **若 G1 过**：起草 Phase 41（跨视角信号 → SoftDist α/权重 → downstream 验证，预注册 real>shuffled + 5-seed），并把"跨视角矛盾检测"写入论文作为核心突出点。
- **若 G1 不过**：转止损——按"适用性边界"重定位论文（统一定律 `球价值=max(0,球结构−下游已有结构)` + 全项目负结果作为证据 + 解释 8 篇表格文献为何不可迁移到图）。我（Claude）据此起草论文骨架与导师话术。

## 给导师的话术（诊断跑完后按结果二选一）

> 我把孙旭第五章的 GEAF 思路和我的 SoftDist 对齐分析后，发现它们的本质区别是：GEAF 的球建在原始特征上、下游分类器也弱，所以球提供了下游拿不到的结构；而我的球建在 GRACE 表示 H 上、下游 GCN 也吃 H，球与下游同源，所以被吸收。要让粒球在图上有独立价值，只能让它去捕捉"原始特征 X 与表示 H 之间的跨视角矛盾"——这是任何单空间方法都看不到的信号。我做了一个可行性诊断，关键看它能否在"X 和 H 两个视角都认为干净"的节点里仍抓出真噪声。[过→粒球作为跨视角矛盾检测器是我们的突出点；不过→我会把整个项目写成"图神经表示下粒球去噪的适用性边界"，给出一个可证伪的统一定律，解释为什么表格上成功的粒球去噪无法迁移到强表示的图节点——这本身是对粒球社区有价值的负结果。]

---

## Phase 40 执行结果备案（2026-06-29 已回填）

结果文件：`runs/cross_view_conflict_20260629/`

### 执行范围

- 已完成：`cora,citeseer` × seeds `0,1,2,3,4`
- 未继续跑：`coauthorcs`
- early-stop 原因：`cora` 与 `citeseer` 两个数据集已经全部未通过 primary G1。预注册 G1 要求至少 `2/3` 数据集通过，因此即使 `coauthorcs` 通过也无法把全局结论从 `STOP` 改成 `GO`。考虑到 `coauthorcs` 的 raw-X 构球较慢，停止继续消耗时间。

### 单元测试与 smoke

- 新增脚本：`scripts/diagnose_cross_view_conflict.py`
- 新增测试：`tests/test_cross_view_conflict.py`
- 验证结果：`py -3 -m pytest tests/test_cross_view_conflict.py -q` -> `5 passed`
- Cora seed0 smoke 正常完成，但信号为负：冲突节点真噪声占比低于非冲突节点，primary 在双干净条件下没有增量。

### G1 判定

| Dataset | Detector | H-clean AUC | X-clean AUC | H∩X minus combo | H pass | X pass | Gate |
|---|---|---:|---:|---:|---:|---:|---|
| Cora | `s_xview` | 0.4485 | 0.5179 | -0.2868 | 0/5 | 0/5 | FAIL |
| Cora | `s_xview_soft` | 0.4098 | 0.5266 | -0.2950 | 0/5 | 1/5 | FAIL |
| CiteSeer | `s_xview` | 0.4959 | 0.4900 | -0.3036 | 0/5 | 0/5 | FAIL |
| CiteSeer | `s_xview_soft` | 0.4878 | 0.4959 | -0.2788 | 0/5 | 0/5 | FAIL |

判定：G1 不通过。两个 primary 都不能在 `kNN-H` 或 `kNN-X` 认为干净的子集里稳定抓出噪声，也远低于 `s_combo_2knn`。

### G2/G3 观察

- G2 不支持：Cora 冲突节点噪声率 `0.3970`，非冲突节点噪声率 `0.4287`，delta `-0.0317`；CiteSeer 冲突节点噪声率 `0.4036`，非冲突节点噪声率 `0.4351`，delta `-0.0314`。也就是说，发生 X/H 局部多数冲突的节点反而没有更脏。
- G3 部分成立但没有价值：primary 与 `kNN-H/X` 的相关性很低，说明它确实不是单空间 kNN 的同源复制；但低相关没有带来有效检测能力，因此不能作为贡献。

### 最终结论

Phase 40 判定为 `STOP`。跨视角粒球冲突没有提供单视角 kNN 拿不到的有效噪声信号。按照本计划的 go/no-go 规则，不进入 Phase 41 downstream，也不再继续开新的 H 空间或 X/H 粒球变体。该结果应作为“图神经表示下粒球去噪适用性边界”的封口证据之一。
