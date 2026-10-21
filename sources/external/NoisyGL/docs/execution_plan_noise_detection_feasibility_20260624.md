# 执行计划：图节点标签噪声检测可行性诊断（Phase 39 · 换题 go/no-go 闸门）

日期：2026-06-24（v2，纳入 GPT 审稿 8 点 + Claude 加严 2 点）
执行者：Codex
前置状态：SoftDist[X‖H] 已冻结为主方法；分类增量在 Phase 24 公平对照（NRGNN 同享 [X‖H]）下被吸收，Phase 34 GBSelect 下游也证明"球离线信号强、下游被 GCN 单视角/样本量损失吸收"。导师建议：(1) 多数据集平均效果要好 (2) 可考虑换任务 (3) 查最新文献。八篇粒球去噪文献 + 两个 GPT + 我方教训三方收敛：**粒球在噪声标签问题里的有效用法是"估计标签可信度 / 检测噪声"，而非硬提分类**。

**本计划是纯离线诊断，唯一目的：决定要不要把论文主任务从"噪声标签下节点分类"换成"图节点训练标签错误检测与清洗"。它是换题的闸门，不是换题后的第一个实验。**

---

## 0. 这份计划回答的唯一问题（判据优先级已锁死）

> 粒球的节点级噪声分数，是不是一个**好检测器**，并且——更关键——它提供的信息是否**正交于"H 空间 kNN 一致性 / GCN 置信度"**？

**第一判据是正交性，不是 AUROC 排名。** 球建在 GRACE H 上，如果球检测分数 ≈ kNN-in-H，那它在分类任务里被 [X‖H]/NRGNN-XH 吸收的病，会在检测任务里被一个 "kNN-in-H baseline" 原样重演。AUROC 再高、被简单 baseline 平替 = 没赢。所以正交性过不了，换题也救不了，必须前置。

---

## 0.1 红线（必读，决定实验对不对）

1. **clean 标签只进评测**：`noisy_flag = (noisy_train != clean_train)` 只用于 AUROC / AUPRC / P@K / 覆盖率 / 保留率诊断，**绝不进入任何一个 score 的计算**。
2. **所有 score 只用 train mask 的 noisy 标签**：球投票、kNN、图邻居、LP，邻居池一律限定 train；val/test 标签不参与任何 score。
3. **不改 SoftDist anchor**，不动任何已注册 predictor。本计划新增独立诊断脚本，除复用 Phase 33 的 10-epoch warmup GCN（仅为 GCN-conf baseline）外不训练任何下游模型。
4. **seed0 单数据集先 smoke**，通过再 3 数据集 × 5 seeds。
5. 新增涉及标签的函数必须配防泄漏单测；测试全绿；research-record 报告。

## 0.2 环境与复用（关键：扩展已有脚本，构球与主方法同源）

- ROCm Python：`C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe`，`cuda:0`
- **构球参数与 embedding 路径不再用 BIG_BALL_PARAMS 回退，而是 `load_conf(None, "frozengrace_gb_gcn_softdist_xh", dataset_key).gb` 取 `max_balls/min_ball_size/split_improvement/embedding_path`**——与冻结的 SoftDist 主方法用同一份 config、建出同一批球，杜绝漂移。已确认三数据集：

  | dataset_key | max_balls | min_ball_size | split_improvement |
  |---|---:|---:|---:|
  | cora | 64 | 32 | 0.05 |
  | citeseer | 64 | 32 | 0.05 |
  | coauthorcs | 256 | 32 | 0.001 |

  **报告里必须输出每个 run 实际使用的构球参数**（防止 Coauthor-CS 误回退到 Cora 口径）。
- NoisyGL 显式 percent split 0.2/0.1/0.7，uniform 0.4
- **新脚本 `scripts/diagnose_label_noise_detection.py`，从 `scripts/diagnose_biggraph_orthogonality.py` 复制扩展**。现成可复用：`load_frozen_embeddings`/`build_ordinary_granular_balls`/`partition_purity_diagnostics`/`_ball_suspicion_scores`/`_knn_suspicion_scores`/`_warmup_gcn_probs`/`_safe_auc`/`_safe_corr`/`_quantile_mask`/`_effective_noise_rate`。新增 `from sklearn.metrics import average_precision_score`。
- **超参一律固定，不网格搜索**：`knn_k=20`（Phase 33 口径）；`ball_neighbor_K=10`（primary，5/20 仅敏感性、不进主判据）；LP 步数 `T=2`（40% 噪声下多步传播会扩散噪声，故取小步）；`pass_auc=0.6`。

## 0.3 undefined / eligibility 处理（防"靠丢难样本刷指标"）

- undefined 情形：球内 train 仅 1 个（无 `p_{B\i}`）、球 train < `min_train_per_ball`、`s_graphdens` 分母为 0、`s_center` 的 `r_B≈0`。
- **统一规则**：undefined score 填中性值 **0.5**（不是 NaN、不跳过）。
- **primary 指标在 all-train 上算**（undefined 计入，按 0.5 处理）；**另报 eligible-only 版本**与每个 detector 的 **`eligible_rate`**。若某 detector 仅在 eligible 子集上好看、all-train 上塌，视为靠丢样本取巧，不予采信。

---

## 1. 节点级噪声分数（detector 候选）

每个 score 越大=越可疑。`B`=节点 i 所在球，`T_B`=球内 train 节点，`p_B=counts/total`（counts 在 `T_B` 上对 noisy 标签计数）。**primary（参与 go 判据）与 exploratory（仅报告/上界）已预注册分离。**

| score | 定义 | 角色 | 信号族 | 来源 |
|---|---|---|---|---|
| `s_ball`（已有，含自身） | `1 − p_B[ỹ_i]` | 参照 | H-球 | Phase 33 现状 |
| **`s_ball_loo`（primary_ball）** | `1 − p_{B\i}[ỹ_i]`，自身从 counts 去掉、total−1 | **primary** | H-球 | web GPT #1（去自确认；选它做 primary 是因为它是 `s_ball` 最直接、最有原理的升级）|
| `s_jsd_loo` | `JSD(onehot(ỹ_i), p_{B\i})` | exploratory | H-球 | MgCNL |
| `s_center` | `1 − exp(−‖h_i−c_B‖²/(r_B²+ε))` | exploratory | H-几何 | G-GBC |
| `s_entropy` | `H(p_{B\i})`（球内广播） | exploratory | H-球 | BNO-GB |
| `s_ballnbr` | `1 − C_B`，`C_B`=H 空间最近 `K=10` 球中多数类与 B 一致比例 | exploratory | H-球间 | BNO-GB |
| **`s_graphdens`（primary_graph）** | `1 − G_B`，`G_B`=B 与"同多数类近邻球"间图边数 / B 与全部近邻球图边数 | **primary** | 含图边（**非纯图、非完全非-H**）| web GPT #3 / GAGB |
| `s_combo_H` | `z(s_ball_loo)+z(s_center)+z(s_entropy)` 均值 | exploratory | H 混合 | 组合 |
| `s_combo_HG` | `s_combo_H + z(s_graphdens)` | exploratory | H+图 | 组合 |

> **`s_graphdens` 表述纪律**：它仍依赖 H 空间构球、H 空间近邻球、noisy majority label，因此只是"含图边信息的候选正交信号"，**不能写成"纯图"或"完全非-H"**。
> **后验偏差纪律（GPT #2）**：go 判据只认 `s_ball_loo` 和 `s_graphdens` 两个预注册 primary。`s_combo_*`、`s_jsd_loo`、`s_center`、"best-of" 作为 exploratory / oracle 上界报告。**若 best-of 好但两个 primary 不好，只能写"存在可挖信号、需另立验证"，不得据此宣布 go。**

## 2. Baseline 检测器（必须广，含决定性 H 对照与对应 shuffled 控制）

| baseline | 定义 | 角色 |
|---|---|---|
| `random` | 均匀随机 | 下界 |
| `s_knn`（已有） | `1 −` (k=20 个 H 最近 train 邻居同 noisy 标签比例) | **决定性 H 对照** |
| `s_gcn_conf`（已有） | `1 − p_warmup(ỹ_i)`，warmup GCN([X‖H]) ep5-10 均值 | **决定性模型对照** |
| `s_gcn_entropy` | warmup 概率分布熵 | 模型不确定度 |
| `s_gcn_smallloss` | 节点 noisy CE loss | 经典准则 |
| `s_graphnbr` | `1 −` (1-hop **train** 邻居同 noisy 标签比例) | **决定性纯图对照**（球图增量要打它）|
| `s_lp` | 轻量 LP：noisy train one-hot 经对称归一化邻接传播 `T=2` 步，`1 − prop[ỹ_i]` | 图传播对照 |
| `s_shuffled` | 打乱球分配上算 `s_ball_loo` | **真球结构对照（配 primary_ball）** |
| `s_graphdens_shuffled` | 打乱球分配上重算 `s_graphdens` | **真球结构对照（配 primary_graph，GPT #5）** |
| NRGNN/RTGNN/TFR/R2LP 可抽取分数 | best-effort | **可选**，不阻塞闸门 |

## 3. 评测指标（每 detector × dataset × seed）

- `AUROC`、`AUPRC`
- **`AUPRC_lift = AUPRC − train_noise_rate`**（随机 AUPRC 基线=噪声率而非 0.5，必须报 lift）
- `Precision@K`/`Recall@K`/`F1@K`，K=真实噪声数；附 top 20/30/40% 三档
- **类别覆盖率**：top-K 标记里覆盖了 C 类中的几类；是否某些类的噪声系统性抓不到
- `retention`/`dropped_clean`（误删干净）在 keep=0.5/0.6/0.7
- 每个 detector 另报 eligible-only 指标 + `eligible_rate`

## 4. ★第一判据：正交性（决定换不换题的核心）

对两个 primary（`s_ball_loo`、`s_graphdens`）各算条件 AUROC：

| 正交性指标 | 子集 | 含义 |
|---|---|---|
| `*_auc_in_knn_clean50` | s_knn 最干净 50% | 在"kNN-in-H 认干净"处还能否抓真噪声 |
| `*_auc_in_gcn_clean50` | s_gcn_conf 最干净 50% | 在"GCN 认干净"处还能否抓真噪声 |
| **`*_auc_in_knn_AND_gcn_clean50`** | **kNN∩GCN 都认干净（GPT #4）** | **最贴近"被 H/GCN 吸收"命门的测试** |
| `corr(*, knn)` / `corr(*, gcn)` | — | 高相关（>~0.8）= 冗余 |
| `knn_auc_in_*_clean50` | 对称项 | H 视角在球干净子集还有没有信息 |

- **intersection 小样本护栏**：knn50∩gcn50 在 Cora 约 25% train（~135 节点）。若该子集内**噪声样本 <8**，`*_auc_in_knn_AND_gcn_clean50` 填 NaN，G1 回退到两个单 clean50 子集判定。
- **图增量佐证（exploratory）**：报告 `s_combo_HG − s_combo_H` 的 AUROC 差，用于 G4 归因（提升是否来自图结构），但不单独决定 go。

## 5. go/no-go 闸门（预注册，按优先级）

均按 **seed 平均** + **paired 方向计数**判定。

| 闸门 | 判据 | 不过的后果 |
|---|---|---|
| **G1 正交性（决定性）** | 对某个 primary：在 `knn_clean50` 与 `gcn_clean50` 上 **mean 条件 AUROC ≥ 0.6** 且 **≥4/5 seeds 为正向(>0.5)**，并 **mean(real − shuffled) ≥ 0.02**（`s_ball_loo` 配 `s_shuffled`，`s_graphdens` 配 `s_graphdens_shuffled`）。**G1a=s_ball_loo、G1b=s_graphdens 分别评，≥1 个 primary 在 ≥2/3 数据集过线即 G1 过**。intersection 子集过线为加分项。 | **不换题**。球检测=H-kNN 平替；正交性结果写入论文 §6 作为"离线可分但被 H 视角吸收"的精确边界 |
| G2 检测质量 | primary 的 AUROC 与 AUPRC_lift 明显 > random，且不被 s_knn/s_gcn_conf 压制（≥最强 baseline 或在其 std 内），≥2/3 数据集 | 检测器本身不够强，换题无卖点 |
| G3 类别覆盖 | top-K 标记覆盖绝大多数类，不塌缩到少数类 | 检测有系统盲区，论文需限定适用范围 |
| **G4 图增量（真正的新意）** | **`s_graphdens` > `s_graphnbr`（纯图 baseline）**，≥2/3 数据集；且 `s_combo_HG − s_combo_H > 0` 佐证 | 图结构无独立增量 → 换题后仍只是"又一个 H-球检测器"，卖点须改为"统一节点级多证据检测框架"而非"图感知" |

**决策矩阵**：
- G1 + G2 + G4 过（G3 至少多数类覆盖）→ **正式换题**：主任务=图节点训练标签错误检测与清洗，分类降为下游 utility（检测→清洗→分类提升闭环）。
- **G1 不过 → 不换题**，老实收口分类论文，正交性边界写入 §6。
- G1/G2 过但 G4 不过 → 换题成立但"图增量"叙事不成立，卖点改为"统一节点级多证据检测框架"。

## 6. 执行顺序

```
1. 复制 diagnose_biggraph_orthogonality.py → diagnose_label_noise_detection.py
   实现 §1 全部 score（primary/exploratory 分离）+ §2 全部 baseline（含两个 shuffled 控制）
   + §3 全部指标（含 AUPRC_lift、类别覆盖、eligible_rate）+ §4 正交性（含 intersection 子集 + 小样本护栏）
   构球参数走 load_conf(softdist_xh) 同源；undefined→0.5；报告输出实际构球参数
   单测：篡改 clean 标签 → 所有 score 不变；score 只用 train；s_ball_loo 在球内全同标签时为 0；
        undefined 填 0.5；coauthorcs 实际用 256/32/0.001 不回退 Cora
2. seed0 单数据集 smoke（cora）：不崩、AUROC 合理、real(s_ball_loo) > shuffled、clean 不进 score
   └─ 异常 → 停，修
3. cora,citeseer,coauthorcs × seeds 0-4 全量 → 按 §5 判 G1-G4
4. 【Phase 39b，仅当 G1 且 G2 过】补 boundary 数据集 = pubmed（3 类、低可聚类，球已知偏弱）
   — 不要求赢，只看失败是否符合正交性解释；防止只在"球本来就强"的三图上自我确认（GPT #8 + Claude 加严）
```

- 输出目录：`runs/noise_detection_feasibility_20260624/`
- 报告：`runs/noise_detection_feasibility_20260624/noise_detection_feasibility_report.md`（每 detector×dataset 的 AUROC/AUPRC_lift/P@K/覆盖率/eligible_rate + §4 正交性表 + §5 闸门判定 + 实际构球参数 + 明确 go/no-go 结论）

## 7. 完成后

- 更新经验文档 `02_experiment_history_and_lessons 11.md`（新增 Phase 39）、记忆 `gb-direction-turning-point-2026-06.md`、新窗口总结、探索轴总览 HTML。
- **若 go**：起草换题定位（检测为主、分类为下游 utility），把 SCGNN / MgCNL / BNO-GB / 王炳贵硕士论文 作为任务框架引文加入 related work；规划检测主表 + 清洗→分类闭环表。
- **若 no-go**：正交性结果进论文 §6 边界分析，分类论文按现状诚实收口；不再沿球检测做调参追赶。

## 给导师的话术（诊断跑完后按结果二选一）

> 这八篇文献共同支持：粒球在噪声标签问题中的有效用法是估计标签可信度 / 检测噪声，而非硬提分类。我先做了可行性诊断，关键看两点：(1) 球的噪声检测能否赢经典 filter 和图 baseline；(2)——更关键——球的检测信号是否独立于"H 空间 kNN 一致性 / GCN 置信度"。因为分类任务失败的根因正是球被基于 H 的简单方法吸收，换任务若不能证明正交性会重蹈覆辙。诊断[漂亮→正式把主任务调整为图节点标签错误检测；平替→老实收口分类论文，把正交性写成方法边界]。

---

## Phase 39 执行结果备案（待填）

结果文件：`runs/noise_detection_feasibility_20260624/`
（Codex 跑完后回填：各闸门判定、go/no-go 结论、两个 primary 检测器的条件 AUC（含 intersection 子集）、图增量是否成立、boundary 数据集失败是否符合正交性解释。）
