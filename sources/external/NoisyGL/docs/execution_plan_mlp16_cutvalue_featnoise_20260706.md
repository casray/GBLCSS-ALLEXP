# 执行计划：Cut-Value 结构量化 + 特征噪声防御臂（Phase MLP-16）

日期：2026-07-06
执行者：Codex
定位：**写作期的 §6 机制量化 + 家族标准协议防御，不是新机制。** 依据 13 篇 GNN-to-MLP 文献全文核验（`related_work_cut_notes_20260702.md` §6）：(P0) Cut Value 是该家族 4 篇论文（GLNN/NOSMOG/VQGraph/SALE-MLP）的标配结构一致性指标，在已有预测输出上直接算，量化"共识软目标给 X-only 学生注入的结构量"；(P1) 特征噪声曲线 `X̃=(1−α)X+αε` 是该家族的标准鲁棒性协议（NOSMOG §5.4 / PGKD §6.2 / VQGraph 附录 C.3 同款），来自这个圈的审稿人大概率按此协议提问。触发依据：硬规则第二条"防御具名审稿攻击"——攻击名 = "graph-free 圈审稿人用家族标准协议问特征噪声/结构一致性"。

## 0. 总红线

1. 不改主表、不改主线 claim、不引入新机制轴；两项产出分别进 §6 机制段与鲁棒性/limitation 段。
2. P0 纯后处理零训练；P1 只扰动**测试期特征**，训练完全复用或按协议重跑，噪声只加 train 标签的红线不变（特征噪声是评测期扰动，与标签噪声正交）。
3. 判读分支预注册（见各节），任何结局如实进论文。
4. <1×std 写 comparable；P1 曲线整条进图，不挑 α 点。

## P0 —— Cut Value 结构一致性量化（零训练成本，纯后处理）

### 动机与公式
- `CV = tr(Ŷᵀ A Ŷ) / tr(Ŷᵀ D Ŷ)`，Ŷ = 模型对全部节点的 softmax 输出（N×C），A = 邻接矩阵，D = 度矩阵。CV∈[0,1]，越高表示预测与图拓扑越一致。GLNN 报过参照量级：SAGE≈0.92、GLNN≈0.90、vanilla MLP≈0.76（五数据集均值，供 sanity 对照，不直接比数字）。
- 论文用途（§6 机制段）：量化"共识软目标在不给学生任何图输入的情况下注入了多少结构 bias"，并与 Demystifying (AAAI26) 的 NTK-邻接对齐理论互为印证。

### 做法
1. 在 `run_standard_downstream.py`（或独立脚本 `scripts/compute_cut_value.py`）加一个后处理模式：加载已训好模型的全节点 softmax 输出（若未落盘则按原 config+seed 重训——弱 MLP 重训便宜，但**优先复用**已有 checkpoint/输出）。
2. 计算矩阵：
```
方法臂: {Ours β0.5, MLP+CE, shuffled, kNN@20-β0.5(MLP-11 现成 config), Ours-distill(T3), GLNN-style(T1)}
  × {Cora, CiteSeer, PubMed, CoauthorCS, AmazonPhoto} × uniform0.4 × seeds 0-4
  （Ours-distill/GLNN-style 仅 Cora/CiteSeer/PubMed——MLP-9 范围）
```
3. A 用 NoisyGL dataloader 的原始邻接（对称化、无自环，与家族文献一致；实现里写清约定）。CV 只做评测，不进任何训练。
4. 单测：(a) 玩具图上手算 CV 对拍；(b) 完美同配预测 CV→高、随机预测 CV→低的方向性断言；(c) Ŷ 行和为 1。

### ★预注册判读

| 观察 | §6 写法 |
|---|---|
| CV(Ours) > CV(CE) 且 > CV(shuffled)，≥4/5 数据集 | 主预期："共识软目标显著提高 X-only 学生的预测-拓扑一致性——结构 bias 经监督侧通道真实传入"；与 Demystifying 理论互印 |
| CV(Ours) ≈ CV(kNN@20) | 预期（两者同为局部共识）；如实写"结构注入量与共识形式无关"，与平替结论一致 |
| CV(Ours-distill) > CV(Ours train-only) | 预期（teacher 蒸馏传更多结构）；作为扩展节的机制补充 |
| CV(Ours) ≤ CV(CE) | 反预期——如实报告并写"共识增益不经由拓扑一致性通道"，§6 措辞相应收窄；不隐藏 |

## P1 —— 特征噪声防御臂（家族标准协议）

### 协议（照抄家族三篇的共同版本）
- `X̃_test = (1−α)·X_test + α·ε`，ε 为各向同性高斯（按特征维度标准化后注入，实现细节与 NOSMOG/PGKD 一致并在报告里写明）；α ∈ {0, 0.1, 0.3, 0.5, 0.7, 0.9}。
- **只扰动评测期输入**：训练在干净特征 + 噪声标签下照旧（train label noise 协议不变）；α=0 即现有主表数字，作锚点校验。
- **我们的 H 不重算**（H 来自训练期干净特征图——这符合部署语义：预训练资产已冻结，测试特征在线到达时已损坏）。此约定在报告与论文中显式声明。

### 矩阵
```
{Ours β0.5, MLP+CE, small-loss, JoCoR, shuffled}
  × {Cora, CiteSeer, PubMed, CoauthorCS, AmazonPhoto}
  × uniform0.4(标签噪声固定) × α∈{0.1,0.3,0.5,0.7,0.9} × seeds 0-4
  = 5 方法 × 5 数据集 × 5 α × 5 seeds = 625 次评测
```
- **实现关键**：训练一次、多 α 评测——每个 (方法,数据集,seed) 只训一个模型（或直接加载主表已有模型），对 5 个 α 各做一次前向评测。实际新训练量 = 0~125 run（视已有模型可否复用），评测极快，总时长半天内。
- α=0 锚点断言：评测值必须逐 seed 等于主表值（防管线错位，MLP-14 教训）。

### ★预注册判读（诚实先验写前面）

先验：**中性偏负**。GLNN Fig.5 已示 MLP 系对特征噪声比 GNN 更脆（X-only 天然依赖特征质量）；我们相对 CE/small-loss 的预期是**同斜率退化**（大家都是 X-only MLP，特征坏了一起坏），共识软目标不针对特征噪声设计，没理由更抗。

| 观察 | 论文写法 |
|---|---|
| Ours 与 baselines 同斜率退化、相对优势保持 | 主预期："标签噪声下的优势在测试特征噪声下保持相对稳定；绝对精度随特征质量退化是 X-only 方法的共同属性（引 GLNN Fig.5）"——进鲁棒性小节或附录 |
| Ours 退化明显更快（斜率差 >1×std） | 如实进 limitation："共识软目标训练的决策边界对测试特征扰动更敏感"；不藏 |
| Ours 退化更慢 | 意外正结果；如实报告但不升级为卖点（机制上没有设计支撑，写"观察到但归因留待未来工作"） |

## 运行量与产出

| 项 | 新训练 | 评测 | 时长 |
|---|---:|---:|---|
| P0 Cut Value | 0（优先复用；缺输出则重训 ≤150 弱 MLP run） | ~150 次 CV 计算 | 数小时 |
| P1 特征噪声 | 0~125 run（视复用） | 625 次前向 | 半天内 |

- 输出目录：`runs/mlp16_cutvalue_featnoise_20260706/`
- 报告 `mlp16_report.md` 必含：
  1. CV 总表（6 臂 × 5 数据集 × 5-seed，mean±std）+ 预注册分支对号入座；
  2. 特征噪声曲线表/图（5 方法 × 5 数据集 × 6 α 含 α=0 锚点）+ 斜率对比 + 分支对号入座；
  3. α=0 锚点校验结果（逐 seed 对照主表）；
  4. 完整性 expected/observed/missing。

## 完成后

- 骨架 v2 §6 补 CV 机制段（按 P0 分支措辞）；鲁棒性/limitation 段补特征噪声句（按 P1 分支）；T2/附录表清单加 CV 表与特征噪声图。
- 更新记忆与 `99_session_handoff.md`。
- **硬规则继续生效**：此后新实验仍须"导师点名"或"防御具名攻击"。

---

## 6. 执行结果备案（Codex 跑完后按 MLP-8~15 格式追加于此）

### 2026-07-06 Codex 执行完成

- 新增实现：
  - `scripts/run_mlp16_cutvalue_featnoise.py`
  - `tests/test_mlp16_cutvalue_featnoise.py`
  - `config/knn_mlp_softdist_beta05/knn_mlp_softdist_beta05_amazonpho.yaml`
- 输出目录：`runs/mlp16_cutvalue_featnoise_20260706/`
- 完整性：180/180 model groups 完成；`mlp16_completeness.csv` missing=0；`mlp16_anchor_check.csv` 中 α=0 锚点失败=0。
- 验证：`python -m pytest tests/test_mlp16_cutvalue_featnoise.py -q`，3 passed。
- 图表：`mlp16_report.md`、`mlp16_feature_noise_curves.png`、`mlp16_cut_value_aggregate.csv`、`mlp16_feature_noise_aggregate.csv`。

#### P0 Cut Value 结论

P0 主预期没有成立。`CV(Ours) > CV(CE) 且 > CV(shuffled)` 只在 CiteSeer/CoauthorCS 局部成立，不满足 ≥4/5 数据集标准：

| 比较 | Cora | CiteSeer | PubMed | CoauthorCS | AmazonPhoto |
|---|---:|---:|---:|---:|---:|
| CV(Ours−CE) | −0.1803 | +0.0394 | −0.0320 | +0.0329 | −0.0015 |
| CV(Ours−Shuffled) | −0.0513 | +0.0477 | −0.0702 | +0.0221 | −0.0288 |
| CV(Ours−kNN) | −0.0526 | −0.0495 | +0.0267 | −0.0068 | −0.0058 |

预注册分支：**反预期分支**。论文 §6 不应写“train-only 共识软目标显著提高拓扑一致性”。更安全写法：

> Cut Value does not consistently increase for the train-only target variant; its gains are therefore not primarily explained by a global topology-smoothness channel. In contrast, the distillation variant strongly increases CV over train-only on Cora/CiteSeer/PubMed, suggesting that teacher distillation carries substantially stronger topology-aligned bias.

支持扩展节的次结论成立：`CV(Ours-distill) > CV(Ours train-only)` 在 Cora/CiteSeer/PubMed 全部 5/5 seeds 为正（+0.2458/+0.2460/+0.1388）。

#### P1 特征噪声结论

P1 符合“X-only 方法共同随特征质量退化”的先验，但不是完全同斜率：

- Ours 在 Cora/CiteSeer/AmazonPhoto 上相对优势基本保持到中高噪声；α=0.9 时 Cora 仅略低于 JoCoR −0.09pp，CiteSeer 仍 +0.29pp，AmazonPhoto 仍 +0.81pp。
- PubMed 继续是边界：α=0 已低于 JoCoR/Small-loss，α=0.3~0.7 差距扩大，α=0.9 时收窄但仍低 −1.40pp。
- CoauthorCS 中等 α 下 JoCoR 更稳（α=0.3/0.5/0.7 分别为 −1.26/−3.05/−0.57pp），α=0.9 Ours 反而 +1.51pp。

α=0→0.9 平均 drop：

| 数据集 | Ours | CE | Small-loss | JoCoR | Shuffled |
|---|---:|---:|---:|---:|---:|
| Cora | 34.0pp | 22.8pp | 31.4pp | 29.8pp | 22.5pp |
| CiteSeer | 32.7pp | 23.8pp | 28.8pp | 29.2pp | 22.6pp |
| PubMed | 31.6pp | 23.1pp | 36.1pp | 36.2pp | 21.7pp |
| CoauthorCS | 59.5pp | 60.3pp | 59.8pp | 59.8pp | 58.5pp |
| AmazonPhoto | 54.8pp | 48.4pp | 51.3pp | 53.7pp | 46.7pp |

预注册分支：**主预期 + limitation 混合**。可写“标签噪声优势在多数数据集/α 区间保持，但 SoftDist 不是特征噪声防御机制；在 PubMed 和 CoauthorCS 中等 α 下存在更敏感或被 JoCoR 反超的边界。”
