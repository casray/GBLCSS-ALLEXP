# GB-GCL 实验历史与经验 - 实验 C/D：噪声标签节点分类主线收官

本文档整理 2026-06-09 至 2026-06-11 这一轮窗口内的全部实验：从 C1 ratio-split 验证出发，经过框架迁移（GRACE 微调管线 → frozen-GRACE + 端到端 GCN）、target 构造打磨（hard → gated → soft → SoftDist）、五个增强机制的系统性排除，到 [X||H] 主线确认和 noise grid 适用边界分析。机制探索在本轮结束时正式收官，最终方法为 **SoftDist [X||H]**。

涉及目录：`gb_denoise_stage_b_compactness`（C1）、`gb_c1_ratio_success_20260609`（C1 快照与改进）、`external/NoisyGL`（主战场）。

---

# 1. 总判断

- Confirmed: C1（GRACE 预训练 + GB 微调）的粒球机制在 ratio split（train 20%）下真实有效（vs noball +19.3/+14.8pp，5/5 seeds），但整个 Phase 2 微调框架弱于端到端 GCN，C1 在 Cora 输 GCN+CE 6.6pp。瓶颈不是球，是训练范式。
- Confirmed: 框架迁移成功——frozen GRACE 只做"建球 + 标签预处理"，分类器换成全新端到端 GCN，是本轮最大的转折。GRACE frozen 表示线性可分性（0.837）本来就高于 noisy GCN（0.810），之前是 Phase 2 微调把好表示用坏了（noball 微调后 0.804→0.570）。
- Confirmed: target 构造的最优形式是 **SoftDist**：multiclass alpha `(margin - 1/C)/(1 - 1/C)` + 球内 train 噪声标签完整分布作为混合目标。hard 校正 NCR 高但 CDR 高（Cora 22%），二值 gate 在两个数据集需要相反的松紧，binary-alpha soft 校正强度不足，SoftDist 是第一个两数据集同时赢 GCN+CE 的设置。
- Confirmed: **[X||H] 特征拼接是第二个真实增量**：H 对 CE 分类器 +1.9/+2.3pp，球在 H 之上仍有 +1.1/+1.3pp，real>shuffled 双数据集 5/5。最终 SoftDist [X||H] 5-seed Cora 0.8137±0.0067，均值超 NRGNN（0.8107±0.0230）且 std 小 3.4 倍。
- Confirmed: 在 SoftDist 之上叠加的全部七个增强方向均判负或无增量（动态 alpha、ball consistency、NRGNN+BallAux、自适应粒度 E1、无标签扩展、early-learning 裁决 ×2）。每次失败都有预注册门槛和诊断数据，构成六个干净的机制边界负结果。
- Confirmed: noise grid 无 regime win。Cora 在 4/5 个 regime 与 NRGNN tie（更稳），CiteSeer 全 regime 输，uniform 0.6 下 offline 校正管线崩溃（输 11pp，real 甚至输 shuffled）。球增量的价值区间是中高噪声（uniform/pair 0.4），低噪声下冗余。
- Decision: **机制探索收官，转入论文写作。** 主 claim：简单非迭代管线在 Cora 与 NRGNN 打平且方差小 3 倍 + 表示阶段对标签噪声构造性免疫（全 regime 复用同一份 H）+ 完整归因链 + 诚实适用边界。

---

# 2. 实验时间线总索引

| # | 日期 | 实验 | 一句话结果 | 决策 |
|---|---|---|---|---|
| 1 | 06-09 | C1 ratio-split 5-seed formal | Cora 0.748/CiteSeer 0.571，vs 三对照全 5/5 正 | C1 机制验证通过 |
| 2 | 06-10 | C1_gated（coverage+confidence gate） | 最佳 0.714 < C1 0.744 | gate 判负 |
| 3 | 06-10 | C1_weighted（可靠性加权 CE） | 0.7433 ≈ C1 0.7443 | 打平，无增量 |
| 4 | 06-10 | NoisyGL 基线摸底 seed0 | GCN+CE Cora 0.810 > C1 0.744 | 框架级问题暴露 |
| 5 | 06-10 | frozen-GRACE 球 + 端到端 GCN（hard） | Cora -1.5pp / CiteSeer +5.0pp，CDR 22%/30% | 有信号，CDR 是病根 |
| 6 | 06-10 | gated hard（min_train=3, tau=0.7） | Cora +0.4pp 但 CiteSeer 崩回 0.631 | 二值 gate 判负 |
| 7 | 06-10 | soft v1（binary alpha） | mean alpha 0.17/0.07，≈不校正 | alpha 公式错 |
| 8 | 06-10 | **SoftDist**（multiclass alpha + 分布 target） | seed0 双赢 GCN+CE（+1.3/+4.3pp） | 主线候选 |
| 9 | 06-10 | SoftDist 5-seed + NRGNN 对照（ROCm） | 0.795/0.674 稳定赢 GCN+CE，输 NRGNN 1.5/2.9pp | 主线确认，差距明确 |
| 10 | 06-10 | 动态 alpha（agree×1.5/disagree×0.5） | 保守版 no-op，fast 版反伤 | 判负：自强化 |
| 11 | 06-10 | ball consistency 正则（λ=0.1） | 完全无增量 | 判负 |
| 12 | 06-10 | NRGNN + BallAux（λ=0.01/0.05/0.1） | 最大 +0.22pp，Cora shuffled 反超 real | 判负：与图传播冗余 |
| 13 | 06-10 | E1 自适应纯度-覆盖率建球（3 变体） | 无变体双赢几何球，CDR 不降反升 | 判负：20% 覆盖票密度不足 |
| 14 | 06-10 | Phase 1 [X\|\|H] seed0 | gcn_xh_ce 0.811/0.689 | 过门 |
| 15 | 06-10 | Phase 2 无标签扩展 precision 诊断 | precision 0.875/0.822，过门 | 进训练 |
| 16 | 06-10 | Phase 3 无标签扩展训练（λ=0.05/0.1） | 全部低于 SoftDist static | 判负：球级 target 无逐节点分辨率 |
| 17 | 06-11 | Phase 5 [X\|\|H] 5-seed | **softdist_xh Cora 0.8137±0.0067 ≥ NRGNN** | 主线最终确认 |
| 18 | 06-11 | Phase 4a early-learning 裁决诊断（XH warmup） | 三门全不过 | 不训练 |
| 19 | 06-11 | Phase 4a-2 同诊断（X-only warmup） | 同样不过，排除 H 耦合解释 | Phase 4 永久关闭 |
| 20 | 06-11 | Phase 6 noise grid（5 regime × 5 法 × 5 seed） | 无 regime win；Cora 4 tie / CiteSeer 5 loss | 适用边界确定，收官 |

---

# 3. 逐阶段详细记录

## 阶段一：C1 ratio-split 验证（实验 1）

**背景**：Planetoid 标准 split 仅 5.2% 标签覆盖，62% 可靠球内 0 个 train 节点，多数投票天然缺票。改用 stratified ratio split（0.2/0.1/0.7）后机制恢复。

**正式结果（5 seeds, symm 0.4, 只污染 train）**：

| 数据集 | C1 | C1_noball | C1_shuffled | C1_topk |
|---|---:|---:|---:|---:|
| Cora | **0.748±0.018** | 0.555±0.027 | 0.594±0.031 | 0.549±0.018 |
| CiteSeer | **0.571±0.026** | 0.424±0.011 | 0.447±0.018 | 0.411±0.019 |

paired delta 全部 5/5 正，min_delta > 9pp。可靠球平均 train 票 Cora 2.8 / CiteSeer 2.6。

**Coverage sensitivity（Cora seed0）**：train_ratio 0.1/0.2/0.6 → C1 0.634/0.744/0.814，增量 +12.7/+17.4/+19.9pp，单调增。

**结论**：球机制真实有效，依赖 local labeled coverage。这组数据后来成为论文的 coverage sensitivity 素材。

## 阶段二：C1 框架内改进（实验 2-3，全部无效）

- **C1_gated**：球内 train ≥ τ_cov 且 margin ≥ τ_conf 才做 soft correction。最佳 0.7143 < C1 0.7443。原因：C1 的 hard majority repair 本身就是有效成分，gating 削弱它反而伤。
- **C1_weighted**：保留 hard repair，按 `w_B = max(floor, min(1, n_train/k_ref)·margin)` 加权 train loss。0.7433 ≈ 0.7443，因为 ratio 0.2 下多数球的 w_B 本来就接近 1，降权的球太少。

**教训**：在弱框架上做精细化是徒劳。此时尚未意识到框架本身是瓶颈。

## 阶段三：基线摸底与框架迁移（实验 4-5，本轮最大转折）

**NoisyGL seed0 基线（percent split 0.2/0.1/0.7, uniform 0.4）**：

| 方法 | Cora | CiteSeer |
|---|---:|---:|
| NRGNN | 0.8359 | 0.7006 |
| GCN+CE | 0.8100 | 0.6327 |
| SCE | 0.7921 | 0.5945 |
| RTGNN | 0.7855 | 0.6645 |
| PIGNN | 0.7847 | 0.6409 |
| CLNode | 0.7451 | 0.5279 |
| C1 (ours) | 0.7443 | 0.5713 |

**关键诊断**：C1 输给最朴素的 GCN+CE。但 GRACE frozen linear_eval = 0.8366 > GCN+CE 0.8100，且 C1_noball 微调后从 linear 0.8039 掉到 classifier 0.5704——**问题不是 GRACE 表示弱，是 Phase 2 低 lr 微调 + noisy CE 把表示用坏了**。球的 +19pp 是在修复 Phase 2 自己造成的损害。

**框架迁移**：GRACE 只负责建球和标签预处理（frozen），分类器换成全新 NoisyGL 端到端 GCN。hard majority 校正 seed0：Cora 0.7947（-1.5pp）/ CiteSeer 0.6830（+5.0pp），shuffled 0.3953/0.3222（差距巨大，球结构信号极强）。诊断：NCR 0.688/0.591，**CDR 0.224/0.298**——纠错能力真实，但误伤干净标签是病根。

## 阶段四：target 构造打磨（实验 6-9，收敛到 SoftDist）

三步迭代，每步修一个明确缺陷：

| 版本 | 缺陷 | Cora | CiteSeer |
|---|---|---:|---:|
| hard | CDR 22%/30% 太高 | 0.7947 | **0.6830** |
| gated hard（二值 gate） | 改 246→22 个，CiteSeer NCR 崩 | **0.8137** | 0.6314 |
| soft v1（binary alpha, 零点 0.5） | 多分类下 margin 多在 0.3-0.5，alpha≈0 | 0.8100 | 0.6379 |
| **SoftDist**（multiclass alpha + ball_dist） | — | **0.8232** | **0.6761** |

SoftDist 的两个修正：alpha 零点从 0.5 改为 1/C（mean alpha 从 0.17/0.07 恢复到 0.47/0.35）；target 从 one-hot majority 改为球内完整分布（保留不确定性）。**这是第一个两数据集同时赢 GCN+CE 的设置。**

**5-seed 确认（ROCm，注意与 CPU 数值不可混比）**：

| 方法 | Cora | CiteSeer |
|---|---:|---:|
| NRGNN | 0.8107±0.0230 | 0.7027±0.0108 |
| SoftDist | 0.7954±0.0094 | 0.6737±0.0113 |
| GCN+CE | 0.7842±0.0151 | 0.6530±0.0121 |
| SoftDist shuffled | 0.7832±0.0167 | 0.6578±0.0116 |

**教训**：gate 是"全有/全无"的粗糙工具，两个数据集需要相反松紧，不存在 universal 阈值；连续插值（alpha）+ 保留分布信息（ball_dist）+ 多分类校准（1/C 零点）才是对的形式。

## 阶段五：动态机制排除（实验 10-12，全部判负）

- **动态 alpha**（GCN 预测同意球投票则 alpha×1.5，否则×0.5）：warmup=20 版 no-op（best epoch 在 3-6，刷新根本没生效——本协议下 GCN 记忆噪声极早）；fast 版（每 epoch 刷新）反伤 Cora。**自强化**：模型一旦同意了可能错误的球多数票，就会进一步强化对它的信任。
- **ball consistency 正则**（球内预测方差惩罚，λ=0.1）：完全无增量（0.8032/0.6748 vs static 0.8042/0.6748）。
- **NRGNN + BallAux**（在 NRGNN loss 上加 λ·CE_soft(logits, ball_target)）：最大增量 +0.22pp（CiteSeer λ=0.1），Cora 大 λ 下 shuffled 反超 real。**球的标签校正信号和 NRGNN 的在线图传播高度冗余**——因为 GRACE H 本身从图结构训练而来，球编码的主要还是图同质性。

**教训**：frozen 一次性球能提供的信息在 SoftDist 处已经饱和；模型自身预测不能作为反馈信号源（自强化）；和图传播方法杂交不可行（信号冗余）。

## 阶段六：E1 自适应纯度-覆盖率建球（实验 13，判负但有论文价值）

**假设**：几何球横跨类边界是 CDR 的病根；标签感知的自适应粒度（混类球分细、干净区域保粗）应降 CDR。预注册约束：分裂只用 noisy train 标签（noisy-supervised ball construction，明确不是无监督 GB）；E1 只动 train target；诊断必报 noisy purity vs true purity。

**三个验收规则全部失败**：

| 变体 | 球数 (Cora/CiteSeer) | Cora | CiteSeer |
|---|---|---:|---:|
| SoftDist 几何（锚点） | 52/64 | 0.8042 | **0.6748** |
| v1 阈值分裂（过细） | 59/139 | 0.8005 | 0.6770 |
| v2 严格增益（过粗） | 6/4 | **0.8158** | 0.6705 |
| v3 多尝试增益（中粗） | 9/6 | 0.7995 | 0.6740 |

CDR 没有任何变体下降（粗变体反升至 0.31-0.45）；v3 Cora shuffled (0.8063) 反超 real。

**失败的统计物理**：20% 覆盖 + 40% 噪声下，noisy purity 天花板 ~0.66，每球 5-15 票时纯度估计 std ±0.15，诊断证实 noisy purity 全程比 true purity 低 0.2-0.3。**标签信号太弱，无法驱动粒度决策**：阈值规则退化为"分到守卫挡住"（CiteSeer NCR 0.59→0.36），增益规则拒绝几乎一切分裂。粒度本身比标签感知更重要，几何 max_balls=64 已近实践最优。

**论文价值**：这个负结果精确解释了为什么 CGBC 式纯度驱动分裂不能搬到图半监督场景——它们的数据集 50-94% 全覆盖，纯度估计天然可靠。

## 阶段七：新信息源（实验 14-19，一胜两负）

E1 后的反思：之前所有失败都在反复加工同一份信息（球内 train 票）。转向三个未用过的信息源。

### 杠杆 2 = [X||H]（成功，成为主线）

GRACE H 之前只用于建球就扔了，分类器只吃 X。改成 GCN 输入 `[X || L2归一化(H)]`：

**Phase 5 五种子确认**：

| 方法 | Cora | CiteSeer |
|---|---:|---:|
| **SoftDist [X\|\|H]** | **0.8137±0.0067** | 0.6891±0.0159 |
| NRGNN | 0.8107±0.0230 | **0.7027±0.0108** |
| GCN+CE [X\|\|H] | 0.8027±0.0069 | 0.6761±0.0181 |
| SoftDist [X\|\|H] shuffled | 0.7921±0.0097 | 0.6696±0.0234 |

paired delta：H-on-CE +1.86pp(5/5)/+2.31pp(4/5)；ball-on-H +1.10pp(4/5)/+1.30pp(5/5)；real−shuffled +2.16/+1.95pp（双 5/5）。**Cora 均值超 NRGNN 且 std 小 3.4 倍。**

**重要教训：Cora seed0 是误导**——seed0 显示 ball-on-H 为 -0.5pp，5-seed 实际 +1.10pp(4/5)。本轮第二次 seed0 误导（第一次是 gated hard 的 Cora +0.4pp）。

### 杠杆 1 = 无标签扩展（判负，结构性原因）

precision 诊断过门（gate min_train=10/margin=0.5：precision 0.875/0.822，coverage 达标，real≫shuffled）。但训练判负：λ=0.05/0.1 全部低于 SoftDist static（0.776-0.784 / 0.668-0.674）。

**失败机理**：球级 target 给同球 ~50 个节点完全相同的分布；模型自身对无标签节点的预测已有 ~81% 准确率**且逐节点分辨率**。用 87% 准确但球内恒定的信号替换它，拿微弱精度优势换掉了分辨率，净亏。与 ball consistency 同病根：**球粒度太粗，不适合做无标签节点的监督信号**。precision 高 ≠ 可用。

### 杠杆 3 = early-learning 双源裁决（判负，两次诊断互为 robustness check）

假设：warmup 期 GCN 置信度（early-learning 现象）与球投票是独立信号，可裁决"球想翻但早期模型支持原标签"的节点以降 CDR。预注册门槛：CDR 相对降 ≥30%（双数据集）、NCR 损失 ≤10%、保护精度 ≥70%。

- **4a（XH warmup）**：无 tau 过三门。Cora 保护节点太少（CDR 降幅最高 25%），CiteSeer 保护精度 48-68%。
- **4a-2（X-only warmup，排除 H 耦合解释）**：结果几乎相同（Cora 26% / CiteSeer 46-69%）。

**结论**：early-learning 置信度无论与球几何耦合还是独立，都不足以安全保护干净标签。本协议下 GCN best epoch 在 3-6，记忆噪声太早，早期信号的窗口本身就很脏。Phase 4 永久关闭，未进入训练（诊断先行省掉了一轮无效训练）。

## 阶段八：noise grid 适用边界（实验 20，收官）

5 regime × 5 方法 × 2 数据集 × 5 seeds。全部 regime 复用同一份 H（label-free，表示阶段对标签噪声构造性免疫——这是与 NRGNN 的本质区别，也是论文卖点）。

**胜负矩阵（ours vs NRGNN，win = mean>0 且 ≥4/5 seeds 正）**：

| Regime | Cora | CiteSeer |
|---|---|---|
| uniform 0.2 | tie (+1.05pp, 3/5) | loss (-0.81pp) |
| uniform 0.4 | tie (+0.31pp, 3/5) | loss (-1.36pp) |
| uniform 0.6 | **loss (-11.20pp)** | loss (-7.12pp) |
| pair 0.2 | tie (-0.32pp) | loss (-3.56pp) |
| pair 0.4 | tie (+0.52pp, 3/5) | loss (-4.23pp) |

**三个发现**：
1. 格局是**数据集分型**不是噪声分型：Cora 全线 tie（且方差普遍更小），CiteSeer 全线 loss。
2. 球增量价值区间是中高噪声：real−shuffled 在 uniform 0.4（+2.2/+2.0pp）和 pair 0.4（+3.6/+2.0pp）最强；uniform 0.2 下 ≈0（标签可靠时球冗余）。"噪声越高球增量越大"仅部分成立（uniform CiteSeer 单调 +0.5→+1.3→+2.2pp，Cora 在 0.6 不再增）。
3. **uniform 0.6 是 offline 校正管线的崩溃点**：Cora real 甚至输 shuffled（0.651 vs 0.687），NRGNN 的 online 机制在此 regime 明确更强。

---

## 阶段九：收官后的穷尽性验证（实验 21-40，2026-06-12 至 06-15）

noise grid 后曾判定"机制收官"，但后续应用户要求，沿四个新轴做了穷尽性验证：文献甄别、逐节点分辨率、SSL 升级、数据集/backbone 扩展、投稿缺口补全、定制 SSL。**全部未改变主方法**，但产出多个论文级发现与诚实边界。

### 9.1 文献甄别（13 篇 GB+DL 论文，3 并行 agent）

12 篇为已判死方向换皮或域不匹配（GAdaBoost/GB-RVFL 的 purity 加权 = SoftDist 已做；GBExplainer/MOGB/SGBGC 的标签驱动分裂 = E1；PGBSF/EvoSampling/GBGM/GBRIP/GBCM-GCN 域不匹配）。唯一幸存 MGNR (TPAMI) 的多粒度一致性 → 进逐节点诊断。

### 9.2 多粒度一致性裁决（实验 21，判负）

fine256/mid64/coarse16 三个 label-blind 几何粒度，跨粒度一致才校正。预注册门槛全不过：**保护精度仅 0.52 < 0.70**。跨尺度分歧标记的是"投票含糊区"，不是"干净标签边界"——前提证伪。

### 9.3 逐节点分辨率（实验 22，判负 + 论文论点）

UnionNET/TSS 启发，把球级恒定 target 改为逐节点：相似度加权投票（A）+ 球心距离调制 alpha（B）。门槛全不过——**CDR 与 NCR 同步崩塌**。机理：球内仅 5-10 票，软加权无分辨率。**三类 gate 信号（票数/时间/尺度）+ 逐节点分辨率四维全否**，共同证明"球校正候选集内干净 vs 噪声不可分"，是 §7 核心论点。distance_bins.csv 量化了 CDR 的空间分布（首次直接测量），是论文素材。

### 9.4 NoisyGL baseline 补全（实验 23，全胜）

补跑 5 个从未跑过的方法 × 5 seeds：

| 方法 | Cora | CiteSeer |
|---|---:|---:|
| UnionNET（最近邻机制） | 0.7837±0.0161 | 0.6867±0.0136 |
| CRGNN | 0.7690±0.0101 | 0.6476±0.0188 |
| CGNN | 0.7389±0.0380 | 0.6135±0.0322 |
| TSS | 0.6385±0.0250 | OOM 失败 |
| R2LP | 0.5745±0.0266 | 0.4653±0.0574 |
| **SoftDist [X\|\|H]** | **0.8137** | **0.6891** |

**全部输给 SoftDist**，包括机制最接近的 UnionNET（双数据集）。主表更稳。

### 9.5 SSL encoder 升级（实验 24-26，判负 + 重要论文发现 C3）

- Phase 8 probe：CCA-SSG 双数据集 probe 超 GRACE（+2.7/+1.6pp），GraphMAE Cora-only，BGRL/可学习τ 全负
- Phase 9 下游：**CCA-SSG/GraphMAE 的 H 在球构造时塌缩成 1 个球**（512 维，几何不可分），real-shuffled 增益归零，CiteSeer 下游反不如 GRACE
- **发现 C3：probe 可分性 ≠ 下游可聚类性**。GRACE 的 InfoNCE uniformity 让表示在超球面均匀展开、天然可分球；这是 SoftDist 对 SSL 的隐含要求。GRACE 保持主线 H。

### 9.6 投稿缺口补全（实验 27-35，Phase 15-19）

| Phase | 内容 | 结果 |
|---|---|---|
| 15 instance 噪声 | instance {0.2,0.4} | 0.4 球贡献真实（Cora +1.7/5/5），0.2 弱；CiteSeer 仍输 NRGNN。强化 regime claim |
| 16 覆盖率曲线（当前管线） | train_ratio {0.05-0.4} | **非单调！** Cora 0.05 +1.5pp → 0.40 -0.99pp。[X\|\|H] 吸收了部分球优势，**修正旧 C1 单调结论** |
| 17 效率表 | wall/params | SoftDist 7.9s/100k/非迭代 vs NRGNN 24.7s/465k/迭代（3.1× 时间、4.6× 参数）；UnionNET >1200s 超时 |
| 18 oracle gap-closure | clean 上界 | Cora 37.8% / CiteSeer 47.4% / PubMed -32.6%（边界）|
| 19 Amazon-Photo（跨域）| 共购图 | 非第三成功（real≈shuffled、SoftDist≈GCN），但赢 NRGNN +0.97pp；球未塌缩（purity 0.86）→ homophily/饱和边界，非几何失败 |

### 9.7 定制 SSL：GRACE + uniformity 正则（实验 36，判负 + 强化 C3）

label-free 显式 uniformity 正则（λ∈{0.1,0.5,1.0}）。Phase 20a 几何诊断**硬判负**：uniformity 指标确实变强（-1.0→-3.6），但**球塌缩到 1**（purity 0.79→0.30）。Cora λ=0.1 probe 甚至略超 GRACE 却仍塌缩——**第三次确认 C3**。关键论点：GRACE 的价值不是 uniformity 最大化，而是 **alignment/uniformity/局部可分性的平衡**，对递归 2-means 球分裂器局部最优。"为什么是 GRACE"从工程选择上升为有原理的分析。

### 9.8 阶段九总结

四个新轴（文献、逐节点、SSL、扩展）全部 resolved，主方法零改动。净增论文资产：UnionNET 等 5 baseline 全胜、instance 噪声谱、效率表硬数字、oracle gap-closure、C3 三重确认（换 SSL × 显式 uniformity × probe≠clusterability）、四维不可分性证明、跨域/覆盖率诚实边界。P1/P2 收尾三项（Phase 21-23）已于 06-15 完成：球数敏感性证实 64 在稳定平台（Cora 16-128 间仅 0.05pp 波动、真实球全面赢 shuffled）；CDR 距离分桶图证实误伤随到球心距离单调上升（Cora 0.0000→0.0153）；TSS CiteSeer CPU 补跑成功（0.5055±0.0510，主表补全）；t-SNE 图已出。

## 阶段十：公平性对照与 2025 外部对比（实验 41-44，2026-06-16，重塑核心 claim）

应导师要求补外部对比。**Phase 24（NRGNN + [X\|\|H] 公平性对照）是整个项目最有价值的一次诚实校准**：

| 数据集 | NRGNN | NRGNN [X\|\|H] | SoftDist [X\|\|H] | SoftDist − NRGNN[X\|\|H] |
|---|---:|---:|---:|---:|
| Cora | 0.8107 | **0.8139** | 0.8137 | −0.02pp（打平，3/5）|
| CiteSeer | 0.7027 | **0.7090** | 0.6891 | **−1.99pp（输，1/5）**|

**结论：一旦让 NRGNN 也用 H，SoftDist 不再领先。** 之前"持平/超过 NRGNN"的表观优势很大部分是 H 的功劳，而 H 任何方法都能用。**主 claim 必须降级，不能再写 "matches/beats NRGNN"。**

**但粒球的受控增量完好且更干净**（控制了 H 变量）：SoftDist − GCN+CE[X\|\|H] = +1.10/+1.30pp（4/5、5/5）；SoftDist − shuffled = +2.16/+1.95pp（双 5/5）。**给定相同 H，球软目标稳定优于纯 CE 和随机球**——这是控制变量后唯一防得住的性能 claim。

- **Phase 25 TFR (CIKM 2025)**：同协议跑通（官方 DBLP 模板迁移到 Cora/CiteSeer，非作者调优）。Cora 0.7758（我们赢 3.8pp）、CiteSeer 0.6979（我们赢 0.88pp）。弱证据，进表需标 caveat。
- **Phase 26 ProCon (IJCAI 2025)**：无官方代码 → 仅 related work，不进主表（未自实现冒充 baseline，处理正确）。
- **Phase 27 GD²**：未跑，可选 related work 引用。

**论文定位最终校准（诚实框架）**：丢弃"性能 SOTA"，改为四条都防得住的组合——(1) 简洁性（非迭代、参数 1/4.6、墙钟 1/3.1、最低方差）；(2) **受控粒球增量**（同 H 下球优于 CE 和随机球，控制变量 claim）；(3) 表示噪声免疫 + 可复用；(4) 丰富机制边界分析。贡献定性从"性能 SOTA"转为"概念 + 效率 + 分析"，适合 KBS/Neural Networks/CIKM。CiteSeer 差距 = NRGNN 在线机制在 H 均衡后确有真实优势。论文初稿摘要/C2/§3/§4 已全部改为此框架。

## 阶段十一：数据集广度反转 + 适用域刻画（实验 45-55，2026-06-16 至 06-18）

补外部数据集时一度判定"球只适用小图"，但坚持"诊断真变量"后发生戏剧性反转，最终把适用边界从模糊观察变成有网格证据的刻画。这是本项目最曲折也最长见识的一段。

### 11.1 大图三连败 → 误判"只适用小图"（实验 45-48）

- PubMed(19.7k,3类)：SoftDist 0.832 输 GCN+CE 0.839（但赢 NRGNN +2.4pp）——边界
- Amazon-Photo(7.6k,8类)：real≈shuffled，球贡献消失——边界
- DBLP(17.7k,4类)：默认 num_balls=1 塌缩。**Phase 32 两层诊断**：调参(max_balls=256/split_improvement=0.001)后球纯度恢复到 0.84，但下游仍 real≈shuffled——"构造可修复，下游被吸收"
- **GBSelect（球+GCN 双视角选干净样本，实验 47-48）**：离线降噪 40%→6% 看着极强,但下游三个数据集三种死法——DBLP/PubMed 命门(vs gcn_only)不过；Amazon 命门过(+3.1pp 5/5)却输全样本 -12pp。**命门对照(gbselect vs gcn_only)拦下了一个假阳性**——若只看"双视角>单视角"会误判成功，加了"vs 全样本"才暴露 selection 范式丢监督的致命伤
- 一度结论："球的两种用法(soft-target + selection)在大图都失败 → 只适用小图"

### 11.2 关键反转：类数才是变量，不是规模（实验 49-51，Phase 35）

**用户坚持"不要太早下定论，想清楚为什么大图失败"——这个坚持救了整条线。** 归因发现：失败的大图(DBLP4类/PubMed3类)共同点其实是**类少**，不是大。现有 5 数据集全挤在对角线（大图恰好都类少），size 与 class-count **完全混淆**。

判别实验：**Coauthor-CS(18.3k,15类) 与 DBLP(17.7k,4类) 同规模只差类数**。
- **CS 球有效**：SoftDist−GCN[X‖H] +0.53pp 5/5、−shuffled +0.55pp 5/5；GBSelect 命门也过
- **推翻"只适用小图"**：类数(非规模)是关键调节变量，大图大门重开，成功数据集 2→3
- Amazon-Computers(13.7k,10类) 精化：real>shuffled +1.8pp 5/5 但输全样本 -0.70pp → 类数充足是必要非充分，强同质+强基线仍吸收

### 11.3 真变量是球可聚类性（实验 52-53，Phase 37）

9 数据集 × 5 候选变量(规模/类数/同质性/noise-harm/正交AUC)网格。**球增量 vs 球纯度/正交AUC = Pearson 0.97-0.99**，远超规模(-0.60)、类数(-0.38)、同质性(0.51)。WikiCS(正交AUC 0.528≈随机) 球处处负增量，干净反例。**结论：决定球有没有用的是 ball-clusterability，规模/类数只是代理变量。**

### 11.4 可部署 proxy：方向对但样本不足（实验 54-55，Phase 37d→38）

- 11.3 的高相关变量是 clean-label 诊断量，**不能部署**（GPT 精确纠正：true_purity 用 clean、AUC 标签是"是否被污染"也需 clean）
- 转测只用 H+noisy 的可部署 proxy。**冻结 3 候选**(vote_margin/vote_entropy/ball_gcn_disagree)防 formula shopping
- 37d 时 ball_gcn_disagree Spearman=-1.00 但 n=5（假象）；**Phase 38 扩到 n=7-8 后塌缩到 -0.54**——证实小样本假象。最佳 vote_entropy -0.79(n=8)、margin +0.67(n=8)，方向对但未达预注册 n≥10 门槛
- **判定：adaptive layer 不评估，降级 future work。** 冻结 proxy 纪律拆穿假信号，反而强化论文诚信

### 11.5 LastFM-Asia：18 类仍失败，直接确认“类数不是充分条件”（实验 56，Phase 39）

为争取第 4 个成功数据集，选择 PyG 中未使用、规模适中且类别数高的 LastFM-Asia（7,624 节点、18 类）做诊断先行验证。PyG 原始 `graphmining.ai` NPZ 在 2026-06-18 已不可用，因此按 FEATHER 论文公开预处理族，从官方 SNAP 原始用户-artist 特征确定性重建 128 维输入：二值 incidence matrix → Scikit-Learn TruncatedSVD(128, randomized, n_iter=5, seed=0)。重建 NPZ SHA256 为 `819a0abebd88ac18c816d706b2d10175514dfaf452f0faef7018e1c9723ff2c2`；该输入与论文协议同类，但不声称与失效 NPZ 字节一致。

GRACE 用预注册的 PubMed fallback 配置完成 1500 epochs，导出 `[7624,256]` 的 label-free `H`。uniform 0.4、seeds 0/1/2 的 Phase 39a 门控结果：

| 指标 | mean ± std | 门槛 | 判定 |
|---|---:|---:|---|
| edge homophily | 0.8739 ± 0.0000 | >0.85 预警 | 高同质/吸收风险 |
| num_balls | 145.33 ± 1.15 | ≥8 | PASS，不塌缩 |
| true weighted purity | 0.2542 ± 0.0006 | ≥0.65 | **FAIL** |
| ball AUC | 0.6399 ± 0.0056 | 诊断 | 弱信号 |
| early GCN AUC | 0.9514 ± 0.0032 | 诊断 | GCN 已极强 |
| ball AUC in GCN-clean50 | 0.5995 ± 0.0498 | ≥0.70 | **FAIL** |

**预注册决策：停止 Phase 39b，不跑下游。** LastFM 的球数量充足但 country 标签纯度极低；artist-interest 几何与国家类别只弱对齐，同时高同质图上的早期 GCN 已吸收大部分可用信号。该结果直接否定“类数高就可能成功”的简化说法，并给 ball-clusterability 中介解释增加一个 18 类反例。成功数据集仍为 Cora/CiteSeer/Coauthor-CS 三个；LastFM 作为“多类但球语义不对齐”的边界点进入 §8。

pair 噪声只记录 future work：系统性 `A→B` 翻转可能在球内制造“伪多数”，使投票一致地带偏；候选解法是用 label-free `H` 几何一致性校验 noisy-label 多数并识别伪多数。本阶段未实现该机制，不能写成已验证结论。

完整证据：`external/NoisyGL/runs/lastfmasia_20260618/phase39_result.md`。

### 11.6b 噪声钟形曲线补全（0.5 噪声率，实验补充，2026-06-18）

补跑 uniform 0.5 + 三成功数据集 pair/instance（150 行），完成"球价值随噪声率"的完整曲线。Cora uniform 全谱球增量(SoftDist−GCN[X‖H])：0.2 −0.37pp(2/5) → 0.4 +1.10pp(4/5) → **0.5 +1.52pp(5/5，最强 uniform 点)** → 0.6 +0.66pp(2/5，real<shuffled)。**干净的倒 U 钟形**：低噪声有害（标签已可信，覆盖它是干扰）、中高噪声(0.4-0.5)峰值（个体不可信，集体共识相对价值最大）、极高噪声(0.6)崩溃（球内多数票本身被噪声系统性带翻）。0.5 是预声明的过渡诊断点，不是新调参点；pair 0.6 正确排除（指定错误类在 >0.5 时成为多数）。

**pair 噪声弱势跨数据集系统确认**：pair 0.2 三数据集处处不稳；pair 0.4 仅 Cora 保住增量，CiteSeer/Coauthor-CS "无稳定增量"。机理：pair 系统性 A→B 翻转在球内造"伪多数"，集体投票被一致带偏（uniform 随机错误能平均掉、pair 系统错误压不过）。future work 解法：用 label-free 的 H 几何校验投票、识别伪多数（需新预注册诊断，未实现）。完整证据：`external/NoisyGL/runs/noise_breadth_20260618/noise_robustness_report.md`。

### 11.6 阶段十一总结

成功数据集 2→3（Cora/CiteSeer/Coauthor-CS，含 18k 大图），消灭"只在小图验证"质疑。LastFM-Asia 又补充了一个 18 类边界点：类数高仍可能因球语义纯度不足和强 GCN 吸收而失败。适用边界从模糊观察升级为：**球增量由 ball-clusterability 中介（诊断量强相关证据），规模/类数只是代理；可部署 proxy 方向正确但 Phase 38 的 n≤8 未达门槛，adaptive 留 future work**。方法论收获：(1) 命门对照拦假阳性（GBSelect Amazon）；(2) 坚持诊断真变量避免早结论（size→class→clusterability 三级纠正）；(3) 冻结候选 + n 门槛拆穿小样本完美相关；(4) 诊断门控在 LastFM 上阻止了无依据的下游扩表。

## 阶段十二：SoftDist-MLP 机制变体线（2026-06-18，归因问题的最终解决）

针对"SoftDist 增量到底来自球、还是来自 [X‖H] 表示、还是来自 GCN 图传播"这个最大归因软肋，主动移除后两者（分类器不输入 H、改普通 MLP 不做图传播），隔离球软目标的独立价值。

**核心证据——球增量随下游强度变化**（这是比单提 MLP 更有力的规律）：

| 下游 | 输入 H | 图传播 | 球增量 (Cora/CiteSeer) |
|---|---|---|---|
| GCN [X‖H] | 是 | 是 | +1.10/+1.30pp（被强表示吸收） |
| GCN(X) | 否 | 是 | +1.13/+2.07pp（5 月已跑） |
| **MLP(X)** | 否 | 否 | **+6~+9pp** |

下游越缺 H 与图传播，球边际贡献越明显。**诚实边界（GPT 纠正）**：这是与结果一致的解释，非受控因果规律（MLP/GCN/特征/优化同时变）；不能说"单调规律"（增量不随噪声率单调）。

**MLP 噪声谱（180 行，6 条件 6/6 稳定）**：clean(安全,+0.4/+2.1pp)、uniform 0.2/0.4/0.5/0.6、pair 0.4、instance 0.4 全部稳定正（vs CE 且 vs shuffled，多数 5/5）。意外：+8pp 不是 uniform 0.4 甜点区,而是已测谱内普遍有效；uniform 0.6 在 MLP 上仍 +6.8pp（与 GCN 上 0.6 崩溃相反——MLP+CE 基线已低到有纠正空间）。claim 限"可迁移到 X-only MLP"，**保留 GAT/APPNP CiteSeer 负结果**，不写"模型无关"。

**效率（下游，GRACE 单列）**：SoftDist-MLP 92k参数/5.68s/140MiB vs NRGNN 465k/29.5s/406MiB。**严格表述**（GPT 纠正）：分类器训练更快+推理无图传播+显存低于GCN/NRGNN；但含球目标构建（~2.5-5.7s）的下游总时间不占优（5.68 vs SoftDist-GCN 4.86s），参数不低于 X-only GCN。不写"下游轻量/端到端碾压"。

**X-ball 完全无 H（负结果，干净判负）**：固定 raw-X 几何建球在 Cora/CiteSeer 均退化单球（纯度 0.30/0.21、real==shuffled），未过门、未跑下游。**诚实边界（GPT 最大纠正点）**：只证明"固定几何+预注册参数下 raw-X 退化单球"，**未**证明 H 理论必需/所有 X-space 聚类失败/topology-ball 必败/无 H 下游失败（下游根本没跑）。

**核心定调句（论文采用）**：SoftDist 的作用不依赖将 H 输入分类器，也不依赖 GCN 消息传播；但当前有效的软目标仍依赖由冻结 H 构造的粒球，固定 raw-X 几何尚不能提供可用替代。

**最终双层定位**：主性能 = SoftDist-GCN[X‖H]；机制/轻量变体 = SoftDist-MLP(X)；X-ball 记负结果；topology-ball 低优先级独立预注册。**实验至此真正收口，进入正文写作。**

## 阶段十三：Posterior-Calibrated SoftDist 诊断门控（2026-06-22，STOP）

受 SCGNN“语义确定域—不确定域—拒绝预测”思想启发，本阶段只检验一个窄问题：现有 SoftDist 是否因忽略球内训练票数而把低支持率球误当作高置信证据。未移植 SCGNN 的标签纯度分球、Anchor 图或在线 LCC。

### 13.1 研究假设与粒球角色

- **研究假设**：在相同 noisy-label margin 下，只有 2–3 个训练票的球应比 8+ 票球表现出更高 CDR 或更差 clean-mass delta；若该差异跨 seed、跨数据集稳定，则 Jeffreys/Dirichlet 后验收缩有实现依据。
- **粒球角色**：冻结 H 上一次性构造的局部语义证据单元；仅训练节点投票，不扩展无标签伪监督，不接入模型预测反馈。
- **方法动机**：当前 `alpha` 只由类别比例决定，无法区分“2:0”与“20 个票形成的多数”；Posterior-SoftDist 原计划通过每类 `+0.5` 伪计数降低低票数球的虚假置信度。
- **预期机制**：如果低票数确实对应不可靠证据，收缩应优先降低这些球的校正强度，在保留高支持率球 NCR 的同时减少 CDR。

### 13.2 协议、代码与预注册标准

- 数据集：Cora、CiteSeer、Coauthor-CS。
- 协议：`uniform 0.4`，percent split `0.2/0.1/0.7`，seeds `0..4`，沿用各数据集现有 SoftDist 的 frozen H、`max_balls`、`min_ball_size`、`split_improvement`。
- 分箱：球内训练票数 `2–3`、`4–7`、`8+`；优先比较 `2–3` 与 `8+`。
- 诊断指标：CDR、NCR、clean-mass delta、alpha、真实多数准确率、覆盖节点数。
- **成功标准**：至少 2/3 数据集满足以下任一项，且方向在至少 4 个 paired seeds 为正：
  - `CDR(2–3) - CDR(8+) >= 0.05`；
  - `clean_mass(8+) - clean_mass(2–3) >= 0.02`。
- **失败标准**：未达到上述门槛则停止，不实现 Posterior-SoftDist、不运行下游训练。
- 新增代码：
  - `external/NoisyGL/utils/frozen_grace_targets.py::support_bin_diagnostics`
  - `external/NoisyGL/scripts/diagnose_posterior_softdist.py`
  - `external/NoisyGL/tests/test_posterior_softdist.py`
- 验证：相关回归测试 `30 passed`；新增诊断测试 `5 passed`。

### 13.3 结果

| 数据集 | 2–3 票箱出现 seeds | 低票箱平均覆盖训练节点 | CDR gap | CDR 正向 seeds | clean-mass gap | mass 正向 seeds | 判定 |
|---|---:|---:|---:|---:|---:|---:|---|
| Cora | 3/5 | 6.00 | +0.0866 | 2/3 | +0.0026 | 1/3 | FAIL |
| CiteSeer | 4/5 | 5.25 | +0.1745 | 3/4 | -0.0112 | 1/4 | FAIL |
| Coauthor-CS | 4/5 | 14.50 | +0.2131 | 3/4 | +0.0123 | 3/4 | FAIL |

关键观察：

1. 低票数球确实偶尔具有更高 CDR，但方向不稳定，没有任何数据集达到 `4` 个正向 paired seeds。
2. clean-mass gap 在三数据集均未达到 `0.02`；CiteSeer 方向甚至相反。
3. `2–3` 票箱只覆盖极少训练节点，并在 Cora 2/5、CiteSeer 1/5、Coauthor-CS 1/5 seeds 中完全缺失。即使局部 CDR 较高，潜在干预覆盖也过小且不稳定。

### 13.4 最终决策与 claim 边界

- **最终决策：STOP / 放弃。** 按预注册规则，不实现 Posterior-SoftDist、Posterior shuffled 或 DirectDist，也不运行下游主对照。
- **已被实验支持的结论**：在当前 frozen-H、20% 标签覆盖和 uniform 0.4 协议下，低票数不是稳定、可部署的粒球可靠性信号。
- **合理解释**：低票数箱样本极少，CDR 比率方差大；同时 clean-mass 未显示一致恶化，因此“高比例不等于高证据”在统计直觉上成立，但没有转化成足以支持新模块的经验信号。
- **尚未验证的假设**：其他先验强度、连续支持率函数或更多数据集可能产生不同结果；本阶段没有证明所有贝叶斯收缩形式理论上无效。
- **被负结果削弱的方向**：把 SCGNN 的确定域/不确定域思想直接落为“按球内票数拒绝或收缩”的主创新。没有新的独立证据前不要复活。
- 完整证据：`external/NoisyGL/runs/posterior_softdist_20260622/`。

## 阶段十四：2025 最新基线补全与 NoisyGL 协议修正（2026-06-22，IMPLEMENTED / FORMAL RUN PENDING）

### 14.1 协议审计发现

- 旧 `run_standard_downstream.py` 调用 `label_process(labels=data.labels, ...)` 时未传 split mask，实际对全图标签加噪。
- 多数 predictor 又用 `self.noisy_label[self.val_mask]` early-stop，因此历史 NoisyGL、TFR 和 XH 对照是“全图污染 + noisy validation”，并非文档声称的 train-only noise。
- 该缺陷只影响由这些 NoisyGL 入口生成的结果；独立 C1 ratio 管线已有 train-mask 防泄漏实现，不能不加区分地一并作废。
- **决策**：受影响的历史数字全部降为 legacy，不进入修正后的正式主表，不与新结果混算 mean/std。

### 14.2 实现内容

- `utils/labelnoise.py`：`label_process` 增加 `eligible_mask`，新正式入口只污染 train 节点。
- `scripts/run_standard_downstream.py` 与 `scripts/run_ssl_swap_downstream.py`：强制显式 `--noise_scope`；CSV 记录 split/noise hash、train/val/test 修改数、config ID、wall time 和可训练参数量。
- TFR：沿用作者官方仓库 <https://github.com/eaglelab-zju/TFR> 的 decoder、feature reconstruction 和高置信伪标签机制，增加 Coauthor-CS 配置及 12 配置调参网格。
- ProCon：按 IJCAI 2025 论文复现 semantic peers、两跳结构加权、GMM clean posterior、prototype EMA 和 BGRL 风格自监督；明确标记 `paper reimplementation`。
- GD²：按 NeurIPS 2025 论文复现 ego/structure/mixed 三视图、低分歧 GMM clean selection、高置信伪标签净化和三项分类损失；明确标记 `paper reimplementation`。
- `scripts/run_latest_baseline_experiments.py`：固定三个数据集、12 配置上限、seed0 clean-validation 选参、冻结配置后 5 seeds、附录基线、协议哈希审计、mean/std 和 paired delta 汇总。

### 14.3 Smoke 结果与状态

- Cora：TFR、ProCon、GD² 均完成 2 epochs。
- Coauthor-CS：三方法均完成 1 epoch；ProCon 有界 peer 构建和 GD² 三视图未 OOM。
- 六个成功行均满足 `modified_val_count=0`、`modified_test_count=0`，同 dataset/seed 的 split/noise hash 一致。
- smoke 仅验证执行、内存和协议，不构成性能结论；2 epoch/1 epoch accuracy 不得进入论文。
- ROCm `2.9.1+rocm7.2.1` / RX 7700 XT smoke 也已完成，6/6 状态为 `ok` 且协议审计通过。
- 正式 108 次调参、135 次主表和 75 次附录运行尚未执行；应继续在同一 ROCm 环境完成，不能把 CPU smoke、ROCm smoke 或历史 legacy 数字混作正式比较。
- 产物目录：`external/NoisyGL/runs/latest_baselines_trainonly_20260622/`。

---

# 4. 大总结：方向判定表

## 4.1 已确认 OK 的方向（论文主线资产）

| 方向 | 证据强度 | 说明 |
|---|---|---|
| **SoftDist [X\|\|H] 主方法** | 5-seed 双数据集 + 完整归因链 | Cora 0.8137±0.0067 ≥ NRGNN，std 小 3.4 倍；非迭代单遍预处理，复杂度远低于 NRGNN |
| frozen GRACE 表示 + 端到端 GCN 框架 | 多轮一致 | GRACE 表示必须 frozen 使用；微调它是反模式 |
| SoftDist target 形式（multiclass alpha + ball_dist） | 4 版迭代收敛 | hard/gated/binary-soft 各有明确缺陷，SoftDist 是唯一双赢形式 |
| [X\|\|H] 特征拼接 | 5/5 与 4/5 seeds | H 对分类器有独立增量（+1.9/+2.3pp），与球增量可叠加 |
| H 的噪声免疫性 | 全 regime 复用同一 H | label-free 预训练 → 表示阶段不暴露于标签噪声，工程与叙事双重价值 |
| ratio split 协议 + coverage sensitivity | 5-seed | 球机制依赖 local labeled coverage；当前 [X\|\|H] 管线下增量**非单调**（低覆盖最强、高覆盖饱和转负），修正了旧 C1 的单调说法 |
| 球增量的噪声 regime 定位 | noise grid + instance | 中高噪声（0.4）最强，低噪声冗余——可写成适用条件 |
| 非迭代低复杂度 | 效率表（Phase 17）| SoftDist 7.9s/100k 参数 vs NRGNN 24.7s/465k（3.1×时间、4.6×参数），UnionNET 超时——简洁性有硬数字 |
| GRACE 不可替代性（C3）| probe×3 / SSL swap / uniformity 反例 | probe 可分性 ≠ 球可聚类性；GRACE 的 alignment-uniformity 平衡对 2-means 球分裂器局部最优。换 SSL 或显式增 uniformity 均使球塌缩 |

## 4.2 已判死的方向（每个都有预注册门槛 + 诊断数据，不要复活）

| 方向 | 判死证据 | 根本原因 |
|---|---|---|
| GRACE encoder 微调（C1 Phase 2 范式） | noball 微调后 0.804→0.570 | noisy CE 低 lr 微调破坏 frozen 表示的可分性 |
| 二值 gate（C1_gated / gated hard） | 两数据集需要相反松紧 | "全改/不改"无中间态，不存在 universal 阈值 |
| 可靠性加权 CE（C1_weighted） | 与 C1 打平 | ratio 0.2 下多数球权重本就≈1 |
| 动态 alpha / 模型预测反馈 | no-op 或反伤 | 自强化 confirmation bias；GCN 记忆噪声太早（best epoch 3-6） |
| ball consistency 正则 | 零增量 | 球粒度太粗，球内恒定约束无逐节点信息 |
| NRGNN 杂交（BallAux） | 最大 +0.22pp，shuffled 反超 | 球信号与图传播冗余（H 本身来自图结构） |
| 自适应纯度-覆盖率建球（E1，3 变体） | CDR 不降反升 | 20% 覆盖下票密度不足：纯度天花板 0.66、每球 5-15 票 std±0.15，标签信号驱动不了粒度 |
| 无标签扩展（球级伪监督） | precision 0.87 过门但训练全负 | 球级 target 无逐节点分辨率，输给模型自身 81% 的逐节点预测 |
| early-learning 裁决（XH 与 X-only warmup） | 双诊断三门全不过 | 早期信号不可靠：Cora 保护太少，CiteSeer 保护精度 <70% |
| 伪标签回流投票池（更早窗口） | precision 75.8% 不足 | 球会放大错误 |
| 纯度驱动分裂直接搬 CGBC | E1 + 文献对比 | CGBC 数据集 50-94% 全覆盖，纯度估计可靠性前提在图半监督下不成立 |
| 多粒度一致性裁决（MGNR 启发） | 保护精度 0.52 < 0.70 | 跨尺度分歧标记投票含糊区，非干净标签边界 |
| 逐节点分辨率（相似度加权/距离调制 alpha）| CDR 与 NCR 同步崩塌 | 球内仅 5-10 票，软加权无逐节点分辨率 |
| Posterior-SoftDist（按球内票数后验收缩） | 3 数据集 × 5 seeds 诊断门控全 FAIL；clean-mass gap 均未达 0.02 | 2–3 票箱覆盖极少且跨 seed 缺失，CDR 方向不稳定，票数不能作为可部署可靠性信号 |
| 换 SSL 方法（CCA-SSG/GraphMAE/BGRL/可学习τ）| 下游球塌缩或 probe 即负 | probe 高 ≠ 球可聚类；512 维表示几何不可分球 |
| 定制 SSL：显式 uniformity 正则 | 球塌缩到 1，purity 0.79→0.30 | uniformity 最大化 ≠ 球可分；GRACE 的几何平衡才是关键 |

## 4.3 仍开放但本轮不做的

- CiteSeer 与 NRGNN 的 1.4pp 差距：唯一框架内说得通的杠杆是提升 CiteSeer 的 GRACE 预训练质量（其 H 增量 +2.3pp > Cora +1.9pp，说明 H 质量是杠杆）——属于调参不是机制，放 future work。
- uniform 0.6 regime：需要 online 机制，超出 offline 预处理框架的能力边界，论文如实写。
- pair 噪声的“伪多数”问题：系统性 class flip 可能让球投票一致带偏；可探索 label-free `H` 几何校验多数，但必须重新走诊断→预注册→对照流程。

---

# 5. 方法论经验（比单个实验结论更值钱）

1. **seed0 会撒谎，且本轮撒了两次**。gated hard 的 Cora +0.4pp（5-seed 不成立）和 XH 的 ball-on-H -0.5pp（5-seed 实际 +1.1pp, 4/5）。结论：seed0 只配做 go/no-go smoke，任何写进决策的 delta 必须 5-seed paired。
2. **诊断先行 + 预注册门槛挡住了两轮无效训练**（early arbitration 两次诊断都没进训练），也暴露了一个反例：**precision 诊断过门但训练仍判负**（无标签扩展）——静态质量指标（precision/coverage）不能代理训练动态（分辨率竞争）。诊断能省钱，但不能替代最小训练验证。
3. **shuffled 对照是本轮最高产的单一工具**。E1 v3 的 Cora shuffled 反超 real、BallAux 大 λ 下 shuffled 反超、uniform 0.6 real 输 shuffled——三次都直接否掉了"看起来还行"的方向。任何球机制 claim 没有 shuffled 对照不可信。
4. **失败要追到机理而不是调参**。本轮每个判负都给出了机制解释（自强化/冗余/票密度/分辨率/记忆过早），这让"永久关闭清单"可信，防止了同一想法换皮复活（球级伪监督与 ball consistency 同病根，第二次直接识别）。
5. **协议坑要写死在文档里**：NoisyGL 必须显式 `--split_type percent --train_percent 0.2 --val_percent 0.1 --test_percent 0.7`，否则静默回退 Planetoid split（本轮烧过一次，E1 第一轮数据作废重跑）；ROCm 与 CPU 数值不可混比（GCN+CE Cora CPU 0.8100 vs ROCm 0.7974）；`gb:` config 键通过 `_conf_value` 静默覆盖 predictor 类属性，对照必须在诊断 CSV 里核实控制字段。
6. **信息源审计比机制创新优先**。阶段五的三连败后，把"管线里还有哪些没用过的信息"列出来（H 进分类器/无标签节点/时间维度early信号），立刻找到了 [X||H] 这个本轮最大单项增量——它不是新机制，只是把已有资产接上。
7. **框架瓶颈 > 机制瓶颈**。C1 阶段在弱框架上做了两轮精细化（gated/weighted）全部白费；基线摸底（实验 4）一次就暴露了真正的瓶颈。新方向应该先打基线再做机制。

---

# 6. 关键资产索引

## 代码（external/NoisyGL）

- `predictor/FrozenGraceGB_GCN_Predictor.py`——全部 frozengrace 方法族（hard/gated/soft/softdist/xh/expansion/adaptive 及 shuffled 对照）
- `utils/frozen_grace_targets.py`——target 构造全家：`apply_ball_soft_targets`（SoftDist 核心）、`build_ball_expansion_soft_targets`、`partition_purity_diagnostics`
- `utils/ordinary_gb.py`——几何建球 + `build_purity_adaptive_granular_balls`（E1，保留为分析工具）+ shuffled 对照
- `scripts/diagnose_ball_expansion.py`、`scripts/diagnose_early_arbitration.py`（含 `--warmup_features {xh,x}`）
- `tests/`——33 个单元测试全绿，含防泄漏测试（建球/扩展/裁决各有篡改 val/test 标签不变性测试）

## 报告（external/NoisyGL/runs/）

| 实验 | 报告 |
|---|---|
| 基线摸底 | `ratio_c1_baselines_20260610/` |
| hard/gated/soft/softdist seed0 | `frozengrace_gb_gcn{,_gated,_soft,_softdist}_ratio_c1_20260610/` |
| SoftDist 5-seed | `softdist_5seed_rocm_20260610/` |
| dynalpha / BC / BallAux | `softdist_dynalpha_seed0_rocm_20260610/`、`softdist_bc_seed0_rocm_20260610/`、`nrgnn_ballaux_seed0_rocm_20260610/` |
| E1 自适应建球 | `adaptive_gb_seed0_rocm_20260610/adaptive_gb_e1_seed0_rocm_report_20260610.md` |
| XH seed0 / 5-seed | `xh_seed0_rocm_20260610/`、`xh_5seed_rocm_20260611/` |
| 扩展诊断/训练 | `ball_expansion_diag_20260610/`、`ball_expansion_seed0_rocm_20260610/` |
| early 裁决 ×2 | `early_arbitration_diag_20260611/`、`early_arbitration_diag_xonly_20260611/` |
| noise grid | `noise_grid_rocm_20260611/`（aggregate/paired_deltas/win_matrix/ball_increment_trend 四个 CSV） |
| LastFM-Asia Phase 39 | `lastfmasia_20260618/phase39_result.md`、`lastfmasia_20260618/orthogonality/` |
| Posterior-SoftDist 支持票数门控 | `posterior_softdist_20260622/run_manifest.md`、`support_bin_report.md`、`support_bin_raw.csv`、`support_bin_aggregate.csv`、`support_bin_gate.csv` |

## C1 阶段（gb_denoise_stage_b_compactness / gb_c1_ratio_success_20260609）

- ratio formal 5-seed：`reports/ratio_formal_*_20260609.csv`、`results_ratio_formal_20260609/`（40 runs 完整工件）
- coverage sensitivity：`reports/ratio_c1_*_20260609.csv`
- C1_gated / C1_weighted：`gb_c1_ratio_success_20260609/results_phase_a/`、`results_phase_weighted/`

## 执行计划文档

- `external/NoisyGL/docs/execution_plan_xh_expansion_20260610.md`——Phase 1-6 全部计划、结果备案与永久关闭清单（含判死证据）

## 环境

- ROCm Python：`Desktop/三论文融合/三论文融合/GRACE-master/.venv-rocm/Scripts/python.exe`（AMD RX 7700 XT, cuda:0）
- frozen H：`gb_graph_aware_minival_6_1/checkpoints/embeddings/{Cora,CiteSeer}/ratio_train0p2_val0p1_split0_pretrain0/H.pt`
- C1 阶段 Python：`/c/Users/Admin/AppData/Local/Programs/Python/Python312/python.exe`（默认 python 是 Store stub）
