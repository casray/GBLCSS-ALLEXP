# 执行计划：写作前最后一道门 —— H-input 公平性控制 + 补全标签率 5-seed 曲线（Phase MLP-6）

日期：2026-07-01
执行者：Codex
定位：**MLP-5 主表已成（Cora/CiteSeer 全胜含最新方法，PubMed 边界，shuffled 15/15）。进入论文写作前，必须堵两个洞：(A) 审稿人第一刀——"你赢只是因为你用了 GRACE-H、baseline 没用"；(B) 低标签率曲线只有 0.05/0.65 两个 5-seed 端点，中间点只在 seed0，画不出带误差棒的曲线。** 两件都算力小，一次跑完即可写作。

## 0. 背景

- 已核对代码：β=0.5 主方法的 **MLP 分类器输入 = 原始 X**（`use_xh_features=False`），和所有 baseline 一样 → **输入特征这一层公平 ✓**。
- **但**：β=0.5 额外通过"球"用了 **GRACE-H（图结构派生表征）**来生成软标签目标；而 small-loss/JoCoR/FINE/CE 只有原始 X。**这是信息不对称。**
- shuffled 对照只证明了"真球 vs 随机球（两者都在 H 上）",**没有**证明"把 H 也给 baseline 当输入会怎样"。→ 缺 H-input 对照。
- 低标签率：MLP-5b 全范围 `{0.05,0.10,0.20,0.35,0.50,0.65}` 只在 seed0；MLP-5 只把 `{0.05,0.65}` 升到 5-seed。中间 4 点缺 5-seed。

## 0.1 红线

1. clean 只进评测；噪声只加 train；同一 X-only 2-layer 弱 MLP backbone、同 config（延续 MLP-1~5）。
2. paired（同 seed 同 split）+ mean±std over 5 seeds + 逐 seed 方向计数；多数据集一致性看 ≥2/3。
3. **Part A 给 baseline 的 H 用 [X‖H] 拼接**（复用现成 `concat_l2_normalized_embeddings` + row-L2-norm，与 `use_xh_features=True` 路径完全一致）。注意：**这比我们的方法更慷慨**——我们的分类器只吃 X（H 仅进球目标），而 baseline 的分类器直接吃到 H。**若 β=0.5(X-input) 仍赢 baseline([X‖H]-input)，是 a fortiori 强证据。**
4. Python：ROCm venv。percent split 显式传参。

## 0.2 复用与新增

- **复用（不重跑）**：MLP-5 主表里 uniform0.4 的 β=0.5、CE、small-loss、JoCoR（X-input）；MLP-5 的 train0.05/0.65 5-seed 端点；FINE(X-input, 自身隐层)。
- **新增（Part A，H-input 变体，加进现有 backbone）**：
  - `mlp_xh_ce`：`_FrozenGraceGBGCNBase` 设 `backbone="mlp", use_xh_features=True, skip_ball_targets=True` = [X‖H] 上的纯 CE-MLP。加 predictor 类 + config（含 `gb.embedding_path` 指向同一 GRACE H）。
  - `mlp_small_loss_xh` / `mlp_jocor_xh`：给 `_MLPNoiseBaselineBase` 加 `use_xh_features` 支持（`method_init` 里 load_frozen_embeddings + concat_l2_normalized_embeddings，更新 `n_feat`），派生两个类 + config。
- **单测**：(a) [X‖H] 拼接与主方法路径同一 L2-norm、同维度；(b) FINE/small-loss/jocor 的 clean 标签不进筛选/分组决策（篡改 clean 结果不变）；(c) H-input 变体与 β0.5 同 MLP 结构（层数/hidden/dropout 一致，只差输入维度）。

## 1. Part A —— H-input 公平性控制（审稿人第一刀）

```
Datasets: Cora, CiteSeer (我们赢的地方，攻击针对这里), PubMed (边界，看 H 是否改变边界故事)
Noise: uniform 0.4  (最核心、最常引用的中噪声)
Seeds: 0-4
Methods:
  SoftDist-MLP β=0.5        (X-input 分类器 + H-balls；复用主表)
  MLP+CE      [X‖H]         (新)      —— 对照 "只给 H、无任何抗噪策略"
  MLP+small-loss [X‖H]      (新)      —— 最强经典对手 + H
  MLP+JoCoR   [X‖H]         (新)      —— 最强经典对手 + H
  (参照, 复用) CE / small-loss / JoCoR (X-input)  —— 展示 "加 H 让 baseline 涨多少"
低标签率延伸 (关键卖点也要扛这刀):
  Cora/CiteSeer × train0.05 × {β0.5(复用), small-loss[X‖H], JoCoR[X‖H]} × seeds0-4
```

**核心对比**：`β0.5 − small-loss[X‖H]`、`β0.5 − JoCoR[X‖H]`、`β0.5 − CE[X‖H]`，配对 delta + 逐 seed 方向。

## 1.1 ★Part A 判据（决定论文 claim 强弱，不是生死闸）

| 观察（Cora/CiteSeer，≥2/2 数据集，≥3/5 seeds） | 论文 claim |
|---|---|
| **β0.5 仍 ≥ 最强 H-input baseline** | **王炸**：在"大家都有 GRACE-H"下，球软标签机制仍赢经典抗噪策略 → 审稿人第一刀彻底挡掉，headline claim（球机制真赢）安全 |
| β0.5 ≈ H-input baseline（打平） | claim 收窄为"球提供与'直接吃 H'相当的收益，且对 shuffled 有稳定增量"——仍可发，但不 claim 碾压经典策略；靠综合卖点 |
| β0.5 < H-input baseline | 诚实承认"主要增益来自 GRACE-H 表征、非球本身独有"→ 回落到图线 Phase24 的受控 claim（球 > CE+H、球 > shuffled）；reframe 为"graph-representation-based 去噪 + 球轻量增量"，headline 改口 |
| PubMed：H-input baseline 是否拉近/拉大差距 | 并入边界分析，说明 PubMed 的强抗噪 baseline + H 一起更强 |

> 备注：`β0.5 > shuffled`（真球 vs 随机球）已在主表 15/15 稳过，Part A 是**另一条正交的公平轴**（H-access），两者都过才无懈可击。

## 2. Part B —— 补全低标签率 5-seed 曲线（画图用）

```
Datasets: Cora, CiteSeer   (低标签率优势只在 citation 型；PubMed 是边界，不补)
Train_percent: 0.10, 0.20, 0.35, 0.50   (补齐中间点；0.05/0.65 复用 MLP-5 端点)
val=0.1, test=剩余 (同 MLP-5b 规则)
Seeds: 0-4
Methods: β=0.5 / small-loss / JoCoR / FINE / β0.5-shuffled   (与端点同一套，保证曲线连续)
```

- 产出：每 (dataset, method) 的 `test acc mean±std vs train_percent` 6 点完整曲线（含误差棒），以及 `β0.5 − small-loss/JoCoR/FINE/shuffled` 的 delta-vs-标签率曲线。
- 看点：低标签率优势的**单调趋势**是否在 5-seed 全程成立（不只是两端点）。

## 3. 执行顺序

```
1. 加 mlp_xh_ce / mlp_small_loss_xh / mlp_jocor_xh (+configs, embedding_path 指向同一 GRACE H)；单测(§0.2)
2. Part A 跑: Cora/CiteSeer/PubMed × uniform0.4 × {CE[X‖H], small-loss[X‖H], JoCoR[X‖H]} × seeds0-4
   + 低标签延伸: Cora/CiteSeer × train0.05 × {small-loss[X‖H], JoCoR[X‖H]} × seeds0-4
   (β0.5 与 X-input baseline 复用 MLP-5，不重跑)
3. Part B 跑: Cora/CiteSeer × {0.10,0.20,0.35,0.50} × {β0.5,small-loss,JoCoR,FINE,shuffled} × seeds0-4
4. 汇总: Part A 配对表(β0.5 vs H-input baseline) + §1.1 判定; Part B 完整 6 点曲线 + delta-vs-标签率
5. 报告 + 明确 §1.1 结局(公平性锁定 / 收窄 / reframe) + 低标签率曲线是否 5-seed 全程成立
```

- 输出目录：`external/NoisyGL/runs/mlp6_fairness_labelrate_20260701/`
- 报告：Part A 配对 delta 表(含逐 seed) + §1.1 判定 + Part B 6 点曲线数据 + 趋势判定 + 数据完整性。

## 4. 完成后

- 更新记忆（MLP-6 结果 + 公平性结局 + 曲线是否成立）。
- **公平性锁定(β0.5 赢 H-input baseline) + 曲线成立** → 实验主体**彻底齐**，进入**论文写作**（Claude 起草：双重鲁棒 Intro + β=0.5 半硬方法 + 主表 + H-input 公平性对照 + 结构表(MLP-2 C2 沿用) + 硬度/加权消融(MLP-4/5a) + 低标签率曲线 + 球必要性(shuffled) + related-work 切割(Graph-MLP/MgCNL/FINE/DivideMix) + 适用域诚实边界(PubMed/Coauthor-CS)）。
- **若 Part A 收窄/reframe** → 先按新 claim 重整定位再写，不带病写作。

---

## Phase MLP-6 执行结果备案
结果文件：`external/NoisyGL/runs/mlp6_fairness_labelrate_20260701/`

完整性：
- MLP-6 新增 run：265/265，missing=0。
- 合并 MLP-5 复用 anchor 后：435/435，missing=0。
- 单测：`tests/test_xh_features.py tests/test_mlp_noise_baselines.py tests/test_mlp6_fairness_labelrate.py`，17 passed。

Part A：H-input 公平性控制结论 = **reframe_needed**。

核心结果（uniform0.4, train0.20, 5 seeds）：
- Cora：β0.5 = 0.5559 ± 0.0150；CE[X‖H] = 0.4854 ± 0.0417；small-loss[X‖H] = 0.5770 ± 0.0288；JoCoR[X‖H] = 0.6013 ± 0.0343。
- CiteSeer：β0.5 = 0.5383 ± 0.0275；CE[X‖H] = 0.5166 ± 0.0422；small-loss[X‖H] = 0.5630 ± 0.0228；JoCoR[X‖H] = 0.5860 ± 0.0230。
- PubMed：β0.5 = 0.7190 ± 0.0088；CE[X‖H] = 0.8368 ± 0.0052；small-loss[X‖H] = 0.8483 ± 0.0050；JoCoR[X‖H] = 0.8479 ± 0.0064。

配对 delta（β0.5 - H-input baseline）：
- Cora vs CE[X‖H]：+0.0705，5/5 positive；vs small-loss[X‖H]：-0.0211，2/5 positive；vs JoCoR[X‖H]：-0.0454，1/5 positive。
- CiteSeer vs CE[X‖H]：+0.0217，4/5 positive；vs small-loss[X‖H]：-0.0247，1/5 positive；vs JoCoR[X‖H]：-0.0477，0/5 positive。
- train0.05 低标签公平延伸也不过门：Cora vs JoCoR[X‖H] -0.0227；CiteSeer vs JoCoR[X‖H] -0.0657。

解释：β0.5 仍稳定赢 X-only CE/small-loss/JoCoR，并且赢 CE[X‖H]；但当 classic robust baseline 也直接吃 `[X‖H]` 后，small-loss[X‖H]/JoCoR[X‖H] 反超。论文不能再写“球软标签机制在所有 baseline 同等 H-access 下仍赢经典抗噪策略”。更稳的表述应改为：**SoftDist 在 X-only 弱 MLP 协议下提供稳定球增量；若把 GRACE-H 直接作为分类器输入，经典 small-loss/JoCoR 能吸收甚至超过这部分收益。因此 H 是主要收益源之一，球机制是轻量标签侧利用 H 的方式，而非在 H-input 强 baseline 上的绝对优势。**

Part B：标签率 5-seed 曲线结论 = **X-only fairness 下成立**。

完整 6 点曲线显示，β0.5 在 Cora/CiteSeer 的 train_percent 0.05/0.10/0.20/0.35/0.50/0.65 上均值都高于 X-only small-loss、JoCoR、FINE 和 shuffled。相对 JoCoR 的增量随标签率升高而缩小：
- Cora：+6.68pp, +4.80pp, +3.87pp, +1.37pp, +1.24pp, +0.18pp。
- CiteSeer：+6.11pp, +6.52pp, +3.93pp, +2.42pp, +2.69pp, +1.35pp。

球必要性（β0.5 - shuffled）在所有 12 个 dataset×label-rate 点都是 5/5 positive：
- Cora：+6.57pp, +9.51pp, +11.96pp, +11.59pp, +13.61pp, +11.49pp。
- CiteSeer：+8.31pp, +10.08pp, +11.48pp, +9.47pp, +7.64pp, +7.32pp。

最终决策：
- 继续保留 MLP 主线，但 claim 必须收窄，不能把 H-input robust baseline 对照隐去。
- 低标签率曲线可以作为 X-only backbone 下的强证据；H-input fairness 结果必须作为 honest control 写入论文或附录。
- 写作前需要重整主叙事：从“球机制绝对赢经典鲁棒策略”改为“GRACE-H 的双重利用：标签侧球软监督在弱 MLP/X-only 协议下稳定有效；直接特征侧使用 H 是更强但更不轻量的 baseline，揭示收益来源边界。”
