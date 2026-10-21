# 执行计划：椭球/GMM 加权在弱MLP下重验（Phase MLP-5a · 被GCN吸收机制的翻案检验）

日期：2026-07-01
执行者：Codex
定位：**最小 smoke，验证两个"当初死于被GCN[X‖H]吸收"的机制(ellipsoid / gmm_resp)，在弱MLP + 新主算法(β=0.5半硬)下能否翻案。** 判它们是"锦上添花进主方法"还是"进消融表(证明简单半硬已足够)"。

## 0. 背景与唯一问题

- 用户核心思路：换标尺(弱MLP)后，那些"死于被GCN吸收"的机制值得重验。按类A/类B原则筛完，**只有 ellipsoid、gmm_resp 是类A(死于被吸收)且没在弱MLP系统验过**；其余(expand/bc/dynalpha/gated/adaptive)是类B(死于机制本身缺陷)，不重验。
- ellipsoid/gmm_resp 本质 = "球内节点贡献不均"的精细加权(椭球用Mahalanobis、GMM用后验)。而 MLP-4 已证"半硬β=0.5 > 纯软"。**新问题：精细加权 + 半硬，会不会比纯半硬更好？** 硬度(β)和加权是正交两个维度，没组合验过。

> **唯一问题**：`ellipsoid+β=0.5` 或 `gmm_resp+β=0.5` 能否在 ≥2/3 数据集稳定 > 纯 `β=0.5`？

## 0.1 诚实预判（写在前，防期望过高）

- 即使翻案，**大概率是小提点，不改变论文走向**（硬度是主因，加权是软标签内部微调）。
- **gmm_resp 撞 G-GBC**(KBS2026, GMM+Mahalanobis 图像噪声)、**ellipsoid 撞 EliDecide** → 新颖性观感弱，审稿会问区别。所以它们的定位是"让主方法更强 + 丰富消融"，**不是核心卖点**。
- 结局二选一都有用：赢→进主方法(半硬+精细加权)；不赢→进消融(证明"简单半硬已足够，精细加权无额外收益"，这是干净的负结果)。

## 0.2 红线

1. **基线用 β=0.5(新主算法)，不是旧软标签 β=0**——问"加权能否超过半硬"，不是"超过纯软"。
2. clean 只进评测；噪声只加 train；同一 X-only 弱MLP backbone、同config。
3. **多数据集一致性**(用户核心纪律)：≥2/3 数据集方向一致才下"加进主方法"结论。
4. seed0 先 smoke，出信号再 5-seed。Python：ROCm venv。percent split 0.2/0.1/0.7 显式传参。

## 0.3 代码状态（核实过）

- **gmm_resp MLP版已有**：`frozengrace_gb_mlp_softdist_gmmresp_Predictor`（+ config），但**没系统跑过**。
- **ellipsoid 只有GCN版**：`frozengrace_gb_gcn_softdist_xh_ellipsoid`（`softdist_weighting="ellipsoid"`）。**需新建 MLP版**：照 gmmresp MLP predictor 的样子，加 `backbone="mlp"` + `softdist_weighting="ellipsoid"` + config(cora/citeseer)。
- 加权逻辑现成：`apply_ball_ellipsoid_soft_targets` / gmm_resp 在 `frozen_grace_targets.py`。
- **★关键待确认(实现时解决)**：ellipsoid/gmm_resp 加权 能否和 `target_beta=0.5` **组合**？两者一个改"软标签分布内部的加权"、一个做"软↔硬插值"。正确组合 = **先按加权算出软分布，再对该分布做 β 插值**(`(1-β)·weighted_dist + β·onehot(majority)`)。Codex 实现时确认加权+β的顺序，并加单测。
  - 若组合工程量大，**退而求其次**：先只跑 `ellipsoid+β=0`(纯软加权) vs `β=0.5` vs `β=0`，看加权在纯软层面有没有用；有用再做加权+半硬。

## 1. 实验矩阵（最小 smoke）

```
Datasets: Cora, CiteSeer
Noise: uniform 0.4   (椭球/GMM最可能有用的中噪声)
Seeds: 0
Methods (全 X-only 弱MLP):
  β=0.5              (新主算法, 基线)
  ellipsoid + β=0.5  (需新建MLP predictor)
  gmm_resp + β=0.5   (MLP版有, 确认配β)
  ellipsoid shuffled + β=0.5   (真结构对照, 保一个)
  [可选] β=0 纯软 + ellipsoid/gmm_resp  (若β组合难，先验纯软层面)
```

## 2. 指标 + 核心对比

- test acc(seed0)每方法。
- `ellipsoid+β0.5 − β0.5`、`gmm_resp+β0.5 − β0.5`(加权是否超过纯半硬)。
- `ellipsoid+β0.5 − ellipsoid-shuffled+β0.5`(加权是否依赖真实球结构)。

## 3. ★判据

seed0 先看方向，出信号再 5-seed 确认（用户纪律：seed0 会撒谎）。

| seed0 观察 | 下一步 |
|---|---|
| ellipsoid 或 gmm_resp + β0.5 明显 > β0.5(两数据集都正) | 5-seed 确认 → 若 ≥2/3 数据集 ≥4/5 seeds 稳定正 → 加进主方法(半硬+精细加权)，写进方法章 |
| 加权 ≈ β0.5(无明显增益) | 进消融表：结论"简单半硬(β0.5)已足够，椭球/GMM 精细加权无额外收益"——干净负结果，且回避了撞 G-GBC/EliDecide 的新颖性风险 |
| 加权 < β0.5(有害) | 明确记录椭球/GMM 在弱MLP下也不行(和GCN下一致)，彻底关闭这两个机制 |
| 加权 > shuffled 但 ≈ β0.5 | 说明加权有球结构信号但被半硬吸收；写进消融讨论 |

## 4. 执行顺序

```
1. 新建 ellipsoid 的 MLP predictor + config(照 gmmresp MLP版)；确认 gmm_resp MLP predictor 可跑
2. 实现/确认 加权 + target_beta 的组合(先算加权软分布→再β插值)，加单测(β=0退化=纯加权软, β=1退化=硬)
   若组合难 → 先跑纯软层面(β=0)版本
3. seed0 跑 Cora/CiteSeer uniform0.4: β0.5 / ellipsoid+β0.5 / gmm_resp+β0.5 / ellipsoid-shuffled+β0.5
4. 按§3判 → 出信号则 5-seed(Cora/CiteSeer/PubMed × uniform0.4)确认
5. 报告 + 明确"加权进主方法 or 进消融 or 关闭"
```

- 输出目录：`external/NoisyGL/runs/mlp_weighting_revalidate_20260701/`
- 报告：各方法 acc + 加权vs半硬delta + 加权vs-shuffled + §3判定 + 加权机制定位(主方法/消融/关闭)。

## 5. 完成后

- 更新记忆（MLP-5a 结果 + ellipsoid/gmm_resp 在弱MLP下翻案与否）。
- **翻案(赢半硬)** → 加进主方法，MLP-5(主表重做)用"半硬+加权"版。
- **不翻案** → 进消融表(简单半硬足够)，主算法维持纯 β=0.5；类B机制(expand/bc/dynalpha/gated/adaptive)按预注册不重验，论文里写"GCN下判负+机制原因，不适用MLP线"作诚实边界。
- 之后才回主线：MLP-5(β=0.5或半硬+加权 重做标签主表) → 补FINE/DivideMix baseline → 真实噪声数据集。

---

## Phase MLP-5a 执行结果备案（已完成）

结果文件：`external/NoisyGL/runs/mlp_weighting_revalidate_20260701/`

实现确认：
- 已实现 `ellipsoid/gmm_resp + target_beta` 组合，顺序为：先构造加权 soft distribution，再做 `(1-β)·weighted_dist + β·onehot(majority)`。
- 已新增 MLP predictor/config：`frozengrace_gb_mlp_softdist_ellipsoid_beta05`、`frozengrace_gb_mlp_softdist_ellipsoid_beta05_shuffled`、`frozengrace_gb_mlp_softdist_gmmresp_beta05`。
- 回归测试通过：`python -m unittest tests.test_frozen_grace_targets tests.test_gmm_resp_softdist tests.test_mlp_softdist tests.test_mlp_weighting_revalidate -v`。

seed0 smoke 设置：
- Datasets: Cora, CiteSeer
- Noise: uniform 0.4, train-only
- Split: percent 0.2/0.1/0.7
- Backbone: X-only weak MLP
- Seed: 0

| Dataset | β=0.5 baseline | ellipsoid+β=0.5 | gmm_resp+β=0.5 | ellipsoid shuffled+β=0.5 |
|---|---:|---:|---:|---:|
| Cora | 0.5546 | 0.5662 | 0.5230 | 0.4459 |
| CiteSeer | 0.5533 | 0.5485 | 0.5451 | 0.4257 |

关键 paired delta：

| Dataset | ellipsoid − β0.5 | gmm_resp − β0.5 | ellipsoid − shuffled |
|---|---:|---:|---:|
| Cora | +1.16pp | -3.17pp | +12.03pp |
| CiteSeer | -0.47pp | -0.82pp | +12.29pp |

判定：
- `ellipsoid+β=0.5` 只在 Cora seed0 赢 β=0.5，在 CiteSeer 小输；不满足“两数据集都正”的 5-seed 触发条件。
- `gmm_resp+β=0.5` 在 Cora/CiteSeer 均输 β=0.5；直接进入负结果/消融，不作为主方法候选。
- `ellipsoid` 相对 shuffled 双数据集均大幅为正（约 +12pp），说明真实球结构仍有强信号；但这部分信号没有稳定超过简单半硬目标，属于“有结构信息，但被 β=0.5 半硬机制吸收”。

最终决策：
- 主算法维持 **β=0.5 半硬目标**，不加入 ellipsoid/gmm_resp 精细加权。
- MLP-5a 不扩 5-seed；ellipsoid/gmm_resp 作为消融或负结果记录，结论是“在弱 MLP 下，简单半硬目标已基本吸收球内精细加权收益”。
- 后续 MLP 主表仍按 β=0.5 执行；β=1 保持 aggressive / medium-noise variant 定位。
