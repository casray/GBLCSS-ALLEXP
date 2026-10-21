# 执行计划：标签率(覆盖率)敏感性 · 多数据集扩范围（Phase MLP-5b）

日期：2026-07-01
执行者：Codex
定位：**用新标尺(弱MLP + β=0.5主算法)重判"标签率敏感性"——这是又一个"当初被GCN吸收"的旧结论,可能挖出"低标签率优势"新卖点。** 多数据集 + 扩大标签率范围。

## 0. 背景与唯一问题

- 旧结论(Phase16, GCN[X‖H]下)："球增量非单调,低覆盖(0.05)最强,高覆盖(0.40)转负——因为分类器通过[X‖H]直接看H吸收了球优势"。**这是类A(被吸收)结论,GCN下判的。**
- 新标尺:弱MLP不吃H、β=0.5半硬主算法。**弱MLP下标签率趋势可能反转**:标签越少→球软标签补充监督越值钱→低标签率下 β=0.5 相对抗噪策略优势可能更大。
- MgCNL式噪声鲁棒论文常报"不同标签率鲁棒性曲线"。**若低/中标签率下 β=0.5 稳定优于抗噪策略,是很好的论文主图/卖点。**

> **唯一问题**:弱MLP + β=0.5 下,随标签率变化,β=0.5(以及相对抗噪策略的增益)呈什么趋势?有没有"低标签率优势"?

## 0.1 红线

1. clean 只进评测;噪声只加 train;同一 X-only 弱MLP backbone、同config(MLP-1~4一致)。
2. **多数据集一致性**(用户核心纪律):趋势结论需 ≥2/3 数据集方向一致。
3. seed0 先 smoke 看趋势,趋势清晰后关键点补 5-seed。Python: ROCm venv。
4. **主算法用 β=0.5**(MLP-4 已定型),不是旧软标签 β=0。

## 0.2 复用(全现成)

- config: `frozengrace_gb_mlp_softdist_beta05`(cora/citeseer/pubmed)、`mlp_ce`、`MLPNoiseBaselines`(small-loss/co-teaching/JoCoR/ELR/SCE)、shuffled。
- 标签率通过 `--split_type percent --train_percent X --val_percent Y --test_percent Z` 传(single_exp.py 已支持)。
- Coauthor-CS 无 β05 config → 需新建(仿 cora,用 CoauthorCS GRACE H);但它慢,**低标签率时它反而快**(train样本少),可纳入。

## 0.3 关键设计:标签率范围 + val/test 怎么随之变

- **标签率(train_percent)范围(扩大)**:`0.05, 0.1, 0.2, 0.35, 0.5, 0.65`。
  - **val/test 规则(每个 train_percent 都要定,三者和≤1)**:固定 val=0.1;test = 1 − train − 0.1。例:train0.05→val0.1→test0.85;train0.65→val0.1→test0.25。**val 固定 0.1 保证选模型条件一致**,test 随 train 变(标准做法,报告里说明)。
  - **低标签率类不平衡护栏**:train0.05 在小数据集(Cora 7类)可能某些类样本极少甚至0。**若某类 train 样本 <3,记录警告**;若整类缺失导致 predictor 报错,该 (dataset, train_percent) 标记 SKIP 并在报告注明(不硬凑)。PubMed(3类,19k节点)低标签率最稳。

## 1. 实验矩阵

```
Datasets: Cora, CiteSeer, PubMed  (+可选 Coauthor-CS,低标签率时不慢)
Noise: uniform 0.4  (主;可选加 uniform0.2 看低噪声下标签率趋势是否不同)
Train_percent: 0.05, 0.1, 0.2, 0.35, 0.5, 0.65   (val=0.1, test=剩余)
Seeds: 0 (先smoke看趋势)
Methods (全 X-only 弱MLP, 同config):
  MLP+CE                (基线)
  SoftDist-MLP β=0.5    (新主算法)
  best anti-noise对照   (small-loss + JoCoR, MLP-1显示这俩最强)
  shuffled β=0.5        (真结构对照)
```
= 3数据集 × 6标签率 × 4方法 × seed0 = 72 run(弱MLP快;PubMed/低标签率都不慢)。

## 2. 指标 + 核心看点

- test acc(seed0)每 (dataset, train_percent, method)。
- **主看**:`β0.5 − CE`、`β0.5 − small-loss`、`β0.5 − JoCoR`、`β0.5 − shuffled` 随 train_percent 的**趋势曲线**。
- **核心问题**:增益(β0.5 相对抗噪策略)是随标签率**单调下降**(低标签率优势,新卖点)、**单调上升**、还是**中间峰**?旧GCN下是"低覆盖最强"——弱MLP下是否复现/反转?

## 3. ★判据(挖卖点 or 记边界)

seed0 看趋势,趋势清晰的关键点(如最优/最差标签率)补 5-seed 确认。

| 观察到的趋势(≥2/3 数据集一致) | 结论 |
|---|---|
| **低标签率(0.05/0.1)下 β0.5 相对抗噪策略优势明显更大** | **挖到"低标签率优势"卖点** → 补 5-seed → 做成论文主图/曲线("标签稀缺时粒球软监督更值钱")。呼应旧Phase16趋势但换成"赢抗噪策略"的新标尺 |
| 增益随标签率大致平坦(各标签率都稳定小赢) | 写"标签率鲁棒":跨标签率一致优于抗噪策略,是稳健性证据(次卖点) |
| 高标签率下 β0.5 增益消失/转负(像旧GCN结论) | 记录边界:"标签充足时球监督收益被稀释",诚实写进适用域;主协议锁在中低标签率 |
| 趋势数据集间不一致 | 只报各数据集曲线,不下统一趋势结论(用户纪律) |

## 4. 执行顺序

```
1. 新建 Coauthor-CS 的 β05 config(仿 cora,用 CoauthorCS GRACE H)——若纳入
2. 写批处理:3(+1)数据集 × 6标签率 × 4方法 × seed0，val=0.1、test=剩余；低标签率类不平衡护栏(某类<3警告，整类缺失SKIP并注明)
3. seed0 跑 → 画 增益-vs-标签率 趋势曲线(每数据集)
4. 按§3判趋势 → 趋势清晰的关键标签率点补 5-seed
5. 报告：增益-vs-标签率 曲线(每数据集×每对照) + 趋势判定 + 是否挖到低标签率卖点 + SKIP/警告记录
```

- 输出目录：`external/NoisyGL/runs/mlp_labelrate_20260701/`
- 报告：每 (dataset, train_percent) 的 acc 表 + β0.5相对CE/small-loss/JoCoR/shuffled 的 delta 随标签率曲线 + 趋势判定 + 卖点/边界结论 + 数据完整性(含SKIP)。

## 5. 完成后

- 更新记忆(MLP-5b 结果 + 标签率趋势 + 是否有低标签率卖点)。
- **有低标签率优势** → 论文加一张标签率鲁棒主图,补进 MLP-5 主表方案。
- **无/边界** → 记录适用域(主协议锁中低标签率),不强做。
- 之后回主线:MLP-5(β=0.5 重做标签主表)→ 补 FINE/DivideMix baseline → 真实噪声数据集 → 写作。

---

## Phase MLP-5b 执行结果备案（seed0 smoke 已完成）

结果文件：`external/NoisyGL/runs/mlp_labelrate_20260701/`

实现与完整性：
- 新增 runner：`scripts/run_mlp_labelrate_20260701.ps1`
- 新增汇总：`scripts/summarize_mlp_labelrate.py`
- 新增公平对照：`frozengrace_gb_mlp_softdist_beta05_shuffled`（同 β=0.5、同 X-only MLP、random size-matched balls）
- 补齐 PubMed 的 `mlp_ce` / `mlp_small_loss` / `mlp_jocor` config
- 运行矩阵：Cora/CiteSeer/PubMed × train_percent {0.05,0.10,0.20,0.35,0.50,0.65} × 5 methods × seed0 = 90/90 run 完成
- `mlp_labelrate_completeness.csv`：缺失 0
- `mlp_labelrate_split_audit.csv`：低类覆盖警告 0；最低类计数 Cora train0.05 仍有 7 个样本，未触发 skip

关键端点结果（test acc, seed0）：

| Dataset | train | CE | small-loss | JoCoR | β=0.5 | β=0.5 shuffled |
|---|---:|---:|---:|---:|---:|---:|
| Cora | 0.05 | 0.3312 | 0.3481 | 0.3412 | 0.4098 | 0.3286 |
| Cora | 0.65 | 0.5126 | 0.6411 | 0.6529 | 0.6514 | 0.5126 |
| CiteSeer | 0.05 | 0.3502 | 0.3771 | 0.3746 | 0.4443 | 0.3336 |
| CiteSeer | 0.65 | 0.5957 | 0.6330 | 0.6173 | 0.6318 | 0.5848 |
| PubMed | 0.05 | 0.5479 | 0.6712 | 0.6908 | 0.6132 | 0.5314 |
| PubMed | 0.65 | 0.7539 | 0.8273 | 0.8217 | 0.8174 | 0.7166 |

趋势摘要（low = train0.05/0.10，mid = 0.20/0.35，high = 0.50/0.65；数值为 β=0.5 相对 baseline 的 mean delta）：

| Dataset | Comparison | Low | Mid | High | Low-High | Direction |
|---|---|---:|---:|---:|---:|---|
| Cora | β0.5 vs small-loss | +0.0535 | +0.0302 | +0.0176 | +0.0358 | low-label advantage |
| Cora | β0.5 vs JoCoR | +0.0553 | +0.0195 | +0.0053 | +0.0501 | low-label advantage |
| CiteSeer | β0.5 vs small-loss | +0.0791 | +0.0644 | +0.0201 | +0.0590 | low-label advantage |
| CiteSeer | β0.5 vs JoCoR | +0.0826 | +0.0644 | +0.0219 | +0.0607 | low-label advantage |
| PubMed | β0.5 vs small-loss | -0.0623 | -0.0374 | -0.0166 | -0.0457 | high-label advantage / still negative |
| PubMed | β0.5 vs JoCoR | -0.0779 | -0.0445 | -0.0098 | -0.0681 | high-label advantage / still negative |

真实球结构对照：
- β=0.5 相对 shuffled 在 3 个数据集、6 个标签率上全部为正。
- 大致范围：Cora +6.65pp 到 +14.68pp；CiteSeer +4.69pp 到 +13.10pp；PubMed +7.21pp 到 +10.08pp。
- 说明 GRACE-H 真实球结构在弱 MLP 下始终提供有效监督信号；失败点不是 “球无效”，而是 PubMed 上 small-loss/JoCoR 的同 backbone 抗噪策略更强。

判定：
- **低标签率优势不是全数据集统一结论。**
- Cora/CiteSeer 上，β=0.5 相对 small-loss/JoCoR 的增益在低标签率明显更大，符合“标签稀缺时粒球监督更值钱”的机制解释。
- PubMed 上，β=0.5 虽稳定赢 CE 和 shuffled，但在所有标签率都输 small-loss/JoCoR；随着标签率升高，负差距缩小。这是明确边界：PubMed 的弱 MLP 抗噪 baseline 本身很强，球监督不足以超过它。
- 因此论文可写为：**在 Cora/CiteSeer 型 citation 数据上存在 seed0 低标签率优势信号；PubMed 是边界数据集，不能写跨数据集统一低标签率优势。**

5-seed 决策：
- 本轮不自动扩全矩阵 5-seed，因为三数据集趋势不完全一致。
- 若要把“低标签率优势”写成论文图，建议只对关键端点补 5-seed：Cora/CiteSeer 的 train0.05 与 train0.65，methods = β0.5 / small-loss / JoCoR / shuffled（可保留 CE）。PubMed 则作为边界，不作为低标签率优势 claim 的支撑。

产物：
- 汇总报告：`runs/mlp_labelrate_20260701/mlp_labelrate_report.md`
- 配对差值：`runs/mlp_labelrate_20260701/mlp_labelrate_paired_deltas.csv`
- 趋势表：`runs/mlp_labelrate_20260701/mlp_labelrate_trends.csv`
- split audit：`runs/mlp_labelrate_20260701/mlp_labelrate_split_audit.csv`
- 曲线图：`runs/mlp_labelrate_20260701/mlp_labelrate_delta_curves_{cora,citeseer,pubmed}.png`
