# 执行计划：目标硬度 β 谱系 5-seed 确认（Phase MLP-4）

日期：2026-07-01
执行者：Codex
定位：**把 MLP-3 seed0 看到的"数据集/噪声依赖硬度"规律，升级到 5-seed，判是否稳（seed0 会撒谎）。** 决定主算法走向：固定硬度 / 固定半硬 / 还是需要自适应硬度。几乎零新开发（MLP-3 的 config/代码/脚本全现成，只扩 seeds）。

## 0. 背景与唯一问题

- MLP-3 seed0 结论：β=0(soft) 6/6 都不是最好（原SoftDist软标签非最优硬度，确证）；β=1 赢 uniform0.4(3/3正)，β=0.5 赢低噪声 uniform0.2；**落在"最优硬度随数据集/噪声变化"**。β=1 全 6/6 > shuffled(+8~15pp)，硬化仍依赖真实球结构。
- **但这是 seed0 单点。历史上 seed0 撒谎多次（gated-hard/ball-on-H/DBLP椭球都被骗过）。** 必须 5-seed 确认这个规律是否稳，才能定主算法。

> **唯一问题**：5-seed 下，"uniform0.4 偏 β=1、uniform0.2 偏 β=0.5、Cora 偏硬、数据集有差异"这个规律稳不稳？

## 0.1 红线

1. β 固定超参，不用 clean/test 调（沿用 MLP-3）。这一轮仍只跑固定 β∈{0,0.5,1}，**不做自适应**（自适应触发条件见§4）。
2. clean 只进评测；噪声只加 train；同一 X-only 弱MLP backbone、同config（MLP-1/2/3 一致）。
3. paired delta（同 seed 同 split 比较），报 mean±std over 5 seeds + 逐 seed 方向计数。

## 0.2 复用（全现成，只扩 seeds）

- config 已建：`frozengrace_gb_mlp_softdist`(β=0) / `frozengrace_gb_mlp_softdist_beta05`(β=0.5) / `frozengrace_gb_mlp_softdist_beta1`(β=1) / `frozengrace_gb_mlp_softdist_beta1_shuffled`，均含 cora/citeseer/pubmed。
- `target_beta` 插值代码已进 `apply_ball_soft_targets`（β=0→distribution，β=1→majority onehot，中间线性）。
- 批处理脚本 `scripts/run_mlp_beta_hardness_20260701.ps1` 已存在——**改 seeds 从 {0} 到 {0,1,2,3,4} 即可**。
- Python：ROCm venv（同前）。NoisyGL percent split 0.2/0.1/0.7 显式传参。

## 1. 实验矩阵

```
Datasets: Cora, CiteSeer, PubMed
Noise: uniform 0.2, uniform 0.4
Seeds: 0,1,2,3,4
Methods: β=0, β=0.5, β=1, β=1-shuffled   (全 X-only 弱MLP, 同config)
```
= 3×2×4×5 = 120 run（弱MLP快，PubMed 在 MLP-3 已验证不慢）。

## 2. 指标

- test acc mean±std（5 seeds）每 (dataset, noise, β)。
- paired delta：β=1 − β=0、β=0.5 − β=0、β=1 − β=0.5、β=1 − β=1-shuffled，各含 mean + 逐seed方向计数(x/5)。

## 3. ★判据（承接 MLP-3 的四种模式，用 5-seed 定）

判"方向稳"用：paired delta 同号 ≥4/5 seeds 才算该设定有稳定偏好。

| 5-seed 观察 | 主算法决策 |
|---|---|
| **β=1 在 ≥2/3 数据集、两个噪声率都稳定 ≥ β=0（≥4/5 seeds）** | 主算法 = hard-majority target（最简单）。进 MLP-5 用 hard 重做 MLP-2 关键主表 |
| **β=0.5 在 ≥2/3 数据集稳定不输两端（both noise）** | 主算法 = 固定半硬 β=0.5。同样进 MLP-5 重做主表 |
| **规律稳但数据集/噪声依赖**（uniform0.4稳偏硬 且 uniform0.2稳偏半硬，且跨数据集方向可复现） | **触发自适应硬度**（MLP-5=adaptive smoke）：按 noisy vote margin/purity 选每球 β（部署期信号，不用clean）。这是论文核心机制卖点 |
| **规律不稳**（seed0 的偏好在 5-seed 下抖动、方向计数 <4/5、数据集间不可复现） | seed0 是假象；退而求稳：选一个"跨设定平均最好且方差小"的固定 β 作主算法（大概率 β=0.5），不追自适应。soft-vs-hard 证据进消融 |

**共同底线**：无论哪种，β=1 > shuffled 若 5-seed 仍成立(≥4/5)，则"硬化依赖真实球结构"这条写进论文（球必要性的强证据）。

## 4. 执行顺序

```
1. 改 run_mlp_beta_hardness_20260701.ps1 的 seeds → {0,1,2,3,4}（或新建 _5seed 脚本，避免覆盖 MLP-3 seed0 结果）
2. 跑 3数据集 × uniform{0.2,0.4} × {β0,β0.5,β1,β1-shuffled} × seeds0-4 = 120 run
3. 汇总：mean±std 表 + paired deltas(含逐seed方向计数) + 跨数据集/跨噪声一致性
4. 按§3判四种模式 → 明确主算法决策(hard / 半硬 / 自适应 / 稳态固定β)
5. 报告
```

- 输出目录：`external/NoisyGL/runs/mlp_beta_hardness_5seed_20260701/`（**别覆盖 MLP-3 的 seed0 目录**）
- 报告：mean±std 表(dataset×noise×β) + paired deltas + 逐seed方向计数 + 一致性分析 + §3 模式判定 + 主算法决策 + β=1-vs-shuffled 是否 5-seed 成立。

## 5. 完成后

- 更新记忆（MLP-4 结果 + 主算法硬度终定 or 转自适应）。
- **固定硬度/半硬胜** → MLP-5：用新主算法重做 MLP-2 受影响的关键主表（只重跑受影响格，不全重跑）。
- **数据集依赖且稳** → MLP-5：自适应硬度 smoke（noisy-margin/purity 选 β，seed0 先看能否自动"0.4变硬/0.2变半硬"且不输固定最优β）。
- 主算法定型后，才补 FINE/DivideMix baseline + 真实噪声数据集（顺序不变）。

---

## Phase MLP-4 执行结果备案（已完成）
结果文件：`external/NoisyGL/runs/mlp_beta_hardness_5seed_20260701/`

执行状态：已完成。完整性：3 数据集 × 2 噪声率 × 4 方法 × 5 seeds = 120/120，无缺失，无 ROCm OOM。

输出文件：

- `runs/mlp_beta_hardness_5seed_20260701/beta_uniform0p2_seeds_0_1_2_3_4.csv`
- `runs/mlp_beta_hardness_5seed_20260701/beta_uniform0p4_seeds_0_1_2_3_4.csv`
- `runs/mlp_beta_hardness_5seed_20260701/mlp_beta_hardness_aggregate.csv`
- `runs/mlp_beta_hardness_5seed_20260701/mlp_beta_hardness_paired_deltas.csv`
- `runs/mlp_beta_hardness_5seed_20260701/mlp_beta_hardness_decision.csv`
- `runs/mlp_beta_hardness_5seed_20260701/mlp_beta_hardness_report.md`

### 5-seed mean±std

| Dataset | Noise | beta=0 soft | beta=0.5 | beta=1 | beta=1 shuffled | mean-best |
|---|---:|---:|---:|---:|---:|---|
| Cora | 0.2 | 0.6141±0.0124 | 0.6294±0.0160 | **0.6368±0.0136** | 0.5121±0.0191 | beta=1 |
| Cora | 0.4 | 0.5184±0.0137 | 0.5559±0.0150 | **0.5679±0.0196** | 0.4178±0.0133 | beta=1 |
| CiteSeer | 0.2 | 0.6229±0.0083 | **0.6338±0.0120** | 0.6289±0.0193 | 0.5338±0.0159 | beta=0.5 |
| CiteSeer | 0.4 | 0.5162±0.0270 | 0.5383±0.0275 | **0.5456±0.0154** | 0.4105±0.0231 | beta=1 |
| PubMed | 0.2 | 0.8223±0.0052 | **0.8251±0.0033** | 0.8125±0.0052 | 0.7416±0.0123 | beta=0.5 |
| PubMed | 0.4 | 0.7011±0.0357 | 0.7187±0.0096 | **0.7424±0.0064** | 0.6192±0.0262 | beta=1 |

### Paired delta 方向计数

| Dataset | Noise | beta0.5-soft | beta1-soft | beta1-beta0.5 | beta1-shuffled |
|---|---:|---:|---:|---:|---:|
| Cora | 0.2 | +1.53pp (5/5) | +2.27pp (5/5) | +0.74pp (4/5) | +12.47pp (5/5) |
| Cora | 0.4 | +3.75pp (5/5) | +4.95pp (5/5) | +1.20pp (5/5) | +15.01pp (5/5) |
| CiteSeer | 0.2 | +1.10pp (4/5) | +0.60pp (2/5) | -0.50pp (2/5) | +9.51pp (5/5) |
| CiteSeer | 0.4 | +2.22pp (5/5) | +2.95pp (5/5) | +0.73pp (3/5) | +13.51pp (5/5) |
| PubMed | 0.2 | +0.28pp (4/5) | -0.98pp (0/5) | -1.26pp (0/5) | +7.10pp (5/5) |
| PubMed | 0.4 | +1.76pp (4/5) | +4.13pp (4/5) | +2.37pp (5/5) | +12.32pp (5/5) |

### 四模式判定

- 固定 hard-majority target 不过门：只有 Cora 同时在 0.2/0.4 稳定支持 beta=1 ≥ beta=0；CiteSeer 0.2 只有 2/5，PubMed 0.2 为 0/5 且均值为负。
- 固定 beta=0.5 “不输两端”也不过门：Cora 和 PubMed 在 uniform0.4 明显偏 beta=1。
- 自适应硬度不触发：uniform0.4 稳定偏 hard 的数据集为 2/3，但 uniform0.2 稳定偏半硬只有 1/3，不满足“低噪声偏半硬、中噪声偏硬”的跨数据集复现。
- seed0 的“硬度依赖”规律部分成立但不够整齐；严格按预注册判据归入“规律不稳/不可直接做自适应”。

### 主算法硬度决策

不追自适应硬度。保守默认主算法建议改为 **beta=0.5 半硬目标**，理由：

1. beta=0.5 是唯一在 6/6 设置中稳定高于旧 beta=0 soft target 的硬度（每格 ≥4/5 seeds 为正）。
2. beta=1 虽然是 4/6 设置的 mean-best，且在 uniform0.4 上很强，但 PubMed uniform0.2 明确为负（0/5，-0.98pp），不适合作为跨噪声默认。
3. beta=1 real > shuffled 在 6/6 设置中全部成立（5/5），可以作为“硬化目标仍依赖真实球结构”的强消融证据，但应写成 aggressive / mid-noise variant，而不是默认主算法。

论文 claim 边界：可以写“target hardening matters; beta=0.5 is a stable fixed default over soft distribution, while full hardening is beneficial mainly in medium-noise regimes.” 不能写“hard majority 全面最优”，也不能写“自适应硬度已经成立”。
