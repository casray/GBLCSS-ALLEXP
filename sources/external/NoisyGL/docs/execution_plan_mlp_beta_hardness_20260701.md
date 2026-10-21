# 执行计划：目标硬度 β 谱系 × 多数据集 seed0 smoke（Phase MLP-3）

日期：2026-07-01
执行者：Codex
定位：**最小算力，定主算法的"目标硬度"**。MLP-2 消融发现 hard-majority 在 Cora/CiteSeer/Coauthor-CS uniform0.4 上 5/5 稳定 > soft distribution（delta −5~−7pp）——**推翻了 SoftDist "软标签更好" 的核心假设**。本实验回答唯一问题：**目标是越硬越好、还是中间最好、还是数据集依赖？** seed0 谱系，30分钟量级。

## 0. 唯一问题 + 用户的关键约束（多数据集，防单数据集翻车）

> 固定超参 β 控制目标硬度：`target_i = (1−β)·soft_distribution_i + β·onehot(majority_i)`。β=0 是当前 SoftDist（软），β=1 是 hard-majority（硬），β=0.5 半硬。**在弱MLP下、多个数据集上，哪个 β 最好？**

**用户的硬约束（必须遵守，是45天最成熟的方法论直觉）**：本方法在不同数据集上表现历来不一致（Phase11/24、MLP-1/2 反复出现"一个数据集好另一个不好"）。**所以绝不能只看单数据集 seed0 定方向。判据必须要求 ≥2/3 数据集方向一致才下结论**（见§4）。这是这个实验相对普通 seed0 smoke 的关键加严。

## 0.1 红线

1. **β 是固定超参，不用 clean/test 调**（否则 cherry-picking）。这一轮只跑固定 β∈{0,0.5,1}，**不做自适应 β**（自适应留到后续，且触发条件见§4）。
2. clean 标签只进评测。噪声只加 train。所有方法同一 X-only 弱MLP backbone、同配置（沿用 MLP-1/2 的 MLP）。
3. seed0 单跑定方向；**任何"改主算法"的结论都要 ≥2/3 数据集一致 + 后续5-seed确认**，不能凭 seed0 单点。
4. Python：`C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe`（ROCm）。NoisyGL percent split 0.2/0.1/0.7 显式传参。

## 0.2 环境与复用（关键：β=0/β=1 已现成，只差 β=0.5 插值）

- **已现成**：
  - β=0（soft）= `frozengrace_gb_mlp_softdist_Predictor`（target_mode="distribution"）。
  - β=1（hard）= `frozengrace_gb_mlp_hard_Predictor`（已注册，target_mode="majority" 或硬 corrected-label）。**先确认它就是 onehot(majority) 软标签、同 MLP backbone。**
  - shuffled 版已有。
- **需新增**：**β=0.5 插值**——在 `apply_ball_soft_targets` 里，最终 target = `(1−β)·distribution_target + β·majority_onehot`，加一个 `target_beta` config 字段（默认按现有行为）。**注意：不要复用已存在的 `dynamic_agreement_beta`（那是动态alpha的agreement权重，与本目标硬度插值无关，混用会出错）——新起名 `target_beta`。**
- 单测：β=0 时 target 完全等于旧 soft distribution；β=1 时等于 onehot(majority)；β=0.5 时是两者均值。篡改 clean 标签 target 不变。

## 0.3 数据集（多数据集是核心，但控算力）

- **主：Cora, CiteSeer**（快，MLP-1/2 已有 config 和趋势）。
- **第三个数据集（满足用户"多数据集"要求，但避开 Coauthor-CS 的2小时）**：
  - **首选 PubMed**——比 Coauthor-CS 快很多，且是"当初判负的大图"，顺便验证大图在新标尺下是否反转。**但风险：PubMed 没有现成 SoftDist-MLP config，且 GRACE H embedding 未确认存在。**
  - **执行前先检查 PubMed GRACE H 是否存在**：
    - 存在 → 仿 cora config 建 PubMed 的 mlp-softdist config，用 PubMed。
    - **不存在 → 退回用 Coauthor-CS 作第三个**（虽慢，但 seed0 单跑 + 只4方法×2噪声=8个run，可接受；或只在 uniform0.4 单噪声跑 Coauthor-CS 省一半）。
  - 不要为 PubMed 现训 GRACE（超出本 smoke 范围）。

## 1. 实验矩阵（最小）

```
Datasets: Cora, CiteSeer, {PubMed 或 Coauthor-CS}   ← 三个
Noise: uniform 0.2, uniform 0.4                      ← 两个(0.4是消融发现硬赢的,0.2是软可能赢的)
Seeds: 0
Methods (全 X-only 弱MLP, 同config):
  β=0   (soft, 当前SoftDist)
  β=0.5 (半硬, 新增插值)
  β=1   (hard majority)
  shuffled (β=1的shuffled球, 真结构对照, 保一个)
```
总 run ≈ 3×2×4 = 24（PubMed快则全跑；若退Coauthor-CS，可只在uniform0.4跑它省一半）。

## 2. 指标

- test acc（seed0 单值，干净 test）。
- 每数据集每噪声率：β=0/0.5/1 的精度 + 相对 β=0（当前soft）的 delta + 是否 > shuffled。

## 3. 关注的核心对比

- **β=1 vs β=0**：硬是否真的赢软（复查 MLP-2 消融在 uniform0.2 是否也成立，还是只 uniform0.4）。
- **β=0.5 vs β=0/β=1**：中间是否更稳（插值有没有价值）。
- **跨数据集一致性**：同一个 β 的胜负方向，在 3 个数据集上是否一致（用户最关心的）。
- **noise-rate 依赖**：uniform0.2 和 0.4 是否偏好不同 β。

## 4. ★判据（升级版，含"多数据集一致性"——用户的核心加严）

**下"改主算法"结论的前提：≥2/3 数据集方向一致，且都 > shuffled。**

| 观察到的模式 | 结论 / 下一步 |
|---|---|
| **β=1 在 ≥2/3 数据集、两个噪声率都 ≥ β=0** | 主算法改 hard-majority MLP（soft 是历史包袱）。下一步：hard 版跑 5-seed 全矩阵确认 |
| **β=0.5 在 ≥2/3 数据集最稳（不输两端）** | 主算法 = soft-hard 插值（固定β）。写 "target hardness interpolation"。5-seed 确认最优 β |
| **不同数据集/噪声偏好不同 β**（用户担心的情况） | **不改成固定硬度**；这本身是发现 → 触发下一轮"按 noisy vote margin/purity 自适应 β"（部署期可得信号，不用clean）。这是论文的一个 selling point（自适应硬度）|
| **β=0（soft）在 ≥2/3 数据集仍最好** | soft 站得住，MLP-2 的 hard-majority 优势只是 uniform0.4 单点/数据集现象；回去做其他改进（覆盖率等）。soft-vs-hard 跨数据集证据进消融表 |
| **单数据集支持硬、其他不支持** | **不改**（正是用户要防的翻车）；记录"硬度是数据集依赖的" → 同样触发自适应 β 探索 |

**关键纪律**：seed0 + 3数据集只用于"定方向 + 排除单数据集假象"，**最终任何主算法变更必须再跑 5-seed 确认**。seed0 会撒谎（历史多次），多数据集一致性只是第一道过滤，不是终判。

## 5. 执行顺序

```
1. 加 target_beta 插值到 apply_ball_soft_targets（β=0→distribution, β=1→majority onehot, 中间线性插值）
   新起名 target_beta，勿混 dynamic_agreement_beta。单测3条(β=0/0.5/1) + clean不进target
2. 检查 PubMed GRACE H 是否存在 → 定第三数据集(PubMed 或 退Coauthor-CS)
3. seed0 跑：3数据集 × uniform{0.2,0.4} × {β0, β0.5, β1, shuffled}
4. 按§4判：看跨数据集一致性 + noise依赖，给出四种模式里的哪一种
5. 报告 + 明确"主算法硬度"决策 + 是否需5-seed确认/自适应β
```

- 输出目录：`external/NoisyGL/runs/mlp_beta_hardness_20260701/`
- 报告：3数据集×2噪声×3β 的 acc 表 + 相对soft的delta + shuffled对照 + 跨数据集一致性分析 + §4 模式判定 + 主算法硬度决策。

## 6. 完成后

- 更新记忆（MLP-3 结果 + 主算法硬度定论 + 是否数据集依赖）。
- **β=1 或 β=0.5 胜**：定新主算法，跑 5-seed 全矩阵确认 → 回 MLP-2 用新主算法重做关键主表（只重跑受影响的，不全重跑）。
- **数据集依赖**：下一轮做"noisy-margin 自适应 β"（部署期信号，不用clean），作为论文自适应卖点。
- **soft 仍最好**：维持 soft，soft-vs-hard 证据进消融，转做覆盖率/其他改进。

---

## Phase MLP-3 执行结果备案（已完成）
结果文件：`external/NoisyGL/runs/mlp_beta_hardness_20260701/`

执行状态：已完成。使用 PubMed 作为第三数据集（PubMed GRACE H 存在），未退回 Coauthor-CS。完整性：3 数据集 × 2 噪声率 × 4 方法 × seed0 = 24/24，无缺失。

输出文件：

- `runs/mlp_beta_hardness_20260701/beta_uniform0p2_seeds_0.csv`
- `runs/mlp_beta_hardness_20260701/beta_uniform0p4_seeds_0.csv`
- `runs/mlp_beta_hardness_20260701/mlp_beta_hardness_aggregate.csv`
- `runs/mlp_beta_hardness_20260701/mlp_beta_hardness_paired_deltas.csv`
- `runs/mlp_beta_hardness_20260701/mlp_beta_hardness_decision.csv`
- `runs/mlp_beta_hardness_20260701/mlp_beta_hardness_report.md`

### 结果表（seed0 test acc）

| Dataset | Noise | beta=0 soft | beta=0.5 | beta=1 | beta=1 shuffled | beta1-soft | beta1-shuffled |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cora | uniform 0.2 | 0.6153 | 0.6306 | **0.6385** | 0.4976 | +2.32pp | +14.09pp |
| Cora | uniform 0.4 | 0.5314 | 0.5546 | **0.5620** | 0.4095 | +3.06pp | +15.25pp |
| CiteSeer | uniform 0.2 | 0.6121 | **0.6151** | 0.6022 | 0.5159 | -0.99pp | +8.63pp |
| CiteSeer | uniform 0.4 | 0.5344 | **0.5533** | 0.5528 | 0.4076 | +1.85pp | +14.52pp |
| PubMed | uniform 0.2 | 0.8187 | **0.8244** | 0.8121 | 0.7310 | -0.66pp | +8.12pp |
| PubMed | uniform 0.4 | 0.6868 | 0.7224 | **0.7443** | 0.6095 | +5.75pp | +13.48pp |

### 跨数据集模式

- beta=0 没有在任何一个设置上最好，说明当前 soft distribution 不是最优硬度。
- beta=1 在 3/6 设置上最好：Cora 两个噪声率、PubMed uniform0.4。
- beta=0.5 在 3/6 设置上最好：CiteSeer 两个噪声率、PubMed uniform0.2。
- beta=1 相对 beta=0 在 4/6 设置为正，尤其 uniform0.4 上 3/3 为正；但在 CiteSeer/PubMed uniform0.2 为负。
- beta=1 real 相对 beta=1 shuffled 在 6/6 设置为正，delta 为 +8.12pp 到 +15.25pp，说明硬化目标依然依赖真实球结构，不是随机分组收益。

### 决策

本轮支持“目标硬化有价值”，但不支持“固定 beta=1 直接替代主算法”。原因是低噪声 uniform0.2 下 CiteSeer 和 PubMed 更偏向 beta=0.5，而 Cora 全部偏向 beta=1；这正好落在 §4 的“不同数据集/噪声偏好不同 beta”模式。

下一步不应回到软标签，也不应直接把主算法写成 hard majority。建议最小后续：

1. 对 beta=0.5 与 beta=1 做 5-seed 确认，先限 Cora/CiteSeer/PubMed × uniform {0.2,0.4}。
2. 若 5-seed 仍显示 uniform0.4 全部偏向 beta=1、uniform0.2 偏向 beta=0.5，则再做部署期自适应硬度（例如按 noisy margin/purity 选 beta）的最小 smoke。
3. 若 5-seed 只有 beta=1 稳定胜出，则主算法可改为 hard-majority target；若 beta=0.5 更稳，则主算法改为固定半硬目标。

论文 claim 边界：目前只能写“seed0 beta spectrum suggests that harder ball targets can improve weak MLP robustness, but the optimal hardness is dataset/noise dependent”。不能写固定 beta 已经确定，也不能写 hard target 全面优于 soft target。
