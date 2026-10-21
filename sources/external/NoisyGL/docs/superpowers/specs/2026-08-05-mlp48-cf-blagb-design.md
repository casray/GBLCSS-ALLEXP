# MLP-48 CF-BLAGB 小幅创新设计规格

## 状态

- 日期：2026-08-05
- 状态：设计待用户确认，尚未实现、尚未运行
- 候选名称：Cross-Fitted Budgeted Label-Assisted Granular Balls（CF-BLAGB）

## 1. 研究问题

正式 GB-LCSS 的粒球完全由冻结 GRACE 表征中的几何关系生成。MLP-46 CF-NAGB 尝试用 20% 含噪训练标签干预分裂是否成立，但候选树从 51/67/128 个叶节点收缩到中位数 7/5/3 个球，导致目标质量明显下降。该结果说明含噪标签并非完全没有分组信号，但不适合充当几何分裂的硬门控。

本实验只回答一个更窄的问题：

> 在保持球数预算和其他 GB-LCSS 部件不变时，能否用跨折含噪标签重新排序几何候选分裂，使有限球预算优先用于标签分布差异更明显的区域？

## 2. 设计目标与非目标

### 2.1 设计目标

1. 只修改第二阶段的粒球生成顺序。
2. 标签只提供有界的分裂优先级修正，不决定某个区域能否继续分裂。
3. 每个数据集和种子的最终球数与正式几何方法一致。
4. 通过五折 cross-fitting，阻止节点自身标签参与自身分区和局部投票。
5. 用离线目标质量诊断先判断方向，不直接扩大为完整下游实验。

### 2.2 非目标

- 不修改 GRACE 预训练或冻结表征 H。
- 不修改 LOO、tie-safe `distribution_only`、beta=0.5、alpha、soft CE 或 X-only MLP。
- 不使用 clean label 选规则、调权重或决定是否通过。
- 不搜索新的球数、最小球规模、几何阈值或标签权重。
- 不把一次 seed0 结果写成论文结论。

## 3. 冻结不变量

以下设置与当前正式 LOO-GB-LCSS 完全一致：

- 表征：逐行 L2 归一化的 frozen GRACE H。
- 噪声：仅污染训练标签。
- 软目标：LOO + tie-safe `distribution_only`。
- beta：0.5。
- alpha：沿用正式节点相关公式，不增加票数感知项。
- 学生：相同 X-only MLP、优化器、split 和评估协议。
- 几何基础：递归 2-means 候选分裂。

## 4. 方法设计

### 4.1 五折隔离

将训练节点按 seed 确定性划分为五折。对第 s 折中的目标节点，使用其余四折的含噪训练标签：

- 计算候选子球的标签后验分布；
- 决定候选分裂的优先级；
- 构造该目标节点的局部共识分布。

第 s 折标签不参与第 s 折节点的分区引导或局部投票。节点原始含噪标签仍只作为正式软目标外层混合项的一部分，这与当前 LOO-GB-LCSS 一致。

### 4.2 标签无关候选树

每一折共享一棵在 H 上构造的标签无关递归 2-means 候选树：

1. 从全部节点构成的根节点开始。
2. 首先严格复现正式几何方法的分裂顺序、随机种子和 `split_improvement=0.05`，将其终止分区作为候选树前缀；该前缀的叶节点数必须等于 K_geo。
3. 从正式前缀继续向下扩展时，只依据几何和 `min_ball_size=32` 生成候选分裂，不再使用几何增益阈值提前停止。
4. 不使用含噪标签决定候选分裂是否生成或保留。
5. 候选树最多生成 128 个叶节点，且必须能够提供至少 K_geo 个叶节点。

其中 K_geo 是相同数据集、相同 seed 下正式几何 GB-LCSS 得到的球数。若候选树因最小球规模限制无法达到 K_geo，则该设置显式记为 `not_constructible`，不得通过修改参数或回退规则补救。

该前缀约束是实现期发现的必要一致性修正。通用 MLP-46 候选树使用不同的节点编号随机种子，在 Cora 上只能生成 51 个叶节点，而正式方法生成 52 个球。不得用降低 `min_ball_size` 的方式弥补该差异；候选树必须先复现正式几何层级，再扩展更深候选。

### 4.3 几何增益

对候选分裂 a -> (l, r)，定义归一化几何紧致度增益：

\[
G(a)=\max\left(0,
\frac{D(a)-\frac{|l|D(l)+|r|D(r)}{|a|}}
{\max(D(a),\varepsilon)}
\right),
\]

其中 D(.) 为节点到所属中心的平均距离。G(a) 仅依赖 H。

### 4.4 跨折标签分布

用其余四折中落入子球的含噪训练标签计数 n_l、n_r 估计子球类别分布。采用固定总强度为 2 的 Dirichlet 先验：

\[
\pi_c=\frac{n_c^{global}+1}{\sum_r n_r^{global}+C},
\qquad
\gamma_c=2\pi_c,
\]

\[
\bar p_l(c)=\frac{n_{l,c}+\gamma_c}{N_l+2},
\qquad
\bar p_r(c)=\frac{n_{r,c}+\gamma_c}{N_r+2}.
\]

该先验只缓和稀疏票数，不新增可调参数。clean label 仅用于 oracle 诊断，不进入正式构造。

### 4.5 有界标签辅助分数

候选分裂得分定义为：

\[
S(a)=G(a)\left(1+
\frac{\operatorname{JSD}(\bar p_l,\bar p_r)}{\log 2}
\right).
\]

由于归一化 JSD 位于 [0,1]，因此：

\[
G(a)\le S(a)\le 2G(a).
\]

标签信号最多将几何优先级放大一倍，不能把零几何收益的分裂变为有效分裂，也不能阻止其他候选继续分裂。

### 4.6 固定预算切树

从根节点开始，每次从当前可分裂叶节点中选择 S(a) 最大者进行分裂，直到恰好得到 K_geo 个叶节点。并列时仅使用确定性的几何键和节点索引排序，不读取类别编号。

因此，CF-BLAGB 与正式方法具有相同的最终球数预算。它改变的是“预算优先分给哪些区域”，而不是“是否缩减预算”。

### 4.7 软目标构造

每一折得到一套分区。对该折目标节点 v：

1. 找到 v 所属球 B_{j,-v}。
2. 只用其他四折的训练节点统计 p_{j,-v}。
3. 按正式 tie-safe LOO 规则得到 q_{j,-v} 和 alpha_{j,-v}。
4. 构造：

\[
t_v=(1-\alpha_{j,-v})e_{\tilde y_v}
+\alpha_{j,-v}q_{j,-v}.
\]

若外部有效票数不足，沿用正式 LOO-GB-LCSS 的 fallback，不新增第二套规则。

## 5. Phase A：离线目标质量 smoke

### 5.1 固定协议

- 数据集：Cora、CiteSeer、CoraFull。
- 噪声：uniform 0.4，train-only。
- Split：percent 0.2/0.1/0.7。
- Seed：0。
- 不训练下游分类器。

### 5.2 实验臂

1. 正式 LOO-GB-LCSS：当前主方法锚点。
2. Cross-fit formal GB：正式几何分区，但用相同五折票源构造目标，用于隔离 cross-fitting 本身的影响。
3. Fixed-budget geometry-only：使用同一候选树和 K_geo，只按 G(a) 切树。
4. CF-BLAGB：按 S(a) 切树。
5. CF-BLAGB shuffled guidance：保持折、类别频数、几何和 K_geo 不变，打乱训练标签与节点的对应关系。
6. Clean-guidance oracle：仅诊断可用上限，不参与判定或论文性能比较。
7. Cross-fit adaptive-k：同票源隔离下的强局部邻域参照，不用于选择标签权重。

### 5.3 报告指标

- target accuracy。
- target Brier score。
- NCR、CDR 和 coverage。
- K_geo、实际球数、每球训练票数分布。
- 与 geometry-only、shuffled、formal anchor 的逐数据集差值。
- clean oracle headroom。
- 构造时间和异常状态。

### 5.4 进入 Phase B 的门槛

以下条件全部满足才训练 seed0 下游分类器：

1. 三个数据集的五折分区均恰好产生 K_geo 个球。
2. CF-BLAGB 相对 fixed-budget geometry-only 和 shuffled guidance，在至少 2/3 数据集上同时降低 Brier。
3. CF-BLAGB 相对 cross-fit formal GB 的三数据集平均 target accuracy 增量不为负。
4. 任一数据集 target accuracy 回退不超过 1.0pp。
5. Clean-guidance oracle 相对 cross-fit formal GB 在至少 2/3 数据集存在正 headroom。
6. 无标签泄漏、NaN、空目标或单球塌缩。

若未通过，不修改先验强度、标签乘子、折数、K_geo 或第二版评分公式。该方向关闭，结论为“含噪标签不适合进入当前粒球构造”。

## 6. Phase B：seed0 下游 smoke

仅 Phase A 通过后运行。沿用相同三个数据集和 uniform 0.4 seed0，比较：

- 正式 LOO-GB-LCSS。
- Cross-fit formal GB。
- Fixed-budget geometry-only。
- CF-BLAGB。
- CF-BLAGB shuffled guidance。
- Cross-fit adaptive-k。

进入五种子的条件：

1. CF-BLAGB 至少在 2/3 数据集超过 cross-fit formal GB。
2. 三数据集平均增量至少 +0.5pp。
3. 任一数据集不得下降超过 1.0pp。
4. CF-BLAGB 相对 shuffled guidance 至少在 2/3 数据集领先 1.0pp。
5. 所有行 split/noise hash 一致，无标签泄漏或预测坍缩。

## 7. Phase C：五种子确认

仅 Phase B 通过后补 seeds 1-4。正式判据为：

1. 至少 2/3 数据集 CF-BLAGB 相对 cross-fit formal GB 的均值为正，且各数据集至少 4/5 seeds 同向。
2. 三数据集平均增量至少 +0.5pp。
3. 任一数据集均值回退不超过 0.5pp。
4. 至少 2/3 数据集 real-shuffled 均值差不低于 1.0pp，且至少 4/5 seeds 同向。

五种子通过只说明该小改动值得进一步扩谱，不直接替换第一篇论文的正式算法。

## 8. 测试要求

实现前必须先写测试，至少覆盖：

1. 候选树构造不读取任何标签。
2. 每折最终球数精确等于 K_geo。
3. 修改 held-out 折标签不改变该折节点的分区或局部票分布。
4. 修改 val/test 标签不改变分区、目标或训练输入。
5. JSD 对称、有限，输入相等时为 0。
6. 对全部候选满足 G(a) <= S(a) <= 2G(a)。
7. geometry-only 等价于令标签乘子恒为 1。
8. shuffled guidance 保持折划分和类别直方图不变。
9. 所有 soft target 非负且逐行和为 1。
10. tie-safe 并列结果不依赖类别编号的先后顺序。

## 9. 预期文件边界

通过书面规格确认后，实施计划预计只涉及：

- 扩展 `utils/noise_aware_gb.py` 或新增独立 CF-BLAGB 模块。
- 新增 `scripts/run_mlp48_cf_blagb.py`。
- 新增 `tests/test_cf_blagb.py` 与 smoke 汇总测试。
- 新增 `runs/mlp48_cf_blagb_20260805/`，保存 raw、aggregate、gate、report 和配置快照。

不得修改正式 predictor 配置或覆盖既有 MLP-46 结果。

## 10. 结论边界与止损

- 成功时只能写：跨折含噪标签可在固定球预算下改善几何分裂优先级。
- 不得写：标签辅助粒球普遍优于 kNN，或找到最优粒球划分。
- 若仅 clean oracle 有效，则说明表征中存在可利用上限，但当前 noisy guidance 不足。
- 若 real 与 shuffled 相当，则说明收益不能归因于真实标签结构。
- 若失败，关闭“直接用含噪标签改进粒球构造”方向，不继续调乘子、阈值或折数。后续若使用去噪教师引导，应作为独立的新方法研究。
