# Stable-GB Reliability：跨视图粒球稳定性驱动的图噪声标签校正诊断与门控

> 文档状态：方法规格 v1，尚未进入实现与新实验。
> 主方向来源：`direction_ranking.md` 的 D2。
> 研究阶段：先做冻结表示上的零训练诊断；只有通过预注册门槛，才允许进入软校正训练。

## 1. 一句话概括

Stable-GB 不把“同球”直接当作正确标签关系，而是把一个节点在多个轻微扰动表示视图中的粒球关系是否稳定，作为“局部标签校正建议是否值得信任”的无监督可靠性信号。

核心问题不是“粒球能否再次检测噪声”，而是：

> 在已有 kNN、模型置信度、loss history、图邻域和单视图粒球统计之后，跨视图粒球稳定性是否仍能增量预测某次局部校正会正确还是会伤害节点？

## 2. 研究定位与边界

### 2.1 为什么从 open-set 转向本方向

原先优先考察的 open-set 局部边界方向已经完成冻结空间 Pilot。结果显示：真实粒球没有稳定优于 KMeans，BANKING77 上甚至显著更差，StackOverflow 上基本完全重合；详见 `pilot_report.md`。同时，PatchCore、cDNP/SupLID、GBDO/GBOC 等工作已覆盖“局部原型/邻域距离用于 OOD”这一大框架。

因此，本轮不继续包装 frozen-space open-set 打分，也不直接进入高成本联合训练。主方向转为文献和现有负结果覆盖较弱的“粒球关系可靠性”：粒球只提供 soft structural confidence，不提供 hard positive、hard negative 或强制 prototype pulling。

### 2.2 本文不声称什么

- 不声称任意粒球划分天然优于 kNN 或 KMeans。
- 不声称冻结表示上的人工扰动等价于真实 GRACE 随机增强视图。
- 不使用 clean label 构造部署时的可靠性分数。
- 不修改 baseline 数据划分、训练协议或评估器。
- 不把单个数据集、单个 seed 或小于标准差的提升写成稳定结论。
- 在诊断门未通过前，不声称 Stable-GB 是有效训练方法。

## 3. 方法小纲要

1. 从现有 GRACE 冻结表示 `H` 出发，构造一组标签无关的轻微特征遮蔽视图。
2. 每个视图独立构建 ordinary granular balls，并为每个节点提取关系持续性、边界稳定性和尺度稳定性。
3. 用单视图粒球生成 leave-one-out 局部标签建议，但不直接采用建议。
4. 在合成噪声 benchmark 中，仅用 clean label 定义“建议正确/错误”的离线诊断真值。
5. 比较基础控制模型与“基础控制 + Stable-GB 特征”的 out-of-fold 增量价值，并与 kNN、KMeans、size-matched shuffle 对照。
6. 只有跨数据集、多噪声 seed 的诊断门通过，才将可靠性转为小权重 soft target mixing；否则对该方向作 No-Go。

## 4. 符号与问题定义

给定图 $G=(V,E)$、冻结节点表示 $H\in\mathbb{R}^{N\times d}$、训练节点集合 $V_L$ 和训练集噪声标签 $\tilde y_i$。clean label $y_i^*$ 只在合成噪声实验的离线评价中可见。

对参考视图构建粒球划分 $a_i^{(0)}$。对训练节点 $i$，在其所在球内排除自身后统计噪声标签分布：

$$
q_i(c)=\frac{\sum_{j\in V_L,\,j\neq i}\mathbb{1}[a_j^{(0)}=a_i^{(0)}]\mathbb{1}[\tilde y_j=c]}
{\sum_{j\in V_L,\,j\neq i}\mathbb{1}[a_j^{(0)}=a_i^{(0)}]}.
$$

若球内除自身外的训练节点少于 `min_train_per_ball - 1`，该节点记为不可建议，不用填零冒充低风险。定义建议类别

$$
\hat c_i=\arg\max_c q_i(c).
$$

主诊断集合只包含发生实际标签改变的节点：

$$
\mathcal C=\{i:\hat c_i\neq \tilde y_i,\;q_i\text{ 可定义}\}.
$$

离线诊断真值为

$$
z_i=\mathbb{1}[\hat c_i=y_i^*],\qquad i\in\mathcal C.
$$

这一定义直接回答“当局部机制想改标签时，它是否改对”，避免把任务退化为一般噪声检测。所有节点上的噪声检测 AUROC 仅作为次要结果。

## 5. 表示视图协议

### 5.1 MVP：冻结 H 的轻微特征遮蔽视图

当前仓库没有可直接复用的“同一已训练 GRACE 编码器在多组随机图增强下导出的表示”。现有 `probe_hspace_multiview_fusion.py` 名称中的 multi-view 指多个检测器的融合，不是增强视图。

因此 MVP 明确采用压力测试视图，而不冒充真实 GRACE view：

- 先对每个节点表示做 L2 normalization。
- 保留原始参考视图 $H^{(0)}=H$。
- 使用 8 个预注册随机种子 `0..7`；每个视图独立将 10% 表示维度置零，再次 L2 normalization。
- 参考视图加 8 个扰动视图，共记为 $M=9$ 个视图。
- 同一组 $H^{(v)}$ 同时供粒球、KMeans 和 kNN 使用，保证视图完全一致。
- 粒球内部二分的随机种子固定为 0，避免把聚类随机性混入表示扰动稳定性；聚类 seed 敏感性另作控制。
- 10% feature masking 与 GRACE 常见的 feature dropping 操作相似，但由于没有重新经过图编码器，只能解释为局部几何压力测试。

不得在看到结果后只保留有利的 masking rate。`p=0.05` 和 `p=0.20` 只允许作为通过主门槛后的敏感性分析，不参与主结果选择。

### 5.2 通过诊断门后的真实视图

若冻结 H 压力测试通过，下一阶段才允许从同一 GRACE checkpoint 导出固定编码器下的真实随机 edge-drop + feature-drop views。此时必须：

- 固定 encoder 权重；
- 保存每个 view 的 augmentation seed 和配置；
- 不用标签选择 view；
- 将“人工 H 扰动结果”和“真实 encoder view 结果”分表报告；
- 若真实 view 复现失败，则人工扰动证据不得升级为方法有效性结论。

## 6. 跨视图粒球可靠性特征

对每个视图独立运行现有 label-blind ordinary granular-ball builder，沿用该数据集已有 `max_balls`、`min_ball_size`、`split_improvement`，不按 clean label 调参。

### 6.1 关系持续性

令 $B_i^{(v)}=\{j:a_j^{(v)}=a_i^{(v)},j\neq i\}$。节点级 co-ball persistence 为全部 $M$ 个视图对之间的平均 Jaccard：

$$
P_i=\frac{2}{M(M-1)}\sum_{0\leq v<u<M}
\frac{|B_i^{(v)}\cap B_i^{(u)}|}{|B_i^{(v)}\cup B_i^{(u)}|}.
$$

空并集按 1 处理，但单例率必须单独报告，防止大量单例虚增稳定性。

### 6.2 边界稳定性

对节点所属球重算中心 $\mu_i^{(v)}$ 与最大半径 $r_i^{(v)}$，定义归一化内部边界裕量：

$$
m_i^{(v)}=\frac{r_i^{(v)}-\|h_i^{(v)}-\mu_i^{(v)}\|_2}{r_i^{(v)}+\epsilon}.
$$

保存 `margin_mean` 与 `margin_std`。稳定且远离局部边界应表现为较高均值、较低标准差。半径为零的单例不直接赋高可靠性，而是标记为 undefined。

### 6.3 尺度稳定性

令 $s_i^{(v)}=|B_i^{(v)}|+1$，记录：

- `log_size_mean`；
- `log_size_std`；
- `singleton_rate`；
- `eligible_view_rate`：满足局部建议最小训练样本数的视图比例。

### 6.4 固定、无监督的主分数

每项先在同一数据集内转为 percentile rank。主分数在看结果前固定为：

$$
R_i^{GB}=\frac{1}{4}\left[
\operatorname{rank}(P_i)+
\operatorname{rank}(\overline m_i)+
1-\operatorname{rank}(\operatorname{std}(m_i))+
1-\operatorname{rank}(\operatorname{std}(\log s_i))
\right].
$$

若任何核心量 undefined，则主分数也设为 missing，并显式报告 coverage；不得用零填充后混入排名。`eligible_view_rate` 和 `singleton_rate` 用于审计，不在主公式中重复加权。

该等权分数是部署候选；使用 clean-label 真值拟合的 logistic regression 只是“这些特征是否包含条件信息”的 oracle diagnostic，不能作为最终方法组件。

## 7. 强基线与负对照

所有比较使用同一数据划分、同一冻结 H、同一组 $M=9$ 个视图和同一候选节点集合。

### 7.1 基础控制特征 M0

- 单视图局部建议置信度：`max(q_i)`、top-1/top-2 margin、球内训练节点数；
- 现有 `s_ball_loo`、center distance、ball entropy；
- kNN 标签怀疑度；
- early-model confidence、entropy、small-loss history；
- 图邻居一致性与节点 degree；
- 数据集与 noise seed 固定效应（跨 seed 模型中）。

基础模型不得含 Stable-GB 跨视图特征。

### 7.2 几何控制

- `M0 + kNN stability`：每个 view 上取固定 k 的邻居集合，计算与 (P_i) 同形式的 Jaccard；k 固定为参考粒球中位大小减 1，并至少为 5。
- `M0 + KMeans stability`：每个 view 的 KMeans 聚类数固定为参考粒球的球数，提取同样的 co-cluster persistence、margin 和 size variation。
- `M0 + GB stability`：加入第 6 节全部 Stable-GB 特征。
- `M0 + all stability`：检验 GB 特征在 kNN/KMeans 稳定性之后是否仍有增量价值。

KMeans 只匹配 cluster count，不能宣称严格 size-matched；实际 cluster-size 分布必须记录。严格 size-matched 对照由 shuffle 完成。

### 7.3 随机负对照

- 对每个 view 独立随机打乱 assignment，同时精确保留该 view 的每个球大小。
- 保持节点数、球数和 size histogram 不变，重算全部稳定性特征。
- 另做 label permutation，仅打乱诊断真值，用于检查 OOF 管道是否泄漏。
- 预期 shuffle 与 label permutation 不产生稳定增益；若它们同样通过，判定管道无效，而不是把真实粒球结果算作成功。

## 8. 诊断模型与无泄漏协议

### 8.1 阶段 A0：可行性与样本量检查

数据集先用 Cora、CiteSeer；noise=`uniform_0.6`，noise seed=0；沿用现有 20%/10%/70% split 和冻结 H。只做以下检查：

- 候选集合 \(\mathcal C\) 数量；
- 正确建议/错误建议数量；
- Stable-GB coverage；
- 球数、大小、半径、单例率；
- 8 个 view 是否可重复生成。

任一数据集若候选少于 50 个、正负类任一少于 10 个或有效 coverage 低于 50%，A0 直接记为“证据不足”，不调门槛救结果。

### 8.2 阶段 A1：seed-0 零训练 kill test

在每个数据集内部做 5-fold stratified out-of-fold logistic regression。StandardScaler 只在训练 fold 拟合；同一节点的 clean/noisy 真值不得跨 fold 泄漏。比较 M0、M0+kNN、M0+KMeans、M0+GB、M0+all。

主指标：候选建议正确性的 AUPRC。
次指标：AUROC、Brier score、log loss、top-30% proposal precision、coverage。
置信区间：以节点为单位的 paired bootstrap，固定 5000 次重采样；报告 seed 和 bootstrap seed。

A1 同时满足以下条件才进入多 seed：

1. Cora 和 CiteSeer 上 `M0+GB` 相对 M0 的 ΔAUPRC 均不低于 +0.02，且 95% paired-bootstrap CI 下界大于 0；
2. `M0+GB` 相对 `max(M0+kNN, M0+KMeans)` 的 ΔAUPRC 在两个数据集均不低于 +0.01；
3. 固定无监督分数 (R_i^{GB}) 的 top-30% proposal precision 相对未筛选候选至少提高 +0.05，两个数据集都成立；
4. size-matched shuffle 的 ΔAUPRC 不超过 +0.005，且真实 GB 至少高于 shuffle +0.02；
5. label permutation 接近随机水平，没有异常正增益。

任何一条失败都停止，不进入训练；可以保留为负结果，但不得在同一 Pilot 中反复换公式。

### 8.3 阶段 A2：多噪声 seed 稳定性

A1 通过后，才扩展 noise seed=`0,1,2,3,4`，仍使用 Cora、CiteSeer 和 `uniform_0.6`。A2 通过条件：

- 每个数据集 mean ΔAUPRC(`M0+GB` vs M0) ≥ +0.015；
- 每个数据集至少 4/5 seeds 为正；
- 按 seed 分层、节点重采样的 hierarchical paired-bootstrap 95% CI 下界 > 0；
- 相对最佳 kNN/KMeans stability 控制，两个数据集的 mean ΔAUPRC 均 ≥ +0.01；
- shuffle 不能表现出相同增益。

这里只能支持“跨视图粒球稳定性包含额外诊断信息”，仍不能支持“训练性能提升”。

## 9. 通过诊断后的软门控训练设计

只有 A2 通过，才进入阶段 B。对候选节点定义保守混合系数：

$$
\alpha_i=\alpha_{max}\cdot R_i^{GB}\cdot \max_c q_i(c),\qquad \alpha_{max}=0.3.
$$

训练目标为

$$
t_i=(1-\alpha_i)\operatorname{onehot}(\tilde y_i)+\alpha_i q_i.
$$

非候选、undefined 或低 coverage 节点保持原噪声标签目标。该设计将粒球限制为弱关系教师：即使可靠性最高，局部分布最多占 30%，不会把 same-ball membership 变成 hard label。

阶段 B 必须至少比较：

- 原始 CE baseline；
- 已有 LCSS / soft-small-loss 基线；
- 无可靠性门控的固定-alpha 局部软目标；
- confidence-only 门控；
- kNN-stability 门控；
- KMeans-stability 门控；
- Stable-GB 门控；
- Stable-GB 去掉 persistence、margin stability、size stability 的逐项消融；
- size-matched shuffled-GB 门控。

训练阶段保持原 split、evaluation、epoch budget、early stopping 和 seed 设置。成功必须看 mean±std、paired seed delta 和置信区间，不以最佳 seed 或最佳 epoch 个例判断。

## 10. 实现隔离与可追溯输出

后续实现只允许新建：

```text
next_paper/
  stable_gb_reliability/
    configs/
    src/
    tests/
  runs/
    stable_gb_reliability_<timestamp>/
```

现有 `external/NoisyGL`、第一篇论文代码、checkpoint、logs 和 results 一律只读。允许通过明确 import 或复制最小必要函数复用以下已核实能力：

- `external/NoisyGL/utils/ordinary_gb.py`：label-blind ordinary GB 与 exact size-matched shuffled assignment；
- `external/NoisyGL/scripts/diagnose_label_noise_detection.py`：已有 LOO 粒球分数、kNN、early GCN、small-loss、图邻居控制；
- 现有 frozen-H artifacts 和 MLP34 trace：只读输入，不覆盖。

每次运行必须写出：

- 完整 `config.json`；
- 输入 artifact 路径及 SHA-256 或大小/mtime 指纹；
- view seeds、noise seeds、split；
- 节点级 `features.csv` 或压缩 NPZ；
- fold 级和 seed 级 raw metrics；
- bootstrap 原始摘要与 CI；
- `summary.csv`、`statistics.json`、`run.log`；
- 自动生成的 success/failure gate report。

## 11. 预期机制与可证伪性

### 11.1 预期机制

若一个局部关系只在单次划分中出现，它更可能来自表示边界抖动、球尺度偶然性或局部混类；若相同邻域关系在多个轻微视图中持续、节点始终位于球内部且尺度稳定，则该局部建议可能更可信。Stable-GB 的价值应体现为在强控制量之后对 proposal correctness 的增量预测，而不是再次展示粒球纯度或噪声检测相关性。

### 11.2 直接证伪条件

- GB 稳定性不优于 kNN/KMeans 稳定性；
- 增益被单视图球大小、模型置信度或 loss history 完全解释；
- shuffle assignment 取得相似增益；
- 只在 Cora 或单个 seed 有效；
- 人工 H 扰动有效，但真实 GRACE views 不复现；
- 诊断有效，但软门控训练没有稳定超过现有 LCSS / soft-small-loss；
- 改善小于 seed std 或 paired CI 跨 0。

满足任一核心证伪条件时，应记录为 No-Go 或降级为普通稳定性分析，不再通过换阈值、换视图 seed 或删数据集救方向。

## 12. Claim–evidence 对照表

| 潜在 claim | 最低所需证据 | 当前状态 |
|---|---|---|
| 跨视图粒球稳定性与局部校正正确性相关 | A1 两数据集、无泄漏 OOF、bootstrap CI、shuffle control | 尚未验证 |
| 该信号不同于一般局部几何稳定性 | A2 中稳定超过 kNN/KMeans controls | 尚未验证 |
| 该信号跨噪声随机性稳定 | 2 数据集 × 5 noise seeds，mean±std 与 hierarchical CI | 尚未验证 |
| Stable-GB 能改善噪声图学习 | 阶段 B 同协议多 seed，超过 CE、LCSS、soft-small-loss 与控制门控 | 尚未验证 |
| 真实 GRACE augmentation 下机制成立 | 固定 encoder 的真实随机增强视图复现 | 尚未验证 |
| 方法可泛化到其他噪声类型/数据集 | uniform 外至少一种噪声与第三数据集 | 尚未验证，不能提前写入摘要 |

## 13. 当前决策

当前决策为：**方法规格已形成，并于 2026-07-14 获用户确认；实验状态仍为尚未验证。**

`experiment_plan.md` 记录的是本目标中已经执行的 frozen open-set Pilot 及其 No-Go 审计，不是 Stable-GB 的正证据。若后续启动 Stable-GB A0/A1，必须另建可执行计划与隔离目录，从零训练诊断开始；不得把 open-set Pilot、人工 feature-masking 设计或本次用户确认写成方法有效性结论。
