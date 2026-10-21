# MLP-46：交叉拟合噪声感知粒球实验计划

## 1. 目标

设计 **Cross-Fitted Noise-Aware Granular Ball，CF-NAGB**。方法仍以冻结 GRACE 表征 \(H\) 的几何结构为主，只使用 20% 含噪训练标签辅助选择粒球层次。

核心目标为：

\[
\operatorname{Acc}(\text{CF-NAGB-LCSS})
>
\operatorname{Acc}(\text{exact kNN}, k=20).
\]

只有在相同信息预算下同时超过固定 \(k=20\) 和噪声感知 adaptive-k，才能声称粒球划分具有独立精度优势。

当前正式方法锚点不得修改：

- `leave_one_out=true`
- `tie_mode=distribution_only`
- `target_mode=distribution`
- \(\beta=0.5\)
- X-only MLP
- train-only noise
- percent split `0.2/0.1/0.7`

### 1.1 历史先验与本轮边界

E1 已经证明，直接用稀疏含噪标签的纯度/覆盖率控制粒球分裂不稳定：在 20% 标签覆盖和 40% 标签噪声下，单球通常只有约 5--15 张有效票，标签纯度的抽样波动足以改变分裂决策。MLP-45 还表明，正式粒球与精确 kNN 的精度通常接近，CoraFull 上粒球存在明显差距。

因此，本轮不是对“标签纯度自适应球”的再次调参，而是只检验一个新的、可证伪的假设：

> 交叉拟合能否消除节点自标签对自身分区的反馈，并且贝叶斯分离证据能否在稀疏含噪票下比固定几何切分产生更可靠的监督目标。

本轮按高门槛 kill-smoke 执行。Phase A 离线目标质量不过门时直接关闭方向，不通过下游精度反复调整候选树深度、最小票数、Bayes factor 或先验强度。

## 2. 方法思想

### 2.1 构造纯几何候选树

读取冻结 GRACE \(H\)，逐行进行 \(L_2\) 归一化。递归 2-means 生成候选树：

- `candidate_max_leaves=128`
- `min_ball_size=32`
- `min_geometry_gain=0.01`
- 候选边界完全不读取标签
- 保存完整父子关系、几何增益和节点 assignment

这一步只产生可选择的几何层次，不直接确定最终粒球。

### 2.2 五折交叉拟合

将 20% 训练节点按含噪类别分层划为 5 折。对第 \(f\) 折节点构造目标时：

- 第 \(f\) 折标签不参与粒球剪枝
- 第 \(f\) 折标签不参与球内投票
- 只使用其余 4 折含噪训练标签
- 节点几何表征仍可参与纯几何候选树

因此，节点自己的含噪标签不会参与确定自己的局部共识。

### 2.3 标签后验

对候选子球 \(B_L,B_R\)，使用构造折标签得到计数 \(\boldsymbol n_L,\boldsymbol n_R\)。采用经验先验：

\[
\pi_c=\frac{N_c+1}{N+C},\qquad
\alpha_c=2\pi_c.
\]

后验均值为：

\[
\bar p_{kc}
=
\frac{n_{kc}+\alpha_c}
{N_k+\sum_c\alpha_c}.
\]

不得直接使用原始标签纯度控制分裂。

### 2.4 贝叶斯剪枝

Dirichlet-multinomial 对数边际似然为：

\[
\log p(\boldsymbol n\mid\boldsymbol\alpha)
=
\log\Gamma(A)-\log\Gamma(A+N)
+\sum_c
\left[
\log\Gamma(\alpha_c+n_c)-\log\Gamma(\alpha_c)
\right].
\]

分离证据为：

\[
\log BF
=
\log p(\boldsymbol n_L\mid\boldsymbol\alpha)
+\log p(\boldsymbol n_R\mid\boldsymbol\alpha)
-\log p(\boldsymbol n_L+\boldsymbol n_R\mid\boldsymbol\alpha).
\]

保留候选分裂必须同时满足：

- 两个子球均至少有 5 个构造折标签
- 候选分裂几何增益不低于 `0.01`
- \(\log BF\ge\log 3\)

否则合并为父球。所有阈值全程冻结，不按数据集或测试精度调整。

### 2.5 软目标

对 held-out 折中的训练节点 \(i\)，找到其折特定粒球，使用其他 4 折标签计算局部分布 \(\boldsymbol q_i\)。随后完全复用正式 GB-LCSS 的：

- alpha 公式
- \(\beta=0.5\) 半硬目标
- `distribution_only` 并列处理
- X-only MLP 与 soft cross-entropy

最终目标仍保留节点自身 noisy one-hot 的外层混合，但自身标签不进入分区和局部分布。

## 3. 实验臂

| 方法 | 作用 |
|---|---|
| Formal LOO GB-LCSS | 当前正式锚点 |
| Cross-fit formal GB | 固定正式几何分区，但每折只用其他 4 折标签投票；隔离交叉拟合本身的影响 |
| Exact kNN-20 LOO | 主要目标对照，必须排除自身 |
| Cross-fit kNN-20 | 与新方法相同的折外投票预算 |
| Cross-fit adaptive-k | 同样使用标签辅助选择邻域尺度 |
| CF-NAGB-LCSS | 新方法 |
| CF-NAGB shuffled-guidance | 只打乱剪枝标签，最终投票仍用真实 noisy labels |
| Fine-tree cross-fit | 保留候选树最深合法切分，不使用标签剪枝 |
| Clean-guidance oracle | 仅在 Phase A 用 clean train labels 剪枝，目标仍按折外 noisy labels 构造；只诊断标签引导分层是否存在上限，不参与训练、调参或正式比较 |

Adaptive-k 使用：

\[
k\in\{5,10,20,40\}.
\]

对每个 \(k\) 计算多数类后验下置信分数：

\[
R_i(k)
=
q_{\max}^{(k)}
-
\sqrt{
\frac{q_{\max}^{(k)}(1-q_{\max}^{(k)})}
{k+2}
}.
\]

选择 \(R_i(k)\) 最大的 \(k\)，并列时选择较大 \(k\)。它同样只能读取其他 4 折含噪训练标签。

所有交叉拟合臂必须复用完全相同的 5 折划分。Adaptive-k 中的 \(k\) 表示可用的“其他 4 折训练节点”邻居数。若可用节点不足则取全部可用节点，不允许回填 held-out 折节点。

`shuffled-guidance` 只在每个构造折内部置换剪枝标签，保持折划分和类别直方图不变；最终局部分布仍由未打乱的折外 noisy labels 计算。

## 4. 实施修改

新增：

- `utils/noise_aware_gb.py`
- `predictor/CFNAGB_MLP_Predictor.py`
- `scripts/run_mlp46_cf_nagb.py`
- `scripts/summarize_mlp46_cf_nagb.py`
- `tests/test_noise_aware_gb.py`
- `tests/test_mlp46_cf_nagb.py`
- `config/cf_nagb_mlp_softdist_beta05_loo/*.yaml`

在 `scripts/run_standard_downstream.py` 注册：

- `cf_nagb_mlp_softdist_beta05_loo`
- `cf_nagb_mlp_softdist_beta05_loo_shuffled_guidance`
- `knn20_crossfit_softdist_beta05`
- `adaptive_knn_crossfit_softdist_beta05`
- `fine_tree_crossfit_softdist_beta05`

不得覆盖现有 predictor、YAML、正式结果或 frozen \(H\)。

## 5. Phase A：单元测试与离线诊断

先运行 Cora、CiteSeer、CoraFull，uniform 0.4，seed 0，不训练分类器。

必须验证：

- 几何候选树在改变任意标签后逐元素不变
- held-out 折标签不影响该折的剪枝和局部分布
- val/test 标签完全不参与
- shuffled guidance 只改变剪枝，不改变候选树
- 每个训练节点恰好获得一个交叉拟合目标
- 所有 target 行和为 1，无 NaN
- 中位最终球数位于 `[8,96]`
- 至少 80% 训练节点获得非零 alpha
- 记录 target accuracy、Brier、NCR、CDR，但 clean label 只用于诊断
- clean-guidance oracle 与 noisy-guidance 使用完全相同的候选树、折划分和冻结阈值

出现单球塌缩、满树不剪枝或覆盖不足时直接停止，不调整第二套阈值。

### 5.1 Phase A 离线止损门

Phase A 只有同时满足以下条件才进入下游训练：

1. CF-NAGB 相对 `Cross-fit formal GB` 和 `shuffled-guidance`，在至少 `2/3` 数据集上使 target accuracy 或 NCR 升高，或使 Brier 或 CDR 降低，且同一数据集没有任一指标沿不利方向恶化超过 1pp。
2. CF-NAGB 相对 `Cross-fit formal GB` 的三数据集平均 Brier 与 CDR不升高，平均 target accuracy 与 NCR 不降低。
3. clean-guidance oracle 在至少 `2/3` 数据集上相对 `Cross-fit formal GB` 改善至少一项目标质量指标。若 clean oracle 也无改善，说明候选树层次本身没有可利用余量，直接永久关闭标签辅助构球方向。
4. 真实 noisy guidance 必须优于 shuffled guidance；仅 oracle 有效而 noisy guidance 无效时，结论为“存在理想标签上限，但 20% 含噪标签不足以实现”，停止下游训练。

上述 clean-label 指标仅执行预注册判定，不用于改变任何参数。Phase A 输出所有逐数据集指标和方向计数，不能只报告通过项。

## 6. Phase B：seed 0 下游 smoke

数据集：

- Cora
- CiteSeer
- CoraFull

协议：

- uniform 0.4
- seed 0
- train-only noise
- split `0.2/0.1/0.7`
- ROCm `cuda:0`

进入 5-seed 的条件：

- CF-NAGB 相对 exact kNN-20 至少 `2/3` 数据集为正
- 三数据集平均增量至少 `+0.5pp`
- 任一数据集不得下降超过 `1pp`
- CF-NAGB 相对 shuffled-guidance 至少 `2/3` 数据集高 `1pp`
- CF-NAGB 相对 Cross-fit formal GB 至少 `2/3` 数据集为正
- 无泄漏、坍缩或协议错误

## 7. Phase C：三数据集五种子终判

补 seeds `1-4`，所有实验臂均运行。

主成功门：

- CF-NAGB − exact kNN-20：至少 `2/3` 数据集 `>=+1pp`
- 通过数据集至少 `4/5 seeds` 正向
- 三数据集平均增量 `>=+1pp`
- CF-NAGB − adaptive-k：至少 `2/3` 数据集为正
- CF-NAGB − shuffled-guidance：至少 `2/3` 数据集 `>=+1pp` 且 `4/5 seeds` 正向
- 相对正式 GB-LCSS 至少 `2/3` 数据集为正
- target NCR、CDR 或 Brier 至少有一项同步改善

## 8. Phase D：冻结后外部验证

Phase C 通过后，不修改任何参数，在以下数据集运行 uniform 0.4、seeds 0-4：

- Coauthor-CS
- Amazon-Photo

总体升级条件：

- 五个数据集中至少 4 个相对 exact kNN-20 为正
- 至少 3 个数据集领先 `>=1pp`
- 相对 adaptive-k 至少 3 个数据集为正
- 不允许任何数据集下降超过 `1pp`
- real guidance 相对 shuffled guidance 至少 4 个数据集为正

通过后才扩展 uniform `{0.2,0.6}`、pair `0.4` 和 instance `0.4`。

## 9. 输出

输出目录：

`runs/mlp46_cf_nagb_20260805/`

必须生成：

- `candidate_tree_diagnostics.csv`
- `crossfit_partition_diagnostics.csv`
- `target_quality.csv`
- `seed0_smoke.csv`
- `five_seed_raw.csv`
- `aggregate.csv`
- `paired_deltas.csv`
- `paired_tests.csv`
- `protocol_audit.csv`
- `completeness.csv`
- `mlp46_report.md`

报告必须逐项给出相对 fixed kNN-20、cross-fit kNN-20、adaptive-k、正式 GB-LCSS 和 shuffled guidance 的差值。
Phase A 还必须单独报告相对 cross-fit formal GB 与 clean-guidance oracle 的目标质量差值。

## 10. 结果解释

- **超过 fixed kNN-20，但不超过 adaptive-k**：新方法有效，但不能声称粒球优于同信息预算 kNN。
- **同时超过 fixed 和 adaptive-k**：粒球获得独立精度证据，可升级为核心机制。
- **只超过 shuffled guidance**：标签辅助真实有效，但仍被 kNN 吸收。
- **不超过当前 GB-LCSS**：关闭标签辅助构球方向。
- **高噪声失败、低中噪声成功**：定位为中等噪声下的标签辅助粒球，不继续追求全噪声统一优势。

## 11. 红线

- 不使用 clean label、val label 或 test label 决定分裂、阈值或超参数
- 不使用测试结果选择 Bayes factor 阈值
- 不降低 kNN 对照的 \(k\) 以制造优势
- exact kNN 必须屏蔽自身节点
- 不允许同一节点标签同时参与自己的剪枝和局部投票
- 不覆盖现有 GB-LCSS 正式方法
- 若未超过 adaptive-k，不得写“粒球优于 kNN”

本计划的核心检验不是“标签能否帮助粒球”，而是：

> 在相同冻结表征、相同含噪标签和相同交叉拟合预算下，区域级噪声感知粒球是否比逐节点近邻更适合构造鲁棒软监督。

## 12. 执行结果与最终决策（2026-08-05）

Phase A 已按冻结计划执行，覆盖 Cora、CiteSeer、CoraFull 的 uniform 0.4、seed 0。9 个单元/判定测试通过，三数据集的 split/noise fingerprint 与 train-only 污染审计均已保存。

### 12.1 关键结果

- 纯几何候选树分别产生 51、67、128 个叶节点；正式固定几何球分别为 52、64、64 个。
- noisy guidance 剪枝后的中位球数仅为 7、5、3，三个数据集全部低于预注册下限 8，判为分区塌缩。
- 相对同折预算的 `Cross-fit formal GB`，CF-NAGB 的三数据集平均 target accuracy 下降 3.36pp，Brier 增加 0.0727，NCR 下降 10.49pp；仅 CDR 因几乎不再改变 argmax 标签而下降 1.99pp。
- CF-NAGB 相对 shuffled guidance 在 3/3 数据集降低 Brier，但两者的 target accuracy、NCR 和 CDR 基本相同。真实 guidance 只是把全局一球平滑收缩为少数粗球，没有产生有效噪声纠正。
- clean-guidance oracle 的中位球数为 11、14、72，但只在 CoraFull 的目标质量上显示有限上限；预注册 oracle headroom 门仅为 1/3，说明失败不只来自 noisy labels，候选层次与当前 Bayes 剪枝准则也没有跨数据集稳定余量。
- `Cross-fit formal GB` 没有塌缩，说明主要失败源不是交叉拟合本身，而是 label-guided pruning。Cross-fit adaptive-k 在目标准确率/Brier 上较强，但 CoraFull 的 CDR 明显升高；该结果只作离线诊断，不外推为下游结论。

### 12.2 预注册判定

- geometry gate：0/3，通过失败。
- noisy guidance vs cross-fit formal：0/3，通过失败。
- noisy guidance vs shuffled guidance：3/3，仅 Brier 改善。
- clean oracle headroom：1/3，通过失败。
- 总判定：`geometry_failed`，Phase A 不通过。

### 12.3 决策

不实现 predictor，不运行 Phase B/C/D，不调整 `min_guidance_per_child`、Bayes factor、prior strength 或候选树参数。当前 CF-NAGB 定义永久关闭；结论限定为：在 20% 含噪训练标签与五折交叉拟合预算下，使用 Dirichlet-multinomial 分离证据剪枝几何球会过度合并区域，目标质量显著弱于固定几何分区，不能作为正式 GB-LCSS 的升级方案。

完整结果：`runs/mlp46_cf_nagb_20260805/mlp46_report.md`。
