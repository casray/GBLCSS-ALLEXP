# MLP-36 审稿防御：LOO、显式平滑与证据口径修订

## 研究假设

当前 LCSS 的球内分布包含节点自身的含噪标签。若局部共识主要来自其他节点，leave-one-out（LOO）后 target 质量与下游精度应保持；若明显下降，则机制应收窄为 group-aware smoothing，而不能只解释为外部邻域纠错。

## 粒球角色

粒球保持为 frozen GRACE-H 上的 label-blind 局部共识单元。本轮不修改分组参数、GRACE、MLP 或 beta，仅改变投票时是否排除当前节点。

## 方法与对照

- `LCSS-current`：现有自票版本，历史锚点。
- `LCSS-LOO`：节点 target 只由同球其他训练节点投票；外部票数少于 2 时退回 noisy one-hot。
- `LCSS-shuffled`：现有随机分组对照。
- `uniform-smoothing`：`(1-epsilon) onehot(y) + epsilon / C`，冻结 `epsilon=0.1`。
- `prior-smoothing`：`(1-epsilon) onehot(y) + epsilon * pi_train_noisy`，冻结 `epsilon=0.1`。

## 协议

- 数据集：Cora、CiteSeer、CoraFull、Coauthor-CS、Amazon-Photo、PubMed。
- 主 smoke：uniform 0.4、train-only、percent 0.2/0.1/0.7、seed 0。
- 复用 frozen H；ROCm Python；GPU `cuda:0`。
- clean label 只用于离线 target-quality 诊断，不进入 target 构造、模型训练或早停。

## 指标

- 下游 test accuracy。
- target Brier、clean-label probability mass gain、target argmax accuracy。
- NCR、CDR、alpha/coverage、current-vs-LOO target L1 差异。
- real LOO vs shuffled、uniform smoothing、prior smoothing。

## Seed0 判定

- LOO 六数据集平均相对 current 不低于 `-0.5pp`。
- 任一核心数据集不下降超过 `1pp`。
- 至少 4/6 数据集 LOO 高于两种显式 smoothing 的较强者。
- 至少 4/6 数据集 LOO 相对 shuffled 为正。

满足后，仅给 LOO 和两种 smoothing 补 seeds 1-4；current/shuffled 按 split/noise hash 复用。未满足则不扩种子，并按 target-quality 证据重写机制。

## 论文修订

- 将 `24/25` 明确标注为经过 uniform-0.4 筛查后完成全谱的数据集上的条件结果。
- 恢复 `[X||H]+Small-loss/JoCoR` 资源对照，claim 收窄到 X-only inference。
- primary families 固定为 LCSS vs CE、shuffled、JoCoR；family 内 Holm 校正并报告 paired 95% CI。
- “相当”改为“数值接近，五种子证据不足以区分”；“上界”改为 H-access reference。
- 修正引用、压缩 Highlights，并单列 preprocessing/training/inference 时间与峰值显存。

## 最终决策规则

- LOO 持平或提升：支持外部局部共识解释，LOO 作为关键消融；是否替换主算法需另行评估重跑成本。
- LOO 略降但仍胜 smoothing/shuffled：表述为局部共识与 group-aware smoothing 的共同作用。
- LOO 明显下降：保留现有算法，但撤回纯外部纠错解释，改写为结构化平滑。

## 实验结果与最终决策（2026-07-15/16）

- 完整性：6 数据集 × 5 方法 × 5 seeds 全部成功；split/noise hash 配对，验证与测试标签零污染。
- LOO 相对 current 的均值增量：Cora `+1.56pp`、CiteSeer `+1.37pp`、CoraFull `-0.15pp`、Coauthor-CS `-0.01pp`、Amazon-Photo `+0.47pp`、PubMed `+0.32pp`；六数据集平均 `+0.60pp`。
- LOO 同时超过 uniform/prior smoothing：`6/6`；LOO 超过 LOO-shuffled：`6/6`，且全部达到至少 `4/5` seeds 正向。
- LOO 的 soft-target Brier 在 `6/6` 数据集改善，但排除自票后目标更激进，NCR 与 CDR 通常同步上升；因此不能写成“LOO 普遍降低误伤”。
- Cora/CiteSeer 的未校正 paired t-test 为 `p=0.024/0.039`，六数据集 family 内 Holm 后均不低于 `0.05`，不作显著性 claim。

**最终决策：** 机制结论为 `external_local_consensus_supported`。LOO 作为关键正向消融写入正文，但不替换默认算法；主方法准确表述为“外部局部共识与组感知平滑的混合”。

附加审稿防御产物：

- 五数据集 × 五噪声设置主比较族 Holm 统计：LCSS vs CE/shuffled/JoCoR 的跨设置平均增量为 `+7.16/+7.83/+2.87pp`，Holm `p<0.05` 为 `24/25、25/25、8/25`。
- seed0 效率：LCSS 推理与同架构 MLP+CE 几乎一致；额外成本集中在一次性构球（约 `3.6--4.5s`）。CoraFull 上当前 MLP 前向慢于 GCN，故只保留“不访问图”的资源 claim，不声称所有数据集必然加速。
- 论文新版本：`docs/paper_final_zh_v13_review_defense_20260715.md`。
