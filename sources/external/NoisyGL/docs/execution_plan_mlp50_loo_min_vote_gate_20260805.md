# MLP-50: LOO 粒球最小有效票数门控

## 研究假设

正式 LOO-GB-LCSS 在删除节点自身投票后，只要求至少两张其他训练票。少量投票形成的局部多数可能具有较高方差。若仅在至少五张其他训练票时启用粒球软目标，可能减少错误修正，同时保留足够的监督覆盖率。

## 粒球角色

粒球仍由 frozen GRACE H 上的递归 2-means 无标签构造。训练标签不参与球的生成、分裂或合并，只决定某个训练节点是否拥有足够的留一投票来启用局部共识目标。

## 冻结设计

- 数据集：Cora、CiteSeer、CoraFull。
- 噪声：uniform 0.4/0.6，train-only。
- 划分：percent 0.2/0.1/0.7。
- Seed：0。
- 正式基线：`s_min=2`，`tie_mode=distribution_only`，`leave_one_out=true`。
- 唯一候选：`s_min=5`。
- 压力臂：`s_min=10`，只用于判断门槛过严时的覆盖率损失，不得被事后选为新默认。
- 每个门槛均构造 real 与 size-matched shuffled 两臂，共 `3 x 2 x 3 x 2 = 36` 条目标诊断记录。

这里的 `s_min` 是删除目标节点自身标签后剩余的其他训练票数。实现中 LOO 会要求球内训练节点总数至少为 `s_min+1`。

## Phase A 通过条件

`s_min=5` 必须同时满足：

1. 全部 36 条记录完整，val/test 标签零污染，协议哈希可追溯。
2. 相比 real `s_min=2`，Brier score 在至少 4/6 个单元下降，六单元平均下降至少 0.005。
3. 目标准确率六单元平均提高至少 0.5pp，且任一单元不得下降超过 1pp。
4. 每个单元的训练目标覆盖率不低于 70%。
5. real `s_min=5` 的 Brier score 在至少 4/6 个单元优于 shuffled `s_min=5`。

任一条件不满足即停止，不训练下游模型，不调整第二个门槛。

## Phase B 条件式下游验证

仅当 Phase A 通过时运行 real `s_min=5` 与 shuffled `s_min=5` 的 seed0 下游，并复用正式 `s_min=2` anchor。进入多种子的条件为：

- real `s_min=5` 相比正式 anchor 的六单元平均提升至少 0.5pp；
- 至少 4/6 单元正向，任一单元回退不超过 1pp；
- real 相比 shuffled 至少 4/6 单元领先 1pp。

## 输出

输出目录为 `runs/mlp50_loo_min_vote_gate_20260805/`，包括：

- `phase_a_raw.csv`
- `support_diagnostics.csv`
- `phase_a_gate.json`
- `mlp50_report.md`
- 条件式 Phase B 文件

## 最终决策

- Phase A 失败：保留正式 `s_min=2`，关闭“忽略少票球”作为当前论文改进方向。
- Phase A 通过、Phase B 失败：少票过滤改善目标但不能转化为分类收益，保留为诊断。
- 两阶段均通过：再制定独立 5-seed 计划，不在本轮直接替换正式方法。

## 实验结果（2026-08-05）

- Phase A 完成 `36/36` 条目标诊断记录，协议完整性与 train-only 噪声检查均通过。
- 三个数据集、两个噪声率的真实粒球中，零训练票粒球均为 `0`。在当前 20% 随机训练覆盖和固定粒球尺度下，“整球没有已知标签”没有形成实际问题。
- `s_min=5` 相比正式 `s_min=2` 的 Brier score 在 `0/6` 单元改善，六单元平均变差 `+0.003886`。
- 目标准确率平均仅提高 `+0.137pp`，低于预注册的 `+0.5pp`；Cora/CiteSeer/CoraFull 的覆盖率仍为 `96.67%/97.74%/99.90%`，说明失败不是大规模丢失监督，而是硬过滤没有提高软目标质量。
- real `s_min=5` 在 Brier 上仍然 `6/6` 优于 size-matched shuffled，证明局部几何信号存在；问题在于正式 LOO 已经通过连续 alpha 对少票共识进行了足够保守的利用。
- `s_min=10` 在 Cora/CiteSeer 将覆盖率降至约 `56%/53%`，Brier 明显恶化，确认更严格门槛会过度丢弃局部监督。

**最终判定：Phase A 预注册门失败（`brier_failed`）。不运行 Phase B，不调整第二个阈值，正式方法保持 `s_min=2`。**

完整证据：`runs/mlp50_loo_min_vote_gate_20260805/mlp50_report.md` 与 `phase_a_gate.json`。
