# MLP-51--53：粒球相对 kNN 的三轮表征与共识增强检验

## 方向定义

本方向检验一个明确问题：在不修改正式 LOO-GB-LCSS 学生、损失和噪声协议的前提下，能否通过新的表征信息或跨分区共识，使粒球目标稳定超过公平的 kNN 目标。

三轮共享协议为 Cora、CiteSeer、CoraFull，train-only uniform 0.4/0.6，percent split 0.2/0.1/0.7，seed 0。所有粒球目标使用 `tie_mode=distribution_only`、`leave_one_out=true`、`beta=0.5`。所有 kNN 对照使用相同表征、相同票数尺度并排除查询节点自身。clean label 仅用于运行后的 target-quality 诊断。

证据目录：

- `external/NoisyGL/runs/mlp51_multipartition_gb_20260806/`
- `external/NoisyGL/runs/mlp52_crosspartition_vote_pool_20260806/`
- `external/NoisyGL/runs/mlp53_nrgnn_latent_balls_20260806/`

## MLP-51：三个 GRACE 分区的最终目标平均

### 假设与粒球角色

不同 GRACE 预训练种子形成的粒球分区 ARI 较低，可能提供互补局部关系。候选将三个独立 H 上的正式 LOO 粒球 target 等权平均。粒球仍是固定预算局部共识单元，没有读取标签构球。

### 代码与配置

- 计划：`external/NoisyGL/docs/execution_plan_mlp51_multipartition_gb_20260806.md`
- 脚本：`external/NoisyGL/scripts/run_mlp51_multipartition_gb.py`
- 测试：`external/NoisyGL/tests/test_mlp51_multipartition_gb.py`
- 三份 H 均来自已归档的独立 GRACE 预训练；没有重训或按结果选择 H。

### 结果

- 相对单 H 粒球，Brier 改善 `6/6`。
- 相对三 H shuffled 粒球，Brier 改善 `6/6`。
- 相对同样三 H 平均的 kNN，只在 `2/6` 单元 Brier 更低。
- CiteSeer u0.4、CiteSeer u0.6、Cora u0.4 的 target accuracy 回退超过 0.5pp。
- 30 行目标结果与每视图诊断完整，无哈希、泄漏或 simplex 错误。

### 结论

**最终决策：放弃最终 target 平均。** 多 H 确实改善校准，但相同收益可被 kNN 获取；同时平均操作把部分有用的离散纠正变成保守概率。该结果只能支持“预训练种子包含互补局部信息”，不能支持粒球独有优势。

## MLP-52：跨分区 LOO 票数池化

### 假设与粒球角色

MLP-51 可能因先形成 target 再平均而过于保守。本轮先汇总三个分区中每个节点的 LOO 标签计数，再统一计算分布、半硬目标和 alpha，希望保留多数纠正能力。

### 代码与配置

- 计划：`external/NoisyGL/docs/execution_plan_mlp52_crosspartition_vote_pool_20260806.md`
- 脚本：`external/NoisyGL/scripts/run_mlp52_crosspartition_vote_pool.py`
- 测试：`external/NoisyGL/tests/test_mlp52_crosspartition_vote_pool.py`
- kNN 使用完全相同的三视图票数池化，是机制镜像而非弱化对照。

### 结果

- 相对正式单 H 粒球，Brier 仅 `1/6` 改善。
- 相对 pooled-kNN，Pareto 胜出 `0/6`。
- 相对 MLP-51 的 final-target averaging，target accuracy 只在 `3/6` 恢复。
- 四个小图单元 target accuracy 相对单 H 回退超过 0.5pp。
- real 相对 shuffled 的 Brier 仍为 `6/6` 更低，说明失败不是因为几何完全无效。

### 结论

**最终决策：放弃跨分区原始票数池化。** 三个中等粒球的票数直接合并后形成了过大的有效邻域，类别分布变散、alpha 变弱，破坏了已知的中等尺度工作区。不得通过事后修改 beta、归一化或粒球大小继续调同一机制。

## MLP-53：NRGNN 鲁棒教师隐层中的粒球

### 假设与粒球角色

前两轮只增加无监督表征视图，没有引入新的抗噪信息。本轮训练 NRGNN 鲁棒图教师，提取验证最优分类器在验证最优重构图上的倒数第二层表示 Z，再在 Z 中构造正式 LOO 粒球。目标是检验经过鲁棒监督塑形的空间是否更适合粒球共识。

### 代码与配置

- 计划：`external/NoisyGL/docs/execution_plan_mlp53_nrgnn_latent_balls_20260806.md`
- 脚本：`external/NoisyGL/scripts/run_mlp53_nrgnn_latent_balls.py`
- 测试：`external/NoisyGL/tests/test_mlp53_nrgnn_latent_balls.py`
- 每个数据集和噪声单元独立重训 NRGNN；保存 model state、重构图、Z 和全节点 teacher probabilities。
- 同空间 kNN 为 exact cosine、vote-matched、`exclude_self=true`。

### 结果

- 30/30 行成功，六份教师工件完整，无完整性失败。
- NRGNN direct probability 相对正式 H-GB 的 Brier 胜出 `5/6`，证明鲁棒教师本身有有效抗噪信息。
- Z-GB 相对正式 H-GB 的 Brier 只胜 `3/6`。
- Z-GB 相对同空间 kNN 的预注册 Pareto 胜出仅 `1/6`。
- Z-GB 相对 shuffled 只胜 `4/6`，未达到 6/6。
- CoraFull 的 Z 空间在 u0.4/u0.6 均只生成 1 个球；GB、matched-k=3958 的 kNN 与 shuffled 全部退化成同一全局标签分布。

### 结论

**最终决策：放弃鲁棒教师隐层再粒球化。** NRGNN 的预测概率有效，但其分类隐层不具备当前递归 2-means 所需的可分裂几何。教师有效不等于教师隐层适合局部区域投票；把 direct teacher 压缩成球 target 反而损失信息。

## 质量审计

- Baseline comparability：三轮均在同一 split/noise hash 内比较；kNN 与 GB 使用相同表征和票数尺度。
- Label leakage：构造阶段不读取 clean label；验证/测试标签未被污染；target-quality 指标在构造完成后计算。
- Statistical boundary：三轮都是预注册 seed0 smoke，只能作机制筛选和负方向关闭，不能形成跨 seed 性能 claim。
- Tests：MLP-53 完成后，相关回归测试 `47 passed`，仅有 torch-geometric 外部弃用警告。

## 跨轮经验

1. 新表征信息可以改善 target calibration，但 kNN 通常能获得同样的信息，不能自动转化为粒球独有优势。
2. 对局部共识而言，先平均 target 比池化原始票数稳健；原始票数池化会把多个中等邻域变成过粗邻域。
3. 分类表征的高预测精度与粒球可分裂性是两种性质。NRGNN direct teacher 很强，但 CoraFull Z-space 可完全塌缩为一球。
4. 当前粒球构造在 frozen GRACE H 的中等尺度平台已经接近其信息上限。继续更换分组规则或表征，若没有粒球独有的新信息，仍会被同空间 kNN 镜像。

## 论文 claim 影响

- 已支持：正式 GRACE-H GB 是局部共识的固定预算、互斥、可审计实现；直接鲁棒图教师是有价值的独立蒸馏路线。
- 未支持：粒球在精度或 target quality 上普遍超过公平 kNN；多分区或鲁棒教师表征能升级正式 GB-LCSS。
- 禁止写入：把 MLP-51--53 seed0 结果描述为主方法提升，或只报告其中有利单元。
- 可作为 future work 的线索：学习专门面向区域共识的表征目标，而不是直接复用分类隐层；这需要新论文级实验体系。

## 方向级最终决策

**Outcome: abandon（当前论文与当前管线）。** 三轮自主迭代均未通过预注册门。停止继续寻找第 4 个粒球变体，不进入下游 5-seed，不修改当前投稿算法。若未来重启，必须引入当前三轮没有的新信息源或新任务设定，而不能继续调整平均、池化、beta、粒度或教师隐层。
