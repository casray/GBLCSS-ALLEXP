# MLP-37：GRACE 预训练种子分层方差

## 研究假设

第一篇 LCSS/SoftDist-MLP 的主结果此前主要条件于一份 frozen GRACE 表征。若局部共识收益不是特定预训练随机种子的偶然结果，则在多个 GRACE seeds 下，LCSS 相对 CE、shuffled 和强鲁棒基线的方向应保持稳定，H-seed 方差也不应显著大于下游训练方差。

## 粒球扮演的角色

粒球仍是 frozen GRACE H 上的 label-blind 局部共识分组。球内仅使用 train noisy labels 构造 soft target；本实验不改变球的角色、参数或目标公式。

## 方法动机

内部二审指出：仅报告 downstream seeds 不能覆盖上游 GRACE 预训练随机性。与其在标题中把方法永久限定为 fixed-H conditional，本实验用受控的 3×3 分层设计直接估计 H-seed 与 downstream-seed 两类波动。

## 代码与配置改动

- 新增 `external/NoisyGL/scripts/pretrain_mlp37_grace_seeds.py`，按冻结配置训练 GRACE seeds 0/1/2，并保存 encoder/projector checkpoint、H、config、loss、时间、峰值显存和 SHA256。
- 新增 `external/NoisyGL/scripts/run_mlp37_grace_seed_variance.py`，运行 LCSS、shuffled、CE 和 JoCoR 的分层下游矩阵。
- 新增 `external/NoisyGL/scripts/summarize_mlp37_grace_seed_variance.py`，输出分层聚合、paired deltas、方差、ARI、成本和预注册门判定。
- 新增 `external/NoisyGL/tests/test_mlp37_grace_seed_variance.py`。同时修复标准 runner 的 CSV UTF-8 写入，并为当前遗留 raw CSV 保留兼容读取。
- 算法配置不变：`beta=0.5`、`max_balls=64`、`min_ball_size=32`、`split_improvement=0.05`。

## 固定协议

- 数据集：Cora、CiteSeer、CoraFull。
- Split：percent 0.2/0.1/0.7。
- Noise：uniform0.4，只污染 train。
- GRACE seeds：0/1/2；downstream seeds：0/1/2。
- CoraFull 复用已预注册的 DBLP-style GRACE 配置，不做超参搜索。

## 预期机制

GRACE H 的具体局部几何会随随机种子变化，但如果方法依赖的是较宽泛的局部共识而非唯一划分，那么精确球 assignment 可以变化，LCSS 相对对照的聚合收益仍应保持同向。

## 跑前成功标准

1. 每个数据集上，LCSS−shuffled 与 LCSS−CE 的均值在全部 3 个 H-seed 上方向一致且为正。
2. H-seed 间精度标准差不超过 `max(0.25pp, 2 × downstream RMS SD)`。
3. 强对照检查：LCSS−JoCoR 在全部 H-seed 上为正。
4. 球数、target quality 和跨 H-seed ARI 只作描述，不作为通过门。

## 失败标准

任一核心方向门或方差门失败，则不得写“跨预训练种子稳健”；摘要与 Highlights 必须提升 fixed-H conditional 限定，并如实披露失败数据集。不得事后修改门槛或追加种子救结果。

## 实验结果

- 完整性：9/9 次 GRACE 预训练、72/72 条下游结果、27/27 个几何单元全部完成；无 NaN、OOM、预测坍缩或 val/test 标签污染。
- LCSS−shuffled 的跨 H-seed 最小均值增量：Cora `+11.43pp`、CiteSeer `+11.61pp`、CoraFull `+4.73pp`。
- LCSS−CE 的跨 H-seed 最小均值增量：`+11.70/+10.81/+4.82pp`。
- LCSS−JoCoR 在 9 个 dataset×H-seed 单元均为正；所有列出的逐下游 seed 比较均 3/3 正向。
- H-seed SD：`0.53/0.07/0.44pp`；downstream RMS SD：`0.87/1.11/0.39pp`，三数据集均通过方差门。
- 跨 H-seed assignment ARI 总均值 `0.300`，范围 `0.247--0.365`。具体划分不是唯一的，但收益稳定。
- 平均 GRACE 预训练成本：Cora `0.07 min/0.40 GiB`、CiteSeer `0.13 min/0.67 GiB`、CoraFull `74.48 min/12.73 GiB`。

## 最终决策

**通过并冻结。** 正文增加“在三个数据集、三个 GRACE 预训练种子下核心增量方向稳定”的分层方差证据；不得推广成全面解决预训练不确定性。Coauthor-CS/Amazon-Photo 仍保留仅归档 H、缺原 checkpoint 的可复现性限制。第一篇不再新增实验、数据集或方法改动，进入匿名工件、HTML 和终审阶段。

## 证据路径

- 计划：`external/NoisyGL/docs/execution_plan_mlp37_grace_seed_variance_20260716.md`
- 结果：`external/NoisyGL/runs/mlp37_grace_seed_variance_20260716/`
- 报告：`external/NoisyGL/runs/mlp37_grace_seed_variance_20260716/mlp37_report.md`
- 论文候选：`external/NoisyGL/docs/paper_final_zh_v15_submission_candidate_20260716.md`
