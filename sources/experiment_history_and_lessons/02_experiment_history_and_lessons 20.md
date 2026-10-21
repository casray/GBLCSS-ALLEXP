# MLP-23 数据集广度筛选与 Coauthor-Physics 大图 GRACE（2026-07-10）

## 研究假设

在固定 LCSS 机制、固定 MLP 骨干和固定 train-only 噪声协议下，优先筛选与现有成功数据集结构相近的新图，至少找到一个可进入论文实验表的数据集，并检验真实局部共识增益能否扩展到更完整的引文图和更大的合著图。

## 粒球角色与方法动机

粒球仍只作为 frozen GRACE H 空间中的 label-blind 局部分组；组内仅统计 train noisy labels 并生成 β=0.5 半硬 soft target，最终分类器为 X-only 两层 MLP。不得根据新数据集调整 `max_balls=64`、`min_ball_size=32`、`split_improvement=0.05` 或 β。

## 固定协议

- split：percent 0.2/0.1/0.7
- noise：train-only uniform 0.4 主筛选，seeds 0-4
- 方法：LCSS、shuffled、MLP+CE 与 10 个同骨干鲁棒策略，共 13 方法
- 主通过门：LCSS−shuffled ≥1pp 且 ≥4/5 seeds 正；LCSS−最强 baseline ≥1pp 且 ≥4/5 seeds 正；65/65 行成功并通过泄漏审计
- 次级广度处置：uniform 0.4 与最强策略相当且真实分组稳定强于 shuffled 时，可进入正文“标准设置广度/适用域”表，但不得增加核心全谱胜出 claim

## 候选与代码改动

按 Cora_ML、Attributed-Wiki、Coauthor-Physics 顺序筛选；CiteSeerFull 作为 Coauthor-Physics 初始 OOM 时的预注册替代候选，最终也完成筛选。

- NoisyGL loader/CLI：`external/NoisyGL/utils/dataloader.py`、`single_exp.py`、`total_exp.py`
- GRACE loader/fallback：`gb_graph_aware_minival_6_1/src/noisy_label_gb_grace/data.py`、`scripts/pretrain_ssl_candidates.py`
- 配置与测试：`external/NoisyGL/config/**/{coraml,attribwiki,citeseerfull,coauthorph}.yaml`、`tests/test_mlp23_dataset_breadth.py`
- 汇总：`external/NoisyGL/scripts/summarize_mlp23_dataset_breadth_20260710.py`

Coauthor-Physics 的 34493 节点使原 N×N InfoNCE 在 12GB ROCm GPU 上 OOM。新增 anchor-chunked contrastive loss（batch=1024）并使用 gradient checkpoint；toy 测试证明分块与原完整 loss 在 `1e-6` 容差内一致，目标函数与 DBLP fallback 的 1000-epoch 超参数不变。最终 H 为 `34493×256`、全有限、行范数约 1，且上游与 NoisyGL 的标签和边索引逐元素一致。

## uniform 0.4 正式结果

| 数据集 | LCSS | 最强对照 | LCSS−最强 | LCSS−shuffled | 判定 |
|---|---:|---:|---:|---:|---|
| Cora_ML | 55.78±3.37 | Small-loss 55.56±3.42 | +0.22pp（2/5） | +6.71pp（5/5） | boundary；均值相当但跨种子不稳 |
| Attributed-Wiki | 65.42±1.20 | LogitClip 67.83±1.51 | −2.41pp（0/5） | +5.94pp（5/5） | boundary；强 baseline 明确占优 |
| CiteSeerFull | 59.40±1.73 | JoCoR 59.48±1.60 | −0.08pp（3/5） | +13.14pp（5/5） | 正文广度表；与最强对照相当 |
| Coauthor-Physics | 93.11±0.25 | JoCoR 93.09±0.28 | +0.02pp（3/5） | +4.21pp（5/5） | 正文广度表；数值第一但不宣称领先 |

证据：`external/NoisyGL/runs/mlp23_dataset_breadth_20260710/mlp23_report.md`、`mlp23_aggregate.csv`、`mlp23_paired_deltas.csv`。

## 选择性噪声率检查

- CiteSeerFull u0.2：LCSS 69.75±0.86，领先 Small-loss +2.28pp（4/5）；u0.6：41.93±3.12，落后 JoCoR −4.71pp（0/5）。
- Coauthor-Physics u0.2：93.94±0.09，落后 Co-teaching −0.69pp（0/5）；u0.6：87.65±1.19，落后 JoCoR −2.29pp（0/5）。
- Cora_ML u0.2：63.50±2.64，落后 CT −1.86pp（0/5），因此停止其后续全谱扩展。
- 上述所有设置中，LCSS 相对 shuffled 均为 5/5 seeds 正向；真实局部结构有效，但分类增量会被强样本选择/鲁棒损失吸收。

## 最终决策

- 核心 5 数据集×5 噪声设置 claim 不变，不把任何 MLP-23 候选包装成新增全谱胜出数据集。
- CiteSeerFull 与 Coauthor-Physics 加入论文正文 uniform0.4 数据集广度/适用域表；Cora_ML 与 Attributed-Wiki 只进附录筛选审计。
- 论文安全表述：局部结构贡献可迁移到 expanded citation graph 与 34k-node coauthor graph，但在这些数据集上只与最强鲁棒策略相当；球可聚类性是必要条件而非充分条件。
- 不继续为凑数据集搜索或调新数据集专属超参。

## 关键证据

- 设计与执行计划：`external/NoisyGL/docs/superpowers/specs/2026-07-10-mlp23-dataset-breadth-design.md`、`docs/superpowers/plans/2026-07-10-mlp23-dataset-breadth.md`
- 结果目录：`external/NoisyGL/runs/mlp23_dataset_breadth_20260710/`
- 论文表：`external/NoisyGL/docs/mlp23_dataset_breadth_table_20260710.md`
- 新论文版本：`external/NoisyGL/docs/paper_final_zh_v5_mlp23_breadth_20260710.md`
