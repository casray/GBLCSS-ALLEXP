# GB Denoise Stage A Design

## Purpose

本实验用于验证一个机制假设：在相同图数据、encoder、augmentation、训练轮数、seed 和粒球生成算法下，negative-sample-free 图自监督表示是否比 InfoNCE / GRACE 表示更适合生成高质量粒球。

该实验只做 Stage A。Stage B 的 GB compactness loss 反向约束 encoder 暂缓，必须等 Stage A 证明 negative-free 表示空间确实带来更稳定或更高质量的构球信号后再设计。

## Paper Motivation

CGBC 使用 SimSiam 风格 negative-sample-free 预训练，再用 GBC layer 生成粒球，并通过 intra-GB compactness 与 inter-GB separation 参与鲁棒训练。GBmooer 使用 hypersphere / angular contrastive learning 后再做 angle-space 粒球和局部 label smoothing。两篇论文共同支持一个谨慎假设：先学习更适合局部聚合的表示，再生成粒球，可能比直接在原始表示或不匹配的对比目标上构球更可靠。

本项目不能直接照搬 CGBC 的 supervised purity、majority-vote GB label 或 same-label inter-GB separation，因为这些机制依赖标签，进入无监督节点级 GCL 预训练会造成 label leakage。本阶段只做 label-free 表示诊断，标签只用于后验诊断指标，不进入训练或构球。

## Research Hypothesis

InfoNCE / GRACE 的负样本排斥力可能提高全局 uniformity，但也可能削弱局部语义邻域紧凑性，从而影响粒球半径、球内一致性和跨 seed / epoch 稳定性。negative-sample-free 图自监督目标可能减少这种局部聚合干扰，因此更适合后续粒球生成。

该假设不是论文既有结论，而是由 CGBC 和 GBmooer 启发出的待验证机制假设。

## Scope

本阶段实现一个可运行实验工程，建议文件夹名为：

```text
gb_denoise_stage_a_negfree_vs_infonce
```

包含内容：

- 最小 GRACE / InfoNCE baseline。
- 最小 negative-free baseline，优先 BGRL 或 SimSiam-style 图自监督。
- 统一数据加载、augmentation、encoder、linear evaluation、seed 管理。
- 统一粒球生成算法。
- 表示空间到粒球质量的诊断脚本。
- 汇总 CSV 和 Markdown 报告。

不包含内容：

- 不实现 GB compactness loss。
- 不实现 CGBC 的 label purity 训练。
- 不用 majority label 更新训练目标。
- 不做 same-ball hard positive、hard negative deletion、prototype pulling。
- 不改变 train / validation / test split。
- 不把标签用于无监督预训练或粒球生成。

## Datasets

第一批数据集：

- Cora
- CiteSeer
- PubMed

如果算力不足，先在 Cora + CiteSeer 跑通完整流程，再补 PubMed。所有结论必须标注数据集覆盖范围，不允许用 Cora-only 结果支撑主 claim。

## Experimental Matrix

| Run group | Representation | Purpose |
|---|---|---|
| `raw_features` | 原始节点特征 | 训练前构球质量基线 |
| `grace_infonce_h` | GRACE encoder output `H` | 普通负样本对比学习主对照 |
| `grace_infonce_z` | GRACE projector output `Z` | 诊断 projector 空间是否适合构球 |
| `negfree_h` | BGRL / SimSiam encoder output `H` | negative-free 主实验组 |
| `negfree_z` | projector / predictor 前表示 | 辅助诊断，不默认作为主结论 |

主比较为 `negfree_h` vs `grace_infonce_h`。`Z` 空间结果只能作为解释或补充，不应替代主比较。

## Shared Controls

所有训练组必须共享：

- 相同数据集和 split。
- 相同 GCN encoder 架构。
- 相同 hidden dimension 和 embedding dimension。
- 相同 edge drop / feature mask augmentation 配置。
- 相同训练 epoch。
- 相同 seed 列表。
- 相同 linear evaluation 协议。
- 相同粒球生成算法、距离度量和粒度参数。

任何协议差异都必须写入 config 和结果报告，并标注为不可直接比较。

## Granular Ball Algorithm

Stage A 使用同一个 label-free 粒球生成器。推荐先实现一个 angle-space 或 normalized embedding 上的递归二分粒球算法：

1. 对输入表示做 L2 normalization。
2. 用 angular distance 或 cosine distance 构造局部粒球。
3. 每个候选球记录 center、radius、average angular distance、size。
4. 根据最小球大小、质量阈值或 split-improvement criterion 停止分裂。
5. 生成结果只依赖 embedding，不依赖标签。

标签只能用于后验 `purity_diagnostic`，用于判断球内语义一致性，不得参与 split、merge、loss 或 early stopping。

## Diagnostics

每个 run 至少输出以下指标：

- `num_balls`
- `mean_ball_size`
- `max_ball_size`
- `singleton_ratio`
- `mean_radius`
- `mean_average_angular_distance`
- `label_purity_diagnostic`
- `assignment_jaccard_across_seeds`
- `assignment_jaccard_across_epochs`
- `embedding_uniformity`
- `effective_rank`
- `embedding_std`
- `linear_eval_accuracy`
- `linear_eval_f1_micro`
- `linear_eval_f1_macro`

如果加入 synthetic edge noise 诊断，还应输出：

- `edge_noise_auc`
- `edge_noise_pr_auc`
- `clean_edge_keep_rate`
- `noisy_edge_keep_rate`

edge noise 诊断是辅助项，不是 Stage A 的必要成功条件。

## Success Criteria

Stage A 通过的最低标准：

- `negfree_h` 在至少两个数据集上相对 `grace_infonce_h` 展示更好的构球质量，例如更低 `mean_radius` / `mean_average_angular_distance`，更高 `label_purity_diagnostic`，或更稳定的 assignment。
- `negfree_h` 不出现明显 collapse，`effective_rank`、`embedding_std`、`embedding_uniformity` 必须处于合理范围。
- `negfree_h` 的 linear evaluation 不明显低于 `grace_infonce_h`。如果构球更紧但下游明显变差，只能说明表示更紧，不能说明表示更适合任务相关粒球。
- 结果必须包含 mean / std 和 per-seed paired delta。

## Failure Criteria

Stage A 失败或降级的条件：

- `negfree_h` 的构球指标没有稳定优于 `grace_infonce_h`。
- `negfree_h` 的半径变小主要来自 representation collapse。
- `negfree_h` 构球更紧但 linear eval 明显下降。
- 优势只出现在单个数据集或单个 seed。
- `grace_infonce_z` 或 `negfree_z` 的结果与 `H` 空间相反，但缺少稳定解释。

失败时，结论应写为：当前证据不支持“negative-free 更适合粒球生成”的主张，Stage B 不应继续。

## Output Files

建议工程输出：

```text
gb_denoise_stage_a_negfree_vs_infonce/
  configs/
  src/
  scripts/
  results/
    stage_a_metrics.csv
    stage_a_summary.md
    runs/
  reports/
```

`stage_a_metrics.csv` 保存所有 run 的结构化结果。`stage_a_summary.md` 汇总每个数据集的主比较、成功/失败判断、异常 run 和下一步决策。

## Claim Boundary

允许的谨慎表述：

- negative-free 表示是否更适合构球，是本阶段要验证的机制假设。
- 如果成功，只能说 negative-free 在本实验协议下提供了更稳定或更紧凑的粒球生成空间。
- 如果失败，应保留为负结果，说明 CGBC 的 negative-free 选择不能简单迁移解释为图节点粒球质量提升。

禁止表述：

- 不得声称 CGBC 已证明 InfoNCE 会破坏粒球。
- 不得声称更小半径一定意味着更好下游性能。
- 不得把 `label_purity_diagnostic` 当作训练依据。
- 不得把单数据集、小于 std 的差异写成稳定提升。

## Deferred Stage B

Stage B 暂缓。只有 Stage A 通过后，才考虑设计 stop-gradient center compactness：

```text
L = L_ssl + lambda * L_gb
L_gb = mean_b q_b * mean_{i in b} angular_distance(z_i, stopgrad(c_b))
```

Stage B 需要单独设计、单独批准，并加入 anti-collapse、no-ball、random-ball 和 shuffled-ball 控制。
