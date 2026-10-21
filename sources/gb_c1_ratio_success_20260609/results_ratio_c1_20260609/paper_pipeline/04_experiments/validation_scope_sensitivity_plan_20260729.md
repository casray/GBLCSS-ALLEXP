# 验证标签噪声敏感性实验计划

## 研究假设

论文当前结果使用干净验证标签选模。本实验检验在训练噪声实例、数据划分、测试标签、方法配置和随机种子保持不变时，将同率噪声扩展到验证标签是否改变 GB-LCSS 的主要相对结论。

## 粒球角色

粒球仍是在冻结 GRACE 表征中构造的标签无关局部语义单元。粒球只聚合含噪训练标签并生成并列安全的 `distribution_only` 软目标。本实验不修改构球、目标公式、`beta`、`alpha` 或学生模型。

## 方法动机

干净验证标签是当前协议中的额外模型选择资源。只报告该协议可能高估方法在未知干净验证集条件下的可用性。敏感性实验关注相对排序是否稳定，不要求 noisy-validation 下绝对精度不下降。

## 固定范围

- 数据集：Cora、CiteSeer、CoraFull。
- 噪声：uniform 0.4。
- 划分：percent 0.2/0.1/0.7。
- seeds：0、1、2、3、4。
- 测试标签：始终干净，只用于最终评估。
- 方法：
  1. GB-LCSS tie-safe `distribution_only`；
  2. shuffled GB-LCSS tie-safe `distribution_only`；
  3. MLP+CE；
  4. JoCoR；
  5. PosteL-noIPL adapted；
  6. 旧 `gsdn` provisional adapted。
- 新增训练：`3 datasets x 6 methods x 5 seeds = 90 runs`。
- 旧 `gsdn` 仅为本轮临时参照，不视为官方或最终忠实复现。
- PubMed 不进入本实验和正文分析。

## 噪声隔离

`train_val` 分两阶段生成：

1. 用原 seed 和训练掩码生成 train-only 噪声。
2. 用 `seed + 1000003` 和验证掩码单独生成验证噪声。
3. 只把第二步的验证节点写回第一步结果。
4. 测试节点保持不变。

每行记录 `train_noise_hash`、`val_noise_hash`、全局 `noise_hash` 和 train/val/test 修改计数。另生成 15 行 train-only 协议指纹，使用相同数据加载器和噪声函数重建历史 clean-validation 行的 train 子哈希，并核对旧全局 `noise_hash`。

## 预期机制

验证标签受噪声污染会改变早停和 checkpoint 选择，因此全部方法都可能下降。若 GB-LCSS 的局部共识信号不依赖干净验证标签，其相对 shuffled 和 MLP+CE 的优势方向应基本保持。实验不预设 GB-LCSS 必须在绝对精度上保持不变。

## 预注册成功标准

主要机制结论通过需同时满足：

1. GB-LCSS 相对 shuffled 在 3/3 数据集上均值为正，且每个数据集至少 4/5 seeds 为正。
2. GB-LCSS 相对 MLP+CE 至少 2/3 数据集均值为正，且通过的数据集至少 4/5 seeds 为正。
3. 任一数据集 GB-LCSS 相对 shuffled 不低于 -0.5 个百分点。
4. 90/90 行完成，无 NaN、OOM、标签泄漏或预测坍缩。
5. 所有 dataset-seed 组的六方法具有相同 split hash、train noise hash 和 train 修改计数。

协议排序稳定性作为辅助判据：

- 对六方法均值排序计算 clean/noisy-validation Kendall tau。
- 比较两个协议的前二方法集合。
- 报告每个方法的协议交互量，不把绝对下降直接解释为方法失效。

## 失败标准

- 任一缺失、重复、失败或非有限结果。
- train-only 与 train-val 的训练噪声实例不一致。
- 验证噪声污染测试节点。
- GB 两臂未使用 `distribution_only`。
- 根据 seed0 或正式结果调整数据集、方法、阈值、派生 seed 或判据。
- 将旧 `gsdn` 写成官方或最终忠实复现。

## 结果分支

- 若主要机制结论通过：正文简要说明结论对验证标签污染具有稳健性，完整表进入附录。
- 若主要机制结论未通过：摘要和实验设置明确限定为 clean-validation 条件结果，正文报告受影响的数据集和比较。

## 实验结果

待运行后写入独立报告，不回写本预注册计划。

## 最终决策

待定。
