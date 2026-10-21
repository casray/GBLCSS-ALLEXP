# MLP-33：高噪声图原生对比、噪声率误设与跨噪声类型验证

## 1. 研究假设

1. `LCSS+soft-small-loss` 在 uniform 0.6 下的稳定协同不等于图原生 SOTA；必须用相同高噪声协议直接比较 NRGNN/NRGNN[X||H]/RTGNN。
2. `forget_rate=auto` 依赖已知合成噪声率；若假设噪声率误差为 0.1 时性能仅小幅变化，该限制可被定量披露，而不是只作为文字 limitation。
3. LCSS target correction 与 small-loss selection 的互补性应当在 pair/instance noise 下继续接受四臂归因检验。

## 2. 粒球角色与方法动机

粒球仅在 frozen GRACE H 中提供 train-noisy-label 的局部共识 soft target；它不进入图原生对照，不改变 split，也不使用 val/test 标签。组合方法以逐节点 soft CE 排序执行 small-loss selection，检验“监督修复 + 训练期样本筛选”是否互补。

## 3. 代码与配置改动

- 未修改 LCSS、NRGNN、RTGNN、GRACE H 或历史配置行为。
- 新增汇总脚本：`scripts/summarize_mlp33_u06_graphnative_sensitivity_noise.py`。
- 新增测试：`tests/test_mlp33_u06_graphnative_sensitivity_noise.py`。
- 原始与汇总输出：`runs/mlp33_u06_graphnative_sensitivity_noise_20260713/`。

固定协议：percent split 0.2/0.1/0.7、`noise_scope=train`、ROCm `cuda:0`。图原生和误设敏感性为 seed0；pair/instance 的 seed0 过门后补齐 seeds 0-4。

## 4. 预注册成功与失败标准

### 4.1 Uniform 0.6 图原生 seed0

若 direct combo 或冻结 X-only distillation 在至少 3/5 数据集胜出或距最佳图方法不超过 2pp，则扩 5 seeds 和 Coauthor-Physics；否则停止。

### 4.2 Pair/instance 0.4 五种子

每种噪声独立满足：

- 至少 3/5 数据集 mean synergy > 0 且至少 4/5 paired seeds 正向；
- 五数据集平均 synergy >= 1pp；
- 任一数据集 mean regression 不低于 -0.5pp；
- 至少 3/5 数据集 real-shuffled >= 1pp 且至少 4/5 seeds 正向。

其中 `synergy = combo - max(LCSS, Small-loss)`。

## 5. 实验结果

### 5.1 Uniform 0.6 图原生对比

| 数据集 | 最佳 X-only | 最佳图原生 | 差值 |
|---|---:|---:|---:|
| Cora | 68.65 | 74.99 | -6.33pp |
| CiteSeer | 59.19 | 68.08 | -8.89pp |
| CoraFull | 52.61 | 58.44 | -5.83pp |
| Amazon-Photo | 84.65 | 88.40 | -3.75pp |
| PubMed | 77.99 | 77.20 | +0.78pp |

只有 1/5 数据集进入 2pp 或胜出，未过门。未扩图原生五种子，也未运行 Coauthor-Physics。

### 5.2 噪声率误设敏感性

真实 uniform 0.6 下，假设 0.5 相对正确 0.6 的差值为 Cora -0.26pp、CiteSeer -1.03pp、CoraFull +0.43pp；假设 0.4 时为 -1.48/-0.52/-1.14pp。结论是误差 0.1 大体可用但并非无损，低估 0.2 在两个数据集造成超过 1pp 回退。

### 5.3 Pair/instance 0.4 五种子

| 噪声 | 通过协同的数据集 | 平均协同 | 最小协同 | real-shuffled 通过 |
|---|---:|---:|---:|---:|
| Pair 0.4 | 4/5 | +1.49pp | -0.10pp | 5/5 |
| Instance 0.4 | 5/5 | +3.04pp | +0.90pp | 5/5 |

Pair 的逐数据集 synergy 为 Cora +1.04、CiteSeer +1.07、CoraFull +0.90、Amazon-Photo -0.10、PubMed +4.52pp。Instance 为 +3.55/+3.57/+2.69/+0.90/+4.47pp。两种噪声均通过冻结门槛。

完整性：pair 与 instance 各 100/100 唯一五种子单元；全部 `status=ok`，val/test modified count 为 0，同数据集/seed 四臂 split/noise hash 一致。

## 6. 最终决策

- **继续**：把 `LCSS+soft-small-loss` 定位为中高噪声 graph-free 鲁棒训练候选；uniform 0.6、pair 0.4、instance 0.4 均已有五种子证据。
- **停止**：不扩本轮 uniform 0.6 图原生五种子，不加入 Coauthor-Physics；现有结果不支持“接近或超过图原生 SOTA”的 claim。
- **披露限制**：small-loss forget rate 使用已知注入噪声率；误设 0.1 多数情况下影响约 1pp 内，但不是严格鲁棒。
- **保持冻结**：不继续调整蒸馏融合、robust teacher、球数、动态 alpha 或 confidence gate。

