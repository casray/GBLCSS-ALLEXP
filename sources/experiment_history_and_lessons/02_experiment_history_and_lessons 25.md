# 实验历史与经验 25：Clean-Val 选择与 ROCm 重复性审计

## 1. 研究假设

MLP-34 证明 noisy-only 球内分歧不能准确估计合成噪声率。本阶段检验两个更保守的问题：

1. 在已有干净验证集的实验协议下，能否把 forget rate 当作普通训练超参选择，从而不需要知道精确注入率？
2. MLP-34 trace 中最大 2.76pp 的复跑差异是否代表整个 ROCm 实验管线的普遍方差？

## 2. 粒球扮演的角色

粒球继续只在 frozen GRACE H 中构造局部共识 soft target。它不用于估计 clean label，
不参与 clean-val rate 选择规则，也不根据验证/测试结果改变分组。Clean-val 只选择
soft-small-loss 的 forget rate。

## 3. 方法动机与预期机制

- 噪声率估计要求恢复一个生成参数，难度高且已被 MLP-34 判负。
- forget rate 的实际作用是控制每轮保留样本比例，因此可把它视为训练强度，而不是噪声率的物理估计。
- 如果 clean-val 选择接近 true-rate oracle 并优于固定默认值，则可以诚实移除“必须知道精确合成噪声率”的依赖，但仍需披露 clean validation assumption 和额外网格成本。
- 重复性审计用于区分标准运行器的真实波动与 trace/instrumentation 路径中的局部偶发现象。

## 4. 代码与配置改动

- 新增 `scripts/run_mlp35_cleanval_grid.py` 和
  `scripts/summarize_mlp35_cleanval_grid.py`。
- 新增 `scripts/run_mlp35_repeatability_audit.py` 和
  `scripts/summarize_mlp35_repeatability_audit.py`。
- 新增 `tests/test_mlp35_cleanval_repeatability.py`。
- 未修改 predictor、LCSS target、GRACE H、历史 YAML 或冻结主方法。

## 5. 固定协议

- 数据集：Cora、CiteSeer、CoraFull、Amazon-Photo、PubMed。
- 噪声：uniform 0.2/0.4/0.6，train-only。
- Split：percent 0.2/0.1/0.7。
- Seeds：0-4。
- 候选 rate：`{0,0.2,0.4,0.6}`；fixed 0.5 只作对照，不参与选择。
- 每个 seed 独立按 clean validation accuracy 选择；完全平局取更小 rate。
- ROCm Python，`cuda:0`。

## 6. 成功与失败标准

### 6.1 Oracle 替代门

- 15 个 setting 的五种子 paired mean 至少 12/15 不低于 oracle 1pp。
- 全局 paired mean 不低于 -1pp。
- 任一 setting paired mean 不低于 -2pp。

### 6.2 固定默认价值门

- 至少 9/15 setting 相对 fixed0.5 非负。
- 全局 selected-fixed0.5 至少 +0.5pp。

### 6.3 重复性门

- 三个审计 setting 的方法内 range 均不超过 1pp。
- paired delta 在三次独立复跑中 3/3 同向。

## 7. 实验结果

### 7.1 完整性

- Seed0 75/75、seeds1-4 300/300、重复性 18/18 运行成功。
- 同一 dataset/noise/seed 内 split/noise hash 一致。
- val/test 标签修改计数均为 0，无重复或缺失单元格。

### 7.2 Clean-val 五种子结果

- Oracle 替代门通过：15/15 setting mean 均在 1pp 内；全局
  selected-oracle `+0.11pp`，最差 setting `-0.46pp`。
- 固定默认价值门通过：12/15 setting 非负；全局 selected-fixed0.5
  `+1.26pp`。
- Exact true rate 只在 45/75 个 seed-level 单元被选中，说明该过程选择的是训练强度，
  不是估计噪声率。
- u0.2 仅 4/25 次选择 rate 0，不能写成“低噪声自动退化为 LCSS”。
- 最差单 seed 为 CiteSeer u0.4 seed4，clean-val 选择 rate0 后比 oracle 低 4.38pp；
  五种子均值为 -0.46pp。该例说明小验证集选择仍有方差。

### 7.3 ROCm 重复性

- Cora pair0.4：combo-Small-loss `+2.53pp`，方法内 range 0，3/3 正向。
- Amazon-Photo pair0.4：combo-Small-loss `+5.28pp`，方法内 range 0，3/3 正向。
- PubMed pair0.4：combo-JoCoR `+3.07pp`，最大方法内 range 0.10pp，3/3 正向。
- 重复性门通过。MLP-34 的 2.76pp trace 差异不是可套用到所有正式结果的全局 ROCm 方差。

## 8. 最终决策

**决策：继续（continue），保留 clean-val tuned 变体；不静默替换冻结的 known-rate 版本。**

可以写：

> 在干净验证集可用时，预声明 forget-rate 网格能够保持 true-rate oracle 水平，
> 并在平均意义上优于固定 0.5 默认值。

不能写：

- 完全 noise-rate-free；
- 估计了真实噪声率；
- 低噪声下必然自动关闭样本选择；
- 单 seed 的验证选择没有风险。

`forget_rate=0` 与历史 LCSS 在 15 个可比哈希中 12 个 test 完全一致；PubMed
最大差异 1.12pp，因此只称“同机制零筛选臂”，不称 bitwise 等价。

## 9. 经验与论文影响

1. unknown-rate 问题可通过合法的 model-selection protocol 缓解，无需继续调失败的 noisy-only 估计公式。
2. 代价是额外候选训练和 clean validation assumption，论文必须同时报告计算开销与设定边界。
3. ROCm 波动必须按具体运行路径审计，不能用单个最大差异给所有弱增量统一降级或免责。
4. Pair 的全局协同仍只有约 1.49pp，应保留“条件性、中等幅度”的五种子措辞；PubMed Pair/Instance 强锚点可单独报告。

## 10. 资产

- 执行备案：`external/NoisyGL/docs/execution_plan_mlp35_cleanval_repeatability_20260714.md`
- 结果：`external/NoisyGL/runs/mlp35_cleanval_repeatability_20260714/`
- 主报告：`external/NoisyGL/runs/mlp35_cleanval_repeatability_20260714/mlp35_report.md`
- 重复性报告：`external/NoisyGL/runs/mlp35_cleanval_repeatability_20260714/mlp35_repeatability_report.md`
- 测试：`external/NoisyGL/tests/test_mlp35_cleanval_repeatability.py`
