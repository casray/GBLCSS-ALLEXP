# MLP-40：Cora_ML 与 Coauthor-Physics 缺失噪声谱 Seed0 闸门

## 研究假设

Cora_ML 与 Coauthor-Physics 在 uniform0.4 下均有真实 LCSS 分组相对 shuffled 的增量。
补齐缺失噪声设置可以检验该结构信号能否稳定转化为相对正式鲁棒 baseline 的优势，并决定
两数据集能否进入核心全谱主表。

## 粒球扮演的角色

粒球继续作为 frozen GRACE H 上 label-blind 的局部共识单元，只用 train noisy labels
形成 train-only soft target。方法参数完全冻结，不为新设置调整粒度、beta 或分类器。

## 方法动机

此前 Cora_ML 实际已完成 uniform0.2/0.4，仅缺 uniform0.6、pair0.4、instance0.4；
Coauthor-Physics 已完成 uniform0.2/0.4/0.6，仅缺 pair0.4、instance0.4。先做 seed0
闸门可用最小成本排除不可能形成稳定全谱结论的数据集。

## 代码与配置改动

- predictor、方法配置和既有 anchor 均未修改。
- 新增 `scripts/summarize_mlp40_coraml_coauthorph_fullspectrum.py`，强制审计 45 行矩阵、
  九臂方法集合、split/noise hash、train-only 污染范围和冻结门。
- 新增 `tests/test_mlp40_coraml_coauthorph_fullspectrum.py`，覆盖完整矩阵、门判定、hash
  冲突和 val/test 污染拒绝。

## 固定协议

- Percent split 0.2/0.1/0.7，`noise_scope=train`，seed0，ROCm `cuda:0`。
- 每个设置九臂：LCSS、size-matched shuffled、MLP+CE、Small-loss、Co-teaching、
  JoCoR、SCE、ELR、LogitClip。
- 共 5 settings × 9 methods = 45 runs；45/45 成功，总墙钟约 457 秒。
- 全部行 hash 配对一致，val/test 标签修改数为 0，modified count 全部只来自 train。

## 成功与失败标准

- Cora_ML：至少 2/3 设置超过最强 baseline、平均至少 +1pp、无设置低于 −1pp、
  3/3 real−shuffled 至少 +1pp。
- Coauthor-Physics：2/2 设置均超过最强 baseline、平均至少 +0.5pp、2/2
  real−shuffled 至少 +1pp。
- 未过门不补 seeds 1–4，不放宽标准。

## 实验结果

| 数据集 | 设置 | LCSS | 最强正式 baseline | Shuffled | LCSS−best | LCSS−shuffled |
|---|---|---:|---|---:|---:|---:|
| Cora_ML | uniform0.6 | 37.88 | JoCoR 42.08 | 31.73 | −4.20pp | +6.15pp |
| Cora_ML | pair0.4 | 51.91 | Small-loss 50.33 | 46.37 | +1.57pp | +5.53pp |
| Cora_ML | instance0.4 | 52.91 | Small-loss 55.53 | 46.56 | −2.62pp | +6.35pp |
| Coauthor-Physics | pair0.4 | 89.39 | JoCoR 89.31 | 83.62 | +0.09pp | +5.78pp |
| Coauthor-Physics | instance0.4 | 93.40 | SCE 92.71 | 87.49 | +0.69pp | +5.91pp |

Cora_ML 只有 1/3 设置赢最强 baseline，平均 −1.75pp，最差 −4.20pp，明确失败。
Coauthor-Physics 2/2 数值正向，但平均只有 +0.39pp，低于冻结的 +0.5pp 线，也失败。
两个数据集的 real−shuffled 均明显为正，说明分组结构不是随机平滑；但这一信号被强鲁棒
优化吸收，不能转化为稳定主表优势。

## 解释边界

这是 seed0 筛选证据，不能据此声称 pair/instance 的多种子均值。可以支持的结论只有：
Cora_ML 不值得继续扩全谱；Coauthor-Physics 的 pair/instance seed0 边际小于预注册的
扩展价值门。不能把 Coauthor-Physics 的 +0.09/+0.69pp 写成稳定胜出。

## 最终决策

**NO-GO，两数据集均不扩 seeds 1–4。** Cora_ML 保留为附录广度/边界数据集；
Coauthor-Physics 保留为已有 uniform 三档的正文广度/适用域数据集，不进入核心成功计数，
也不补成正式五设置全谱表。第一篇论文核心主表和 24/25 条件性口径不变。

## 证据路径

- 计划：`external/NoisyGL/docs/execution_plan_mlp40_coraml_coauthorph_fullspectrum_20260717.md`
- 原始与汇总：`external/NoisyGL/runs/mlp40_coraml_coauthorph_fullspectrum_20260717/`
- 汇总报告：`external/NoisyGL/runs/mlp40_coraml_coauthorph_fullspectrum_20260717/mlp40_report.md`
