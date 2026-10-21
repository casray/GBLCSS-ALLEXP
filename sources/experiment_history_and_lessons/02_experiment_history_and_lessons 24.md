# 实验历史与经验 24：强基线、噪声率估计与选择机制诊断

## 1. 研究问题

本阶段补齐第二篇候选方法的三个证据缺口：PubMed 的 JoCoR 强基线、unknown noise rate 的 noisy-only 估计、以及 LCSS 与 Small-loss 协同的选择机制。

## 2. 固定协议

- percent split 0.2/0.1/0.7，train-only noise，ROCm GPU。
- LCSS/GRACE H/64-32-0.05/beta=0.5/MLP 全部冻结。
- 性能与估计使用 seeds 0-4；mask 机制诊断只用 seed0。

## 3. 核心结果

1. PubMed 上，组合相对 JoCoR：Pair 0.4 为 +3.78pp、5/5；Instance 0.4 为 +4.05pp、5/5；Uniform 0.6 为 +1.79pp、3/5。故只支持 Pair/Instance 条件性边界消除，不支持三噪声全胜。
2. LOO 球投票 rho-hat 系统性高估：Cora/CiteSeer/CoraFull mean abs error 为 0.120/0.132/0.233，虽 coverage 近 100%。B1 失败，B2 未运行。
3. 组合在 Pair/Instance 的 selected-target accuracy 相对 Small-loss 平均提高 3.60/6.60pp，real 对 shuffled 10/10 正向；但 hard-repaired noisy retention 门仅 1/10。
4. Soft target 的主要作用不是大量硬翻标签，而是增加 clean-class probability mass、重排样本损失并产生更稳定的选择 mask。Pair/Instance final turnover 从 Small-loss 的 0.280/0.243 降至组合的 0.105/0.083。
5. 饱和相关性多指标方向冲突，不能建立一般规律。
6. Trace 回归中 40/45 行与历史精度逐位相同，45/45 的哈希与选择数量一致；5 行受 ROCm 非确定性影响，最大偏差 2.76pp。重复同一 trace 可恢复历史值，因此保留首次正式 trace 并披露该限制，不做择优覆盖。

## 4. 决策与经验

- 第二篇继续以“局部共识目标修复 + 小损失选择的协同”为主线。
- known noise rate 仍是诚实限制；禁止把失败的 LOO 估计器写成可部署模块。
- 机制措辞改为“soft target reshapes loss ranking and improves selected supervision”，不再写“保留更多硬纠正噪声节点”。
- Phase C clean label 只用于事后解释，必须与五种子性能证据分开标注。

## 5. 资产

- 执行备案：`external/NoisyGL/docs/execution_plan_mlp34_pubmed_rhohat_mechanism_20260714.md`
- 结果：`external/NoisyGL/runs/mlp34_pubmed_rhohat_mechanism_20260714/`
- 工具：`utils/lcss_noise_rate.py`、`utils/selection_trace.py`
- 测试：`tests/test_mlp34_noise_rate_estimator.py`、`tests/test_mlp34_selection_trace.py`、`tests/test_mlp34_summary.py`
