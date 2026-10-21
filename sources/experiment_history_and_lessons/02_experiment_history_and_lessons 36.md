# MLP-60：KRD 与 VQGraph 忠实协议移植 smoke

## 研究假设

论文现有 KRD/VQGraph 数字来自机制代理。完整移植官方核心训练流程后，同一噪声协议下的结果可能发生实质变化。

## 粒球扮演的角色

粒球不进入两个基线，只提供同协议的球式共识教师参照。

## 方法动机与代码改动

- KRD 使用官方的邻域知识点采样、teacher-student KL 和动态可靠性更新。
- VQGraph 使用官方离散 `VectorQuantize`、图 tokenizer、软标签与软码字蒸馏。
- 两条方法在 `predictor/G2MFaithful_Predictor.py` 中以新名称实现，不覆盖旧代理。
- VQGraph Cora 参数按论文 Table 10/12 固定；仓库 CLI 默认值不作为主实验参数。
- 新增配置、runner 注册和 9 项行为测试。

## 实验协议

Cora，uniform 0.4，仅训练标签加噪，20/10/70 划分，seed 0，clean validation 选模，学生推理只输入节点属性。

## 预期机制与判据

若忠实移植相对旧代理改变至少 1 个百分点，说明旧代理不足以代表该方法，值得考虑 5-seed 正式运行。所有结果必须通过 split/noise hash、污染范围和 X-only 推理审计。

## 实验结果

- KRD faithful：72.45%，旧代理 68.44%，提高 4.01 个百分点。
- VQGraph faithful：相同 seed 的两次 GPU 运行分别为 64.91% 和 66.91%，旧代理为 76.36%。
- 球式共识教师同 seed：81.42%。
- VQGraph clean-label sanity check：82.48%，说明最终实现能够正常训练。相同 seed 的 noisy GPU 结果相差 2.01 个百分点，尚不能视为稳定估计。
- 两条有效 smoke 使用相同 split/noise hash，污染计数均为训练/验证/测试 = 230/0/0。

## 最终决策

- **KRD：继续条件成立。** 若论文保留 KRD，应运行 5 seeds 并用忠实协议移植替换旧代理。
- **VQGraph：暂缓扩展。** 若保留为正式比较，应先处理或量化 GPU 非确定性，再补 5 seeds；否则删除该数值行，不能以旧软码本代理冒充原方法。
- 单 seed 不进入论文，不修改当前稿件。

## 证据位置

- 计划：`external/NoisyGL/docs/execution_plan_mlp60_krd_vqgraph_faithful_seed0_20260905.md`
- 结果与报告：`external/NoisyGL/runs/mlp60_krd_vqgraph_faithful_seed0_20260905/`
- 测试：`external/NoisyGL/tests/test_mlp60_krd_vqgraph_faithful.py`
