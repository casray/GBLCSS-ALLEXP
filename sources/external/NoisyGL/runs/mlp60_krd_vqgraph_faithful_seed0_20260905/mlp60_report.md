# MLP-60: KRD 与 VQGraph 官方核心机制协议移植 smoke

## 结论

本实验完成了 KRD 和 VQGraph 的 Cora、uniform 0.4、seed 0 最小运行。两条路线均通过协议与数值有效性检查，但结果指向不同结论：

- KRD 忠实协议移植达到 72.45%，比旧 `krd_adapted` 的同 seed 结果高 4.01 个百分点，说明旧机制代理低估了 KRD。
- VQGraph 在两次同 seed GPU 运行中分别达到 64.91% 和 66.91%，均低于旧 `vqgraph_adapted`。clean-label 诊断达到 82.48%，说明最终移植能够复现 VQGraph 在标准监督下的基本行为；40% 标签噪声下的下降首先出现在其图 tokenizer 教师。
- 同 seed 的正式球式共识教师为 81.42%。KRD 低 8.97 个百分点；VQGraph 两次运行低 14.51--16.52 个百分点。

KRD 重复运行逐位一致。VQGraph 的相同 seed GPU 结果相差 2.01 个百分点，因此只能作为不稳定的单 seed 诊断，不能写成论文均值或稳定排序。

## 研究假设

当前论文中的 KRD 与 VQGraph 行来自机制代理。若完整保留官方训练目标、可靠性更新和离散码本，结果可能发生实质变化。

## 粒球扮演的角色

粒球不参与 KRD 或 VQGraph 的训练。正式球式共识教师只作为同一划分、噪声 realization 和 X-only 学生接口下的参照。

## 实现依据

- KRD：官方仓库 `LirongWu/KRD`，commit `32be7e42ab65b1d246cf1591d8bc48dfe7fb6b21`。
- VQGraph：官方仓库 `YangLing0818/VQGraph`，commit `60aa1c99782d542c316a37f0394eba934bcf97b0`。
- VQGraph 的离散量化直接调用官方仓库随附的 `vq.py::VectorQuantize`。
- VQGraph 的 Cora 主配置依据论文 Table 10 和 Table 12，而非仓库 CLI 默认值：2048 个码字，教师隐藏维度 128、dropout 0、学习率 0.01、权重衰减 0.0005；学生隐藏维度 128、dropout 0.4、学习率 0.005、权重衰减 0.001。
- 官方量化器依赖 `einops`；本次环境使用 `einops==0.8.2`。

该实验仍是官方核心机制在论文统一协议下的移植，不是官方 headline 数字复现。数据版本、20/10/70 划分、标签噪声和 clean-validation checkpoint selection 均服从本项目协议。

## 调试过程与作废结果

初版 VQGraph 移植误用了仓库示例默认配置，包括 5000 个码字、GCN 示例教师和错误的 Cora 学生优化参数；同时使用了手写近似 EMA 量化器。该版本在 uniform 0.4 和 clean label 下分别得到 29.97% 和 44.33%，不具备证据效力。

换用官方 `VectorQuantize` 并补齐无向边和自环后，clean-label 结果提高到 51.35%，但配置仍未对齐论文主实验。进一步按论文 Table 10/12 修正后，clean-label 结果达到 82.48%，教师达到 81.79%。因此只有 paper-configuration 版本进入本报告的有效结果表。

## 正式 smoke 协议

- 数据集：Cora。
- 划分：train/validation/test = 0.2/0.1/0.7。
- 噪声：uniform 0.4，仅污染训练标签。
- 种子：0。
- 模型选择：干净验证标签。
- 学生推理：仅输入节点属性，不访问图。

## 技术有效性审计

| 检查项 | KRD | VQGraph |
|---|---:|---:|
| 运行状态 | 两次 ok 且结果逐位一致 | 两次 ok，但结果不一致 |
| split hash 一致 | 是 | 是 |
| noise hash 一致 | 是 | 是 |
| 修改训练/验证/测试标签数 | 230/0/0 | 230/0/0 |
| checkpoint selection | clean validation | clean validation |
| 训练使用图 | 是 | 是 |
| 推理使用图 | 否 | 否 |
| 使用无标签节点 | 是 | 是 |
| 方法特有诊断 | 98 次可靠性更新，0 次拟合失败 | 129 个活跃码字 |

共同 split hash：`fe241a5bd04c1aba7c5312aa410fc4c653a2c14dd68e58e9015b5271eb4c67f9`。

共同 noise hash：`eb786f64ad9bf24efea0826e657708af4a1f081023c1eca47b0106fc2765d3a1`。

## 结果

| 方法 | seed 0 精度 | 旧代理 | 相对旧代理 | 球式共识教师 | 相对球式教师 |
|---|---:|---:|---:|---:|---:|
| KRD faithful port | 72.45 | 68.44 | +4.01 | 81.42 | -8.97 |
| VQGraph faithful port, GPU run A | 64.91 | 76.36 | -11.45 | 81.42 | -16.52 |
| VQGraph faithful port, GPU run B | 66.91 | 76.36 | -9.45 | 81.42 | -14.51 |

VQGraph clean-label 诊断为 82.48%，明显高于两次 uniform 0.4 运行。这一差异说明标签噪声影响了该移植中的教师与学生，但单 seed 和 GPU 波动不支持更一般的鲁棒性结论。

## 最终决策

1. **KRD：建议继续 5 seeds，但只在论文继续保留 KRD 行的前提下。** 忠实移植相对旧代理改变了 4.01 个百分点，已超过预注册的 1 个百分点门槛。正式稿不能再把旧代理结果当作 KRD 的忠实结果。
2. **VQGraph：暂不扩展。** 忠实移植通过了 clean-label sanity check，但相同 seed 的两次 GPU 运行相差 2.01 个百分点，且两次均明显弱于旧代理和球式教师。若论文坚持保留 VQGraph，应先解决或量化运行非确定性，再做 5 seeds；若不准备完成该比较，应删除 VQGraph 数值行，而不是继续使用表现更好的软码本代理代替原方法。
3. **不修改论文。** 本轮仅完成 seed-0 决策实验。任何表格替换都需要成对 5-seed 结果。

## 证据文件

- `krd_seed0.csv`：KRD 有效 smoke。
- `krd_seed0_repeat.csv`：KRD 同 seed 确定性复核。
- `vqgraph_paper_config_uniform0p4_seed0.csv`：VQGraph 最新 GPU smoke；需结合报告中的同 seed 波动解读。
- `vqgraph_paper_config_clean_seed0.csv`：VQGraph clean-label sanity check。
- `mlp60_smoke_audit.csv`：同 seed 对账与协议摘要。
- `execution_plan_mlp60_krd_vqgraph_faithful_seed0_20260905.md`：预注册计划。
