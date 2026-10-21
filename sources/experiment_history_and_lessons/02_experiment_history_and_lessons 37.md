# MLP-61 KRD/VQGraph 五种子正式复现记录

## 研究假设

旧稿中的 KRD 与 VQGraph 仅是机制适配。MLP-61 检验采用原论文核心模块和论文参数后，两条图到 MLP 路线在统一噪声协议下的真实表现，并与球式共识教师作同 seed 配对比较。

## 粒球扮演的角色

粒球不参与 KRD 或 VQGraph。现有球式共识教师仅作为同一数据划分、噪声实现、验证规则和属性输入学生协议下的参照。

## 代码与配置

- KRD 核心来源：`LirongWu/KRD`，commit `32be7e42ab65b1d246cf1591d8bc48dfe7fb6b21`。
- VQGraph 核心来源：`YangLing0818/VQGraph`，commit `60aa1c99782d542c316a37f0394eba934bcf97b0`。
- 统一协议：20%/10%/70% 划分，uniform 0.4 与 0.6，噪声只注入训练标签，seeds 0-4，干净验证集选模，学生推理只输入节点属性。
- 执行设备：AMD Radeon RX 7700 XT，PyTorch ROCm，`cuda:0`。未使用 CPU 回退。

## 结果

KRD 完成 50/50 条。相对球式共识教师，10 个数据集-噪声单元中只有 CiteSeer/uniform 0.6 更高（+2.51 pp），其余 9 个单元更低。50 个配对 seed 中 KRD 胜 6 个。

VQGraph 在 Cora、CiteSeer 和 Amazon-Photo 完成 30 条，在 6 个单元中均低于球式教师。Amazon-Photo 出现明显种子崩溃，uniform 0.4 的两次崩溃仅激活 1-2 个码字，精度约 31.4%。该现象只能作为诊断关联，不能写成因果结论。

CoraFull 与 Coauthor-CS 未获得 VQGraph 正式结果。当前 PyG 端口为全图训练，特征相关层和图重构张量在这两个数据集上不可行。官方实现依赖 DGL 邻居采样，而现有 Windows ROCm/PyTorch 环境无法提供兼容运行时。为保持 GPU 执行和方法身份，没有使用 CPU、PCA、缩小码本或简化结构补跑。

## 审计

- 80/80 保留结果状态正常。
- 所有保留单元均覆盖 5 seeds。
- 80/80 与球式教师的 split hash 和 noise hash 一致。
- 80/80 训练使用图，学生推理不使用图。
- KRD 每条记录均发生可靠性更新且无更新失败。
- VQGraph 每条保留记录均有非零激活码字。

## 结论与决策

1. KRD 可以作为五数据集的 official-core protocol port 报告，但必须说明 CoraFull 使用预注册的 Coauthor-CS 参数映射。
2. VQGraph 不能写成五数据集忠实复现。若保留，只能作为三数据集补充结果并披露规模限制；否则应从主表定量行删除。
3. 本轮不能因结果有利而选择性保留。KRD 在 CiteSeer/uniform 0.6 的正结果必须与其余负结果一并报告。
4. 论文暂不修改，等待作者决定 VQGraph 的呈现方式。

正式报告：`external/NoisyGL/runs/mlp61_krd_vqgraph_faithful_5x5_20260906/mlp61_report.md`。
