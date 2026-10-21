# MLP-58 正式 GB-LCSS 教师设计

## 目标

验证正文正式目标 $t_v$ 能否直接训练 GCN(X) 教师，并通过全节点蒸馏提高 X-only 学生精度。

## 方法

- 粒球：冻结 GRACE H 上的标签无关几何划分。
- 教师输入：仅原始节点属性 X 与邻接矩阵 A。
- 教师监督：`target_beta=0.5`、`leave_one_out=true`、`tie_mode=distribution_only`，与基础 GB-LCSS 完全一致。
- 学生：两层 X-only MLP，在全部节点上拟合教师预测。
- 对照：使用相同球规模分布和正式目标规则的 shuffled 分组教师。

## 实验范围

Cora、CiteSeer、CoraFull，uniform 0.4/0.6，train-only noise，percent 0.2/0.1/0.7，seeds 0--4，共 60 条新记录。

## 判据

1. 相对 GCN(X)+CE：每个噪声率至少 2/3 数据集平均提高不低于 1pp，且至少 4/5 seeds 正向。
2. 相对当前 beta=0 X-only 球式教师：不得出现超过 1pp 的系统性平均回退。
3. 正式真实粒球相对 shuffled：每个噪声率至少 2/3 数据集领先不低于 1pp，且至少 4/5 seeds 正向。

## 解释边界

通过说明正式 GB-LCSS 目标可以迁移到图教师。失败则说明基础 MLP 与图教师对目标锐度的偏好不同，不否定基础 GB-LCSS。
