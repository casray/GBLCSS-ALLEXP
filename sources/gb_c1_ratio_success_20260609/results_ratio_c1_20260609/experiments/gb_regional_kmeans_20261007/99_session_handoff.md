# 匹配区域数 k-means 对照：交接

2026-10-07。用户批准设计并开始下一轮控制实验；已完成 30 个 KM 学生训练和一个 GB 工程回放，没有剩余训练进程。

## 本轮目标与来源

问：固定 ERASE 教师和学生协议后，GB 区域蒸馏的增益是否可以由相同 GRACE 空间的 k-means 分组重现？
前序：`../gb_regional_distillation_seed0_20261007/` 为五臂 seed0 探索；`../gb_regional_stability_20261007/` 为学生 seeds 1/2 和统一密集矩阵数值实现的 seed0 参考。
必须复用后者的数值版本和活动清单 `frozen_v2_inputs.json`，不能混入旧 index_add 的 GB seed0 数字。
本轮读了主实验 seed0 配置/成员缓存、NoisyGL 表示加载及归一化/聚类实现、文献地图和历史33。

## 协议

五数据集 × uniform0.4/0.6 × student seeds0/1/2，教师/划分/噪声/表示均固定seed0。
同一行归一化 GRACE 缓存；KM 区域数匹配 GB：Cora52，其余64。
k-means++，random_state0，n_init10，max_iter300，tol1e-4，Lloyd。每数据集一个分组跨学生seed/噪声复用。
CPU四线程聚类，ROCm GPU学生。MLP64两层BN/dropout.5，Adam.01/.0005，200轮，全节点教师CE+区域KL，lambda1，首次严格验证最优选模。
学生seed只改变初始化和dropout，不能称为三独立协议seed。

## 主要结论与负结果

30对平均：GB−R0 +0.296753pp，KM−R0 +0.281181pp，GB−KM +0.015572pp。
GB−KM14胜1平15负，三seed均值分别−0.043119/+0.052792/+0.037042。
10格三seed均值：GB−R0 10正，KM−R0 9正1负，GB−KM4正6负。
通用区域机制gate通过；粒球独立优势gate失败（三个要求均未达到）。
Cora0.4 KM−R0均值−0.404573，seed0−1.635884pp；不得隐藏。
GB相对KM优势在Cora较强，CoraFull很小；CiteSeer、Coauthor-CS、Photo六格KM均值更高。
固定20/50/100/200轮 GB−KM分别−0.213932/−0.100445/−0.085514/−0.181286pp。
结果削弱GB特有精度优势，支持继续研究通用区域蒸馏。不宣称统计等价或显著性。

## 验证及文件

`test_kmeans.py`损失/梯度/编号置换通过；`replay_check.json` GB回放概率逐元素一致，验证轨迹/最佳轮次一致。
150新检查点GPU重载概率差0，全部200轮、选模、初始化/RNG/协议和活动输入hash核验通过。
`PLAN.md`、`run_kmeans.py`、`analyze.py`、`diagnostics.py`、`COMMANDS.md`；`frozen_inputs.json`活动清单。
`partitions/`五分组；`runs/`30次完整结果；`engineering/GB_replay/`工程复现。
`paired_results.csv`30行，`cell_summary.csv`10格mean/std，`summary.json`、`实验结论.md`。
`fixed_epoch_results.csv`、`selected_diagnostics.csv`、`teacher_results.csv`、`diagnostics_summary.json`。
KM轨迹字段`real_region_kl`是KM实际区域项；比较共同GB/KM指标用selected_diagnostics.csv。

## 下一窗口第一步

先读实验结论。当前决策为修改定位、暂缓GB独有精度claim。优先设计固定教师/学生的更强蒸馏控制，再看区域项是否仍增加收益；尚未启动下一阶段。
本轮没有教师重训、教师空间重分组、新损失、kNN、超参搜索或投稿稿件更改。
独立划分/污染/教师seed、分组seed变化、其他教师及效率实测仍未验证。
