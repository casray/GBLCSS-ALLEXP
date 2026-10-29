# 2026-10-29 最近实验与阶段判断

更新日期：2026-10-29。该页汇总 2026-10-27 至 2026-10-29 新增的实验记录；每个分支的完整配置、代码、结果表、审计和负结果见下方链接。不同分支协议各自独立，指标按原报告解释。

## 本轮重点结果

| 实验 | 结果与决策 |
|---|---|
| GCN + SUPPORT/D1/C1 三 seed | 五图带噪汇总中，联合学生为 77.464 ± 0.449% Accuracy、70.862 ± 1.401% Macro-F1；CE 学生为 74.905 ± 0.655% / 67.987 ± 1.001%。详见报告中的逐图、逐条件配对结果及全部负差。 |
| R2 教师蒸馏补齐 | 五图 clean/U.4/U.6 汇总中，完整教师蒸馏学生为 73.692 ± 0.488%，CE 教师蒸馏学生为 71.621 ± 0.597%；相对 CE 学生配对增量为 +2.071 ± 0.136 pp。当前范围没有 Pair 条件，完整表见报告。 |
| 固定特征与粒球目标交叉 | 在 Coauthor-CS 上，固定 GraphECL 学生并换成 GRACE 目标，测试增量 +0.704 pp；Photo 上特征来源与目标来源的作用不同，不能用单一排序概括。详见交叉矩阵。 |
| R2 读出和单位化 | 效果依图而变。五图 SUPPORT 单位化测试平均 +0.470 pp，但验证平均 −0.113 pp；Cora/CS 为明显正向，CiteSeer/CoraFull 为负，Photo接近零。暂不全图替换默认读出。 |
| GB-ContextECL | 五图原生 ECL+SUPPORT 平均 74.018%；固定旧目标为 73.994%（−0.023 pp），新表示重新构球为 73.952%（−0.065 pp），主推进门未通过。CS 重新构球 +0.342 pp，Photo −0.728 pp；暂停当前规格扩展，保留数据集分化供机制诊断。 |
| 路线 1：粒球关系蒸馏筛查 | 缓存级增量 AUROC 为 +0.007148，低于预设 +0.01 门槛；K-means 原型对照为 −0.000769，Brier 回归改善量为 −0.000558。停止当前 KL 定义的编码器训练。 |
| 交叉拟合标签感知对比 | 相对同预算折内投票，CS/Photo 测试差为 +0.043 ± 0.039 / +0.411 ± 0.158 pp；验证宏增量 +0.251 pp，低于 +1 pp 推进门。相对原全量支持，CS −0.062 pp、Photo −0.439 pp；Photo 目标 Brier 恶化 +0.002723。当前两折规格暂缓。 |

## 新增实验目录

| 目录 | 入口 |
|---|---|
| [gcn_support_d1_c1_3seed_20261027](<sources/gb_c1_ratio_success_20260609/results_ratio_c1_20260609/experiments/gcn_support_d1_c1_3seed_20261027>) | [REPORT_zh.md](<sources/gb_c1_ratio_success_20260609/results_ratio_c1_20260609/experiments/gcn_support_d1_c1_3seed_20261027/REPORT_zh.md>); [HANDOFF.md](<sources/gb_c1_ratio_success_20260609/results_ratio_c1_20260609/experiments/gcn_support_d1_c1_3seed_20261027/HANDOFF.md>); [PLAN.md](<sources/gb_c1_ratio_success_20260609/results_ratio_c1_20260609/experiments/gcn_support_d1_c1_3seed_20261027/PLAN.md>) |
| [graphecl_gb_context_20261028](<sources/gb_c1_ratio_success_20260609/results_ratio_c1_20260609/experiments/graphecl_gb_context_20261028>) | [REPORT_zh.md](<sources/gb_c1_ratio_success_20260609/results_ratio_c1_20260609/experiments/graphecl_gb_context_20261028/REPORT_zh.md>); [PLAN.md](<sources/gb_c1_ratio_success_20260609/results_ratio_c1_20260609/experiments/graphecl_gb_context_20261028/PLAN.md>); [COST_zh.md](<sources/gb_c1_ratio_success_20260609/results_ratio_c1_20260609/experiments/graphecl_gb_context_20261028/budget/COST_zh.md>) |
| [graphecl_honest_pilot_20261029](<sources/gb_c1_ratio_success_20260609/results_ratio_c1_20260609/experiments/graphecl_honest_pilot_20261029>) | [RESULT_INTERPRETATION_zh.md](<sources/gb_c1_ratio_success_20260609/results_ratio_c1_20260609/experiments/graphecl_honest_pilot_20261029/RESULT_INTERPRETATION_zh.md>); [REPORT_zh.md](<sources/gb_c1_ratio_success_20260609/results_ratio_c1_20260609/experiments/graphecl_honest_pilot_20261029/REPORT_zh.md>); [PLAN.md](<sources/gb_c1_ratio_success_20260609/results_ratio_c1_20260609/experiments/graphecl_honest_pilot_20261029/PLAN.md>) |
| [r2_teacher_dist_csphoto_20261027](<sources/gb_c1_ratio_success_20260609/results_ratio_c1_20260609/experiments/r2_teacher_dist_csphoto_20261027>) | [REPORT_zh.md](<sources/gb_c1_ratio_success_20260609/results_ratio_c1_20260609/experiments/r2_teacher_dist_csphoto_20261027/REPORT_zh.md>); [HANDOFF.md](<sources/gb_c1_ratio_success_20260609/results_ratio_c1_20260609/experiments/r2_teacher_dist_csphoto_20261027/HANDOFF.md>); [PLAN.md](<sources/gb_c1_ratio_success_20260609/results_ratio_c1_20260609/experiments/r2_teacher_dist_csphoto_20261027/PLAN.md>) |
| [ssl_backbone_csphoto_20261027](<sources/gb_c1_ratio_success_20260609/results_ratio_c1_20260609/experiments/ssl_backbone_csphoto_20261027>) | [run.py](<sources/gb_c1_ratio_success_20260609/results_ratio_c1_20260609/experiments/ssl_backbone_csphoto_20261027/run.py>); [contrastive_family.py](<sources/gb_c1_ratio_success_20260609/results_ratio_c1_20260609/experiments/ssl_backbone_csphoto_20261027/contrastive_family.py>); [downstream.py](<sources/gb_c1_ratio_success_20260609/results_ratio_c1_20260609/experiments/ssl_backbone_csphoto_20261027/downstream.py>) |
| [ssl_cross_sources_20261027](<sources/gb_c1_ratio_success_20260609/results_ratio_c1_20260609/experiments/ssl_cross_sources_20261027>) | [REPORT_zh.md](<sources/gb_c1_ratio_success_20260609/results_ratio_c1_20260609/experiments/ssl_cross_sources_20261027/REPORT_zh.md>); [PLAN.md](<sources/gb_c1_ratio_success_20260609/results_ratio_c1_20260609/experiments/ssl_cross_sources_20261027/PLAN.md>) |
| [ssl_noisy_norm_extension_20261028](<sources/gb_c1_ratio_success_20260609/results_ratio_c1_20260609/experiments/ssl_noisy_norm_extension_20261028>) | [REPORT_zh.md](<sources/gb_c1_ratio_success_20260609/results_ratio_c1_20260609/experiments/ssl_noisy_norm_extension_20261028/REPORT_zh.md>); [PLAN.md](<sources/gb_c1_ratio_success_20260609/results_ratio_c1_20260609/experiments/ssl_noisy_norm_extension_20261028/PLAN.md>) |
| [ssl_readout_control_20261027](<sources/gb_c1_ratio_success_20260609/results_ratio_c1_20260609/experiments/ssl_readout_control_20261027>) | [REPORT_zh.md](<sources/gb_c1_ratio_success_20260609/results_ratio_c1_20260609/experiments/ssl_readout_control_20261027/REPORT_zh.md>); [PLAN.md](<sources/gb_c1_ratio_success_20260609/results_ratio_c1_20260609/experiments/ssl_readout_control_20261027/PLAN.md>) |
| [ssl_readout_factorial_20261027](<sources/gb_c1_ratio_success_20260609/results_ratio_c1_20260609/experiments/ssl_readout_factorial_20261027>) | [REPORT_zh.md](<sources/gb_c1_ratio_success_20260609/results_ratio_c1_20260609/experiments/ssl_readout_factorial_20261027/REPORT_zh.md>); [PLAN.md](<sources/gb_c1_ratio_success_20260609/results_ratio_c1_20260609/experiments/ssl_readout_factorial_20261027/PLAN.md>) |

## 项目接续文档

- [文献与研究地图](<sources/gb_c1_ratio_success_20260609/results_ratio_c1_20260609/01_literature_and_research_map.md>)：更新至 2026-10-29。
- [实验历史与经验](<sources/gb_c1_ratio_success_20260609/results_ratio_c1_20260609/02_experiment_history_and_lessons.md>)：包含本轮结果、门槛和停止决定。
- [会话交接](<sources/gb_c1_ratio_success_20260609/results_ratio_c1_20260609/99_session_handoff.md>)：供下一轮研究接续。
- [实验目录索引](EXPERIMENT_INDEX.md) 提供完整归档入口；[逐文件清单](manifests/update_20261029/files.csv)记录源路径、字节数与 SHA-256。

## 归档边界

本次新增 9 个实验/代码目录，共 281 个文本文件；另更新 3 份项目上下文。模型权重、数组缓存、原始数据未上传，仍依项目既有清单留在本地。
