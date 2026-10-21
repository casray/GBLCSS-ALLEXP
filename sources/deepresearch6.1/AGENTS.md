# Project Guidance for Codex

## Research Goal

本项目目标是基于 GRACE 或类似节点级图对比学习框架，研究 Granular Ball Computing 与 Graph Contrastive Learning 的有效结合方式，形成一篇可投稿的研一阶段论文。

后续 Codex 在提出新方法、修改代码、跑实验、总结结果之前，必须先理解本项目的研究目标、已有文献材料、已有实验经验、失败模式和科研边界。

## Required Reading Before Any New Experiment

任何新实验前，Codex 必须优先阅读以下材料：

- `01_literature_and_research_map.md`
- `02_experiment_history_and_lessons.md`
- 项目中已有的重要实验日志、配置文件、结果文件和总结文档

## Document Lineage and Usage

- `01_literature_and_research_map_ai_optimized.md` 是由 `01_literature_and_research_map.md` 整合形成的 AI-facing 优化版。需要快速判断实验方向时可先读优化版；涉及论文 claim、逐篇文献证据、公式、数据集细节或不确定信息时，必须回查 `01_literature_and_research_map.md`。
- `03_cross_direction_meta_review.md` 是由 `02_experiment_history_and_lessons.md`、`02_experiment_history_and_lessons 2.md`、`02_experiment_history_and_lessons 3.md`、`02_experiment_history_and_lessons 4.md`、`02_experiment_history_and_lessons 5.md`、`02_experiment_history_and_lessons 6.md` 整合形成的跨方向复盘材料。
- `deep-research-report.md` 是由 `02_experiment_history_and_lessons.md`、`02_experiment_history_and_lessons 2.md`、`02_experiment_history_and_lessons 3.md`、`02_experiment_history_and_lessons 4.md`、`02_experiment_history_and_lessons 5.md`、`02_experiment_history_and_lessons 6.md` 整合形成的深度研究报告。
- 上述整合版文档可以用于快速建立全局判断，但不得替代原始实验历史文件中的配置、日志、结果表、seed、mean/std 和失败记录追溯。

TODO: 如果 `01_literature_and_research_map.md` 不存在，需要先创建并整理文献脉络、研究问题和可行方向。

TODO: 如果 `02_experiment_history_and_lessons.md` 不存在，需要先创建并整理已有实验、负结果、失败原因和经验边界。

TODO: 如果实验日志、配置、结果表或阶段总结缺失，需要先补齐最小可追溯记录，再设计新实验。

## Current Research Preference

当前研究倾向是将粒球优先视为：

- 局部语义关系单元
- 置信度信号
- soft structural constraint
- weak view relation teacher
- robust local structure prior

当前不优先将粒球视为：

- hard positive label
- hard negative filter
- 过强 prototype pulling 模块

除非有明确的置信度建模、消融实验和跨数据集证据，不应把粒球机制设计成强制标签或强拉近约束。

## Avoid Repeating Failed Patterns

后续实验必须避免重复以下模式：

- 不要把 same ball membership 直接当成二值正样本标签，除非有置信度控制。
- 不要只用粒球做简单负样本过滤。
- 不要随意修改 baseline evaluation。
- 不要改变 train / validation / test split。
- 不要只凭单个数据集提升判断成功。
- 不要比较不同 seed 或不同训练协议下的结果。
- 不要把小于 std 波动范围的提升描述成显著改进。

如果确实需要触碰上述边界，必须先写明原因、风险、对照设置和可验证标准。

## Required Format for Every New Experiment

每个新实验必须记录以下内容：

- 研究假设
- 粒球扮演的角色
- 方法动机
- 代码改动
- 配置改动
- 预期机制
- 成功标准
- 失败标准
- 实验结果
- 最终决策：继续、修改、暂缓或放弃

缺少上述信息的实验不得作为论文结论依据。

## Experiment Quality Rules

所有实验必须保留并可追溯以下信息：

- config
- raw log
- result table
- mean and std
- seed setting
- baseline setting
- success or failure report

实验比较必须使用相同数据集划分、相同训练协议、相同评估流程和可比 seed 设置。若任何条件不同，必须明确标注为不可直接比较。

## Paper Claim Boundary

Codex 在任何实验总结、阶段总结或论文草稿中，必须区分以下四类内容：

- 已经被实验支持的结论
- 仅为合理猜测的解释
- 尚未验证的假设
- 已经被负结果削弱的方向

不得把猜测写成结论，不得隐藏负结果，不得将单次运行或小于标准差波动范围的提升包装成稳定改进。
