# 投稿完整性审计：最终闸门（2026-07-11）

审计对象：`docs/paper_final_zh_v6_table34_completed_20260710.md` 及其引用的正式实验产物。历史探索目录只在被正文引用时纳入；未进入论文证据链的失败分支不视为投稿结果。

## 一句话结论

**论文核心数值不是编造的，主协议也可由当前代码复现；但当前稿件仍不能投稿。** 阻塞原因不是主方法数值，而是若干基线与所署论文名称不相符、正文存在可由代码直接证伪的机制事实、复杂度引用外推错误，以及参考文献未完成。

## 已通过项目

1. **核心数值可追溯。** 主表 1625 行、表 3 的 225 行、表 4 的 150 行、表 5 的 200 行均追溯到 seed-level CSV；正文对应单元格与重算 mean±std 的 mismatch 为 0。
2. **协议为 train-only。** 所有进入正文的正式行均为 `noise_scope=train`，`modified_val_count=modified_test_count=0`，且配对方法的 split/noise hash 一致。
3. **独立重跑通过。** Cora、CiteSeer、CoraFull 的 LCSS、shuffled、MLP+CE seed0 共 9 行，由当前代码重跑后 test 浮点值、split hash、noise hash 与历史原始行完全一致。
4. **主方法无 clean-label 训练泄漏。** clean labels 只用于 NCR/CDR、purity 等事后诊断。现有回归测试确认任意篡改 clean labels 后，几何 SoftDist 的 `soft_target/alpha/pseudo_label` 不变；assignment 本身只由 H 几何构造。
5. **GRACE H 来源可信。** 五份正式 H 都是有限、行归一化张量；NoisyGL 与上游 GRACE 的标签顺序、特征行和规范化边集合逐项一致。Cora/CiteSeer 可由 checkpoint 数值重提取，最大误差约 1e-7。
6. **旧入口的全标签污染已修复。** `single_exp.py` 与 `total_exp.py` 原本未传 `eligible_mask`，会污染 val/test；现已固定为 `eligible_mask=train_masks` 并有回归测试。该问题没有污染已审计正式表，因为正式表使用的是 `run_standard_downstream.py`。

## P0：投稿前必须处理

### P0-1 基线名称与实现不匹配

Phase 3 的论文-代码逐项审计显示：

- 主表中的 CT、ANL、FINE、DivideMix-lite 未实现对应论文的核心算法，不能继续作为这些正式方法的实验结果；
- Table 3 的 Graph-MLP、FF-G2M、KRD、VQGraph 与原文存在结构性差异，其中 FF-G2M/KRD/VQGraph 缺失的恰是各论文定义方法身份的核心模块；
- GSDN、GLNN-style 可在明确写 `adapted/style` 的前提下作为机制对照，但不能声称官方复现；
- TFR 是官方代码移植，ProCon/GD2 是论文导向的本地重实现，可保留为清楚标注的跨部署参考，但不能与官方 tuned 数字等同。

处置只能二选一：

1. 忠实移植并重跑；或
2. 删除正式方法名，改成不借用论文品牌的本地机制变体，并降低“胜过 11 个方法/强 G2M 基线”的 claim。

仅写 `adapted core implementation` 不足以解决问题，因为当前若干实现并未保留 core。

### P0-2 正文机制事实与正式代码冲突

必须修正以下句子：

- `H in R^{N x 256}`：Cora 正式 H 是 128 维；应写 `H in R^{N x d_h}` 并披露各数据集配置来源。
- `实际分组数 22-133`：当前实现有硬上限 64，不可能产生 133 个球。
- `规模上限在多数数据集未触发`：正式 25 次重建中 19/25 触顶；CoraFull、Coauthor-CS、Amazon-Photo 均为 5/5 触顶。
- `学生约九万参数`：实际分别为 Cora 92,359、CiteSeer 237,574、CoraFull 562,182、Coauthor-CS 436,687、Amazon-Photo 48,392。应按数据集报告范围，而非写单一数字。
- 大图“大球”不能只归因于自然几何停止；固定 `B_max=64` 是直接原因之一。

### P0-3 复杂度公式不成立

正文 `O(N d log B_max) [34]` 不是当前代码的已证明复杂度。[34] 分析的是其 GBC/GBkNN 框架。当前实现每次成功分裂前扫描当前全部分区，并最多执行 `B_max-1` 次分裂；应保守写为 `O(B_max N d + T_2means)`，当 `B_max=64` 固定时对 N 近似线性。

kNN 的 `O(|V_L|^2 d)` 只适用于当前显式相似度矩阵实现，不能暗示所有 kNN/ANN 都是二次复杂度。

### P0-4 文献表不完整或有错误

- `[25]`, `[28]`, `[30]-[32]`, `[35]-[36]` 仍是中文占位符；
- SA-MLP 已有 TMLR 2024 正式版本；
- MgCNL 是 IEEE TASE，不是 IEEE TII；
- RKD-MLP 是 OpenReview submission，不能伪装成已发表论文；
- 表 5 的 TFR、ProCon、GD2、CLNode、CRGNN 缺正式引用；
- Granular Ball Sampling 应补正式 TNNLS 34(4), 2023 卷期与 DOI。

### P0-5 first claim 必须保留限定

GSDN 已经做过 graph-free MLP 的非对称训练标签噪声实验。不能写“首次研究 graph-free label noise”。安全版本是：

> 据我们所知，首次在该部署设定下将训练标签噪声作为中心问题，并在多噪声类型/噪声率下与标签噪声学习策略族进行统一协议的系统比较。

即便这样，也要在附录给出文献筛选矩阵，避免仅凭“13 篇”口头断言。

## P1：强烈建议投稿前处理

1. **SSL 随机性未进入误差条。** 每个数据集的 5 个下游 seed 共用同一份 H；std 只覆盖 split/noise/model randomness，不覆盖 GRACE pretraining randomness。正文必须披露，不能称完整管线稳定性。
2. **v2 H 缺 encoder checkpoint。** CoraFull/Coauthor-CS/Amazon-Photo 仅保存 H、metadata、losses，不能从保存权重精确再生成。应补保存 checkpoint 的预训练脚本，并最好重训一次验证 hash/下游容差；至少发布现有 H hash。
3. **teacher 每 epoch 读取 test accuracy。** 当前 checkpoint 由 val 选择，test 没参与自动选择，因此没有观察到直接 test leakage；但循环内计算 test 不符合严格评测卫生，易引发人为调参质疑。应改为加载最佳 val checkpoint 后只评一次 test。
4. **隐私措辞过强。** 无图推理减少部署端对边的访问，但不是隐私算法。写“图结构可能因隐私/访问控制边界不可用”，不要写“方法保护图隐私”。
5. **结构噪声结果是部署属性。** X-only student 对推理邻接扰动不变属于 by-construction invariance，不是学得的结构去噪能力；不能包装成“双重鲁棒方法”。
6. **监督表征抹平局部簇是机制解释，不是已证实普遍定律。** 当前实验只支持“在本管线和已测设置下发生塌缩”，应降低因果措辞。
7. **Table 5 的 CLNode 当前实现使用 val labels 参与预模型选择和 difficulty 构造。** 它已从正文数值行排除；在修复并重跑前不得恢复。

## P2：论文质量改进

1. 报告每个方法的训练资源：是否用图、是否用全部无标签节点、是否用 teacher、推理是否严格 X-only。
2. 在主表中增加逐 seed paired delta 或附录原始 seed 表；5-seed 下不要用“显著”描述仅略高于 std 的差值。
3. 将 `LCSS` 的贡献明确写成局部共识软监督配方。kNN/kmeans 平替证据意味着不能把“粒球优于邻域方法”作为 claim。
4. 对参数量、训练时间、H 预训练成本分别报告；不要用 Cora 的 92k 参数代表全部数据集。
5. 明确 H 为同一数据集跨噪声类型/噪声率复用的固定 artifact，并提供 SHA256 manifest。

## 投稿闸门

当前状态：**FAIL / DO NOT SUBMIT**。

最低通过条件：

1. 删除或忠实重跑 P0-1 中不忠实的正式方法行；
2. 修正 H 维度、球数、参数量、复杂度四类事实；
3. 完成全部参考文献并补 Table 5 引用；
4. 收窄 first/privacy/structure-robustness claim；
5. 重新生成正文表格并再跑数值一致性测试。

满足以上条件后，核心 LCSS 数值、train-only 协议、H 对齐和独立复现证据足以支撑一篇诚实的 scope-limited 论文。

## 审计产物索引

- `docs/submission_integrity_audit_phase1_20260710.md`
- `docs/submission_integrity_audit_phase2_tables3_5_20260711.md`
- `docs/submission_integrity_audit_phase3_method_fidelity_20260711.md`
- `docs/submission_integrity_audit_phase4_grace_provenance_20260711.md`
- `docs/submission_integrity_audit_phase5_independent_rerun_20260711.md`
- `docs/submission_integrity_audit_phase6_literature_claims_20260711.md`
- `runs/submission_integrity_audit_20260710/`
