# Issue Traceability

本表保留三份 source review 的全部 32 个原始 issue；一行对应一个原始 ID。`Decision` 同时记录综合裁决、最严格 claim action 与所属 consolidation package。初始状态全部为 `OPEN`；MET-01 在完成现有资产审计且确认 tie 非零并具有 material 影响前，不得改为 `REQUIRES_EXPERIMENT`。

| Issue ID | Severity | Decision | Target section | Verification | Status |
|---|---|---|---|---|---|
| NAR-01 | P0 | ACCEPTED: DELETE/WEAKEN → PKG-P0-01 | 摘要 L19；引言 L29；§2.3 L67 | 全文排他性 novelty 词搜索；所有保留差异均有原论文证据或非排他表述 | OPEN |
| NAR-02 | P1 | MODIFIED: WEAKEN → PKG-P1-01 | 标题；摘要；贡献 2；§5.3.2；§7 | 标题—摘要—贡献—结论统一为局部共识框架的粒球实例；明确不主张优于 kNN | OPEN |
| NAR-03 | P1 | ACCEPTED: WEAKEN → PKG-P1-02 | Highlight 5；摘要；§1；§6；§7 | 蒸馏被标为 secondary extension；CoraFull 反例与 kNN 近似结果未遗漏 | OPEN |
| NAR-04 | P1 | MODIFIED: WEAKEN → PKG-P1-03 | Highlights；摘要；§1；§5.1；§7 | 24/25、14/15、五主加六扩展和 transductive X-only 适用域逐处一致 | OPEN |
| NAR-05 | P1 | MODIFIED: WEAKEN → PKG-P1-03 | 摘要；引言；§5.5；§7 | 删除隐私保证与普遍低时延；保留 CoraFull 不更快及指定硬件边界 | OPEN |
| NAR-06 | P1 | MODIFIED: TEST → PKG-P1-01 | §2.2–2.4；引言；贡献 | 最近工作矩阵逐项有原论文 locator；未核验项不进入事实性正文 | OPEN |
| NAR-07 | P1 | ACCEPTED: KEEP/RESTRUCTURE → PKG-P1-04 | §5.1–5.2 Table 3；§6 Table 5–6 | 正式方法、代理和跨资源参照视觉分区；每表只陈述允许问题 | OPEN |
| NAR-08 | P2 | ACCEPTED: WEAKEN → PKG-P2-01 | 摘要；§1 贡献；§7 | 三条贡献互斥可映射；蒸馏单列 secondary extension | OPEN |
| NAR-09 | P2 | MODIFIED: WEAKEN/DELETE → PKG-P1-08 | §5.3.2；§5.4 | kmeans、noisy-CE GRACE、MLP-hidden 无 raw claim 全部删除；其余限定数据集与 seed | OPEN |
| NAR-10 | P2 | ACCEPTED: DELETE → PKG-P2-01 | §8；附录索引 | 删除链接占位与“当前、已补齐、暂不进入”等内部制作态措辞 | OPEN |
| MET-01 | P1 | MODIFIED: TEST → PKG-P1-09 | §3.4.1–3.4.2；算法 1；附录 | 先冻结 tie 暴露率、节点占比、α 质量和置换审计；仅非零且 material 时升级实验 | OPEN |
| MET-02 | P1 | ACCEPTED: WEAKEN → PKG-P1-05 | 摘要；§1；§3.4.2；§4.2–4.3 | 全文区分构球 label-independent、noisy-label loss 与 clean-validation selection | OPEN |
| MET-03 | P1 | MODIFIED: TEST → PKG-P1-06 | §4.2–4.4；§5.1 | 25 settings 的 method/seed 行全部连接 split/noise hash、污染计数和 source SHA-256 | OPEN |
| MET-04 | P1 | ACCEPTED: WEAKEN → PKG-P1-07 | §4.4；§8；material supplement | downstream replay 与端到端复现能力分级；资产存在性和 hash 逐项核验 | OPEN |
| MET-05 | P1 | MODIFIED: TEST → PKG-P1-06 | §4.1；§4.4；附录 | 冻结表含候选网格、开发集、选择指标、规则、时间点及 dataset-specific 参数 | OPEN |
| MET-06 | P1 | ACCEPTED: WEAKEN → PKG-P1-02 | §3.5；§6 Table 5 | 列明基础到蒸馏同时变化的五项；所有机制句降为 arm-level association | OPEN |
| MET-07 | P1 | MODIFIED: WEAKEN → PKG-P1-08 | §5.3.1 | “独立监督”改为额外非随机信号；LOO 对默认目标未过 Holm 与 post-hoc 边界齐全 | OPEN |
| MET-08 | P1 | MODIFIED: DELETE/WEAKEN → PKG-P1-08 | §5.3.2；§7 | 删除 kmeans；约 10 票限定 Cora/CiteSeer；large-ball 同时变化参数全部披露 | OPEN |
| MET-09 | P1 | MODIFIED: DELETE → PKG-P1-08 | §5.4 | 仅保留 Cora seed-0 可追溯 SSL 几何诊断；无 raw 精确数值和跨数据集泛化均删除 | OPEN |
| MET-10 | P2 | MODIFIED: KEEP/FORMALIZE → PKG-P2-02 | §3.1；§3.5；算法 1/2 | 符号、LOO 节点公式、checkpoint 指标与 epoch tie rule 均可独立实现 | OPEN |
| MET-11 | P2 | MODIFIED: WEAKEN → PKG-P2-02 | §3.6；§5.5 | 复杂度显式含候选、n_init、迭代与排序；seed-0 表称 profile 且环境字段闭合 | OPEN |
| EXP-01 | P1 | ACCEPTED: WEAKEN → PKG-P1-05 | §1；§3.4.2；§4.2–4.3 | 摘要、算法、协议均明确 clean validation 参与 checkpoint selection | OPEN |
| EXP-02 | P1 | MODIFIED: DELETE → PKG-P1-08 | §5.4 | 删除 noisy-CE GRACE 与 MLP-hidden 缺失资产 claim；CiteSeer GraphMAE 反例不被泛化 | OPEN |
| EXP-03 | P1 | MODIFIED: WEAKEN → PKG-P1-08 | §5.3.2 | kNN test 峰值标为探索性；验证峰值差异、数据集票数差异和多参数共变被披露 | OPEN |
| EXP-04 | P1 | MODIFIED: TEST → PKG-P1-06 | §4.2–4.4；§5.1；附录 A | protocol matrix 覆盖 pair/instance 和五主数据集；同 realization 可逐行审计 | OPEN |
| EXP-05 | P1 | MODIFIED: TEST → PKG-P1-06 | §4.4；§5.1；附录 B | 三个 25-test 族标为作者定义且非外部预注册；现有 75 p 值完成单一 Holm 敏感性 | OPEN |
| EXP-06 | P1 | ACCEPTED: WEAKEN → PKG-P1-04 | §5.2 Table 3；§6 Table 5–6 | 代理不参与正式粗体；跨资源差值按全精度重算；无等效或同资源胜负语言 | OPEN |
| EXP-07 | P1 | MODIFIED: WEAKEN/CORRECT → PKG-P1-08 | §5.3.6 | 5% 相对最强 selector 的范围改为约 5.6–5.8 pp，并核对候选集合 | OPEN |
| EXP-08 | P1 | MODIFIED: WEAKEN → PKG-P1-03 | 摘要；贡献 3；§4.1；§5.1；§7 | 统一为五主完整谱加六扩展 uniform 0.4；无时间戳时不称“冻结后” | OPEN |
| EXP-09 | P2 | MODIFIED: WEAKEN → PKG-P2-02 | §4.4；§5.4 | 稳定性句均限定三数据集、三个 H-seed、uniform 0.4；ARI 不外推 | OPEN |
| EXP-10 | P1 | ACCEPTED: TEST → PKG-P1-07 | Table 1–6；§8；附录索引 | `manuscript_table_lineage.csv` 逐 cell 含 raw path/hash/filter/formula；表号映射与派生列物化 | OPEN |
| EXP-11 | P2 | MODIFIED: WEAKEN → PKG-P1-08 | §5.3.1；附录 | Brier/NCR/CDR 的公式、分母、范围和 post-hoc clean-label 边界完整 | OPEN |

Validation target: source issues = 32；source P0/P1 = 25；traceability rows = 32；traceability P0/P1 = 25；silent drops = 0。

Initial status count: `OPEN=32`；`ACKNOWLEDGED=0`；`REQUIRES_EXPERIMENT=0`。
