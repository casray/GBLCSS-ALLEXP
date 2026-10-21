# Revision Roadmap for the Single Lead Writer

## 1. Purpose and hard boundaries

本文档是 Task 6 单一主笔的施工图，不直接重写 v29，也不替代 claim–evidence ledger。综合输入为三份已通过质量门的独立 review、`review_quality_gate.md`、`claim_evidence_ledger.md` 与 `material_manifest.md`。

- 32 个 source issues 全部保留在 `issue_traceability.md`，不因去重而消失。
- 当前阶段不运行任何新训练实验，不修改 split、baseline evaluation、seed 协议或 v29。
- 能由现有日志、CSV、配置和统计输出完成的 protocol、tie、lineage 与 75-test 审计，不算新实验。
- 文献原文核验与系统检索审计不是实验，但投稿前必须完成；未完成前只能使用非排他性 novelty 表述。
- 所有改稿严格按 `P0 factual/integrity → P1 story/method/evidence → P2 prose/formatting` 顺序执行。
- 重复问题按最高 severity 进入施工顺序；删除无证据 claim 优先于补写解释，严格 claim decision 优先于宽松措辞。

## 2. Consolidation packages in required order

| Package | Source IDs | Highest severity | Decision | Exact target sections | Minimal repair | Key evidence retained | Verification | New experiment |
|---|---|:---:|---|---|---|---|---|---|
| PKG-P0-01 Novelty integrity | NAR-01 | P0 | **ACCEPTED — DELETE/WEAKEN** | 摘要 L19；引言 L29；§2.3 L67 | 删除“尚缺乏系统研究”“尚未得到系统研究”等排他性 research-gap 句；改为“相对较少被直接评估的资源约束组合”，并保留 GSDN 等邻近工作。 | C-022=`UNVERIFIED`；manifest 将 literature map 定位为线索而非正式证据；当前没有数据库、检索式、日期、纳排标准和逐篇定位。 | 全文搜索“首次、尚无、尚未、缺乏系统研究、系统研究”等排他词；逐句检查均有可审计证据或已删除。 | **NO**；系统文献核验不是实验。 |
| PKG-P1-01 Core positioning and nearest-work boundary | NAR-02, NAR-06 | P1 | **MODIFIED — WEAKEN now, TEST literature facts before submission** | 标题；摘要；§1 贡献；§2.2–2.4；§5.3.2；§7 | 主身份改为“几何局部共识软监督框架，粒球为固定预算实例”；不主张粒球优于 kNN。区分继承的递归粒球划分与新增的置信度软目标/资源流。相关工作末加入按 noisy labels、训练期图、推理期图/H、标签无关分组、组级 soft target、全节点蒸馏和粒球角色组织的最近工作矩阵。 | C-001 支持软关系/置信度角色；C-010 显示所测四数据集 k=20 kNN 均高于默认粒球；C-024 只支持“与票数不足解释相容”；C-022 禁止无检索的全球首创；原论文事实当前均需核验。 | 标题—摘要—贡献—结论使用同一从属定位；矩阵每个事实有原论文 locator；无“粒球特异优势”或全球首创句；kNN 反证不被隐藏。 | **NO**；原论文核验/系统检索不是实验。 |
| PKG-P1-02 Distillation credit assignment | NAR-03, MET-06 | P1 | **ACCEPTED — WEAKEN** | Highlight 5；摘要；引言；§3.5；§6 Table 5–6；§7 | 将蒸馏定义为资源更强的 secondary extension。明确基础版到蒸馏版同时改变 `target_beta`、教师输入、图传播、监督覆盖和学生损失；只写 arm-level association：X∥H 与全节点蒸馏缩小差距，局部共识在部分高噪声设置有附加收益。保留 CoraFull 反例和 kNN 教师近似结果。 | C-007=`PARTIAL`；C-019 确认教师 X∥H、`target_beta=0`、学生 X-only；`table4_recomputed.csv` 显示 CoraFull 球式学生低于 GCN[X∥H]+CE 与 kNN 教师；六个 kNN/球式差值均小于 1 pp。 | 从 Highlights 到结论不出现“共识教师单独造成全部提升”“all-node coverage 已被独立识别”或统计等效措辞；Table 5 每个比较只回答其可识别问题。 | **NO**；若未来坚持唯一因果归因，才需 factorial experiment，但本轮删除该强 claim。 |
| PKG-P1-03 Scope, deployment motivation, and dataset coverage | NAR-04, NAR-05, EXP-08 | P1 | **MODIFIED — WEAKEN and make counts explicit** | Highlights；摘要；§1；§3.1；§4.1；§5.1 Table 1–2；§5.5；§7 | 统一写成“同图 transductive、训练期可用图、推理期 X-only”；摘要/贡献明确“5 个主数据集的五噪声谱 + 6 个扩展数据集的 uniform 0.4 宽度审计”。用 24/25、14/15 的精确适用域替代“多数/绝大多数”。删除隐私保证和普遍低时延暗示；说明 X-only 消除邻接依赖但 CoraFull 上不保证更快。无时间戳时把“参数冻结后”改为“使用同一固定配置”。 | C-003 仅支持五主数据集 24/25 与三数据集 14/15；C-005 显示六扩展数据集混合结果；C-018 仅支持指定硬件 profile，CoraFull X-only MLP 慢于 GCN；C-019/C-020 限定训练资源与 transductive 协议；C-015/C-016 否定普遍 clean/feature-noise 增益。 | 摘要、贡献、§4.1、结果和结论的数据集/噪声范围完全一致；搜索“11 个数据集”逐处附覆盖结构；任何 latency/privacy 句均不越过 C-018/C-020。 | **NO** |
| PKG-P1-04 Comparison identity and resource-fair presentation | NAR-07, EXP-06 | P1 | **ACCEPTED — KEEP evidence, restructure presentation** | §5.1–5.2，尤其 Table 3；§6 Table 5–6 | 结果按三个问题重排：同骨干鲁棒监督、局部共识必要条件、增加图教师后的资源差距。Table 3 将 identity-preserving/ported 与 mechanism proxy 分块，只在正式可比块标粗；代理移补充材料。Table 6 明示“跨部署资源差距”，逐数据集报告与最强图依赖参照的差值，不做同资源胜负或等效判断。 | C-006 只允许正式 graph-free 组内排名，KRD/VQGraph 不是官方结果；C-008 给出五数据集跨资源差值并禁止公平排名；external Table 4/5 与 manuscript Table 5/6 存在编号映射。 | 每个表格首句只陈述该表允许回答的问题；代理不参与正式粗体/排名；Table 6 差值从全精度单元重算并注明四舍五入。 | **NO** |
| PKG-P1-05 Clean-validation resource boundary | MET-02, EXP-01 | P1 | **ACCEPTED — WEAKEN** | 摘要；§1；§3.3–3.4.2；§4.2–4.3；算法与资源表 | 全文统一为：“GRACE/构球成员关系标签无关；软目标和训练损失不读取 clean train labels；实验使用独立 clean validation labels 做 checkpoint/超参数选择。”不得把完整开发—训练—选模流程称为 label-free 或完全不依赖干净标签。 | C-021 的允许边界；`protocol_audit_report.md` 明确 clean validation 用于 model/config selection；v29 L381/L385 已承认验证选模。 | 搜索“标签无关、干净标签、label-free、不依赖”等词；每处主体和阶段明确，且算法/实验协议/摘要无矛盾。 | **NO** |
| PKG-P1-06 Protocol, freeze, and statistical audit closure | MET-03, MET-05, EXP-04, EXP-05 | P1 | **MODIFIED — TEST with existing assets** | §4.1–4.4；§5.1；附录 A–B；投稿补充材料 | 生成 machine-readable protocol ledger，将 25 settings 的每个 method/seed 绑定到 split/noise hash、污染计数、source SHA-256，instance 记录 generator/realization hash；补 downstream 参数开发—冻结表，区分 dataset-specific encoder 配置。显著性写明三个 25-test 族为作者定义、非外部预注册；用现有 75 个 p 值做单一 Holm 敏感性重算，并同时保留效应量、95% CI 与方向计数。 | `formal_main_raw.csv` 只直接闭合部分数据集/setting 的 hash；`primary_statistics_raw.csv` 无 hash；phase4/mlp21/mlp23 缺部分审计列；C-002 支持窄协议边界；C-004 支持 24/8/25 的三个族内结果；EXP-05 已确认算术正确但整体 75-test FWER 未控制；manifest 未给可审计冻结时间戳/完整 beta ledger。 | 唯一键覆盖 25 settings × 5 seeds × 对应方法且无重复/缺行；同一 setting/seed 的 split/noise hash 唯一；val/test 修改数为 0；冻结表列候选集、选择规则和允许 dataset-specific 项；75-test 重算由现有数据可复现。 | **NO**；协议导出与统计敏感性重算不是新训练。 |
| PKG-P1-07 Reproduction scope and table lineage | MET-04, EXP-10 | P1 | **ACCEPTED — WEAKEN “complete reproduction” and close assets** | §4.4；§8；附录索引；supplement/asset manifest | 明确区分 exact downstream replay 与 encoder-to-result end-to-end reproduction；当前缺 checkpoint/源码/环境/全配置时禁止“完整复现”。补可执行源码入口、环境版本、命令、seed 派生、全数据集配置/ID 映射和统计脚本；生成逐 manuscript cell 的 `manuscript_table_lineage.csv`，物化 Table 4 派生列与 external/manuscript table 编号映射。 | manifest 仅显式列一份 Cora 蒸馏 YAML；Coauthor-CS/Amazon-Photo 缺 encoder checkpoint；`tables3_5_evidence_manifest.csv` 是文件级而非逐 cell lineage；Table 4 派生列未物化；external Table 4/5 实际对应 manuscript Table 5/6。 | 从原始表到每个 manuscript cell 可按 path/hash/filter/formula 回查；发布资产清单逐项存在且 hash 固定；所有“完整/端到端复现”措辞与实际两级能力一致。 | **NO**；这是资产归档和 lineage 生成。 |
| PKG-P1-08 Local mechanism evidence hygiene | NAR-09, MET-07, MET-08, MET-09, EXP-02, EXP-03, EXP-07, EXP-11 | P1 | **MODIFIED — DELETE unsupported claims; WEAKEN the remainder** | §5.3.1；§5.3.2；§5.3.6；§5.4；§7 | 删除无 raw 的 kmeans、noisy-CE GRACE 52→1/纯度 0.79→0.30/−12.6 pp、MLP-hidden 单球。kNN k=20 只作 test-curve 探索诊断，不称验证选择最优。票数结论限定为数据集特定且只是相容解释；large-ball 同时改变的参数全部披露。LOO 改为“其他成员的额外、非随机信号”，注明相对默认精度未通过 Holm；补 Brier/NCR/CDR 公式和 post-hoc clean-label 边界。标签率数字更正为约 0.2–1.4 pp 到 5.6–5.8 pp。SSL 诊断只保留 Cora seed 0 的可追溯事实，并明确 CiteSeer 不复现同一模式。 | C-009 支持 LOO 相对 random/smoothing 及 NCR/CDR 权衡，但不支持“独立”或唯一机制；C-010/C-024 支持有限 kNN/票数观察，禁止 kmeans 与唯一因果；C-025=`PARTIAL`，只支持 Cora seed-0 CCA-SSG/GraphMAE 几何诊断；`mlp6_aggregate.csv` 给出最强 selector 的 5% 差值 5.61/5.80 pp；CiteSeer GraphMAE 为 64 球。 | 逐句回查 ledger locator；全文不存在三类缺 raw claim；每个数字有数据集/seed/protocol 限定；LOO 指标公式可复算；5% “最强 selector”数字与候选集合一致。 | **NO** |
| PKG-P1-09 Tie safety and confidence terminology | MET-01 | P1 | **MODIFIED — AUDIT FIRST, conditional escalation only** | §3.4.1–3.4.2；算法 1；附录 tie audit | 先从现有正式目标/球内计数/生成日志导出 dataset×setting×seed 审计：tie 球数、受影响节点数/占比、受影响 `Σα` 占总 `Σα`、最小索引选中类别分布，并做类别置换目标检查。无论结果如何，把最小索引规则改为 tie-safe 定义，并把 `α` 称为“启发式共识权重”，除非补出统计校准依据。 | v29 明确最小索引 tie-break、`β=0.5`、`s_min=2`；当 C>2 且两票平分时 `α>0`，目标依类别编号变化。当前没有 tie 暴露率与影响量级证据，不能声称已污染正式性能。 | 先冻结审计表与 material 判据；若 tie=0 或影响可忽略，只验证定义/披露；只有 tie 非零且按预先写明的判据对正式目标 materially relevant，状态才改为 `REQUIRES_EXPERIMENT`，先做代表设置 tie-safe 对照，敏感后再扩展。 | **CONDITIONAL, CURRENTLY NO**；当前只做现有资产审计。 |
| PKG-P2-01 Contribution architecture and submission cleanliness | NAR-08, NAR-10 | P2 | **ACCEPTED — WEAKEN/DELETE** | 摘要；§1 贡献；§7；§8；附录索引 | 将贡献拆成任务/接口、粒球实例化与机制审计三条互斥陈述；蒸馏单列 secondary extension。删除“链接将在投稿阶段补入”“当前已补齐”“暂不进入正文”等内部制作状态，只保留投稿时真实可访问的资产范围。 | C-001 支持局部关系/软目标；C-019 把蒸馏定义为独立扩展；manifest 已承担资产制作状态追踪，正文不需复制流水账。 | 三条贡献无重复、可分别映射到方法和证据；正文无“当前、已补齐、暂不、待补链接”等制作态措辞。 | **NO** |
| PKG-P2-02 Method formalism and reporting precision | MET-10, MET-11, EXP-09 | P2 | **MODIFIED — KEEP method, add formal detail, WEAKEN complexity/stability** | §3.1；§3.5–3.6；算法 1/2；§4.4；§5.4–5.5；附录 | 补验证/测试集合、`β_T, τ, ε, n_min, B_max, δ, F`，给 LOO 节点级公式与 teacher/student checkpoint 选择、epoch tie rule；单列蒸馏伪代码。复杂度显式包含候选尝试、`n_init=10`、最大迭代和排序成本，仅在固定预算下称对 N 近线性；seed-0 成本表称 profile，补硬件、ROCm、PyTorch/PyG、dtype、warm-up、同步与重复次数。H-seed 结论限定为三个数据集、三个已测 H-seed、uniform 0.4 下增益方向为正；ARI 只描述性能相对离散分区更稳定的观察。 | raw 中存在 teacher/student best epoch 但稿件缺选择规则；式 (18) 将主要未知成本吸入 `T_2means`；C-018 只支持指定环境观测成本；latency raw 的 hardware/software 字段不足，FP32 MiB/throughput 尚未物化。C-017 与 `mlp37_variance.csv` 仅支持 H-seed std 约 0.53/0.07/0.44 pp、ARI 约 0.25–0.36 的窄诊断。 | 符号首次出现均定义；两套算法可独立实现；复杂度假设逐项列明；资源表与物化资产逐单元一致且不作跨平台外推；稳定性句均携带数据集、noise、H-seed 数并不外推。 | **NO** |

## 3. Conflict adjudication

| Conflicting issue IDs | Decision | Reason | Evidence used | Risk if wrong |
|---|---|---|---|---|
| NAR-02, MET-08, EXP-03 | **以“局部共识框架的粒球实例”作为主定位；明确默认粒球在所测四数据集不如 k=20 kNN，不主张粒球特异优势。** | 标题中心性不能压过直接反证；粒球仍可作为固定预算、标签无关分组实例保留，但创新必须落在软监督接口与资源流。 | C-001；C-010；`mlp11_knn_curve.csv`；C-024=`PARTIAL`；`mlp13_stageB_summary.csv`。 | 若判断错误，可能过度弱化粒球贡献；但反向错误会把可复算的 kNN 优势隐藏成 novelty overclaim，拒稿风险更高。 |
| NAR-03, MET-06 | **蒸馏降为 secondary extension；只报告 X∥H、图传播、目标与覆盖共同改变后的 arm-level association，不做单因素 credit assignment。** | 现有 Table 5 不是闭合 factorial design，无法把增益唯一归给粒球共识、H 或 all-node coverage。 | C-007=`PARTIAL`；C-019；Cora YAML `target_beta=0, teacher_use_xh=true`；`table4_recomputed.csv`。 | 若判断错误，可能低估共识教师作用；若继续强归因，则会把未识别因果写成已证机制。 |
| MET-01 | **tie audit 先行；初始状态保持 OPEN。只有 tie 非零且 materially affects 正式目标时，才改为 REQUIRES_EXPERIMENT。** | 数学上的类别置换非等变性已成立，但经验暴露率未知；直接重跑会违反最小修复与当前不跑新训练的边界。 | v29 的最小索引 tie-break、`β=0.5`、`s_min=2`；C-001/C-021 不能消除该偏置；quality gate 明确认可 audit-first 路径。 | 若 material 判据过松，可能漏掉真实性能偏置；若过严或未审计就重跑，会扩大实验范围且无法判断必要性。 |
| EXP-05 | **保留三个作者定义的 25-test Holm 族，但明确非外部预注册；另用现有 75 个 p 值提供单一 Holm 敏感性分析。** | 现有 24/8/25 算术正确，但它不是全论文 75-test 的统一 FWER 保证；75-test 重算低成本且不需训练。 | C-004；`primary_statistics_raw.csv`；`primary_statistics_holm.csv`；quality gate 对 75-test 可执行性的确认。 | 若只给三个族，可能高估总体显著性；若只给 75-test 而删掉预定义比较结构，会丢失可解释的 comparator-specific 信息。 |
| NAR-01, NAR-06 | **立即弱化排他性 novelty；并行完成系统检索与原论文事实核验。检索完成前不恢复“首次/尚无/尚未系统研究”。** | 文字修复可立即消除 P0；系统检索用于建立可防御差异矩阵，但其缺失不能由措辞推断填补。 | C-022=`UNVERIFIED`；manifest 的 literature-map 边界；GSDN 等已知邻近工作。 | 若只弱化不核验，related-work 差异仍可能错误；若等待检索才删 overclaim，P0 会持续污染稿件。 |
| MET-04, EXP-10 | **当前只称“部分 downstream replay 可复现”；只有源码、环境、配置、checkpoint、命令、统计脚本和逐 cell lineage 闭合后，才允许称对应范围可端到端复现。** | 现有 H/CSV/hash 足以复核部分结果，但 Coauthor-CS/Amazon-Photo 缺 encoder checkpoint，且缺完整执行入口与逐 cell lineage。 | material manifest；`grace_pretrain_manifest.csv`；`tables3_5_evidence_manifest.csv`；Table 4 派生列与表号错位记录。 | 若过度保守，只损失宣传强度；若过度声称完整复现，读者无法从原始图重建结果，构成直接 integrity 风险。 |

## 4. Old ID → Consolidated package mapping

| Old ID | Consolidated package |
|---|---|
| NAR-01 | PKG-P0-01 |
| NAR-02 | PKG-P1-01 |
| NAR-03 | PKG-P1-02 |
| NAR-04 | PKG-P1-03 |
| NAR-05 | PKG-P1-03 |
| NAR-06 | PKG-P1-01 |
| NAR-07 | PKG-P1-04 |
| NAR-08 | PKG-P2-01 |
| NAR-09 | PKG-P1-08 |
| NAR-10 | PKG-P2-01 |
| MET-01 | PKG-P1-09 |
| MET-02 | PKG-P1-05 |
| MET-03 | PKG-P1-06 |
| MET-04 | PKG-P1-07 |
| MET-05 | PKG-P1-06 |
| MET-06 | PKG-P1-02 |
| MET-07 | PKG-P1-08 |
| MET-08 | PKG-P1-08 |
| MET-09 | PKG-P1-08 |
| MET-10 | PKG-P2-02 |
| MET-11 | PKG-P2-02 |
| EXP-01 | PKG-P1-05 |
| EXP-02 | PKG-P1-08 |
| EXP-03 | PKG-P1-08 |
| EXP-04 | PKG-P1-06 |
| EXP-05 | PKG-P1-06 |
| EXP-06 | PKG-P1-04 |
| EXP-07 | PKG-P1-08 |
| EXP-08 | PKG-P1-03 |
| EXP-09 | PKG-P2-02 |
| EXP-10 | PKG-P1-07 |
| EXP-11 | PKG-P1-08 |

## 5. Task 6 execution gate

建议**有条件进入 Task 6**，由单一主笔按以下顺序施工：

1. 先完成 PKG-P0-01，并对标题、摘要、引言、相关工作和结论做一次 novelty 词汇审计。
2. 再完成所有 P1 story/method/evidence packages。PKG-P1-09 必须先做现有资产 tie 审计；在结果满足“非零且 material”前，不得启动或承诺新训练。
3. 现有资产审计与 75-test Holm 敏感性可在 Task 6 中作为非训练型证据闭环并行完成，但相关数字进入 v30 前必须冻结 locator。
4. 最后处理 P2 formalism、prose 与 submission-cleanliness；P2 不得反向放宽 P0/P1 已收缩的 claim。
5. Task 6 每次修改都回填 `issue_traceability.md`；任何无法验证的项保持 `OPEN`，不得以“已知限制”代替修复。

进入 Task 6 不表示论文已可投稿；它只表示 review synthesis 已闭环，且 32 个 source issues 均有明确施工归属。
