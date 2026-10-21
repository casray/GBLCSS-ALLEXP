# Review Quality Gate

审计对象：

- `paper_final_zh_v29_human_style_20260727.md`
- `paper_pipeline/00_evidence/claim_evidence_ledger.md`
- `paper_pipeline/00_evidence/material_manifest.md`
- `paper_pipeline/01_reviews/narrative_novelty_review.md`
- `paper_pipeline/01_reviews/method_reproducibility_review.md`
- `paper_pipeline/01_reviews/experiments_statistics_review.md`

本门禁只判断三份 review 是否满足 Task 4 的规格、证据纪律和可执行性，不重新审稿，也不把重复 issue 自动判为无效。三份 review 共 32 条 issue；每条均具备 `Severity`、`Manuscript location`、`Problem`、`Evidence`、`Rejection impact`、`Minimal fix`、`New experiment required`、`Claim decision` 八个字段，形式 schema 为 32/32 完整。

## Gate summary

| Review | Spec compliance | Evidence discipline | Actionability | Gate |
|---|:---:|:---:|:---:|:---:|
| Narrative / novelty | PASS | PASS | PASS | PASS |
| Method / reproducibility | PASS | PASS | PASS | PASS |
| Experiments / statistics | PASS | PASS | PASS | PASS |

判定原则：

1. P0/P1 必须有 manuscript locator、可追溯证据、拒稿影响和最低必要修复。
2. 数值或资产事实必须来自 ledger/manifest，或来自 ledger 指向的资产；不能把尚未验证的解释写成实验结论。
3. `Minimal fix` 必须是修复当前 claim 所需的最小动作，不能未经发生率审计就直接要求新增训练实验。
4. 同一问题被不同 reviewer 从叙事、方法或统计角度重复发现，不构成失败。

## 1. Narrative and novelty review

**Spec compliance: PASS**

10/10 条 issue 的 schema 完整，severity count 与正文一致：P0 1、P1 6、P2 3。每个 P0/P1 均给出证据、拒稿影响、最小修复、是否需要新实验和 claim decision。

**Evidence discipline: PASS**

该报告没有把未核验的全球新颖性写成事实。NAR-01 与 NAR-06 明确把全球文献覆盖标为 `UNVERIFIED`，并将修复限定为删除排他性表述或完成投稿前原文核验。性能、机制和资源边界均可回到 C-003/C-005/C-006/C-007/C-008/C-010/C-018/C-019/C-020/C-024/C-025 或 v29 表格。NAR-03 对蒸馏归因采用“主要增量/部分设置附加收益”的收缩解释，没有把 H、全节点覆盖或粒球写成已被独立识别的唯一因果因素。

P0 严重性合理。NAR-01 直接对应 ledger 唯一 P0 `C-022`：v29 摘要和引言保留“尚缺乏/尚未得到系统研究”的排他性 research-gap claim，而 ledger 明确将其标为 `UNVERIFIED` 并禁止该类表述。其最小修复是删除或弱化措辞，不要求新实验，也没有扩大研究范围。

**Actionability: PASS**

10 条修复均可直接落到题目、摘要、贡献、相关工作、表格组织、机制解释或资产表述。所有 `New experiment required` 均为 `NO`，与问题性质一致。NAR-02、NAR-03、NAR-04、NAR-07 给出的重写边界具体；NAR-09 明确列出应删的 kmeans、noisy-CE GRACE、MLP-hidden 数值，避免泛泛要求“补充更多分析”。

**Required corrections:**

- None.

## 2. Methodology and reproducibility review

**Spec compliance: PASS**

11/11 条 issue 的 schema 完整，P1/P2 分级清楚；每条 P1 均给出具体证据和修复动作。报告也明确区分已支持、仅相关性支持、尚未识别和当前应删除的内容。没有 P0，符合当前 ledger 中只有 `C-022` 为 P0 的证据分级；方法报告没有把可通过披露或资产补齐修复的问题升级成 P0。

**Evidence discipline: PASS**

MET-02、MET-03、MET-06、MET-07、MET-08、MET-09 分别受 C-021、C-002、C-007/C-019、C-009、C-024、C-025 约束，均把观察与因果结论分开。MET-06 明确指出基础版到蒸馏版同时改变目标、教师输入、图传播、监督覆盖和学生损失，因而只允许 arm-level association；MET-07 没有把 clean-label post-hoc diagnostics 当作训练期信息；MET-09 没有把 Cora seed-0 诊断泛化到 CiteSeer 或全部 SSL 表征。

MET-01 的数学缺陷本身成立：v29 明确采用最小索引 tie-break、\(\beta=0.5\)、\(s_{\min}=2\)，因此存在类别置换不等价的 tie 目标。报告没有声称现有增益已经由该偏置造成，而是指出当前证据无法判断其影响。问题在修复顺序，不在该缺陷是否存在。

**Actionability: PASS**

MET-01 已改为最低必要路径：先从现有正式软目标、球内标签计数或目标生成日志审计 dataset×setting×seed 级 tie 暴露率、受影响节点占比、\(\sum\alpha\) 质量和类别分布；`New experiment required` 为 `NO`。报告仅在 tie 非零且足以改变正式目标时升级到代表性 tie-safe 对照，并仅在代表设置显示敏感时扩展到完整五种子范围。该顺序正确区分了资产审计、算法定义修正和条件性新增实验。

其余 10 条 issue 的修复同样可执行：MET-03/04/05 需要机器可读 protocol/config/reproduction ledger，MET-06/07/08/09 主要收缩因果和证据 claim，MET-10/11 补符号、选择规则和复杂度/计时边界；均未提出离题扩展。

**Required corrections:**

- None.

## 3. Experiments and statistics review

**Spec compliance: PASS**

11/11 条 issue 的八字段 schema 完整，issue count self-check 与正文一致：P0 0、P1 9、P2 2。EXP-01 与 EXP-02 已从 P0 降为 P1，符合二者均可通过局部 claim 收缩或删除修复、且不使主性能结果失效的影响范围。ledger 唯一 P0 `C-022` 仍由 NAR-01 正确承接。

**Evidence discipline: PASS**

EXP-01 至 EXP-09、EXP-11 的问题核心均可由 v29、ledger 或 ledger 指向资产支持，且普遍能够区分结果、解释和未验证假设。EXP-05 明确把 24/8/25 限定为三个作者定义的 25-test 族内结果，没有声称现有算术错误；EXP-07 准确识别了“最强样本选择方法”口径与 6.7 pp 的数字错配。

EXP-10 已删除过时的 PubMed locator 指控，并明确确认当前 C-005 通过 `mlp_maintable_combined_raw.csv` 闭合 PubMed 的 claim-level locator。它只保留仍成立的逐 manuscript cell lineage 缺口、external/manuscript table 编号错位和 Table 4 派生列未物化问题，均与当前 manifest 一致。

**Actionability: PASS**

EXP-05 的 75-test 建议可用现有数据完成，不需要新训练实验。`primary_statistics_raw.csv` 已提供三组比较所需的 25 settings × 5 paired seeds，现有流程也已产生逐 setting 的 paired \(t\) p-value；可以把三组共 75 个 p-value 合并后重新执行一次 Holm，作为整体 FWER 敏感性分析。最低文字修复仍应先说明三个族为作者定义而非外部预注册，并把显著性限定到各自 family；75-test 结果是低成本的统计重算，不是扩大实验范围。

EXP-07 的修复具体且可执行：把 5% 标签率相对最强 selector 的范围改为约 5.6–5.8 pp，而不是混用相对非最强 JoCoR 的 6.7 pp。

EXP-10 的 minimal fix 已收窄为生成 `manuscript_table_lineage.csv`、显式记录 external/manuscript table 映射、物化 Table 4 派生列并记录环境与公式；不再要求重复修复 PubMed locator。11 条 issue 的 `New experiment required` 均为 `NO`，修复均可通过文字、现有数据重算或资产归档完成。

**Required corrections:**

- None.

## Duplicate groups

以下为明显重复或部分重叠的 issue 组。重复没有被计为失败，也没有从有效 issue 数中扣除：

- clean-validation / label-free boundary：MET-02、EXP-01。
- 25-setting split/noise hash closure：MET-03、EXP-04。
- C-024 kmeans / vote-count mechanism：NAR-09（部分）、MET-08、EXP-03。
- C-025 unsupported representation diagnostics：NAR-09（部分）、MET-09、EXP-02。
- distillation credit assignment：NAR-03、MET-06。
- 11-dataset scope and frozen-setting wording：NAR-04、EXP-08。
- formal/proxy/cross-resource comparison identity：NAR-07、EXP-06。
- LOO “independent supervision” overclaim：MET-07、EXP-11。
- reproducibility and table lineage：MET-04、EXP-10（部分重叠但修复粒度不同）。

## Overall gate

**Overall gate: PASS**

- Issue records reviewed: 32
- Issue schema complete: 32/32
- 有效 issue 计数：32/32 均有有效问题核心
- 完全无效 issue ID：None
- 门禁层面需修 issue ID：None
- 合理 P0：`NAR-01`，对应 ledger 唯一 P0 `C-022`
- Experiments severity count：P0 0、P1 9、P2 2
- 可由现有数据完成的统计建议：`EXP-05` 的合并 75-test Holm 敏感性分析
- 已确认的数字错配：`EXP-07`
- 已确认 claim-level locator 闭合、逐单元 lineage 仍待修复：`EXP-10`

**Task 5: ALLOWED**

三份 review 的 `Spec compliance`、`Evidence discipline`、`Actionability` 均为 PASS，允许进入 Task 5。该结论只表示 review 报告通过质量门禁；三份 review 中列出的稿件与资产修复仍需在 Task 5 按优先级综合，不代表 v29 已达到投稿完成状态。
