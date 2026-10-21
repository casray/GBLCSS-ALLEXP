# Information Sciences Paper Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将现有中文 v29 主稿转化为一篇证据可追溯、经多代理独立审稿、适配 Elsevier *Information Sciences* 的高质量论文。

**Architecture:** 以冻结的 claim–evidence ledger 作为唯一事实接口，三个互不可见的审稿代理分别评估叙事与创新性、方法与复现性、实验与统计；综合编辑只合并问题，单一主笔独占主稿写权限，独立终审从零复核。所有新产物写入 `paper_pipeline/`，v29 永不覆盖。

**Tech Stack:** Markdown、PowerShell、ripgrep、Git、Codex local subagents、项目内 CSV/JSON/日志。

---

## File Map

- Create: `paper_pipeline/README.md` — 流水线状态、输入、产物、阶段门禁和版本谱系。
- Create: `paper_pipeline/00_evidence/material_manifest.md` — 输入材料清单、用途和可信级别。
- Create: `paper_pipeline/00_evidence/claim_evidence_ledger.md` — 唯一事实底座。
- Create: `paper_pipeline/01_reviews/narrative_novelty_review.md` — 叙事与创新性独立审稿。
- Create: `paper_pipeline/01_reviews/method_reproducibility_review.md` — 方法与复现性独立审稿。
- Create: `paper_pipeline/01_reviews/experiments_statistics_review.md` — 实验与统计独立审稿。
- Create: `paper_pipeline/02_synthesis/revision_roadmap.md` — 去重、裁决后的修订路线图。
- Create: `paper_pipeline/02_synthesis/issue_traceability.md` — issue 状态与稿件变更追踪。
- Create: `paper_final_zh_v30_information_sciences.md` — 第一轮结构与证据修订稿。
- Create: `paper_pipeline/03_integrity/pre_review_integrity_report.md` — v30 完整性复核。
- Create: `paper_pipeline/04_final/information_sciences_fit_report.md` — 目标期刊适配报告。
- Create later: `paper_final_en_v1_information_sciences.md` — 中文科学内容冻结后的英文稿，不在第一轮审稿前生成。

## Task 1: 初始化可追溯流水线

**Files:**

- Create: `paper_pipeline/README.md`
- Create: `paper_pipeline/00_evidence/material_manifest.md`

- [ ] **Step 1: 记录 canonical manuscript 和禁止覆盖规则**

`paper_pipeline/README.md` 必须明确：

```text
Canonical baseline: paper_final_zh_v29_human_style_20260727.md
Target journal: Information Sciences
Baseline policy: v29 is immutable; every revision uses a new filename
Fact policy: claims may enter a manuscript only through claim_evidence_ledger.md
Browser policy: Web Search only; never use Codex IAB
```

- [ ] **Step 2: 建立材料 manifest**

至少登记以下输入：

```text
paper_final_zh_v29_human_style_20260727.md
experiments_results_conclusion_audit_20260727.md
xhs_ai_paper_writing_synthesis_20260727.md
01_literature_and_research_map.md
latest_baselines_trainonly_20260622_continuation/formal_main_raw.csv
latest_baselines_trainonly_20260622_continuation/paired_deltas.csv
latest_baselines_trainonly_20260622_continuation/appendix_raw.csv
latest_baselines_trainonly_20260622_continuation/protocol_audit_report.md
latest_baselines_trainonly_20260622_continuation/r2lp_and_legacy_baseline_audit_20260623.md
```

每项标记 `PRIMARY_RESULT`、`AUDIT`、`MANUSCRIPT`、`LITERATURE_MAP` 或 `STYLE_GUIDANCE`。

- [ ] **Step 3: 验证所有必需输入存在**

Run:

```powershell
$required=@(
  'paper_final_zh_v29_human_style_20260727.md',
  'experiments_results_conclusion_audit_20260727.md',
  'xhs_ai_paper_writing_synthesis_20260727.md',
  '01_literature_and_research_map.md',
  'latest_baselines_trainonly_20260622_continuation\formal_main_raw.csv',
  'latest_baselines_trainonly_20260622_continuation\paired_deltas.csv',
  'latest_baselines_trainonly_20260622_continuation\appendix_raw.csv',
  'latest_baselines_trainonly_20260622_continuation\protocol_audit_report.md'
)
$required | Where-Object { -not (Test-Path -LiteralPath $_) }
```

Expected: no output.

- [ ] **Step 4: 验证 v29 未被修改**

Run:

```powershell
git status --short -- 'gb_c1_ratio_success_20260609/results_ratio_c1_20260609/paper_final_zh_v29_human_style_20260727.md'
```

Expected: v29 不因本流水线产生新的修改。

## Task 2: 建立 claim–evidence ledger

**Files:**

- Create: `paper_pipeline/00_evidence/claim_evidence_ledger.md`
- Read: `paper_final_zh_v29_human_style_20260727.md`
- Read: `experiments_results_conclusion_audit_20260727.md`
- Read: `latest_baselines_trainonly_20260622_continuation/*.csv`
- Read: `latest_baselines_trainonly_20260622_continuation/*.json`
- Read: `latest_baselines_trainonly_20260622_continuation/*.md`

- [ ] **Step 1: 从摘要、引言、结果和结论提取主要 claim**

每个 claim 使用以下完整 schema：

```markdown
### C-001

- Section:
- Narrowest defensible claim:
- Claim type: PERFORMANCE | MECHANISM | EFFICIENCY | SCOPE | NOVELTY
- Evidence files:
- Evidence locator:
- Protocol: dataset / noise / rate / split / seeds / baseline
- Status: SUPPORTED | PARTIAL | UNVERIFIED | CONTRADICTED
- Allowed interpretation:
- Forbidden expansion:
- Manuscript action: KEEP | WEAKEN | DELETE | TEST
```

- [ ] **Step 2: 将每个性能数字追溯到正式结果文件**

必须优先使用 `formal_main_raw.csv`、`paired_deltas.csv` 和 `appendix_raw.csv`；审计报告只用于解释协议，不能替代原始表。

- [ ] **Step 3: 核对训练节点噪声、split 和 seed 约束**

验证以下边界是否在 claim 中得到保留：

```text
噪声仅施加于训练节点
主结论使用相同数据划分和评估协议
正式比较使用可比 seed
小于标准差的变化不描述为稳定显著提升
```

- [ ] **Step 4: 生成 ledger 状态统计**

在文件末尾报告：

```text
SUPPORTED:
PARTIAL:
UNVERIFIED:
CONTRADICTED:
P0 blockers:
```

数字必须与 ledger 条目逐项计数一致。

- [ ] **Step 5: 运行结构验证**

Run:

```powershell
rg -n '^### C-[0-9]{3}$|^- Status:|^- Manuscript action:' -- 'paper_pipeline/00_evidence/claim_evidence_ledger.md'
```

Expected: 每个 `C-xxx` 后恰有一个 Status 和一个 Manuscript action。

## Task 3: 三路独立并行审稿

**Files:**

- Create: `paper_pipeline/01_reviews/narrative_novelty_review.md`
- Create: `paper_pipeline/01_reviews/method_reproducibility_review.md`
- Create: `paper_pipeline/01_reviews/experiments_statistics_review.md`
- Read only: v29、ledger、manifest 和各自所需的证据文件

- [ ] **Step 1: 启动 Narrative & Novelty Reviewer**

只评估：

```text
研究问题是否清晰
Information Sciences scope 是否匹配
创新点是否独立且不被夸大
题目—摘要—引言—结论是否承诺一致
方法是否被叙述成可推广洞见而非局部工程补丁
```

- [ ] **Step 2: 启动 Method & Reproducibility Reviewer**

只评估：

```text
问题定义、符号和公式
粒球构建是否真正标签无关
训练期与推理期图资源边界
教师、学生和蒸馏流程
复杂度、超参数和复现信息
潜在标签泄漏或协议不一致
```

- [ ] **Step 3: 启动 Experiments & Statistics Reviewer**

只评估：

```text
baseline 公平性
split、noise、seed 和评估一致性
mean/std 与 paired delta
主结果、消融、负结果和部署开销
“显著、稳定、普遍、优于”等词的证据强度
```

- [ ] **Step 4: 强制统一 issue schema**

每条问题必须完整包含：

```markdown
### <ROLE>-<NN>

- Severity: P0 | P1 | P2
- Manuscript location:
- Problem:
- Evidence:
- Rejection impact:
- Minimal fix:
- New experiment required: YES | NO
- Claim decision: KEEP | WEAKEN | DELETE | TEST
```

- [ ] **Step 5: 验证三份报告彼此独立且完整**

Run:

```powershell
$reviews=Get-ChildItem -LiteralPath 'paper_pipeline\01_reviews' -Filter '*_review.md'
$reviews.Count
$reviews | ForEach-Object {
  "$($_.Name): $((Select-String -LiteralPath $_.FullName -Pattern '^- Severity:' -AllMatches).Count) issues"
}
```

Expected: exactly 3 review files；每份至少包含一个明确审稿结论，若无 issue 必须解释为何不存在拒稿风险。

## Task 4: 审稿规格与质量复核

**Files:**

- Review: `paper_pipeline/01_reviews/*.md`
- Create: `paper_pipeline/01_reviews/review_quality_gate.md`

- [ ] **Step 1: 规格复核**

逐份确认审稿代理没有：

```text
修改 v29
引入 ledger 外的新事实
把推测写成实验结论
遗漏 P0/P1 的证据或最小修复
提出与论文目标无关的扩展
```

- [ ] **Step 2: 质量复核**

检查三份报告是否真正独立、具体、可执行；重复措辞或无定位的泛泛建议标记为无效 issue。

- [ ] **Step 3: 写入复核结论**

`review_quality_gate.md` 对每份报告给出：

```text
Spec compliance: PASS | FAIL
Evidence discipline: PASS | FAIL
Actionability: PASS | FAIL
Required corrections:
```

只有三项均为 PASS 的报告才能进入综合。

## Task 5: 综合 revision roadmap

**Files:**

- Create: `paper_pipeline/02_synthesis/revision_roadmap.md`
- Create: `paper_pipeline/02_synthesis/issue_traceability.md`
- Read: all three accepted review reports

- [ ] **Step 1: 合并重复 issue**

重复项保留最高 Severity、全部证据和最严格的 claim decision，不复制代理原句到主稿。

- [ ] **Step 2: 裁决冲突**

每个冲突必须记录：

```text
Conflicting issue IDs
Decision
Reason
Evidence used
Risk if decision is wrong
```

- [ ] **Step 3: 固定修订顺序**

顺序必须为：

```text
P0 factual/integrity
P1 story/method/evidence
P2 prose/formatting
```

- [ ] **Step 4: 建立 issue traceability**

每个 issue 使用：

```text
Issue ID | Severity | Decision | Target section | Verification | Status
```

初始 Status 只能为 `OPEN`、`ACKNOWLEDGED` 或 `REQUIRES_EXPERIMENT`。

- [ ] **Step 5: 验证没有静默丢弃 P0/P1**

Run:

```powershell
$source=(rg '^- Severity: P[01]' 'paper_pipeline/01_reviews' | Measure-Object).Count
$mapped=(rg '\| P[01] \|' 'paper_pipeline/02_synthesis/issue_traceability.md' | Measure-Object).Count
"source=$source mapped=$mapped"
```

Expected: 若合并了重复 issue，roadmap 必须列出旧 ID 到新 ID 的映射；计数差异可被映射表完全解释。

## Task 6: 单一主笔生成 v30

**Files:**

- Create: `paper_final_zh_v30_information_sciences.md`
- Modify: `paper_pipeline/02_synthesis/issue_traceability.md`
- Read only: `paper_final_zh_v29_human_style_20260727.md`

- [ ] **Step 1: 复制 v29 为新版本**

Run:

```powershell
Copy-Item -LiteralPath 'paper_final_zh_v29_human_style_20260727.md' -Destination 'paper_final_zh_v30_information_sciences.md'
```

Expected: v29 和初始 v30 的 SHA-256 相同；随后只修改 v30。

- [ ] **Step 2: 按 P0 → P1 → P2 顺序修订**

每次变更必须关联 issue ID。不得添加 ledger 中不存在的数字、文献或机制结论。

- [ ] **Step 3: 执行章节契约**

```text
Abstract: problem, gap, method idea, principal evidence, implication
Introduction: deployment problem, gap, insight, overview, evidence preview, contributions
Related Work: mechanism-oriented comparison and unresolved gap
Method: inputs, outputs, stages, objectives, inference resources
Experiments: reproducible protocol and fairness conditions
Results: takeaway, quantification, explanation, exception
Conclusion: answer, strongest evidence, limitation, future work
```

- [ ] **Step 4: 更新 issue 状态**

状态只能变为：

```text
RESOLVED
PARTIAL
ACKNOWLEDGED
REQUIRES_EXPERIMENT
```

每个 `RESOLVED` 必须带 v30 的章节或行定位。

- [ ] **Step 5: 验证 v29 未变化且 v30 结构完整**

Run:

```powershell
rg -n '^## (Highlights|摘要|1 引言|2 相关工作|3 方法|4 实验设置|5 实验结果与分析|6 扩展|7 结论|参考文献)' -- 'paper_final_zh_v30_information_sciences.md'
git status --short -- 'gb_c1_ratio_success_20260609/results_ratio_c1_20260609/paper_final_zh_v29_human_style_20260727.md'
```

Expected: v30 包含全部核心章节；v29 没有因本任务产生新改动。

## Task 7: 预审完整性与去 AI 写作门禁

**Files:**

- Create: `paper_pipeline/03_integrity/pre_review_integrity_report.md`
- Read: v30、ledger、issue traceability、原始结果

- [ ] **Step 1: 从零复核 claim–evidence**

不得只检查已知 issue。重新抽取 v30 摘要、引言、结果和结论 claim，与 ledger 对照。

- [ ] **Step 2: 复核数字、表格、图和引用**

所有不一致按 P0 记录。无法直接核实的引用标记 `UNVERIFIED`，不得推定正确。

- [ ] **Step 3: 检查七类 AI 科研失败**

至少覆盖：

```text
实现或协议误读
幻觉结果
依赖未经核实的二手摘要
把 bug 当洞见
方法细节捏造
claim 与证据脱节
整篇叙事被早期错误框架锁定
```

- [ ] **Step 4: 执行 Elsevier prose gate**

检查：

```text
正面科学 claim 优先
每段一个任务
限制不重复
内部实验词汇已删除
负结果仍保留
claim 使用最窄可支持措辞
```

- [ ] **Step 5: 运行本地 linter**

Run:

```powershell
py -3 'C:\Users\Admin\.codex\skills\elsevier-human-paper-writing\scripts\lint_manuscript.py' 'paper_final_zh_v30_information_sciences.md'
```

Expected: linter 输出被逐项人工判断；不能为了清零机械删除科学边界。

## Task 8: Information Sciences 适配与阶段交付

**Files:**

- Create: `paper_pipeline/04_final/information_sciences_fit_report.md`
- Update: `paper_pipeline/README.md`

- [ ] **Step 1: 检查 scope fit**

报告必须回答：

```text
论文如何贡献于 intelligent information processing
理论贡献与实践部署价值分别是什么
为何不是只针对 GRACE 的局部技巧
哪些结论仍只在当前数据和协议内成立
```

- [ ] **Step 2: 建立提交前实时核对清单**

只把官方实时页面已经确认的规则标记为 `VERIFIED`。暂时无法直接核实的摘要字数、关键词数量、匿名要求和图表限制标记为 `RECHECK_BEFORE_SUBMISSION`。

- [ ] **Step 3: 更新流水线状态**

在 `paper_pipeline/README.md` 记录：

```text
Stage completed
Artifacts
P0 count
P1 unresolved count
New experiments required
Next gate
```

- [ ] **Step 4: 运行最终文件与污染检查**

Run:

```powershell
rg --files 'paper_pipeline'
rg -n 'T[B]D|T[O]DO|待补|编造|run_[0-9]+|seed0 smoke|内部版本' -- 'paper_pipeline' 'paper_final_zh_v30_information_sciences.md'
git diff --check
```

Expected: 全部计划产物存在；没有未解释的占位符或内部实验语言；无 whitespace error。

- [ ] **Step 5: 阶段交付**

只有满足以下条件才进入英文重写：

```text
P0 = 0
所有主要 claim 均为 SUPPORTED 或明确降级后的 PARTIAL
三路审稿均通过质量门禁
v29 未被覆盖
用户确认中文科学内容冻结
```

英文重写是下一阶段，不在中文 v30 尚有 P0 时提前启动。
