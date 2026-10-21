# Information Sciences 论文流水线

## 固定配置

- Canonical baseline：`paper_final_zh_v29_human_style_20260727.md`
- Target journal：Elsevier *Information Sciences*
- Research question：在训练标签受污染且部署阶段不可访问图结构的节点分类中，如何从冻结表征空间提取可靠的局部语义共识，并将其转化为节点监督和无图学生模型可用的软监督信号？
- Baseline policy：v29 永不覆盖；每轮修订使用新文件名。
- Fact policy：性能、机制、效率和 scope claim 只有经过 `00_evidence/claim_evidence_ledger.md` 核验后才能进入新稿。
- Writing policy：三个审稿代理只写报告；综合编辑只写路线图；单一主笔独占 canonical manuscript 写权限。
- Browser policy：公开资料只使用 Web Search；禁止调用 Codex/ChatGPT 内置浏览器 IAB。

## 版本谱系

```text
paper_final_zh_v29_human_style_20260727.md
  -> claim–evidence audit
  -> three independent reviews
  -> revision roadmap
  -> paper_final_zh_v30_information_sciences.md
  -> independent integrity review
  -> Chinese scientific-content freeze
  -> paper_final_en_v1_information_sciences.md
```

## 当前状态

| Stage | Status | Required artifact | Gate |
|---|---|---|---|
| 0. Design | COMPLETE | `docs/superpowers/specs/2026-07-27-information-sciences-multi-agent-paper-design.md` | User approved |
| 1. Execution plan | COMPLETE | `docs/superpowers/plans/2026-07-27-information-sciences-paper-pipeline.md` | Self-review passed |
| 2. Evidence freeze | COMPLETE | `00_evidence/claim_evidence_ledger.md` | 25 claims classified; independent gate passed |
| 3. Independent review | COMPLETE | `01_reviews/*.md` | 3/3 reports passed all quality gates |
| 4. Synthesis | IN PROGRESS | `02_synthesis/revision_roadmap.md` | No P0/P1 silently dropped |
| 5. Lead revision | PENDING | `paper_final_zh_v30_information_sciences.md` | v29 unchanged |
| 6. Integrity | PENDING | `03_integrity/pre_review_integrity_report.md` | P0 = 0 |
| 7. Journal fit | PENDING | `04_final/information_sciences_fit_report.md` | Chinese content ready to freeze |

## 阶段门禁

进入英文重写前必须同时满足：

1. P0 issue 为零。
2. 摘要和引言中的所有主要 claim 为 `SUPPORTED`，或已收窄为有明确边界的 `PARTIAL`。
3. 三路审稿报告均通过规格、证据纪律和可执行性复核。
4. `paper_final_zh_v29_human_style_20260727.md` 的 SHA-256 仍为 `BAB34F9D35A54C2CC7D3CFC58DBF61B72BA8D8C2C400FAA744C01C6C80129665`。
5. 用户确认中文科学内容冻结。

## 当前统计

- Claim status：21 `SUPPORTED` / 3 `PARTIAL` / 1 `UNVERIFIED` / 0 `CONTRADICTED`
- P0 count：1（C-022，排他性新颖性表述缺系统检索证据）
- P1 unresolved count：24（待 synthesis 去重、裁决和映射）
- New experiments required：当前无无条件新训练实验；MET-01 先做现有资产 tie 审计，再条件升级
- Next gate：32/32 原始 issue 完整映射到 revision roadmap 与 traceability
