---
name: elsevier-human-paper-writing
description: Use when drafting, revising, translating, or auditing Elsevier-style computer science and AI manuscripts, especially when prose is defensive, repetitive, template-like, overqualified, overly detailed, or reads like a reviewer response or experiment log.
---

# Elsevier Human Paper Writing

Write the manuscript as a scientific argument, not as a reviewer response, lab notebook, or record of discussions with an AI.

For a full-manuscript revision or de-AI audit, read [references/rules.md](references/rules.md) before editing.

## Core Rules

1. State the positive scientific claim first.
2. Put scope conditions inside the claim instead of appending defensive denials.
3. Centralize necessary limitations. State a protocol boundary once in the setup and a substantive limitation once in the conclusion.
4. Give each section one question and each paragraph one task.
5. Introduce motivation before technical detail.
6. Preserve every independent fact disclosure, negative result, protocol condition, citation, number, and table entry.
7. Remove internal experiment identifiers, run directories, smoke-test language, and revision history.
8. Do not hide unfavorable evidence. Report it directly and explain what it establishes.

## Required Workflow

1. **Set the section contract.** Write one sentence describing what the section must establish.
2. **Map claims to evidence.** Every performance or mechanism claim must point to a table, figure, equation, or citation.
3. **Rewrite directly.** Replace “we do not claim X; rather, Y” with “we study Y under condition Z.”
4. **Control detail order.** Start with the scientific question, then method or comparison, then the minimum evidence needed.
5. **Consolidate disclosure.** Move repeated caveats to the experimental protocol or limitations paragraph.
6. **Run the prose gate.** Check paragraph purpose, claim strength, transitions, terminology, and evidence.
7. **Run the linter.**

```powershell
py -3 scripts/lint_manuscript.py path\to\manuscript.md
```

Treat linter findings as revision prompts, not blind deletion commands.

## Section Contracts

- **Abstract:** problem, gap, method idea, principal evidence, implication. Avoid ablation inventories and defensive qualification.
- **Introduction:** deployment problem, research gap, central insight, method overview, evidence preview, concise contributions.
- **Related work:** organize by research question or mechanism. End each subsection with the precise unresolved gap.
- **Method:** define inputs, outputs, stages, objectives, and inference resources. Do not narrate failed variants.
- **Experimental setup:** make the protocol reproducible. Put fairness conditions and resource boundaries here.
- **Results:** lead with the takeaway, quantify it, then explain the mechanism or exception. Do not recite every cell.
- **Extension:** explain the new bottleneck first, then show how the extension addresses it.
- **Conclusion:** answer the research question, summarize the strongest evidence, then state the main limitations and corresponding future work.

## Final Gate

Before accepting a revision, verify:

- The main contribution is understandable without reading the appendix.
- No paragraph begins by denying a claim that the paper never needed to make.
- Necessary limitations remain visible but are not repeated.
- The text contains no internal project vocabulary.
- Claims use the narrowest wording supported by the reported protocol.
- Tables, numbers, citations, and equations are unchanged unless the task explicitly requires factual edits.

