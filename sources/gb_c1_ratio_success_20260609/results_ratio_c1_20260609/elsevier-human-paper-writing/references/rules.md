# Elsevier Manuscript Style Rules

## Evidence Base

These rules synthesize:

- The Xiaohongshu post “AI改论文时你遇到‘防御性写作’了吗” and its publicly visible comments.
- High-liked comments emphasizing direct argumentation, natural transitions, delayed technical detail, paragraph-level focus, and separation of drafting from criticism and formatting.
- Counter-comments noting that necessary scope and limitation statements should not disappear.
- Standard Elsevier-oriented manuscript practice: concise section purpose, evidence-matched claims, reproducible methods, results separated from interpretation, and limitations tied to future work.

The public page displayed 87 comments, but only comments visible without account login were inspected. Do not present the comment synthesis as a complete census.

## What The Main Post Identifies

AI commonly treats manuscript revision as local risk patching:

1. It sees a possible objection.
2. It inserts a denial or disclaimer.
3. It repeats the same boundary around later claims.
4. The manuscript becomes safer sentence by sentence but weaker as a whole.

Common forms include:

- “本文并不声称……”
- “本文并不主张……”
- “这并不意味着……”
- “我们的目的不是……而是……”
- “该结果不等同于……”

These phrases spend words on claims the paper is not making. They also alert readers to weaknesses before the scientific contribution is clear.

## What The Comments Add

### Broad agreement

- Repeated “不是 X，而是 Y” constructions sound machine-generated.
- Excessive “however”, “therefore”, “rather than”, and “not only ... but also” makes the argument mechanical.
- Technical detail should not appear before the reader knows why it matters.
- Every paragraph should have one job and should not repeat the preceding paragraph.
- Discussions with an AI are private reasoning. They should not all be copied into the manuscript.
- Internal experiment labels such as `MLP-24`, `Phase 3`, and `seed0 smoke` must never enter the paper.

### Necessary counterpoint

Some scope and limitation statements are required. The solution is not to delete them all:

- Define the protocol once in Experimental Setup.
- State a result-specific exception where the result is discussed.
- Consolidate broader limitations in the Conclusion.
- Do not repeat the same caveat in the Abstract, Introduction, Results, and Conclusion.

## Desired Reader Experience

The final manuscript should feel:

- **Direct:** it says what was studied and found before discussing exclusions.
- **Focused:** each section advances one part of the argument.
- **Natural:** transitions follow scientific logic rather than template phrases.
- **Calibrated:** claim strength matches the data and protocol.
- **Reproducible:** setup and implementation details are complete but placed where needed.
- **Honest:** unfavorable comparisons and failure cases remain visible.
- **Selective:** the paper includes facts needed by readers, not every concern raised during development.

## Rewrite Patterns

### Scope condition

Avoid:

> 本文并不声称该方法适用于所有场景，而是关注训练期有图、推理期仅使用属性的设置。

Prefer:

> 本文研究训练阶段可利用图结构、推理阶段仅使用节点属性的节点分类问题。

### Evidence boundary

Avoid:

> 该结果不能证明粒球在所有数据集上都优于 kNN。

Prefer:

> 在当前协议下，粒球与匹配邻域规模的 kNN 获得相近精度。

### Negative result

Avoid:

> 尽管 PubMed 上结果不理想，但这并不意味着方法无效。

Prefer:

> PubMed 上的增益低于其他数据集，表明局部共识依赖冻结表征的可聚类性。

### Limitation

Avoid repeating:

> 本文不处理真实噪声。本文不研究归纳场景。本文不声称可扩展到超大图。

Prefer one consolidated statement:

> 当前评估采用合成标签噪声和传导式划分，真实噪声、归纳节点与超大图上的扩展性仍待研究。

## Paragraph Task Matrix

| Section | Paragraph task |
|---|---|
| Abstract | Establish the problem, method, evidence, and significance |
| Introduction | Move from deployment need to unresolved research question |
| Related work | Explain what a family solves and what remains unresolved |
| Method | Define one stage or one objective at a time |
| Setup | Specify data, noise, baselines, metrics, and implementation |
| Results | State one takeaway, support it, and interpret it |
| Conclusion | Answer the research question and delimit the evidence |

## Non-Manuscript Material

Keep these out of the paper:

- Internal phase names and experiment codes.
- Why a run was started, stopped, or reprioritized.
- Conversations with supervisors, reviewers, or AI systems.
- Unselected hypotheses and speculative branches.
- Defensive explanations for objections not raised by the evidence.
- File paths, run folders, machine names, and local environment details.

They belong in experiment logs, response letters, or project documentation.

