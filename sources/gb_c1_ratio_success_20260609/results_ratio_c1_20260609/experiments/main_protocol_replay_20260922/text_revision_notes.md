# Text Revision Notes

## Scope and changes

- Edited only Sections 4.3.3 and 4.3.4 in `GB-LCSS_Elsevier_LaTeX_final_integrated_v2_20260921/main.tex` and Appendix I in that package's `supplementary.tex`.
- Tightened the representation-comparison purpose, separated the parameter-study setup from its pending findings, and clarified the paired controls and target-error analysis.
- Preserved existing result sentences, numbers, equations, citations, table inputs, and the separate pretraining-seed experiment. Added `REPLAY-RESULT` comments to identify findings requiring review.
- Retained the old CPU target-construction description in the body. `REPLAY-PROTOCOL` comments identify the conditional update; they do not assert that the new run has finished.

## Pending results and dependencies

- Section 4.3.3 / Table 6: verify all representation-specific ball counts, single-ball cases, accuracy rankings and gaps, and the concluding interpretation.
- Section 4.3.4 / Appendix I.1 / Table I.1: verify the largest losses, three-point range, budget and sharpening findings, per-dataset accuracy differences, paired mean/std, Brier scores, and voting coverage.
- Appendix I.2 / Table I.2: verify all per-seed error components, five-seed ratios and net reductions, random-grouping comparisons, and positive-margin proportions. These require target diagnostics, not just accuracy summaries.
- The lead agent owns the completed summary and numerical/table updates for the five-dataset, uniform-0.4, five-seed, 18-configuration replay through the original NoisyGL entry point. The intended setup uses GPU target construction and MLP training, original frozen GRACE representations, and CPU KMeans. Confirm the actual software/thread settings before replacing the old protocol text and tables together.
- Per task handoff, default main-table summaries are reproducible for Cora, CiteSeer, CoraFull, and Coauthor-CS; Amazon-Photo seed 1 grouping still differs. This text-only task did not independently verify those results. Do not state that all datasets agree with the main table.

## Backup and verification

- Both original TeX files were copied without overwrite to `text_backup/20260922_173319_024_1f37284b/` under this experiment directory before editing.
- No experiment scripts or table files were edited. No compilation or experiments were run. Existing result text remains provisional until the lead agent completes the checks above; TeX comments are not visible in a rendered manuscript.
