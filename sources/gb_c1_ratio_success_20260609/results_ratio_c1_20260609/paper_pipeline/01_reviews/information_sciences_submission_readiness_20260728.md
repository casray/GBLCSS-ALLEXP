# Information Sciences submission-readiness review

Date: 2026-07-28  
Manuscript target: `paper_final_zh_v31_tiesafe_nearestwork_20260728.md`  
Experiment state: `FORMAL_GRID_COMPLETE_AND_AUDITED`

## Verdict

The manuscript fits the scope of *Information Sciences*. The nearest-work experiment blocker is closed, but author-side submission metadata and repository deposition remain incomplete. The current recommendation is **complete the v31 Chinese evidence merge, then finish the author-controlled submission package**.

The formal common-protocol grid changed the performance claim:

> GB-LCSS stably outperforms PosteL, ALS and SALS, but it is not the overall best recent-work method. GSDN has the higher mean in 15 of 25 settings, while GB-LCSS has the higher mean in 10.

The manuscript must therefore avoid “overall best”, “outperforms all same-resource methods” and SOTA wording. It may report the stable positive differences against PosteL/ALS/SALS and the dataset-dependent trade-off with GSDN.

## Static work completed without running experiments

1. Rewrote the abstract to remove runner/fingerprint/history-log language and retain the task, narrow mechanism, primary evidence and transductive boundary.
2. Reorganized the introduction around three scientific contributions: supervision interface, tie-safe target mechanism and the empirical boundary that real grouping beats random grouping without proving granular-ball superiority.
3. Added PGBSF, SCGNN, SAGHL and GB-CGNN to the granular-ball graph-learning review with verified mechanism and resource distinctions.
4. Changed “fixed budget” to “ball-count upper bound” where the algorithm only fixes \(B_{\max}\).
5. Corrected “five noise types” to five settings covering uniform, pair and instance-dependent mechanisms.
6. Reframed Eq. (10) as an uncalibrated heuristic mixing weight. It does not correct class priors, ball sample size or finite-sample maximum-vote bias.
7. Added the archived GRACE encoder, optimizer and augmentation configurations, including the DBLP/PubMed fallback policies and the checkpoint limitations for Coauthor-CS and Amazon-Photo.
8. Renamed the old §5.2 as a historical resource-stratified diagnosis. The old table is moved to the supplement and must not be used for the current recent-work ranking.
9. Corrected CoraFull 4.47→4.48 pp and ARI 0.25–0.36→0.25–0.37 to match the frozen ledger.
10. Added Data availability, Code and result artifacts, Reproducibility scope, Competing interest and generative-AI declaration sections.

## Formal recent-work grid closed

### GSDN

- Identity: `equation-grounded common-protocol adaptation`, independent and not official code.
- Unique undirected relations are reconstructed from paired directed arcs.
- Eq. (7) retains both directions.
- Eq. (8) supervises from every labeled endpoint; a higher-index labeled endpoint and two labeled endpoints are explicit regression cases.
- Target CE is counted once per unique labeled batch endpoint.
- The per-batch unique-relation normalizer is disclosed as a common-protocol adaptation rather than an equation-identical reproduction.
- Verification: 31 CPU tests passed and `py_compile` passed.

### Recent-work grid orchestrator and evidence

- Non-dry-run `NotImplementedError` removed.
- 25-command fail-fast execution, exact-complete resume, immutable raw capture, 375+125 evidence assembly, protocol audit and statistical outputs implemented.
- Verification: 14 CPU tests passed and `py_compile` passed.
- Formal execution: 25/25 command blocks and 25/25 immutable raw CSV files.
- Native recent-work evidence: 500/500 rows.
- Five-method evidence including frozen GB-LCSS: 625/625 rows.
- Protocol audit: 125/125 dataset--noise--seed groups passed.
- Exact resume: all 25 blocks were verified and skipped; no training subprocess was launched.
- GB-LCSS paired mean differences: +5.202 pp vs PosteL, +7.215 pp vs ALS, +4.202 pp vs SALS and -0.698 pp vs GSDN.
- Highest mean count: GB-LCSS 10/25, GSDN 15/25.
- Detailed evidence: `paper_pipeline/04_experiments/nearest_work_full_grid_final_report_20260729.md`.

## Remaining submission blockers that require author input, not experiments

1. Create the actual anonymous review repository.
2. Freeze a commit/tag and choose a software license.
3. Deposit the reproducibility package in a long-term archive and obtain a DOI, or document the journal-approved alternative.
4. Replace the manuscript’s availability placeholders with the real URL, tag and DOI.
5. Have every author confirm the competing-interest statement.
6. Fill CRediT roles using actual author contributions.
7. Confirm the funding statement and grant numbers; if there was no specific funding, use Elsevier’s recommended no-funding statement.
8. Confirm all author names, affiliations and the corresponding-author address.
9. Produce the separate editable Highlights file and verify the English version of every bullet against the journal character limit.

## Policy notes

- Elsevier’s June 2026 policy requires disclosure when generative AI substantively changes manuscript structure or sentence organization. The manuscript now contains a declaration immediately before the references:  
  https://www.elsevier.com/en-au/about/policies-and-standards/generative-ai-policies-for-journals
- Elsevier requires all authors to provide competing-interest information to the corresponding author and encourages a CRediT contribution statement:  
  https://www.elsevier.com/en-in/subject/next/guide-for-authors
- Elsevier data statements should either give an accessible location or truthfully explain why the research data are not accessible:  
  https://www.elsevier.com/en-gb/researcher/author/tools-and-resources/research-data/data-statement

## Closed experiment gates

1. Corrected-GSDN ROCm functional execution completed.
2. Status, split/noise hashes, train-only contamination and X-only inference were audited.
3. The formal-grid output directory and input snapshot manifest were frozen.
4. The 25-command PosteL/GSDN/ALS/SALS grid completed.
5. The 500-row native grid, 625-row five-method evidence and paired statistics were audited.
6. The remaining manuscript task is to merge these results into v31 with narrower claims.
