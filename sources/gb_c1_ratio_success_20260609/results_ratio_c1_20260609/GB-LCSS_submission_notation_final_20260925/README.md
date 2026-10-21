# GB-LCSS manuscript package - 2026-09-25

Based on `GB-LCSS_submission_final_20260924`; the original package is unchanged.

## Reference trimming - 2026-09-25

- Removed citations to GAT, Zhang et al. (2021), FINE, and ProGMLP, reducing the compiled reference list from 55 to 51 entries. Uncited BibTeX records are retained.
- The introduction now cites ELR for memorization of corrupted labels. Sections 2.1 and 2.3 remove the corresponding method descriptions and adjust the surrounding sentences.
- GNN-CFGD, SALE-MLP, experimental baseline citations, and foundational references are retained. Citation numbering is regenerated in `main.bbl` and `main.pdf`.
- Formulas, experimental values, figures, tables, the supplement, and Highlights are unchanged in this round.

## Notation revision - 2026-09-25

- Ball-dependent quantities use index j; node-dependent quantities use index n; class components use index i. Voting nodes inside sums use index ell.
- Definitions, all three training losses, Proposition 1, Corollary 1, Appendix H, the related experimental formulas, both algorithms, and Table 1 use the same notation.
- Maxima use the compact class index i, whose range is defined in Section 3.1. Competing-class maxima retain the exclusion of the true class.
- Brier scores identify their reference node. Expectations use brackets, and complexity products use thin spaces.
- Section 3.1 introduces the task, objects, and notation in sequence; redundant omitted-index explanations are removed.
- Figure artwork is unchanged. The Figure 1 caption explains that the schematic omits node and ball indices.
- Experimental values, theoretical assumptions, formula labels, references, Highlights, and float-placement settings are unchanged. The notes below record earlier revision rounds.

## Contents

- `main.tex` / `main.pdf`: main manuscript.
- `supplementary.tex` / `supplementary.pdf`: Appendices A-I.
- `highlights.tex` / `highlights.pdf` / `highlights.txt`: highlights source, preview, and editable plain text.
- `references.bib`, `gblcss-numeric.bst`, and `main.bbl`: bibliography sources and compiled bibliography.
- `tables/`, `algorithms/`, `figures/`: manuscript dependencies.
- `figures/source/`: editable author-revised v20 Figure 1 and PNG preview.

## Final changes

1. Section 4.3.3 specifies three pretraining seeds and three downstream seeds for each representation.
2. Section 4.3.2 cites both tables needed for the comparison with random grouping.
3. Table 1 shortens the description of the GRACE representation matrix and vector.

4. Supplementary language edits and the two edited table captions/continuation notices are integrated from the language-polished package.
5. Appendix H clarifies preservation of the most-voted classes and the conditional expected-error comparison.
6. Section 4.3.1 specifies the fixed coefficient of 0.1 used by the uniform/prior smoothing controls.
7. Highlights use four verb-led statements, each within 85 characters.
8. Figure 1 retains the author-revised first page; the extra second page is removed from its PDF only. Editable artwork is unchanged.

Experimental results and mathematical formulas are unchanged. The three queried differences (12.35, 1.53, and -1.61 percentage points) were verified from unrounded per-seed results and retained. Minor differences from subtracting rounded table entries are expected.

## Compilation

Use XeLaTeX and BibTeX for `main.tex`; compile `supplementary.tex` and `highlights.tex` separately with XeLaTeX. This package is also built with Tectonic.

## Submission scope

This is the manuscript/source package, not a completed journal submission. Confirm author approvals, required declarations and submission forms, and the supervisor-requested similarity check before uploading. No author roles or declarations have been invented or added.

- `supplementary_configurations.zip` contains configuration snapshots and runtime-setting records, not a complete code release. The Cora GRACE learning rate has been verified against the main pretraining checkpoint and corrected in Table F.3.
- Confirm author contributions, competing interests, and data availability with the authors.
- AI-use statements are left to the authors, who will describe all tools used before submission. No AI-use statement is included in the current manuscript or supplement.
- Submit Highlights as a separate editable file accepted by the journal system; the PDF is a preview.
- Figure 1 now uses the author's v20 saved on 2026-09-24 at 17:41. No artwork content was edited; the exported empty second PDF page was omitted.

## Final close-out

- Removed the redundant beta-endpoint and tied-vote prose; the cases in the formula and algorithm are unchanged.
- Corrected the two summaries of PosteL, ALS, and SALS to avoid describing PosteL as simple neighbor-label smoothing.
- Table 8 now identifies the reported quantity as the test accuracy of deployed MLPs.
- Tables E.1 and I.1 start with enough room for multiple rows.
- Removed the Appendix F AI-assistance note; the author-edited main manuscript also omits the earlier declaration and Figure 1 attribution. The authors will provide their own statements.
- Retained experimental values, Highlights, and the allocator-reported 12.73 GiB value.

Before upload, the authors still need to confirm competing interests, CRediT roles, tool versions, author approvals, and the supervisor-requested similarity check. No conflict-of-interest or authorship-role statements have been invented.

## Appendix language revision

- Appendix B retains unadjusted confidence intervals and Holm-adjusted p-values as distinct quantities. The seed-count column is now `Wins (seeds)`.
- Appendix D uses complete paragraphs for the leave-one-out, clean-label, and tied-vote checks. It distinguishes retaining the vote distribution from constructing the final soft target.
- Appendix F removes duplicated configuration descriptions. Table F.2 explains the discarded-example fraction, and Table F.3 carries a compact note on dimensions and augmented views.
- Appendices H and I tighten the interpretation of the proof and additional results without changing the displayed equations or experimental table values.
- Main manuscript prose, Highlights, and Figure 1 are unchanged in this round.

## Selected final corrections

- Table F.3 now reports `5e-4` for Cora, matching `config.learning_rate = 0.0005` in the main GRACE checkpoint (`grace/Cora/ratio_train0p2_val0p1_split0_pretrain0/checkpoint.pt`) and the configuration archive.
- Table D.1 gives dataset-wise leave-one-out gains and clean-label differences. Values were checked against `mlp44_seed_deltas.csv` and `mlp46_loo_clean_control_20260730/clean_uniform0p0.csv`; the table distinguishes their averaging scopes.
- The Figure 5 legend spells out `Granular ball partition`. Table 6 defines the abbreviation GB locally.
- Contribution (3) and Section 4.1.3 name the same three comparison groups. Distillation remains a separate extension, not a claim of evaluation under every main noise setting.
- Section 3.3.1 introduces the leave-one-out rule. Tables and legends explicitly abbreviate random size-matched grouping as `Random grouping`.
- Proposition 1 groups its numerical conditions together and retains independent symmetric label corruption as an assumption.
- Loss equations, Figure 1, Figure 4, and appendix ordering are unchanged in this revision.


## Evidence-bound language revision

- Main text and supplement integrate the agreed concision edits. Method definitions, protocol details, negative results, and theoretical conditions are retained.
- A Luna (max) read-only review checked suggestions 24, 67, 68, 69, and 81 before editing. The leave-one-out motivation in Section 3.3.1 is retained; its conditional role is explained in Section 3.4.
- Discussion distinguishes observed associations from causal explanations and reports student, not teacher, accuracy. No unverified reason for selecting CiteSeerFull has been added.
- Abstract, main results, conclusion, and Highlights consistently report the highest mean in 24 of 25 settings. Highlights retain four verb-led statements.
- Equations, algorithms, bibliography, all tables, all figures, and their captions are unchanged in this revision.
