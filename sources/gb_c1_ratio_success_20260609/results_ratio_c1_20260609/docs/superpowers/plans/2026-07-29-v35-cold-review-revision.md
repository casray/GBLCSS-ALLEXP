# V35 Cold-Review Revision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Revise the tie-safe GB-LCSS manuscript with auditable current-rule evidence for vote support, grouping alternatives, representation sources, and GRACE-seed stability, while closing the reproducibility and citation gaps identified by the cold review.

**Architecture:** Preserve the immutable tie-safe v2 main grid and reuse already completed LOO, smoothing, breadth, H-access, and beta-zero teacher results. Add only the missing diagnostics and current-rule comparisons in a new MLP-43 experiment bundle. Generate a new manuscript version from v34 without overwriting historical manuscripts or result directories.

**Tech Stack:** Python 3, PyTorch, scikit-learn, pandas, pytest, NoisyGL standard runner, Markdown manuscript and audit scripts.

## Execution Status

- Completed on 2026-07-30: Tasks 1--10.
- Task 7 closed with 20 audited representation-source rows under the
  pre-registered seed-0 geometric gate.
- Task 10 closed with 30/30 corrected kNN-teacher rows and a hash-matched
  150-row five-arm teacher grid.
- v35 remains the pre-completion snapshot. The completed manuscript is
  `paper_final_zh_v36_experiments_completed_20260730.md`.
- Final closure record:
  `paper_pipeline/01_reviews/v36_deferred_experiments_closure_20260730.md`.

---

## Frozen Scope

- Main datasets: `cora,citeseer,corafull,coauthorcs,amazonpho`.
- Main diagnostic setting: train-only `uniform 0.4`, percent split `0.2/0.1/0.7`, seeds `0-4`.
- Main GB method: `frozengrace_gb_mlp_softdist_beta05`, `tie_mode=distribution_only`.
- Frozen construction: `max_balls=64`, `min_ball_size=32`, `split_improvement=0.05`, `target_beta=0.5`.
- Do not rerun the immutable 500-row main grid.
- Do not rerun LOO, smoothing, breadth, H-access controls, or beta-zero teacher distillation when their existing artifacts pass lineage checks.
- Do not modify GSDN in this revision.
- Do not add inductive, heterophilic, real-noise, ten-seed, or graph-denoising-teacher experiments.

## Task 1: Audit Existing Tie-Safe Assets

**Files:**
- Read: `paper_pipeline/04_experiments/tie_safe_v2_manuscript_impact_matrix_20260728.md`
- Read: `external/NoisyGL/runs/v31_tiesafe_affected_20260728/*.csv`
- Create: `paper_pipeline/04_experiments/mlp43_existing_asset_audit_20260729.md`

- [ ] Verify row counts: LOO `60/60`, smoothing `60/60`, breadth `60/60`.
- [ ] Verify `status=ok`, seeds `0-4`, train-only noise, and zero validation/test modifications.
- [ ] Recompute aggregate values for LOO, smoothing, and six breadth datasets.
- [ ] Record which manuscript cells can be replaced without new training.
- [ ] Verify teacher configs use `target_beta=0`, making the tie branch inactive.

## Task 2: Add Vote-Support Diagnostics

**Files:**
- Modify: `external/NoisyGL/utils/frozen_grace_targets.py`
- Create: `external/NoisyGL/scripts/run_mlp43_vote_support_review.py`
- Create: `external/NoisyGL/tests/test_mlp43_vote_support_review.py`

- [ ] Extend support diagnostics to accept `tie_mode` and report bins `2`, `3-4`, `5-9`, `10-19`, and `20+`.
- [ ] Report both ball counts and target-node coverage for every support bin.
- [ ] Add diagnostics for changed nodes, NCR/CDR, mean alpha, clean-mass delta, and target accuracy by support bin.
- [ ] Write failing tests for exact support-bin boundaries and class-permutation-safe targets.
- [ ] Run:

```powershell
python -m pytest tests/test_mlp43_vote_support_review.py tests/test_frozen_grace_targets.py -q
```

- [ ] Run the diagnostic on the five main datasets and seeds `0-4`.
- [ ] Save raw and aggregate CSV files under `runs/mlp43_cold_review_defense_20260729/`.

## Task 3: Run the s_min Sensitivity

**Files:**
- Create: `external/NoisyGL/scripts/run_mlp43_smin_sensitivity.py`
- Extend: `external/NoisyGL/tests/test_mlp43_vote_support_review.py`

- [ ] Register the frozen grid `s_min in {2,3,5,10}`.
- [ ] Reuse the existing current-rule `s_min=2` rows after split/noise hash verification.
- [ ] Run seed-0 smoke for `s_min in {3,5,10}` on five datasets.
- [ ] Check for NaN, collapse, missing coverage, and val/test leakage.
- [ ] Run seeds `1-4` only after the smoke passes.
- [ ] Output mean/std, paired deltas versus `s_min=2`, positive-seed counts, and support coverage.
- [ ] Do not change the main method regardless of this experiment without a separate adoption decision.

## Task 4: Fix and Verify kNN Target Construction

**Files:**
- Modify: `external/NoisyGL/utils/frozen_grace_targets.py`
- Modify: `external/NoisyGL/predictor/FrozenGraceGB_GCN_Predictor.py`
- Modify: `external/NoisyGL/scripts/run_standard_downstream.py`
- Modify/Create: `external/NoisyGL/config/knn_mlp_softdist_beta05/*.yaml`
- Modify/Create: `external/NoisyGL/config/kmeans_mlp_softdist_beta05/*.yaml`
- Create: `external/NoisyGL/tests/test_mlp43_tiesafe_grouping_alternatives.py`

- [ ] Write a failing test showing that the query node is currently returned as its own first neighbor.
- [ ] Add an explicit `exclude_self` option; corrected formal runs must set it to true.
- [ ] Cap external-neighbor `k` at `n_train-1` and handle `n_train<=1`.
- [ ] Add `tie_mode` to node-distribution targets and implement `distribution_only` semantics.
- [ ] Pass `tie_mode=distribution_only` through k-means targets.
- [ ] Record `knn_exclude_self`, `tie_mode`, matched `k`, and matching diagnostics in CSV output.
- [ ] Preserve legacy behavior unless the corrected option is explicitly enabled.
- [ ] Run focused tests before downstream training.

## Task 5: Run Tie-Safe kNN and k-Means

**Files:**
- Create: `external/NoisyGL/scripts/summarize_mlp43_grouping_alternatives.py`

- [ ] Run corrected vote-matched kNN and matched-cluster k-means on five datasets, uniform 0.4, seeds `0-4`.
- [ ] Reuse current-rule GB-LCSS and shuffled rows after hash verification.
- [ ] Reject any run where kNN includes self or where split/noise hashes differ.
- [ ] Report mean/std and paired differences against GB-LCSS.
- [ ] Treat the results as implementation alternatives, not as a requirement that GB win.

## Task 6: Refresh GRACE-Seed Stability

**Files:**
- Create: `external/NoisyGL/scripts/run_mlp43_grace_seed_tiesafe.py`
- Create: `external/NoisyGL/tests/test_mlp43_grace_seed_tiesafe.py`

- [ ] Reuse the archived three GRACE embeddings for Cora, CiteSeer, and CoraFull.
- [ ] Run `3 H seeds x 3 downstream seeds x 3 datasets` with `distribution_only`.
- [ ] Verify embedding row order, finite values, and H manifest hashes.
- [ ] Report fixed-H downstream variance, cross-H variance, and paired gains over CE/shuffled where available.
- [ ] Replace the old-rule wording only after all 27 runs pass.

## Task 7: Representation-Source Causal Table

**Files:**
- Create: `external/NoisyGL/scripts/run_mlp43_representation_source_smoke.py`
- Create: `external/NoisyGL/tests/test_mlp43_representation_source.py`

- [ ] Inventory the exact historical implementations and artifacts for raw X, adjacency propagation, supervised GCN, and GRACE.
- [ ] Freeze a common ball budget and final target rule.
- [ ] Run seed 0 on five datasets for all auditable sources.
- [ ] Stop an arm explicitly if its artifact or node-order provenance cannot be verified.
- [ ] Expand an arm to five seeds only if seed 0 yields a non-degenerate partition and identical split/noise hashes.
- [ ] Report geometry diagnostics separately from downstream accuracy.

## Task 8: Revise the Manuscript

**Files:**
- Read: `paper_final_zh_v34_cold_review_ready_20260729.md`
- Create: `paper_final_zh_v35_cold_review_revised_20260729.md`

- [ ] Add formal references for PosteL, ALS, and SALS.
- [ ] Cite or remove references `[38]` and `[39]`; add dataset-source citations.
- [ ] Add a hyperparameter provenance subsection distinguishing development, frozen evaluation, and post-hoc diagnostics.
- [ ] Add a reproducibility table for GRACE and baseline-specific parameters.
- [ ] Clarify that H-only heads use the same noisy train labels and split.
- [ ] Define the serving scope as fixed-graph transductive X-only inference in the abstract, introduction, setup, and conclusion.
- [ ] Expand the 2-means complexity expression or remove the opaque placeholder.
- [ ] Remove the throughput promise unless measured throughput is added.
- [ ] Replace stale LOO, breadth, and H-access values with verified v2 values.
- [ ] Explain that beta-zero teacher targets are mathematically unaffected by the tie-safe branch.
- [ ] Add the vote-support and s_min results without adopting a new alpha formula unless separately approved.
- [ ] Replace historical kNN/k-means and GRACE-seed claims with current-rule results.
- [ ] Present representation-source evidence as an attribution result, not a causal claim beyond the tested sources.

## Task 9: Final Audits

**Files:**
- Extend: `paper_pipeline/tests/test_audit_v32_manuscript.py` or create a v35-specific audit.
- Create: `paper_pipeline/04_experiments/v35_evidence_registry_20260729.csv`
- Create: `paper_pipeline/01_reviews/v35_nonexperimental_revision_closure_20260729.md`

- [ ] Run manuscript lint.
- [ ] Verify every numeric claim against its registered CSV source.
- [ ] Check that no old `min_index` result is labeled as current tie-safe evidence.
- [ ] Check all cited reference numbers and all references cited in the body.
- [ ] Check table numbering, appendix cross-references, formulas, and figure paths.
- [ ] Record unresolved items explicitly; do not silently omit failed or unavailable evidence.

## Task 10: Correct the kNN Teacher Control

**Files:**
- Modify: `external/NoisyGL/config/mlp_distill_knn_xh/*.yaml`
- Extend: `external/NoisyGL/tests/test_mlp43_tiesafe_grouping_alternatives.py`
- Create: `external/NoisyGL/runs/mlp43_cold_review_defense_20260729/knn_teacher_exclude_self_*.csv`

- [ ] Set `exclude_self=true` for the kNN teacher target builder.
- [ ] Verify that the teacher still uses `target_beta=0`.
- [ ] Rerun only the kNN teacher arm on Cora, CiteSeer, and CoraFull for uniform 0.4/0.6 and seeds `0-4`.
- [ ] Reuse the other four Table 5 arms without retraining.
- [ ] Replace the kNN row and its paired comparison in Table 5.
- [ ] Record this as a correctness refresh of the control, not a new method experiment.

## Adoption Rules

- Keep `s_min=2` if sensitivity differences are small or inconsistent and low-support nodes form a minor fraction.
- Do not introduce an n-aware alpha unless diagnostics show substantial low-support exposure and a separately preregistered candidate passes.
- Keep the paper claim centered on local consensus; GB remains a fixed-budget implementation even if kNN or k-means is numerically close.
- If representation-source evidence shows X-only geometry is comparable to GRACE, remove graph-structure causal language and describe the source as frozen representation geometry.
- If cross-GRACE-seed variance is materially larger than fixed-H variance, elevate fixed-H conditioning in the abstract and limitations.
