# MLP-35 Clean-Val And Repeatability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Build and execute a leakage-safe clean-validation forget-rate selector and a within-seed ROCm repeatability audit without changing the frozen LCSS predictor.

**Architecture:** Reuse `scripts.run_standard_downstream.run_one` with `model.forget_rate` overrides. Keep execution, clean-val selection, and repeatability aggregation in separate scripts so selection logic can be unit-tested without GPU or dataset downloads.

**Tech Stack:** Python 3, PyTorch/ROCm, CSV/JSON, pytest, existing NoisyGL runner and configs.

---

### Task 1: Clean-Val Selection Tests

**Files:**
- Create: `tests/test_mlp35_cleanval_repeatability.py`
- Create: `scripts/summarize_mlp35_cleanval_grid.py`

- [x] Write tests proving that `select_by_clean_validation(rows)` chooses the highest `valid` candidate and uses the smaller forget rate for exact ties.
- [x] Write a test that changes every candidate `test` value while preserving `valid` and asserts the chosen forget rate is unchanged.
- [x] Write tests rejecting missing rates, duplicate cells, split/noise hash mismatch, non-train noise, and val/test label modification.
- [x] Run `python -m pytest tests/test_mlp35_cleanval_repeatability.py -q` and verify failure because the summary module is absent.
- [x] Implement `validate_candidate_rows`, `select_by_clean_validation`, `build_cleanval_summary`, and JSON/CSV report output.
- [x] Re-run the targeted tests and verify they pass.

### Task 2: Resumable Grid Runner

**Files:**
- Create: `scripts/run_mlp35_cleanval_grid.py`
- Test: `tests/test_mlp35_cleanval_repeatability.py`

- [x] Add a failing test for deterministic `grid_cells(datasets, rates, seeds, forget_rates)` ordering and exact cardinality.
- [x] Run the targeted test and verify the missing function failure.
- [x] Implement `grid_cells`, CSV resume-key loading, `Namespace` construction, `run_one` calls, explicit `requested_forget_rate`, and write-after-each-run behavior.
- [x] Re-run the targeted tests and verify they pass.
- [x] Run `python -m py_compile scripts/run_mlp35_cleanval_grid.py scripts/summarize_mlp35_cleanval_grid.py`.

### Task 3: Seed0 Grid And Gate

**Files:**
- Create: `runs/mlp35_cleanval_repeatability_20260714/cleanval_seed0_raw.csv`
- Create: `runs/mlp35_cleanval_repeatability_20260714/mlp35_report.md`

- [x] Run the ROCm seed0 command for five datasets, three uniform rates, and five requested forget rates.
- [x] Summarize candidates using only `valid`, join exact-hash LCSS anchors from MLP-30, and write the preregistered gate.
- [x] If the gate fails, stop Phase B and record `abandon` without changing the grid.

### Task 4: Five-Seed Expansion When Triggered

**Files:**
- Create: `runs/mlp35_cleanval_repeatability_20260714/cleanval_fiveseed_raw.csv`
- Update: `runs/mlp35_cleanval_repeatability_20260714/mlp35_report.md`

- [x] If Phase A passes, run seeds 1-4 with the identical grid and append seed0.
- [x] Validate 375 unique cells, per-seed hash pairing, and zero val/test corruption.
- [x] Report selected-oracle and selected-fixed0.5 paired means and both frozen gates.

### Task 5: Repeatability Audit

**Files:**
- Create: `scripts/run_mlp35_repeatability_audit.py`
- Create: `scripts/summarize_mlp35_repeatability_audit.py`
- Update: `tests/test_mlp35_cleanval_repeatability.py`

- [x] Add failing tests for three-replicate completeness and hash mismatch rejection.
- [x] Implement the fixed three-setting runner and summary functions.
- [x] Run Cora Pair, Amazon-Photo Pair, and PubMed Pair with three same-seed replicates per method.
- [x] Report per-method ranges and paired-delta ranges without labelling them as variance estimates.

### Task 6: Verification And Durable Context

**Files:**
- Update: `docs/execution_plan_mlp35_cleanval_repeatability_20260714.md`
- Update: `docs/next_paper_research_agenda_20260712.md`
- Update: `docs/99_session_handoff.md`
- Create: `C:/Users/Admin/Desktop/实验8去噪5.19-/02_experiment_history_and_lessons 25.md`

- [x] Run targeted MLP-35 tests plus MLP-31/33/34 regression tests.
- [x] Run `py_compile` on all new scripts.
- [x] Record raw commands, row counts, hash audits, mean/std, gates, negative results, and final decision.
- [x] Update the second-paper claim: PubMed strong evidence is Pair/Instance, uniform 0.6 remains unstable; mechanism wording is soft-mass/loss-ranking, not hard repair.
- [x] Preserve the first-paper frozen method and tables unchanged.

