# MLP-39 LOO-LCSS + Soft-Small-Loss Smoke Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and evaluate two LOO-LCSS soft-small-loss variants on a six-run seed0 smoke.

**Architecture:** Reuse `_LCSSRobustBase` and the existing LOO target path in `utils/frozen_grace_targets.py`. New predictor classes only select the existing robust strategy; configs enable `gb.leave_one_out=true`. A dedicated summarizer joins six new rows with exact-hash MLP-30/36 anchors and applies the preregistered gate.

**Tech Stack:** Python, PyTorch, PyG, pandas, pytest, YAML, ROCm.

---

### Task 1: Freeze experiment contract

**Files:**
- Create: `docs/execution_plan_mlp39_loo_soft_small_loss_smoke_20260717.md`
- Create: `tests/test_mlp39_loo_soft_small_loss.py`

- [ ] Write tests asserting both methods are registered, both predictor classes use `soft_small_loss`, only the shuffled class enables `random_control`, and all six configs enable `leave_one_out`.
- [ ] Run `python -m pytest tests/test_mlp39_loo_soft_small_loss.py -q` and verify failure because the new methods do not exist.

### Task 2: Add minimal predictor and configs

**Files:**
- Modify: `predictor/LCSSRobust_Predictor.py`
- Modify: `scripts/run_standard_downstream.py`
- Create: `config/frozengrace_gb_mlp_softdist_beta05_small_loss_loo/*.yaml`
- Create: `config/frozengrace_gb_mlp_softdist_beta05_small_loss_loo_shuffled/*.yaml`

- [ ] Add two subclasses of `_LCSSRobustBase`; both use `robust_strategy="soft_small_loss"`, and only shuffled sets `random_control=True`.
- [ ] Register both method names.
- [ ] Copy the frozen Cora/CiteSeer/CoraFull soft-small-loss configs and add `gb.leave_one_out: true` without changing any other method parameter.
- [ ] Run the MLP-39 tests and existing MLP-28/36 target tests.

### Task 3: Add exact-hash summary

**Files:**
- Create: `scripts/summarize_mlp39_loo_soft_small_loss.py`
- Test: `tests/test_mlp39_loo_soft_small_loss.py`

- [ ] Add failing synthetic tests for six-row completeness, anchor hash mismatch rejection, paired deltas, and gate evaluation.
- [ ] Implement loading of the new smoke CSV, MLP-30 anchors, and MLP-36 LOO anchors.
- [ ] Output `mlp39_combined.csv`, `mlp39_deltas.csv`, `mlp39_gate.csv`, and `mlp39_report.md`.
- [ ] Verify tests pass.

### Task 4: Run formal smoke

**Files:**
- Create: `runs/mlp39_loo_soft_small_loss_smoke_20260717/seed0.csv`

- [ ] Run the two methods on Cora/CiteSeer/CoraFull, seed0, uniform0.4, train-only, percent 0.2/0.1/0.7, `cuda:0`.
- [ ] Summarize against MLP-30/36 anchors.
- [ ] Verify 6/6 status ok, hashes match, val/test modified counts are zero, and no prediction collapse occurred.

### Task 5: Record decision

**Files:**
- Modify: `docs/execution_plan_mlp39_loo_soft_small_loss_smoke_20260717.md`
- Create: `02_experiment_history_and_lessons 29.md` if the experiment changes the direction decision.

- [ ] Record all per-dataset values and every gate.
- [ ] If the gate passes, permit only paired 5-seed expansion. If it fails, close this exact combination without tuning.
- [ ] Run the final targeted test suite and synchronize project handoff context.
