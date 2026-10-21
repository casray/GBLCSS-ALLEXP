# MLP-58 Formal GB-LCSS Teacher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and evaluate an X-input GCN teacher trained by the formal leave-one-out, tie-safe GB-LCSS target.

**Architecture:** Reuse the existing distillation predictor and target builder. Add two narrowly scoped predictor identities and six configs per identity, then run paired real/shuffled teacher experiments and summarize against audited controls.

**Tech Stack:** Python, PyTorch, PyTorch Geometric, pandas, pytest, YAML.

---

### Task 1: Lock method identity with tests

**Files:**
- Create: `tests/test_mlp58_formal_lcss_teacher.py`

- [x] Test predictor registration, X-only teacher input, formal target fields, and shuffled identity.
- [x] Run the test and verify failure because the new predictors/configs do not exist.

### Task 2: Implement predictors and configs

**Files:**
- Modify: `predictor/FrozenGraceGB_GCN_Predictor.py`
- Modify: `scripts/run_standard_downstream.py`
- Create: `config/mlp_distill_formal_lcss_x/*.yaml`
- Create: `config/mlp_distill_formal_lcss_x_shuffled/*.yaml`

- [x] Add minimal subclasses for real and shuffled formal targets.
- [x] Register both methods.
- [x] Add Cora, CiteSeer, and CoraFull configs with beta 0.5, LOO, tie-safe targets, and X-only teacher input.
- [x] Run tests and verify pass.

### Task 3: Smoke and formal runs

- [x] Run Cora uniform 0.4 seed0 smoke for both methods.
- [x] Audit status, dimensions, target rule, leakage fields, and split/noise pairing.
- [x] Run 60 formal records over three datasets, two rates, two methods, and five seeds.
- [x] Rerun 30 current-pipeline GCN(X)+CE controls so every formal comparison retains split/noise hashes.

### Task 4: Summarize and decide

**Files:**
- Create: `scripts/summarize_mlp58_formal_lcss_teacher.py`
- Create: `tests/test_mlp58_formal_lcss_summary.py`
- Create: `runs/mlp58_formal_lcss_teacher_20260813/mlp58_report.md`

- [x] Write failing summary tests for completeness and paired gates.
- [x] Implement aggregate, paired delta, protocol audit, and gate outputs.
- [x] Run the summary and apply the pre-registered decision without tuning.
- [x] Update the evidence ledger; update the manuscript only if the result changes its scientific account.
