# Robust-Student Distillation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and evaluate an X-only MLP student trained jointly by a fixed GCN[X||H] teacher and LCSS soft-small-loss supervision.

**Architecture:** Subclass the existing plain `mlp_distill_gcn_xh_Predictor`, reuse its teacher training and frozen teacher probabilities, and override only the student optimization loop. Real and shuffled subclasses share all code and differ only in the existing `random_control` flag.

**Tech Stack:** Python, PyTorch, NoisyGL predictors/configs, pytest, ROCm GPU.

---

### Task 1: Student loss primitives

**Files:**
- Create: `predictor/LCSSRobustStudentDistill_Predictor.py`
- Test: `tests/test_mlp32_robust_student_distill.py`

- [ ] Write failing tests for a joint KD plus selected soft-CE loss, equal weights, and shuffled/real class registration.
- [ ] Run `python -m pytest tests/test_mlp32_robust_student_distill.py -q` and verify failure due to the missing module.
- [ ] Implement `robust_student_distill_loss` using existing `soft_cross_entropy_per_sample` and `select_soft_small_loss_indices`.
- [ ] Re-run the focused tests and verify they pass.

### Task 2: Predictor and diagnostics

**Files:**
- Modify: `predictor/LCSSRobustStudentDistill_Predictor.py`
- Modify: `scripts/run_standard_downstream.py`
- Test: `tests/test_mlp32_robust_student_distill.py`

- [ ] Add the real and shuffled predictor subclasses and register both method names.
- [ ] Train the plain CE GCN[X||H] teacher using inherited code, then optimize the X-only student with equal-weight KD and robust LCSS loss.
- [ ] Record KD/robust weights, active forget rate, selected ratio, teacher test, and student input dimension.
- [ ] Verify student model input is X-only and clean labels do not enter loss construction.

### Task 3: Six-dataset configs

**Files:**
- Create: `config/mlp_distill_gcn_xh_lcss_small_loss/*.yaml`
- Create: `config/mlp_distill_gcn_xh_lcss_small_loss_shuffled/*.yaml`
- Test: `tests/test_mlp32_robust_student_distill.py`

- [ ] Create Cora, CiteSeer, CoraFull, Amazon-Photo, PubMed, and Coauthor-Physics configs by copying the corresponding plain-distillation teacher settings and fixed LCSS settings.
- [ ] Set `student_kd_weight=1`, `student_robust_weight=1`, `student_forget_rate=auto`, `student_num_gradual=10`, and `student_exponent=1`.
- [ ] Load every config in the test and verify embedding paths exist.

### Task 4: Seed0 smoke and summary

**Files:**
- Create: `scripts/summarize_mlp32_robust_student_distill.py`
- Create: `runs/mlp32_robust_student_distill_20260712/`
- Test: `tests/test_mlp32_robust_student_distill.py`

- [ ] Run plain distillation plus real/shuffled robust-student distillation on six datasets, uniform0.4 seed0, ROCm `cuda:0`.
- [ ] Reuse MLP-30 direct rows only after exact protocol checks.
- [ ] Produce per-dataset deltas, teacher retention, gate decisions, and a Markdown report.
- [ ] Stop the distillation branch if the preregistered seed0 gate fails.

### Task 5: Graph-native audit and completion

**Files:**
- Create: `scripts/audit_mlp32_graph_native.py`
- Create: `runs/mlp32_robust_student_distill_20260712/graph_native_inventory.csv`

- [ ] Inventory exact-protocol NRGNN/NRGNN[X||H] rows for the six datasets.
- [ ] Reuse compatible five-seed rows; run only missing dataset/seed cells.
- [ ] Add other graph-native methods only when an existing runnable predictor and matching config are present.
- [ ] Generate a cross-resource table with training-graph, inference-graph, H-access, and X-only inference columns.

### Task 6: Verification and research record

**Files:**
- Modify: `docs/next_paper_research_agenda_20260712.md`
- Modify: `docs/99_session_handoff.md`
- Modify: `C:/Users/Admin/Desktop/实验8去噪5.19-/项目核心总结_新窗口上下文.md`

- [ ] Run focused and neighboring regression tests.
- [ ] Audit row counts, duplicate cells, status, leakage fields, and protocol hashes.
- [ ] Record the exact pass/fail result without changing the frozen current manuscript.

