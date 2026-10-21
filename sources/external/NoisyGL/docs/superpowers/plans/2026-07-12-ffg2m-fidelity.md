# FF-G2M Fidelity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a paper-faithful, common-backbone reimplementation of FF-G2M under the train-only noisy-label protocol and remove teacher-loop test peeking.

**Architecture:** A new `ffg2m_reimplemented` ID trains the existing GCN teacher on noisy train labels, selects its checkpoint only by clean validation accuracy, and trains an X-only MLP with the official hard-label, low-frequency neighborhood KL, and high-frequency edge-difference KL objectives. The historical `ffg2m_adapted` MSE proxy remains unchanged and is excluded from official ranking.

**Tech Stack:** Python, PyTorch, NoisyGL sparse adjacency, ROCm.

---

### Task 1: Remove teacher-loop test peeking

**Files:**
- Modify: `tests/test_mlp19_g2m_baselines.py`
- Modify: `predictor/G2MBaselines_Predictor.py`

- [ ] Add a test asserting `_train_teacher` evaluates the test mask exactly once, after the best validation checkpoint is restored.
- [ ] Verify RED against the current per-epoch test evaluation.
- [ ] Refactor teacher training to use validation only during model selection and compute test accuracy once afterward.
- [ ] Verify GREEN.

### Task 2: Implement exact FF-G2M objectives

**Files:**
- Modify: `tests/test_mlp19_g2m_baselines.py`
- Modify: `predictor/G2MBaselines_Predictor.py`

- [ ] Add independent reference tests for LFD and HFD:
  - LFD: `KL(log_softmax(student[src]), log_softmax(teacher[dst]))` on edges with self-loops.
  - HFD: KL between `log_softmax(abs(student[src]-student[dst])/tau)` and the teacher counterpart on non-self edges.
- [ ] Verify RED because `ffg2m_low_frequency_loss` and `ffg2m_high_frequency_loss` do not exist.
- [ ] Implement both helpers and the total `lambda*CE + (1-lambda)*(LFD+HFD)` objective without reading clean labels.
- [ ] Verify GREEN.

### Task 3: Add the independent method ID and configs

**Files:**
- Modify: `predictor/G2MBaselines_Predictor.py`
- Modify: `scripts/run_standard_downstream.py`
- Modify: `scripts/run_ssl_swap_downstream.py`
- Modify: `single_exp.py`
- Modify: `total_exp.py`
- Create: `config/ffg2m_reimplemented/ffg2m_reimplemented_{cora,citeseer,corafull,coauthorcs,amazonpho}.yaml`
- Modify: `tests/test_mlp19_g2m_baselines.py`

- [ ] Add failing registration/config tests.
- [ ] Register `ffg2m_reimplemented` with `ffg2m_lambda=0.4`, `ffg2m_tau=1.5`, common 2-layer width-64 student, 2-layer width-64 GCN teacher, and 200 epochs for both stages.
- [ ] Mark diagnostics as `paper_reimplementation=1`, `official_code_port=0`, and `inference_uses_graph=0`.
- [ ] Pass registration and regression tests.

### Task 4: GPU smoke and five-seed gate

**Files:**
- Create: `runs/mlp26_ffg2m_fidelity_20260712/`
- Create: `docs/execution_plan_mlp26_ffg2m_fidelity_20260712.md`

- [ ] Run Cora/CiteSeer seed0 under uniform 0.4 train-only noise.
- [ ] Require finite teacher/student metrics, no OOM, and exact LFD/HFD diagnostics.
- [ ] If feasible, run the five core datasets and seeds 0--4.
- [ ] Replace the proxy's formal table role only after 25/25 rows succeed.

---

Official code provenance: `LirongWu/FF-G2M`, commit `47edcab20551cfecddf045199613a4d0f8db9b97`.
