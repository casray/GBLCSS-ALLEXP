# FINE Fidelity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an MLP-adapted FINE baseline that preserves the official class-wise uncentered SVD projection and two-component GMM selector.

**Architecture:** The historical `mlp_fine` top-fraction proxy remains unchanged. A new `mlp_fine_official` predictor computes hidden representations each epoch, derives each noisy class's leading right singular vector from the uncentered feature matrix, scores normalized samples by absolute projection, and trains on samples assigned with posterior above 0.5 to the higher-mean per-class GMM component.

**Tech Stack:** Python, PyTorch, NumPy, scikit-learn, NoisyGL MLP runner, ROCm.

---

### Task 1: Lock the FINE selector identity

**Files:**
- Modify: `tests/test_mlp_noise_baselines.py`
- Modify: `predictor/MLPNoiseBaselines_Predictor.py`

- [ ] Add tests proving the direction comes from uncentered SVD, scores are absolute normalized projections, and per-class GMM chooses the higher-mean component.
- [ ] Run the tests and verify they fail because `fine_official_scores` and `fine_gmm_clean_mask` do not exist.
- [ ] Implement both helpers without accepting clean labels.
- [ ] Run focused tests and verify they pass.

### Task 2: Add the separate method ID and configs

**Files:**
- Modify: `predictor/MLPNoiseBaselines_Predictor.py`
- Modify: `scripts/run_standard_downstream.py`
- Modify: `scripts/run_ssl_swap_downstream.py`
- Modify: `single_exp.py`
- Modify: `total_exp.py`
- Create: `config/mlp_fine_official/mlp_fine_official_{cora,citeseer,corafull,coauthorcs,amazonpho}.yaml`
- Modify: `tests/test_mlp_noise_baselines.py`

- [ ] Add failing registration/config tests.
- [ ] Add strategy `fine_official`, threshold 0.5, fixed random seed inherited from the paired run, and same MLP optimizer/backbone.
- [ ] Train on all nodes during a 10-epoch warmup, then dynamically recompute the official selector each epoch; if a class is too small or GMM fails, retain that class rather than using clean-label information.
- [ ] Add selection diagnostics to runner output and pass regression tests.

### Task 3: GPU smoke and expansion decision

**Files:**
- Create: `runs/mlp25_fine_fidelity_20260712/smoke_seed0.csv`
- Create: `docs/execution_plan_mlp25_fine_fidelity_20260712.md`

- [ ] Run Cora/CiteSeer seed0 under uniform 0.4 train-only noise.
- [ ] Require finite output, nonempty selection, and no class completely removed.
- [ ] If feasible, run the five core datasets and seeds 0--4; otherwise remove the formal FINE row and retain only the clearly named proxy.

---

Official code provenance: `Kthyeon/FINE_official`, commit `d5ab341dec5356d5f9a2a8ebc3ba8e846188af61`.
