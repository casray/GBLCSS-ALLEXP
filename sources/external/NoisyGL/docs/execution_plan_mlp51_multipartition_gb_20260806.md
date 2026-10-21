# MLP-51 Multi-Partition Granular-Ball Consensus Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` and `superpowers:test-driven-development` to implement this plan task-by-task.

**Goal:** Test whether averaging tie-safe leave-one-out granular-ball targets from three independently pretrained GRACE representation spaces reduces single-partition boundary error and outperforms a fair three-view exact-kNN target ensemble.

**Architecture:** Each archived GRACE seed defines an independent label-free partition with the frozen GB parameters. Every view produces a full LOO-GB-LCSS target, then the candidate averages the three final target distributions with fixed equal weights. The fair kNN mirror builds an exclude-self, vote-matched exact-kNN target in each same H view and averages those three targets. This phase is target-only; no classifier is trained unless the frozen gate passes.

**Tech Stack:** Python, PyTorch, NumPy, pandas, existing NoisyGL loaders and target builders, pytest.

---

## Frozen protocol

- Datasets: `cora,citeseer,corafull`.
- Noise: train-only uniform `0.4,0.6`.
- Split: percent `0.2/0.1/0.7`.
- Noise/split seed: `0`.
- H views: archived GRACE seeds `0,1,2` from `runs/mlp37_grace_seed_variance_20260716/grace_pretrain_manifest.csv`.
- GB per view: `max_balls=64`, `min_ball_size=32`, `split_improvement=0.05`.
- Target: `target_beta=0.5`, `leave_one_out=true`, `tie_mode=distribution_only`, `min_train_per_ball=2`.
- kNN: exact cosine similarity on normalized H, query self excluded, `k` matched independently to each H view's mean number of training votes per ball.
- Ensemble weights: exactly `1/3`; no learned weighting, validation selection, confidence gate, or dataset-specific parameter.

## Methods

1. `single_h0_gb`: formal LOO-GB target from archived H seed 0.
2. `mp_gb_3h`: equal average of LOO-GB targets from H seeds 0, 1, and 2.
3. `single_h0_knn`: exclude-self exact-kNN target from H seed 0.
4. `mp_knn_3h`: equal average of exact-kNN targets from the same three H views.
5. `mp_gb_3h_shuffled`: equal average of size-matched shuffled LOO-GB targets, with a distinct deterministic shuffle per H seed.

## Preregistered Phase-A gate

All integrity checks are mandatory. The candidate passes only if all conditions hold:

1. `mp_gb_3h` Brier is lower than `single_h0_gb` in at least `4/6` cells.
2. `mp_gb_3h` Brier is lower than `mp_knn_3h` in at least `4/6` cells.
3. No cell has `mp_gb_3h` target accuracy more than `0.5pp` below `single_h0_gb`.
4. `mp_gb_3h` Brier is lower than `mp_gb_3h_shuffled` in all `6/6` cells.
5. Every row has clean validation/test labels, matching split/noise hashes within a cell, finite targets, row sums equal to one, and complete H provenance.

No mean-improvement exception may override a failed count gate.

## Interpretation rules

- If `mp_gb_3h` beats single-view GB but not `mp_knn_3h`, the evidence supports multi-representation ensembling, not a granular-ball advantage.
- If it beats both and passes all gates, run a downstream seed-0 smoke before any five-seed expansion.
- If it misses one count gate by one cell and has no adverse accuracy cell, classify as near-pass and design round 2 from the observed failure mode.
- Otherwise close this formulation and move to one evidence-driven second-priority direction.

## Task 1: Tests first

**Files:**
- Create: `tests/test_mlp51_multipartition_gb.py`
- Create later: `scripts/run_mlp51_multipartition_gb.py`

- [ ] Add a failing test that equal target averaging preserves shape, row sums, and exact arithmetic mean.
- [ ] Add a failing test that the gate rejects incomplete grids.
- [ ] Add a failing test that the gate passes exactly at `4/6` GB and kNN Brier wins, `6/6` shuffled wins, and no target-accuracy drop beyond `0.5pp`.
- [ ] Add a failing test that a `0.51pp` target-accuracy regression fails the gate.
- [ ] Run `python -m pytest tests/test_mlp51_multipartition_gb.py -q` and confirm failure because the runner module is absent.

## Task 2: Minimal target-only runner

**Files:**
- Create: `scripts/run_mlp51_multipartition_gb.py`

- [ ] Implement `average_soft_targets(targets)` with finite/shape/simplex validation.
- [ ] Implement `evaluate_phase_a_gate(rows)` with exact completeness and integrity checks.
- [ ] Load and SHA256-verify the archived H manifest; verify node count and finite row-normalized embeddings.
- [ ] Build per-H real GB, shuffled GB, and exclude-self vote-matched exact-kNN targets using existing target builders.
- [ ] Average only final soft targets; never average clean-label diagnostics or use clean labels in construction.
- [ ] Emit raw rows, per-view diagnostics, gate JSON, and Markdown report.
- [ ] Run the focused tests and then the existing target tests.

## Task 3: Execute Phase A

- [ ] Run with the configured ROCm Python on `cuda:0` and output to `runs/mlp51_multipartition_gb_20260806/`.
- [ ] Audit 30 method rows, 18 per-view target records for GB and kNN provenance, six cells, hashes, target simplex, and label leakage.
- [ ] Generate `phase_a_raw.csv`, `per_view_diagnostics.csv`, `phase_a_gate.json`, and `mlp51_report.md`.
- [ ] Report the complete table and gate result before choosing round 2.

## Task 4: Conditional continuation

- [ ] If Phase A passes, write a separate downstream preregistration before training any MLP.
- [ ] If Phase A is a near-pass or failure, extract the failure pattern by dataset, noise rate, Brier, NCR, and CDR; select exactly one next direction.
- [ ] Never open a fourth innovation round.

