# Posterior-Calibrated SoftDist Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Diagnose whether low-support granular balls are over-trusted, then conditionally add and evaluate a fixed Jeffreys-prior Posterior-SoftDist target.

**Architecture:** Add a pure support-bin diagnostic beside the existing target diagnostics and expose it through a standalone script that reuses the current dataset, noise, embedding, and ball builders. Only if the preregistered diagnostic gate passes, extend the existing target builder with explicit target-estimator modes and register three new downstream methods without changing current SoftDist behavior.

**Tech Stack:** Python 3.12, PyTorch, NumPy, unittest/pytest, NoisyGL configuration and runner utilities.

---

### Task 1: Support-bin diagnostic primitives

**Files:**
- Modify: `utils/frozen_grace_targets.py`
- Test: `tests/test_posterior_softdist.py`

- [ ] **Step 1: Write failing tests**

Add tests that construct three balls with support sizes `3`, `5`, and `8`, and assert
that the diagnostic returns bins `2-3`, `4-7`, and `8+`, uses only train nodes, and
reports NCR, CDR, clean-mass delta, margin, and majority accuracy.

- [ ] **Step 2: Verify RED**

Run:

```powershell
& 'C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe' -m pytest tests/test_posterior_softdist.py -q
```

Expected: FAIL because `support_bin_diagnostics` is not defined.

- [ ] **Step 3: Implement minimal pure diagnostic**

Add `support_bin_diagnostics(...) -> list[dict]` that:

- validates assignment length;
- converts the train mask with `_mask_to_index_array`;
- builds current empirical SoftDist targets;
- aggregates only eligible train nodes by support bin;
- uses clean labels only in returned diagnostic metrics;
- never changes targets or assignments.

- [ ] **Step 4: Verify GREEN**

Run the same pytest command. Expected: all tests in the new file pass.

### Task 2: Standalone Stage-A diagnostic

**Files:**
- Create: `scripts/diagnose_posterior_softdist.py`
- Modify: `tests/test_posterior_softdist.py`

- [ ] **Step 1: Write failing aggregation/gate tests**

Test five paired seeds where two datasets satisfy the preregistered low-support gap
and one does not. Assert that the gate passes only when two datasets have threshold
gaps with at least `4/5` paired seed directions.

- [ ] **Step 2: Verify RED**

Run the new test file. Expected: FAIL because aggregation and gate functions are
missing.

- [ ] **Step 3: Implement script**

The script must:

- load dataset-specific SoftDist configs to obtain embedding path and ball parameters;
- inject uniform `0.4` noise only into training labels;
- build the same geometric balls as current SoftDist;
- write raw, aggregate, and report files after every dataset/seed;
- record the exact protocol and the automatic Stage-B go/no-go decision.

- [ ] **Step 4: Verify GREEN and CLI help**

Run:

```powershell
& 'C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe' -m pytest tests/test_posterior_softdist.py -q
& 'C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe' scripts/diagnose_posterior_softdist.py --help
```

Expected: tests pass and CLI help exits `0`.

### Task 3: Execute Stage-A gate

**Files:**
- Create: `runs/posterior_softdist_20260622/support_bin_raw.csv`
- Create: `runs/posterior_softdist_20260622/support_bin_aggregate.csv`
- Create: `runs/posterior_softdist_20260622/support_bin_report.md`

- [ ] **Step 1: Run five-seed diagnostic**

```powershell
& 'C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe' scripts/diagnose_posterior_softdist.py `
  --datasets cora,citeseer,coauthorcs `
  --seeds 0,1,2,3,4 `
  --noise_type uniform `
  --noise_rate 0.4 `
  --device cuda:0 `
  --split_type percent `
  --train_percent 0.2 `
  --val_percent 0.1 `
  --test_percent 0.7 `
  --output_dir runs/posterior_softdist_20260622
```

- [ ] **Step 2: Apply gate**

If the report says `STOP`, do not implement Tasks 4-7. Record the negative result.
If it says `PROCEED`, continue.

### Task 4: Posterior and DirectDist target estimators

**Files:**
- Modify: `utils/frozen_grace_targets.py`
- Modify: `tests/test_posterior_softdist.py`

- [ ] **Step 1: Write failing target tests**

Assert:

- a `2:0` binary vote produces posterior distribution `[0.8333, 0.1667]` and alpha
  `0.6667`;
- large-support posterior targets approach empirical SoftDist;
- `direct_distribution` replaces eligible train targets with empirical distribution;
- val/test label changes do not affect targets;
- default empirical residual mode exactly matches current SoftDist.

- [ ] **Step 2: Verify RED**

Run the new test file. Expected: FAIL because estimator modes are unsupported.

- [ ] **Step 3: Implement minimal estimator modes**

Extend `apply_ball_soft_targets` with:

```python
distribution_estimator="empirical"  # empirical | jeffreys
mixing_mode="residual"              # residual | direct
prior_count=0.5
```

Keep defaults identical to current behavior.

- [ ] **Step 4: Verify GREEN plus regression suite**

```powershell
& 'C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe' -m pytest tests/test_posterior_softdist.py tests/test_frozen_grace_targets.py tests/test_featuregb_mlp.py -q
```

Expected: all selected tests pass.

### Task 5: Register downstream methods and configs

**Files:**
- Modify: `predictor/FrozenGraceGB_GCN_Predictor.py`
- Modify: `single_exp.py`
- Modify: `scripts/run_standard_downstream.py`
- Create: `config/frozengrace_gb_gcn_postsoftdist_xh/*.yaml`
- Create: `config/frozengrace_gb_gcn_postsoftdist_xh_shuffled/*.yaml`
- Create: `config/frozengrace_gb_gcn_directdist_xh/*.yaml`
- Modify: `tests/test_posterior_softdist.py`

- [ ] **Step 1: Write failing registration/config tests**

Assert the three methods are registered in both runners and that configs exist for
Cora, CiteSeer, and Coauthor-CS with the same anchor parameters as SoftDist.

- [ ] **Step 2: Verify RED**

Run the new test file. Expected: registration/config assertions fail.

- [ ] **Step 3: Add predictor subclasses and configs**

The predictor reads the three new config keys and passes them to the target builder.
No other optimizer, model, split, or embedding setting changes.

- [ ] **Step 4: Verify GREEN**

Run registration tests and `tests/test_standard_runner_baselines.py`.

### Task 6: Downstream runner summary

**Files:**
- Create: `scripts/summarize_posterior_softdist.py`
- Modify: `tests/test_posterior_softdist.py`

- [ ] **Step 1: Write failing paired-summary tests**

Test paired deltas, positive-seed counts, coverage preservation, and the four
preregistered success conditions.

- [ ] **Step 2: Verify RED**

Run the test file. Expected: summary functions are missing.

- [ ] **Step 3: Implement summarizer**

Read standard-runner CSV plus diagnostics CSV and emit aggregate, paired deltas, and
the final decision report.

- [ ] **Step 4: Verify GREEN**

Run the new test file and CLI `--help`.

### Task 7: Execute downstream experiment if Stage A passes

**Files:**
- Create: `runs/posterior_softdist_20260622/downstream_raw.csv`
- Create: `runs/posterior_softdist_20260622/downstream_aggregate.csv`
- Create: `runs/posterior_softdist_20260622/paired_deltas.csv`
- Create: `runs/posterior_softdist_20260622/posterior_softdist_report.md`

- [ ] **Step 1: Run four methods**

```powershell
& 'C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe' scripts/run_standard_downstream.py `
  --datasets cora,citeseer,coauthorcs `
  --methods frozengrace_gb_gcn_softdist_xh,frozengrace_gb_gcn_postsoftdist_xh,frozengrace_gb_gcn_postsoftdist_xh_shuffled,frozengrace_gb_gcn_directdist_xh `
  --seeds 0,1,2,3,4 `
  --noise_type uniform `
  --noise_rate 0.4 `
  --device cuda:0 `
  --split_type percent `
  --train_percent 0.2 `
  --val_percent 0.1 `
  --test_percent 0.7 `
  --stop_on_error `
  --output runs/posterior_softdist_20260622/downstream_raw.csv
```

- [ ] **Step 2: Summarize and verify**

Run the summarizer and confirm every dataset/method has exactly five successful rows,
all output files are non-empty, and the report applies the preregistered criteria
without post-hoc threshold changes.

### Task 8: Final verification and research record

**Files:**
- Modify: `02_experiment_history_and_lessons 11.md` only after results are available.

- [ ] **Step 1: Run full relevant tests**

```powershell
& 'C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe' -m pytest tests/test_posterior_softdist.py tests/test_frozen_grace_targets.py tests/test_featuregb_mlp.py tests/test_standard_runner_baselines.py -q
```

- [ ] **Step 2: Audit protocol and outputs**

Check split, train-only noise, seeds, embedding paths, ball parameters, row counts,
failed status rows, and real-versus-shuffled pairing.

- [ ] **Step 3: Record decision**

Append hypothesis, granular-ball role, code/config changes, expected mechanism,
success/failure standards, results, and final decision (`continue`, `modify`,
`defer`, or `abandon`) to the experiment history.

