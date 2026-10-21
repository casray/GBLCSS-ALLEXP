# ANL-CE and GMM+CT Fidelity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add auditable MLP-adapted implementations of ANL-CE and GMM+CT without changing the semantics or historical results of the existing proxy methods `mlp_anl` and `mlp_ct`.

**Architecture:** The exact method identities are exposed under new IDs, `mlp_anl_ce` and `mlp_gmm_ct`. ANL-CE uses the paper's NCE+NNCE objective and optional L1 term. GMM+CT tracks annotated-class probability margins against every competing class, applies the Mann-Kendall statistic, and unions CT-selected samples with a two-component low-loss GMM selector; its exact quadratic-in-epochs update receives an explicit feasibility gate before any large-dataset run.

**Tech Stack:** Python, PyTorch, SciPy, scikit-learn, pytest/unittest, NoisyGL full-batch MLP runner, ROCm PyTorch.

---

### Task 1: Lock the ANL-CE loss identity with tests

**Files:**
- Modify: `tests/test_mlp_noise_baselines.py`
- Modify: `predictor/MLPNoiseBaselines_Predictor.py`

- [ ] **Step 1: Add a failing reference-formula test**

Add a test that computes NNCE independently from clamped log-probabilities and checks `normalized_negative_cross_entropy_loss` element-for-element. Add a second test that verifies `active_negative_loss` equals `alpha*NCE + beta*NNCE + delta*L1(model)`.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```powershell
python -m pytest tests/test_mlp_noise_baselines.py -k "nnce or anl_ce" -q
```

Expected: collection/import failure because the faithful NNCE helper and new signature do not exist.

- [ ] **Step 3: Implement the exact ANL-CE formula**

Implement:

```python
def normalized_negative_cross_entropy_loss(logits, labels, min_prob=1e-7):
    log_probs = F.log_softmax(logits, dim=1).clamp(min=math.log(min_prob))
    normalizer = -log_probs.sum(dim=1)
    target_loss = -log_probs.gather(1, labels[:, None]).squeeze(1)
    a = -math.log(min_prob)
    return (1.0 - (a - target_loss) / (logits.shape[1] * a - normalizer)).mean()
```

Add `anl_ce_loss` with `model=None`, `l1_weight=0`, and `min_prob=1e-7`. Keep `active_negative_loss` and the old complementary-loss helper unchanged for historical proxy compatibility.

- [ ] **Step 4: Run focused and baseline tests and verify GREEN**

Run:

```powershell
python -m pytest tests/test_mlp_noise_baselines.py -k "nnce or anl_ce or active_negative" -q
```

Expected: all selected tests pass.

### Task 2: Lock the GMM+CT selector with tests

**Files:**
- Modify: `tests/test_mlp_noise_baselines.py`
- Modify: `predictor/MLPNoiseBaselines_Predictor.py`

- [ ] **Step 1: Add failing tests for margin history and selection union**

Add deterministic tests for:

1. `confidence_gap(probabilities, noisy_labels)` returning `p_y-p_c` for every class.
2. Incremental Mann-Kendall scores matching a direct pairwise implementation over a short sequence.
3. The annotated class being excluded before taking the minimum competing-class Z statistic.
4. `gmm_ct_mask == gmm_mask | ct_mask` and no clean labels being accepted by the API.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
python -m pytest tests/test_mlp_noise_baselines.py -k "confidence_gap or mann_kendall or gmm_ct" -q
```

Expected: collection/import failure because the helpers do not exist.

- [ ] **Step 3: Implement the selector helpers and state**

Implement exact CT updates over train nodes only. Store probability-gap history on CPU to avoid reserving GPU memory; compute the exact score by comparing the current gap with all previous epochs. Compute the Mann-Kendall variance `n(n-1)(2n+5)/18`, use `scipy.stats.norm.ppf(1-alpha)`, and union the CT mask with the existing two-component low-loss GMM mask.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run:

```powershell
python -m pytest tests/test_mlp_noise_baselines.py -k "confidence_gap or mann_kendall or gmm_ct" -q
```

Expected: all selected tests pass.

### Task 3: Add new predictor IDs without mutating historical proxies

**Files:**
- Modify: `predictor/MLPNoiseBaselines_Predictor.py`
- Modify: `scripts/run_standard_downstream.py`
- Modify: `scripts/run_ssl_swap_downstream.py`
- Modify: `single_exp.py`
- Modify: `total_exp.py`
- Create: `config/mlp_anl_ce/mlp_anl_ce_{cora,citeseer,pubmed,coauthorcs,amazonpho}.yaml`
- Create: `config/mlp_gmm_ct/mlp_gmm_ct_{cora,citeseer,pubmed,coauthorcs,amazonpho}.yaml`
- Modify: `tests/test_mlp_noise_baselines.py`

- [ ] **Step 1: Add failing registration and configuration tests**

Require both method IDs to resolve to `predictor.MLPNoiseBaselines_Predictor`, require strategies `anl_ce` and `gmm_ct`, and verify the five core dataset configs use the same 2-layer, width-64, dropout-0.5 MLP and 200-epoch training protocol.

- [ ] **Step 2: Run registration tests and verify RED**

Run:

```powershell
python -m pytest tests/test_mlp_noise_baselines.py -k "faithful_baselines" -q
```

Expected: missing method/config failures.

- [ ] **Step 3: Add predictors, runner registrations, and frozen configs**

Use these pre-registered adaptations for all datasets without tuning:

- ANL-CE: `alpha=5`, `beta=5`, `delta=5e-5`, `min_prob=1e-7`; exact paper loss identity, project MLP/Adam backbone retained for fairness.
- GMM+CT: `warmup=30`, `alpha=0.01`, `p_threshold=0.5`, `reg_covar=5e-4`; full-batch epoch is treated as the CT observation step, and the base optimizer/backbone remains fixed.

Record these as MLP-adapted implementations, not bitwise official image-classification reproductions.

- [ ] **Step 4: Run registration tests and verify GREEN**

Run:

```powershell
python -m pytest tests/test_mlp_noise_baselines.py -k "faithful_baselines" -q
```

Expected: all selected tests pass.

### Task 4: Run regression tests and seed-0 GPU feasibility smoke

**Files:**
- Create: `runs/mlp24_faithful_anl_ct_20260712/smoke_seed0.csv`
- Create: `docs/execution_plan_mlp24_faithful_anl_ct_20260712.md`

- [ ] **Step 1: Run all relevant tests**

```powershell
python -m pytest tests/test_mlp_noise_baselines.py tests/test_standard_runner_baselines.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Run Cora/CiteSeer seed-0 smoke on ROCm**

```powershell
& "C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe" scripts/run_standard_downstream.py `
  --datasets cora,citeseer `
  --methods mlp_anl_ce,mlp_gmm_ct `
  --seeds 0 `
  --noise_type uniform `
  --noise_rate 0.4 `
  --noise_scope train `
  --device cuda:0 `
  --split_type percent `
  --train_percent 0.2 `
  --val_percent 0.1 `
  --test_percent 0.7 `
  --output runs/mlp24_faithful_anl_ct_20260712/smoke_seed0.csv `
  --stop_on_error
```

Pass conditions: four successful finite rows, no single-class prediction collapse, CT selected-set diagnostics finite and nonempty, and per-run wall time recorded.

- [ ] **Step 3: Apply the CT feasibility gate**

Estimate the exact history memory and observed runtime for PubMed, Coauthor-CS, Amazon-Photo, and CoraFull. Continue CT beyond the smoke only if projected memory is below 2 GiB and projected per-run time is below 30 minutes. Otherwise retain the explicit `GMM+CT (MLP-adapted, small-graph feasibility only)` result and do not place it in the five-dataset main table.

- [ ] **Step 4: Record the experiment conclusion**

The execution report must distinguish method fidelity, protocol adaptation, smoke accuracy, selector coverage, runtime, and whether each method is eligible for five-seed expansion.

---

## Self-review

- Spec coverage: exact ANL identity, exact CT statistic, new non-conflicting IDs, configs, tests, GPU smoke, and scalability gate are all assigned tasks.
- Claim boundary: results are described as MLP-adapted implementations; no official-reproduction claim is allowed.
- Historical safety: `mlp_anl` and `mlp_ct` remain unchanged and their old result rows remain traceable.
- Failure handling: a CT scalability failure is reported as an applicability/engineering boundary, not silently replaced by an approximate selector.
