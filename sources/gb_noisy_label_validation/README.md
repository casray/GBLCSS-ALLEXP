# Noisy-label GB-GRACE Validation

This project tests whether granular balls built on frozen GRACE encoder
representations can improve noisy-label node classification.

The GRACE upstream source is kept under `upstream/GRACE` and should remain a
baseline dependency. New noisy-label, granular-ball smoothing, classifier,
logging, and experiment orchestration code lives outside upstream.

## Upstream

- Repository: https://github.com/CRIPAC-DIG/GRACE
- Commit used: `b3b5ac3fcbaabbb50e8bd69a075b46cd82a50378`
- `upstream/GRACE` is the only upstream source used by this project. Keep it
  unmodified and put all experiment code outside that directory.

## Quick Start

Create/use a Python environment inside this project with PyTorch, PyG, sklearn,
PyYAML, and pytest installed, then run commands from this directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The formal gate runner uses `.\.venv\Scripts\python.exe` by default. It refuses
to start if the project-local environment is missing required packages. Do not
use virtual environments from other desktop projects.

```powershell
.\.venv\Scripts\python.exe scripts/run_vanilla_grace.py --dataset Cora --split-seed 0 --pretrain-seed 0
.\.venv\Scripts\python.exe scripts/extract_embeddings.py --dataset Cora --split-seed 0 --pretrain-seed 0
.\.venv\Scripts\python.exe scripts/run_noisy_classifier.py --dataset Cora --split-seed 0 --pretrain-seed 0 --noise-rate 0.2 --noise-seed 0 --classifier-seed 0
```

Run the gate experiment orchestration:

```powershell
.\.venv\Scripts\python.exe scripts/run_gate_experiment.py --datasets Cora CiteSeer --noise-rates 0 0.2 0.4 --seeds 0 1 2 3 4 --mode frozen --alphas 0.3 0.5 --save-diagnostics
```

Run the medium-label ratio-split gate separately from Planetoid:

```powershell
.\.venv\Scripts\python.exe scripts/run_gate_experiment.py --datasets Cora CiteSeer --split-mode ratio --train-ratios 0.1 0.2 --val-ratio 0.1 --noise-rates 0 0.2 0.4 --seeds 0 1 2 3 4 --mode frozen --alphas 0.3 0.5 --save-diagnostics
```

If the local project `.venv` is not available but the desktop Python already
has PyTorch, PyG, sklearn, and PyYAML installed, use:

```powershell
python scripts/run_gate_experiment.py --allow-external-python --python C:\Users\Admin\AppData\Local\Programs\Python\Python312\python.exe --datasets Cora CiteSeer --split-mode ratio --train-ratios 0.1 0.2 --val-ratio 0.1 --noise-rates 0.2 0.4 --seeds 0 1 2 3 4 --mode frozen --alphas 0.3 --methods ce global kmeans topk_matched gb gb_prior gb_ca_ce gb_ca_global --save-diagnostics
```

This validation branch extends the original Experiment 7 gate. The default
methods are `ce`, `global`, `kmeans`, `topk_matched`, `gb` strict, auxiliary
`gb_prior`, `gb_ca_ce`, and `gb_ca_global`.

- `gb`: strict ball-only smoothing; kept as the historical reference.
- `gb_prior`: prior-corrected diagnostic; not evidence for strict GB.
- `gb_ca_ce`: coverage-aware GB smoothing with noisy one-hot fallback.
- `gb_ca_global`: coverage-aware GB smoothing with global-prior fallback.

The primary alpha is `0.3`; `0.5` is sensitivity only.

Use `--dry-run` on `run_gate_experiment.py` to print planned runs without
training. The formal seed protocol is `split_seed=0`,
`pretrain_seed=noise_seed=classifier_seed=seed`. Existing complete checkpoints,
embeddings, and result rows are skipped unless `--force` is passed.

Planetoid artifacts keep the legacy `split0` path. Ratio split artifacts use a
protocol-specific token such as `ratio_train0p1_val0p1_split0`, so summaries and
paired deltas do not mix the two protocols.

`topk_matched` uses `k = round(mean(|B_train|-1 for eligible GB balls))`, at
least 1, selects only train-node neighbors, and excludes the node itself. If
primary-alpha `gb` strict coverage averages below 0.30 on Cora/CiteSeer, treat
Planetoid as label-sparse and plan a separate medium-label protocol.

Summarize per-run rows into mean/std tables and paired deltas:

```powershell
.\.venv\Scripts\python.exe scripts/summarize_gate_results.py
```

The current success gate is stricter than the old Experiment 7 gate:
`gb_ca_ce` or `gb_ca_global` must beat `ce`, `global`, `kmeans`, and
`topk_matched` on both Cora and CiteSeer under the same split, seeds, noisy
labels, and frozen GRACE embeddings. If CiteSeer remains unstable or
kmeans/topk matched ties the coverage-aware GB variants, the result should be
reported as a weak/negative validation rather than a main-method claim.
