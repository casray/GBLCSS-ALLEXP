# GSSC Official-Core GPU Adapter

## Status and Frozen Source

- Implementation is ready for parent-controlled execution. The parent has reported a successful Cora smoke with two epochs per phase (six epochs total), and has started the five-graph grid. Those are parent executions, not executions performed by this adapter task.
- This task performed syntax compilation and small non-training formula/API tests only. No GPU training or formal experiment was launched here. GPU end-to-end coverage is explicitly opt-in.
- At the parent's freeze request, no further edits to `gssc_adapter.py` or `test_gssc.py` are allowed. Only this notes file was completed afterward.
- Frozen adapter SHA256: `bd9ed5ee5e5f43e08f91633f7f3f99a5494447c77f10f547570701c5bf685754`.
- Frozen tests SHA256: `37fe87c5c3ca1e0826def3a7606debb4308dbb98f8453127f26480c47639de9f`.
- Adapter last write: `2026-10-09 12:26:54` (host local time). Tests last write: `2026-10-09 12:26:31`.
- Long-running Python processes retain their imported source. Parent should match the version imported by each process against its own launch/provenance record; the hashes above identify the final on-disk version, not an assertion about an already-running process.

## Author Provenance

- Repository: [LirongWu/GSSC](https://github.com/LirongWu/GSSC).
- Verified local author commit: `197a7f55e67e8d683dcd88149b788b76c39f058f`.
- Read first: `vendor/gssc/README.md`, `code/main.py`, `code/model.py`, `code/dataset.py`, `code/utils.py`, and `param/best_parameters.json`.
- Vendor was not modified; its `git status --short` was empty after implementation.
- Run diagnostics/checkpoints include SHA256 hashes of the author main/model/parameter files. No vendor code is dynamically patched or imported into the training loop.
- Project rules, the literature map, and relevant sections of `../../experiment_history_and_lessons/02_experiment_history_and_lessons.md` were consulted. No baseline evaluation, split, noise generation, or parent source was changed.

## API and Label Isolation

```python
result = run(data, config, outdir)
```

- The parent supplies a CUDA PyG Data with raw main-experiment `x` and `edge_index`, noisy labels in `y` only at training nodes, ordered split indices and corresponding masks, ordered `val_y`, explicit `num_classes`, and `name`.
- Features/edges are used as received: no normalization, self-loop insertion, coalescing, symmetrization, PCA, feature reduction, new split, or noise regeneration.
- All input tensors must be on the same CUDA device. CPU inputs fail before creating artifacts. PyTorch's `cuda` API also covers the installed ROCm environment; no DGL, Pyro, torch_sparse, or torch_scatter runtime is required.
- Non-training `y` must be exactly -1. Split masks are checked against their indices; disjointness, index uniqueness, finite features, ranges, and ordered validation-label length are checked.
- `data.val_y` is read only in epoch-end validation under `no_grad`. It never enters the classifier, center, mixup, sampler pretraining, or outer homophily loss. The outer target is the classifier's predicted class, not a validation label or a validation meta-loss.
- Test indices/masks are read only for input consistency checks. Test labels do not exist at this interface and are never loaded. No true-label homophily diagnostics are computed.
- RNG is not reset inside `run`; parent resets seed 0 before each call. A nonzero requested seed is rejected. No adapter network calls, telemetry, dataset downloads, or test-based parameter selection occur.

## Preserved Core

1. NeighSparse has two consecutive linear layers without an intervening activation. For each existing directed edge, `p = (1-alpha) * sigmoid(h_src dot h_dst) + alpha`. The graph prior weight is one, as in the author's `adj_weight`.
2. Sampling uses Torch RelaxedBernoulli, `clamp_probs`, then hard rounding with the identity straight-through backward. This matches the sampling path in [Pyro's official source](https://docs.pyro.ai/en/1.9.2/_modules/pyro/distributions/relaxed_straight_through.html). Pyro's `_unquantize` metadata is not needed because this algorithm never calls distribution `log_prob`.
3. The classifier retains the author's linear stem, BatchNorm/LeakyReLU/Dropout/Linear blocks, and separate `out` and `inf` heads. `out` supplies final probabilities. This is not a plain supervised MLP substitute.
4. MixupScale retains shared linear/BatchNorm transforms, concatenated endpoint attention, and `beta = scale + (1-scale)*sigmoid(attention)`. Both directed virtual-node feature mixtures are retained.
5. Self-contrast is exactly `MSE(y1,z3) + MSE(y2,z4) - MSE(softmax(y1),softmax(zneg)) - MSE(softmax(y2),softmax(zneg))`. The random negative node for each edge is part of the author algorithm; it is not an added subsampling approximation.
6. Noisy supervised CE is applied to labeled endpoints and their corresponding virtual-node heads, with the author's node/edge class-frequency weighting. The per-class noisy-training-to-pseudo-prediction center MSE and its separate model optimizer step are retained.
7. Main epochs alternate classifier/mixup updates with a sampler update minimizing `mean(nonhomophilous weights) / mean(homophilous weights + 1e-12)`. The sum ratio is logged separately. Hard selected weights remain connected to the sampler's relaxed gradient. A forward mean ratio of one is therefore not evidence of zero sampler gradient.
8. This is the author's first-order bilevel alternation using hard `argmax` pseudo-labels, not differentiation through inner optimizer steps. Sampler pretraining uses the author's unusual BCE-with-logits on the hard straight-through sampled values against ones; it is intentionally not replaced by a more conventional objective.

## Training Mode and Optimizers

- `model.train()` is explicit in every usable edge minibatch and before main-phase pseudo-prediction. `model.update` only evaluates the `out` head, matching the author.
- Pseudo-prediction is under `torch.no_grad()` but remains in training mode: BatchNorm running statistics update and dropout remains active. Validation is in eval mode and does not update BatchNorm. The next epoch restores training mode.
- Main model, sampler, and mixup optimizers are instantiated before pretraining. Each pretraining phase uses separate optimizers; main-phase Adam moments start fresh, matching the author source.
- Model and sampler: Adam lr=0.01, weight decay=0.0005. MixupScale: author-specific Adam lr=0.0001, weight decay=0.0005. The special mixup learning rate is not forced to the common lr.

## Fixed Protocol and Predeclared Mapping

All five graphs use hidden=64, dropout=0.5, two layers, common Adam settings, uniform 0.4 training-only label noise supplied externally, and seed 0. These common hidden/dropout settings deliberately replace the author's larger hidden widths and Citeseer dropout=0.7.

| Parent name | Author parameter entry | batch_size | scale | alpha | temp | Mapping status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| cora / Cora | cora | 512 | 0.8 | 0.1 | 0.5 | Official dataset entry |
| citeseer / CiteSeer | citeseer | 1024 | 0.7 | 0.1 | 0.8 | Official dataset entry |
| corafull / CoraFull | cora | 512 | 0.8 | 0.1 | 0.5 | Preset fallback: no official CoraFull entry |
| coauthorcs / CoauthorCS | coauthor-cs | 4096 | 0.8 | 0.3 | 1.0 | Official dataset entry |
| amazonpho / Photo | cora | 512 | 0.8 | 0.1 | 0.5 | Preset fallback: no official Photo entry |

Mapping is based only on the dataset name and was fixed before results. Unsupported names raise an error. There is no nearest-test-score selection. Explicit algorithm-parameter overrides are recorded in `config.parameter_overrides`; default official parameter values are otherwise retained. Common protocol values are enforced, with `lr` and parent's `learning_rate` supported as equivalent aliases.

## Stage Budgets and Selection Deviations

| Phase | Formal default | Parent smoke with epochs=2, smoke=True | Eligible for best checkpoint |
| --- | ---: | ---: | --- |
| EdgeSampler pretraining | 100 | 2 | No: classifier is still untrained |
| Classifier/Mixup pretraining | 100 | 2 | Yes |
| Main bilevel/self-contrast phase | 200 | 2 | Yes |
| Total logged epochs | 400 | 6 | 300 / 4 eligible classifier epochs |

- **Budget deviation from a strict 200-total-supervised-epoch baseline:** GSSC retains the required author classifier warmup in addition to the 200-epoch main budget. It therefore has 300 supervised classifier epochs, plus 100 sampler-only epochs, by default. Do not describe this as an equal-total-training-budget comparison. Config and diagnostics report all phases explicitly.
- `epochs` overrides the main phase. With `smoke=True`, unspecified `ep_epochs` and `cl_epochs` are each capped at `epochs`. Without the smoke flag, setting `epochs=2` alone does NOT cap the 100-epoch pretraining phases. Explicit phase overrides are supported.
- Author early stopping after 50 non-improving epochs is disabled to honor the fixed parent budgets.
- Selection uses clean validation accuracy only, and `>=` selects the last tie, replacing the author's strict `>` rule. The best classifier checkpoint is selected across classifier pretraining and main phases; a warmup winner is retained rather than forced to a worse main checkpoint.
- `best_epoch` is one-based cumulative classifier epoch: 1..300 formally or 1..4 in the parent smoke. `diagnostics.best_phase` and `best_phase_epoch` locate it within the appropriate phase. It is NOT a main-only 1..200 index.
- Each JSONL row carries `global_epoch` (all phases), `epoch` (classifier counter, null for sampler pretraining), `phase`, `phase_epoch`, `loss`, and `val`. Edge pretraining validation is logged but is not checkpoint-eligible.
- Main/classifier `loss` is mean minibatch edge-total loss plus the separate center loss, not one combined full-batch optimization objective. Individual components, batches, processed/selected edges, homophily losses and degeneracy counters are also logged.

## Sparse Memory Port and Numerical Differences

- The author already samples only existing edges; there is no N-by-N candidate adjacency to approximate. Every supplied edge remains eligible each main epoch. No node/edge truncation or added subset selection is used.
- Exact chunked dot-product forward and scatter-add adjoint replace materializing edge endpoint activations for all E edges. Default edge_chunk_size=8192; reducing this only changes the workspace size and floating-point accumulation order, not the graph, scores, objective, or gradient formula.
- The sampler uses O(N*hidden + E + chunk_size*hidden) extra memory, excluding raw feature storage and parameters. Classifier steps also need edge-minibatch feature tensors and full-node center/prediction activations. There is still full-graph work; large-graph speed/memory is not claimed to be measured in this task. CUDA OOM is not caught for CPU fallback or reduced-data retry.
- GPU `randperm` and GPU negative-node generation replace CPU DataLoader shuffling/CPU negative indices. The distributions and epoch batching are preserved, but the old Torch 1.6/DGL/Pyro implementation and this Torch 2.9/ROCm implementation are not bitwise reproducible against each other.
- Class counts use explicit integer bincount over noisy training labels rather than `histc` with implicit range. Absent/uncovered classes receive finite zero edge weights instead of division-by-zero/NaN; uncovered noisy-training classes are logged. For fully covered classes the weighting formula is unchanged.
- A singleton remainder minibatch is skipped and counted. This follows the author's main loop; the same guard is also applied to pretraining to avoid BatchNorm's batch-size-one error. No substitute BatchNorm mode or repeated node is used.
- If no sampled edge exists, the center step still runs, and the event is recorded. The homophily step is skipped whenever either same-class or different-class selected edges are absent, as in the author code.
- Eval/pseudo-label gradients unused by the author are avoided with no_grad; this does not change optimizer targets or train-mode BatchNorm/dropout semantics.

## Artifacts and Execution

- `probabilities.pt`: plain CPU Tensor of shape [N, num_classes], from the selected checkpoint's out-head softmax, all nodes included.
- `checkpoint.pt`: `state` dictionaries for model/mixup/sampler including BatchNorm buffers; resolved `config`; `best_epoch`, `best_val`, and diagnostics. Optimizer/RNG state is not saved: this is an inference checkpoint, not an exact mid-training-resume artifact.
- `epochs.jsonl`: one flushed record per completed epoch, across all phases. Reusing an outdir containing any of these artifacts raises FileExistsError to avoid mixing runs.
- Return keys: `best_epoch`, `best_val`, `parameters` (all three trainable modules), `config`, absolute `probabilities_file`, and `diagnostics`. Per-module parameter counts are exposed separately.
- Import/run assumes parent owns seed reset, GPU scheduling, official dataset/split/noise audit, and final clean test evaluation. This task does not launch `run_screen.py` or the grid.

Non-GPU tests (no bytecode in the experiment directory):

```powershell
& 'D:/codex_temp/noisygl-rocm721/Scripts/python.exe' -B -m unittest discover -s experiments/new_graph_mlp_seed0_20261009 -p test_gssc.py -v
```

Optional tiny synthetic GPU API test, ONLY when scheduled by parent:

```powershell
$env:GSSC_RUN_GPU_TESTS = '1'
& 'D:/codex_temp/noisygl-rocm721/Scripts/python.exe' -B -m unittest discover -s experiments/new_graph_mlp_seed0_20261009 -p test_gssc.py -v
Remove-Item Env:GSSC_RUN_GPU_TESTS
```

## Verification and Claim Boundary

- Initial missing-adapter tests failed as expected before implementation.
- A clamp regression specifically exposed a lower-bound derivative discrepancy (`tiny` versus Pyro/Torch `eps`); this was repaired before the parent source freeze. The final sampler uses actual `clamp_probs`.
- Latest task-local result: 14 discovered tests, 13 passed, one GPU API test intentionally skipped. Coverage includes author model/mixup state/output equivalence, exact chunk score/gradient comparison, full sampler draw/parameter-gradient comparison, extreme clamp derivatives, ST gradient, homophily gradient, self-contrast loss, train-only endpoint counts, absent-class guards, ordered val_y-only evaluation, parent smoke aliases, and CPU-run rejection before writing.
- Syntax compilation passed. A final post-freeze compilation is only a read of source with temporary .pyc outputs, not a source edit. The installed PyG import emits an upstream deprecation warning unrelated to adapter behavior.
- Parent reported Cora smoke validation 0.6222. This is an integration observation only, not a formal five-graph result, improvement, significance claim, or parameter-selection input. No mean/std or multi-seed evidence exists from this task.
- Experiment hypothesis: faithful official-core GSSC can be assessed under the parent's raw-feature/train-only-noise protocol without label leakage or algorithmic simplification. The implementation/tests support formula and interface fidelity, not robustness or superiority under uniform0.4.
- Granular-ball role: none; GSSC is an external graph-informed MLP baseline, not a new GB method. Method motivation, code/config changes, expected mechanism, and port limitations are recorded above.
- Integration success criterion: finite all-node CPU probabilities matching the selected validation checkpoint; complete phase logs; preserved sampler gradient and no forbidden labels in optimization. Failure criteria: leakage, CPU fallback, non-finite loss, API mismatch, or OOM; no silent algorithm/data reduction is permitted.
- Final decision: adapter ready, source frozen; parent owns the already-started formal runs and detailed per-dataset conclusions. Formal outcomes and multi-seed claims remain unverified here.
