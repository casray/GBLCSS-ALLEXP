# TINED official-core main-protocol adapter

## Parent integration correction

The first real Cora smoke failed before training because the adapter incorrectly required exhaustive split coverage. The historical protocol floors each percentage split independently; unused nodes remain unlabeled graph context. Parent repaired only the validation condition to require disjoint splits, retaining the original indices/masks and label isolation. A regression test first reproduced the error and then passed, including rejection of overlapping splits. The full quick suite now has 25 tests: 23 passed and two optional GPU unit tests skipped. Initial worker hashes below precede this interface-only repair; formal launch hashes in `adapter_provenance.json` identify the final source. Failed smoke artifacts are preserved in `excluded_runs/cora_tined_split_validation_failure`.

Date: 2026-10-09. Status: implementation and CPU checks completed; GPU validation
and formal fits are exclusively scheduled by the parent. No formal fit was run here.

## Scope and provenance

- Write set: `tined_adapter.py`, `test_tined.py`, this note. No vendor/parent edits.
- Vendor: `vendor/tined`, upstream `scottjiao/TINED_ICML25`, cloned commit
  `ff5652cfb9b0724fa2ac99e23a3d88df67653741`.
- DGL import fails with `ModuleNotFoundError` in the existing Python environment.
  This is a **torch sparse SAGE port with an AST-reused official student class**,
  not direct execution or full reproduction of the original DGL pipeline.
- `MLP_from_sequence_of_layers` alone is extracted from `models.py`; its official
  constructor and attribute-only `forward` are executed unmodified. Only device-safe
  injection, target loading, and sparse DE calculation are supplied by the adapter.
  No vendor imports, vendor patching, graph positional encoding, PCA, or downloads.
- Source SHA-256 values for the relevant official files are saved in diagnostics.

## Why the teacher is SAGE

This is the author's **TINED experiment driver default**, not a claim that SAGE is
the only theoretically supported teacher. `tasks/3_run_all_TINED.py` specifies
`settings=["SAGE-transductive","SAGE-inductive"]`, selects `models_list=["ours"]`,
and reads `best_records[setting][model][dataset]`. The adapter follows the
`SAGE-transductive -> ours` branch. The author's `SAGE` class constructs
`SAGEConv(..., "gcn")` rather than standard mean GraphSAGE, and `tran.conf.yaml`
sets global two layers and the SAGE base optimizer settings.

The graph operation is exactly

`GP(H)[v] = (sum_(u->v) H[u] + H[v]) / (in_degree(v) + 1)`.

This is not PyG's usual GCN symmetric normalization. The parent raw directed edge
multiset is preserved, including duplicate edges and any existing self-loops.
The formula contributes its own self term. No new symmetrization or feature
normalization is introduced. Xavier-uniform/ReLU gain and zero teacher biases
follow the author's SAGE initialization.

## Main protocol and fixed presets

Public entry point: `run(data, config, outdir) -> dict`, compatible with the parent
`common.py` and `run_screen.py`. Public execution rejects CPU data/fallback.
Defaults: seed 0, teacher 200 epochs, student 200 epochs, hidden 64, dropout .5,
Adam learning rate .01, weight decay .0005, no early stopping. `config['epochs']`
sets both short-smoke budgets; explicit `teacher_epochs`/`student_epochs` can also
be supplied. Departures from 200+200 are recorded as `non_main_budget`.

Author-specific choices are fixed before training from
`best_records/best_records.json`, `SAGE-transductive/ours`:

| Parent dataset | Author preset | lambda | DE weight | DE edge ratio | DE log | FT lr multiplier | GP init | Student norm |
|---|---|---:|---:|---:|---|---:|---|---|
| cora | cora | 1 | 1 | 1 | true | 1 | identity | none |
| citeseer | citeseer | .5 | 10 | .1 | true | .5 | identity | none |
| corafull | cora (unsupported fallback) | 1 | 1 | 1 | true | 1 | identity | none |
| coauthorcs | cora (unsupported fallback) | 1 | 1 | 1 | true | 1 | identity | none |
| amazonpho | a-photo | .5 | .1 | .1 | false | .1 | identity | layer |

All use author batch size 512. `sqrt_DER=True` comes from the actual
`train_student.py` parser default; the task's `squared_DER` search-name entry is
not an implemented parser option and is not silently substituted.
Teacher norm stays `none` as in the base SAGE configuration. Parent optimizer,
hidden size, dropout and epoch budget override author base settings; the original
base values and preset/fallback decision are recorded in the effective config.
Injected FT group lr is base lr times the fixed author multiplier; GP/norm group
uses base lr. Conflicting core/position-encoding overrides raise explicit errors.

## Retained core operations

1. Train the two-layer SAGE-gcn teacher with CE on `y[train_index]` only. Use full
   sparse neighbors, then restore the last highest clean-validation-accuracy epoch.
2. Determine each teacher layer's ordering: FT then GP if `input_dim > output_dim`,
   otherwise GP then FT. The student has one learnable FC for each operation,
   **four FC layers**, not a two-layer replacement. Equal dimensions use GP first.
   CoraFull's output 70 > hidden 64 therefore has GP before its final FT.
3. Copy teacher FT weights and biases into distinct student Parameters with
   `copy_`; do not alias teacher storage. Initialize the additional square GP FCs
   as identity/zero bias. Preserve official student ReLU/norm/dropout after every
   non-final FC, even between a teacher layer's FT and GP.
4. Restore teacher checkpoint before constructing its ordered DE targets. As in
   official SAGE `layer_info`, teacher FT/GP targets use pre-deferred-bias features;
   teacher biases are added after each complete SAGE layer and copied into FT FCs.
5. Each student epoch performs ordered train-index CE mini-batch optimizer steps,
   then all-node teacher KL mini-batch steps, then a **separate full-graph DE step**.
   KL is `batchmean`, `log_target=True`, temperature 1. CE weight is `1-lambda`;
   KL weight is `lambda`. For lambda 1, retain the author's zero-weight CE optimizer
   steps because Adam decay/momentum still affects parameters. Do not merge losses.
6. The DE definition uses row normalization with a **detached** row norm:
   `E(H)=sum_(u->v)||H[u]/||H[u]|| - H[v]/||H[v]||||^2 / N`.
   Zero row norms are replaced with 1. Both FT and GP energy ratios are retained.
   Let `r=log(E_out)-log(E_in)`, target `t` from the corresponding teacher operation,
   and `p=.5` for sqrt_DER. The penalty is `DE_weight * sum((p*r-p*t)^2)` in log
   mode or `DE_weight * sum((exp(p*r)-exp(p*t))^2)` otherwise. It is a sum, not a mean.
7. Sparse adjacency handles GP. Sparse incidence `B` handles DE by computing
   `||B*normalized_H||^2/N` in 64-feature-column blocks. Custom first-order backward
   implements `2*B.T*B*normalized_H/(N*detached_norm)`, also in column blocks.
   No dense NxN graph or full E-by-raw-feature tensor is formed. Full-graph raw-X
   input energy is cached once because it is constant. No nodes/features are dropped.
8. The .1 DE edge ratio for supported presets samples exactly the author's fraction,
   drops isolated nodes only for that auxiliary edge-subgraph DE calculation, and
   remaps node IDs. Teacher, CE/KL and predictions still use every parent node.

## Label and selection boundary

- `y` must be noisy training labels and -1 everywhere else. `num_classes` is
  required explicitly; it is not inferred from the masked label vector.
- Indices/masks must be disjoint, exhaustive, consistent, and on the CUDA device.
  The parent-provided index order is never changed. In particular `val_y` is read
  directly against logits indexed by that exact `val_index`, not sorted indices.
- `val_y` enters accuracy/validation loss and checkpoint selection only, never
  CE/KL/DE gradients. Validation loss is diagnostic; accuracy is the selector.
- Each phase selects with `val_accuracy >= best_val`, keeping the last tied epoch.
- Test labels are neither received nor requested. No test accuracy is computed
  inside the adapter. Transductive teacher predictions/DE on test-node attributes
  are label-free and are not checkpoint selection.
- Deploy with `student(raw_x)` only. `edge_index` is not part of the student forward.

## Artifacts and return value

- `probabilities.pt`: CPU-serialized tensor [all parent nodes, num_classes] from
  the selected student, with finite normalized probabilities in original node order.
  CPU serialization is not a CPU training/inference fallback.
- `checkpoint.pt`: `teacher_state`, `student_state` (`state` alias), effective
  `config`, `teacher_config`/`student_config`, phase best epoch/accuracy, diagnostics.
  Restore teacher architecture from `teacher_config['dims']`, instantiate
  `TINEDStudent(teacher, checkpoint['config'])`, load `student_state`, call `eval()`.
- `epochs.jsonl`: one teacher/student record for each epoch, phase, loss, val,
  val_loss, selected flag, plus CE/KL/DE and ordered student log energy ratios.
  Failures preserve the prefix log; non-finite objectives/gradients raise errors.
- Return: `best_epoch`, `best_val` (accuracy fraction), `parameters` (actually used
  deploy layers/norms), `config`, `probabilities_file`, `checkpoint_file`, `diagnostics`.
  Diagnostics separately count teacher parameters and all student parameters;
  official unused feature-distillation heads/final norm are not hidden in totals.
- Existing artifact paths are rejected; do not overwrite a previous run. Checkpoints
  provide selected states, not full optimizer/RNG resume state.

## Differences and limitations

This is a controlled official-core comparison under the parent's noisy-label
protocol, not reproduction of the paper's clean-label data split or reported scores.
The main departures are the parent split/noise/budgets/base hyperparameters,
accuracy/last-tie selection instead of the author's validation-loss/patience,
and full-neighbor teacher training instead of sampled fan_out=5,5.
The sparse teacher is algebraically the author's full-neighbor SAGE-gcn operator,
not statistically identical to its neighbor-sampled training trajectory.
Fixed DE edge sampling uses seeded torch device `randperm` rather than NumPy choice;
the sampling distribution/fraction is retained, not identical random draws.

The official student zero-energy branch uses log-ratio=1; that value is preserved
but created on the correct device. Teacher zero-energy ratios would divide by zero
in the official analyzer; this adapter records degenerate layers and uses neutral
target log-ratio=0 for them. No claim of exact official zero-case reproduction.
Custom DE backward is first-order, which suffices for Adam; higher-order derivative
use is outside this adapter's contract. ROCm/CUDA sparse-kernel correctness and
whole-graph memory/runtime still require parent GPU smoke; no CPU/PCA/subset fallback.

## Verification and research boundary

Environment checked without GPU allocation: Python 3.12.10,
PyTorch 2.9.1+rocm7.2.1, PyG 2.7.0; DGL unavailable.
The initial red test correctly failed because the adapter did not exist.
CPU checks cover sparse GP with duplicate/self edges, DE value and detached-norm
gradient against direct edge differences, official AST DE/gradient parity, cached
energy parity, full FT/GP sequence including equal dims and 70 classes, nonaliasing
injection, teacher ordered energy targets, two teacher + two student synthetic
epochs, ordered validation labels, train-only contract, and last-tie selection.
CUDA API/artifact/reload smoke is opt-in via `TINED_GPU_SMOKE=1` and is not executed
by this worker while the parent owns GPU. This is mathematical/unit verification,
not CPU fallback for production experiments and not an accuracy experiment.

Run CPU tests (no GPU):

```powershell
& 'D:/codex_temp/noisygl-rocm721/Scripts/python.exe' -B -m unittest discover -s experiments/new_graph_mlp_seed0_20261009 -p test_tined.py -v
```

After the parent releases GPU, its normal command is:

```powershell
& 'D:/codex_temp/noisygl-rocm721/Scripts/python.exe' -B experiments/new_graph_mlp_seed0_20261009/run_screen.py --method tined --dataset cora --smoke
```

Prior reading: current literature map; upstream experiment-history main file and
records 40/41; strong-KD and regional-distillation handoffs; parent common/runner.
The latest independent-seed regional-DIST evidence failed its practical-gain gate.
No GB module, repair labels, thresholds, or test-guided hyperparameter search are
introduced here. This adapter establishes a baseline testable by the parent.
Supported now: the listed CPU mathematical/short-synthetic invariants only.
Unverified: actual GPU execution, formal five-dataset performance, multi-seed
stability, superiority over any method. No mean/std or significance claim is made.
Decision: ready for parent-controlled GPU smoke, then sequential formal training
only if that smoke succeeds; no automatic experiment launch by this worker.

## Source freeze

Source frozen on 2026-10-09 after the parent's final instruction. No further
features or GPU jobs are launched by this worker. Parent schedule: finish the
exclusive GSSC grid, run TINED Cora smoke, then five datasets at 200 teacher plus
200 student epochs if smoke succeeds.

Final local verification: 9 unittest cases, **8 passed / 1 GPU case skipped**;
both Python files passed `py_compile` with bytecode confined to a temporary
directory. `git -C vendor/tined status --short` was empty. The earlier parent
snapshot of 6 passed / 1 skipped preceded two additional CPU reference checks.

Frozen SHA-256:

- `tined_adapter.py`: `709f59e574634485c90be7d5151c176d60595aa15ad780aa50665c15f41d3b9d`
- `test_tined.py`: `4213c5d6af97ba80c07ba62fe90cd6248a41bfee75074ae2fb713cf552180b59`

Ready means ready for the parent's GPU smoke, not a claim that GPU execution or
formal benchmark accuracy has already been verified.
