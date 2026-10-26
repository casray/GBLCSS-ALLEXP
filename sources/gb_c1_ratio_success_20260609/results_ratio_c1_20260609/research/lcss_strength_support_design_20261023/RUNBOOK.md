# LCSS Adaptive Support and Node Control Execution Plan

> For agentic workers: use superpowers:executing-plans for implementation checkpoints. The user authorized one gpt-6.1-sol/high executor; keep GPU fitting serial.

**Goal:** Implement and complete the authorized seed0 experiment queue, preserving paired inputs and validation-only selection.
**Architecture:** Read-only adapters for historical data/models; isolated target/control kernels, fit runner, selectors, evaluator and audit in one new experiment directory.
**Tech Stack:** Existing Python/PyTorch ROCm environment, NumPy, scikit-learn, PowerShell. No new external service.

## 1. Before Dispatch

- [ ] Read `SCHEDULED_PLAN.md`, `config.scheduled.json`, `EVIDENCE.md`, and `HANDOFF.md` in this directory. Old PLAN supplies A/B details only.
- [ ] Check UTC >= `2026-10-23T20:35:00Z`. Do not start training early.
- [ ] Use timezone-aware timestamps for the guard, e.g. Python `datetime.fromisoformat(value.replace("Z", "+00:00"))` versus `datetime.now(timezone.utc)`. PowerShell ConvertFrom-Json may already produce a UTC DateTime; cast that value to DateTimeOffset instead of parsing its culture-formatted string again.
- [ ] Inspect the experiment directory for active owner/agent/run lock, completed cell manifests and final report. Do not duplicate a live or completed queue.
- [ ] The parent spawns exactly one executor with `model="gpt-6.1-sol"`, `reasoning_effort="high"`. Log the returned agent id. If unavailable, report the model problem and keep pending; do not silently substitute.
- [ ] Parent handles schedule/status and later result review; the executor owns the new experiment directory. No overlapping fitting workers.

New experiment root (create only after the time guard):

`C:/Users/Admin/Desktop/实验8去噪5.19-/gb_c1_ratio_success_20260609/results_ratio_c1_20260609/experiments/lcss_strength_support_20261024/`

## 2. Source Inspection and Minimal Adapter

Read the existing implementation before making adapters:

- `experiments/r2_body_comparison_20261023/run.py`: R2 heads, strict validation selection.
- `experiments/r2_body_comparison_20261023/external_v2.py`: SimMLP encoder and author linear head.
- `experiments/simmlp_tined_audit_seed0_20261009/simmlp_adapter.py`: author bridge.
- `experiments/sc_support_channel_pilot_20261023/fit_pipeline.py`: data, labels, historical encoders.
- `experiments/representation_supervision_20261021/core.py`: R0/R2 students and base targets.
- `experiments/neighbor_support_ablation_20261022/kernels.py`: SUPPORT cost.
- `experiments/neighbor_preserving_ball_20261022/generator.py`: deep-tree DP truncation.

Verified input locations:

- E1 data/labels/geometry: `experiments/sc_support_channel_pilot_20261023/diagnostics/e1/<dataset>/seed0/`.
- Historical SUPPORT32 comes from `geometry.pt` field `legacy32`; distinguish deep-tree S32.
- KM: `experiments/r2_body_comparison_20261023/geometry/<dataset>/seed0.pt`.
- SimMLP checkpoint: `experiments/simmlp_tined_audit_seed0_20261009/runs/<dataset>/simmlp/checkpoint.pt`.

Resolve actual tensor keys and file names from these sources; do not infer missing keys. Local torch checkpoints may be loaded as trusted project artifacts, but training adapters must expose only explicit field allowlists.

Create a minimal isolated package under the new experiment root:

- `protocol.py`: manifests, paths, input loading, validation subset, safe resume.
- `targets.py`: m/d, lambda, MASS, B2 factorization, grouping, exact-mass controls.
- `fit.py`: existing R0/R2/SimMLP adapters, shared initialization, per-candidate RNG and optimizer resume.
- `selectors.py`: B1 choices, PROBE with causal log prefix, C2 dataset-held-out predictors.
- `meta_gate.py`: D2 functional linear-head virtual update and sigmoid gate.
- `run.py`: bounded stage/cell queue, not-before and exclusive run lock.
- `analyze.py`: frozen-choice scoring and detailed Chinese report.
- `audit.py` and `tests/`: invariants and independent recomputation.

These are planned files; this runbook does not claim that they exist yet. Use apply_patch for manual edits. Before edits list exact files and intended changes. No historical source modification, package installation, repository push or manuscript work.

## 3. Minimal Tests Before GPU Fits

- [ ] Target tests cover lambda 0/1, q[z]=1, ties and insufficient votes; reconstructed targets sum to 1 and match historical targets at lambda1.
- [ ] B2 common-mask tests verify all controlled arms equal N32 outside J; DM is separately named from native targets.
- [ ] Mean/shuffle tests verify bounded weights and exact sum(m*lambda), including all-zero m and all-zero/all-one weights.
- [ ] Selection tests include validation ties, CE fallback, original versus BALANCED policy, and absence of test/true-noise fields.
- [ ] PROBE tests ensure epoch25 selection never reads later epochs and mandatory anchor survives. Continuous versus interrupted training must agree within recorded numerical tolerance using each arm's own RNG state.
- [ ] C2 split tests prove all rows of the held-out graph are excluded from predictor/scaler fitting.
- [ ] D2 tests prove train/meta/selection/test disjointness; virtual-gradient path reaches gate; real-head target is detached; frozen encoder/BN unchanged.
- [ ] Time guard and exclusive-lock tests use fake clock/state, not a real training launch.

First read-only environment command, to execute after not_before:

```powershell
& 'D:/codex_temp/noisygl-rocm721/Scripts/python.exe' -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO_GPU')"
```

If GPU is in use, inspect relevant running project jobs and utilization, record the condition and wait without terminating others. If no GPU is available, leave pending and report; avoid silently switching all fits to CPU.

After the new files exist, run their fastest unit tests before any GPU smoke. The executor must record the actual test command and observed output, not just copy an intended command into the report.

## 4. Execution Order

- [ ] Freeze active config, source snapshots and input hashes. Capture user authorization and not-before in run manifest.
- [ ] At most 12 two-epoch engineering smokes across adapters/meta/RNG recovery; no scientific screening from these scores.
- [ ] Execute A and B1, then B2. Failures in one scientific arm do not cancel an independent planned stage.
- [ ] Execute C1 PROBE in real prefix order, save selections, then complete full-grid audit. Reuse only exactly matching B2/B1 cells with recorded aliases.
- [ ] Fit C2 predictors with grouped folds and no test fields.
- [ ] Execute D1 four rules, freeze per-cell selected rule and run its exact-mass controls.
- [ ] Execute D2 nine arms under the distinct meta/selection protocol; never reuse full-validation-selected old heads.
- [ ] Freeze every stage's selections before loading test labels. Save probabilities before post-hoc truth diagnostics.
- [ ] Run independent audit, derive per-condition tables, costs and node-level summaries, then write a detailed Chinese report.

All GPU fits serial. Repeated scheduling must see lock/manifests and attach to the existing work, not spawn another queue. Keep engineering failures and repaired versions visible. A numerical or API repair may preserve the scientific specification; a proposed algorithm change requires a new plan rather than silently changing this one.

## 5. Required Artifacts

- Material passport with authorized scope, worker model/effort and actual execution times.
- Full resolved config plus original plan copies and SHA256 source/input manifests.
- Per-cell raw logs, initialization hash, runtime, optimizer settings, selected epoch, checkpoint and full-node probabilities.
- `selections/` with timestamps and validation-only records; PROBE decisions precede full-grid completion.
- `results.csv`, `summary.csv`, `paired_differences.csv`, per-class recalls, clean/macro-F1 warnings, actual support distributions.
- C: selected k/lambda, regret versus FULL_VAL, resource accounting, held-out-dataset predictor results.
- D: group/gate distributions, mean transferred mass, matched/shuffled controls, clean-meta-label controls.
- `failures.jsonl`, `audit.json`, `REPORT_zh.md`, `HANDOFF.md`, terminal completion marker containing completed/planned counts.

For seed0 set seed standard deviation to missing, not zero. Explain what worked, where, against which control, cost and negative conditions. Keep all four condition types and every prescribed dataset visible. No automatic confirmation-seed expansion or new method combinations this run.

The main agent verifies worker output and samples reloaded results before claiming completion. Keep working through any launched fit session; do not hand back a completion message with required processes still running.
