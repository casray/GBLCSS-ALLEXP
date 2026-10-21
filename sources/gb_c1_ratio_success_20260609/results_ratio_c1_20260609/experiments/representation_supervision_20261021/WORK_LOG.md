# Execution work log

- Design approved in the current conversation. Scope is the first single-seed representation/supervision screen, legacy bridge, and H references.
- Completed: historical context and source review; existing ROCm GPU environment probe; 14 behavioral/reference tests; 12 input protocol checks; frozen source/input manifest.
- Completed: three 3-epoch engineering fits, including X-only export equality and frozen-encoder checks. Engineering results are excluded from the formal grid.
- Completed: four validation-only weight-development fits. Lambda=1.0 was selected and frozen before the main grid; test labels were not used.
- Completed: 144 main + 24 legacy + 24 H-probe cells, with three reusable label-free pretraining runs. Grid wall time 511.259 seconds. Progress status is complete; raw epoch logs are saved by the runner.
- Added during execution: audit_independent.py is a separate post-grid checker. It is not imported by training and cannot affect losses, inputs, checkpoint choice or hyperparameters. Its scope is independent metric recomputation and protocol/state checks.
- Completed: post-grid scoring, 192 exact checkpoint reloads, independent192 accuracy recomputation, 52 frozen source/input hashes, selection and budget checks, detailed Chinese interpretation and context handoff.

Decision: retain the R2 representation-training signal; current fixed LCSS has +1.320pp noisy gain but fails the full clean-damage gate. R2+small-loss is not a default combination. No new branch or seed expansion launched.

Preparation format failure and its fix are retained in ENGINEERING.md and failure_prepare.json. No training failure has been reported at this point.
