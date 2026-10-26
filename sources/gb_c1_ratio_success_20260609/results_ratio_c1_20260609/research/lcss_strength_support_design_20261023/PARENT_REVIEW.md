# Main-session independent review

Status: COMPLETE_PARENT_VERIFIED. This file records checks by the coordinating session independently of the worker's audit. Independent script and results are parent_audit.py and PARENT_AUDIT.json.

## Dispatch

- Requested start: 2026-10-24 04:35 Asia/Shanghai (2026-10-23T20:35:00Z).
- Time guard passed before dispatch; observed UTC 20:35:50 and 20:36:44.
- Worker: Aquinas, `01a16a7a-f07c-7e52-8ba3-873a720e7212`, gpt-6.1-sol, high.
- The new experiment directory was absent and Get-Process found no python/pythonw processes before dispatch. CIM enumeration was denied, so GPU availability is separately checked by the worker.
- Worker handoff reports PyTorch 2.9.1+rocm7.2.1 / RX 7700 XT available. Formal GPU launch independently confirmed from environment/events and stage A results at 2026-10-23T20:50:33Z.

## Source inspection completed

- `representation_supervision_20261021/core.py`: historical LOO targets, tie handling, alpha, insufficient-vote fallback, AlignedMLP and encoder/BN freezing.
- `GB-LCSS_code/src/gblcss/models.py`: deployed MLP has linear/ReLU/BN/dropout/linear structure and consumes attributes only.
- `r2_body_comparison_20261023/run.py`: historical R2 frozen-head Adam training, numpy target construction and first strict validation-best selection.
- `r2_body_comparison_20261023/external_v2.py`: author SimMLP encoder/head bridge, AdamW, old last-tie selection.
- `sc_support_channel_pilot_20261023/fit_pipeline.py`: historical data loading, encoder selection, strict selection and deployment checks.
- `sc_support_channel_pilot_20261023/run_diagnostics.py`: E1 geometry keys, deep-tree partitions and legacy32 are separate inputs.
- Active plan, config, runbook, evidence and handoff read. Active scheduled scope supersedes old not-authorized and hard-stage-gate text.

## Checks communicated to worker

1. A reruns CE with strict first-best selection; old external SimMLP results used last-tie selection. Preserve AdamW versus R0/R2 Adam distinction.
2. Frozen R2 head training keeps encoder/BN in evaluation mode and has no training dropout. Preserve identical paired head initialization.
3. Bridge lambda=1 to the latest numpy-built historical targets; document floating-point tolerance rather than adjusting scientific parameters.
4. C1 must execute genuine candidate prefixes even when a matching B2 200-epoch fit is already available. Reuse input/init only; retain separate C1 fitting cost. Matching CE anchors may be reused with an alias record.

## Pending independent completion checks

- [x] Inspect new source for field allowlists and validation-only selection: training exposes observed train and prescribed validation labels only; C1 freezes the real prefix race before full audit; D2 splits original validation.
- [x] Verify unit-test and bounded GPU-smoke output, including RNG continuation and D2 meta-gradient tests: worker final record has 11 passing tests, 5 two-epoch smoke fits and 72 continuation bridges with probability difference zero.
- [x] Reconcile completed fits, reuse aliases, planned scope and all failures: independently counted A80/B1 336/B2 216/C1 576/anchors40/D1 144/D2 108 = 1500 unique fits. One 122-update interrupted attempt retained separately.
- [x] Verify shared initial states, historical input hashes, legacy/deep-tree identity and frozen encoders: worker audit has 13 shared initialization groups, historical bridges, 1607 input hashes; parent rehashed all frozen sources and selections.
- [x] Check C1 prefix decisions: independently reconstructed 768 prefix scores from log prefixes, checked anchor inclusion and immutable decision hashes. 1000 candidate updates + CE200 versus full candidate grid3600; all full-grid training actually executed.
- [x] Check C2 folds: independently recomputed all 20 training-fold feature means and target means, with entire held-out graph excluded.
- [x] Check D1/D2 controls: parent independently checked 4824 transferred-mass comparisons and all 12 disjoint meta/selection splits. Direct clean-meta CE/GCE controls present.
- [x] Independent metrics: 36 probability samples recalculated with sklearn accuracy/macro-F1, exact agreement. Direct functional CPU checkpoint evaluation for 31 R0/R2 samples, maximum probability difference 1.0728836059570312e-6, identical test predicted classes. Recomputed all 828 paired rows.
- [x] Seed0 standard deviations are blank; all four conditions and prescribed dataset arms retained. CS R0 historical SUPPORT32 comparison absent as disclosed; no unsupported fit added.
- [x] Worker returned completed, run.lock absent, Get-Process found no python/pythonw before parent CPU audit; audit itself exited 0.

## Final outcome

Completed. D1 R0 is the leading confirmation candidate: mean noisy gain over validation-selected global strength +1.123pp; by graph Cora +2.058, CoraFull +0.751, Photo +0.560. D1 R2 mean +0.475pp. Matched-mass controls support studying node allocation further; Photo has two negative macro-F1 comparisons.

C1 closely tracks finite FULL_VAL in this pilot and improves R0 over fixed deep-tree32 by +0.564pp. C2 predictors and current D2 meta gate underperform their relevant simple controls. No additional seeds, combinations, manuscript changes or uploads were launched.

Engineering repairs preserved separately: D2 static frequency normalization before any D2 fit; Windows path/alias normalization during postprocessing. schedule_receipt.json remains an immutable dispatch-time source snapshot; live completion is in EXECUTION_STATE.json.
