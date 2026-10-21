# Session handoff: 2026-10-09

## Request and completion

Run SimMLP next, then investigate TINED's low CoraFull result. Completed.
No manuscript edits, no original result changes, no extra seeds or full-grid replay.
All training subprocesses finished. The read-only TINED audit agent was closed.

## Read first

- REPORT.md: measured results, adaptations and diagnosis boundaries.
- PLAN.md: hypothesis, fixed settings and success/failure criteria.
- results.csv and artifact_audit.json: verified saved results and reload audit.
- tined_diagnostics/summary.json and *_phases.json: validation-only diagnostics.
- ../new_graph_mlp_seed0_20261009/99_session_handoff.md: prior GSSC/TINED/DGD state.

## New files

simmlp_adapter.py, run_simmlp.py, test_simmlp.py, diagnose_tined.py,
report_results.py, PLAN.md, REPORT.md, results.csv, artifact_audit.json,
runs/, smoke/, tined_diagnostics/, vendor/simmlp/.

Official SimMLP source: https://github.com/Zehong-Wang/SimMLP
Commit b460fa5caf8ba3381aa00fb11c99a9b85faf17a0; vendor working tree clean.
AST-selected unchanged core definitions; nonparametric aggregation is a sparse
PyTorch port. Exact main split/noise fingerprints and raw attributes are reused.
Author test-based outer pretraining selection is deliberately not reused.
Budget 200 final-epoch label-free pretraining + 200 validation-selected noisy-label
linear probe, hidden64/encoder dropout.5/lr.01/wd.0005, official AdamW+schedule.
Other method knobs use dataset presets; CoraFull uses predeclared Cora fallback.

## Supported evidence

SimMLP test percentages (uniform.4, seed0): Cora81.3193, CiteSeer70.1031,
CoraFull49.3179, Coauthor-CS92.4881, Photo88.7021. All five checkpoint reloads,
source hashes, protocol fingerprints and selected epochs passed artifact audit.
Three fast unit tests pass; real Cora GPU smoke completed. No mean/std inference.

TINED: frozen saved teacher (validation61.1420); three20-epoch student diagnoses,
identical initial weights/seed/data, only validation evaluated, no test-label read.
Baseline reproduces best epoch4/validation18.6963 and reaches one-class validation
prediction at epoch20/5.0531. DE coefficient0 improves best validation25.9222;
student dropout0 improves best validation54.1182, final52.0970. All initial
students have validation51.9454. Thus imposed dropout.5 is a key causal factor
in this particular adaptation; it is not evidence that official TINED fails.
Author Cora preset has student dropout0. No official CoraFull preset exists.

## Hypotheses and caveats

Saved epoch4 eval DE gradient norm5.9641 vs KL.17749, cosine-.34403;
supports objective conflict, not proof across all training updates.
Three student dropout locations (four FCs) vs one teacher dropout location;
CoraFull70classes exceed hidden64, causing correct official FT-GP-GP-FT ordering.
Zero-weight CE still executes official Adam steps; initial validation51.95 drops
to30.62 after first CE stage. Weight decay/momentum effects and lr remain relevant.
Full DGL training parity, other seeds and other teacher settings remain untested.
Original TINED test19.1916 is unchanged; diagnosis has no replacement test result.
DGD remains blocked as previously documented; no new DGD work authorized here.

## Next decision

Before a formal TINED rerun, agree on method-specific student dropout (prefer
author preset) while retaining main split/noise/width and validation-only selection.
Do not silently replace original scores with diagnostics or tune against tests.

## Commands

Runtime: D:/codex_temp/noisygl-rocm721/Scripts/python.exe
Workdir: this experiment folder.

```powershell
& 'D:\codex_temp\noisygl-rocm721\Scripts\python.exe' -X utf8 -m unittest test_simmlp -v
& 'D:\codex_temp\noisygl-rocm721\Scripts\python.exe' -X utf8 -u run_simmlp.py --dataset cora --smoke
& 'D:\codex_temp\noisygl-rocm721\Scripts\python.exe' -X utf8 -u run_simmlp.py --grid
& 'D:\codex_temp\noisygl-rocm721\Scripts\python.exe' -X utf8 -u diagnose_tined.py
& 'D:\codex_temp\noisygl-rocm721\Scripts\python.exe' -X utf8 -u report_results.py
```

Completed formal runs skip reruns; diagnostics refuses overwriting summary.
report_results.py verifies existing checkpoints without training.
