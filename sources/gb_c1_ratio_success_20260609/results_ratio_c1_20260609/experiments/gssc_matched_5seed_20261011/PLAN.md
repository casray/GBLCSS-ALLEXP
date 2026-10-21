# GSSC five-seed extension implementation plan

> Execute inline using the existing approved design and focused test-driven checks.

Goal: Five main datasets, uniform0.4, seeds0..4, same approved GSSC matched adapter.
Status2026-10-11: Paused at user's request to release GPU. Four datasets complete (seeds0..4); Amazon-Photo seed0 complete, seed1 interrupted and preserved, seeds2..4 pending.21/25 records complete. No automatic resume.
Architecture: Import frozen previous adapter and audited protocol; add seed-aware orchestration/reuse verification/reporting only. Original directories/training code unchanged.
Tech stack: Existing PyTorch ROCm/PyG runtime; hashes/CSV/statistics.

## Frozen experiment design

Two deployed FC layers,hidden64,ReLU-BN-dropout0.5. Native auxiliary head,mixup,sampler,losses,presets retained.100 label-free edge +100 supervised warmup +100 supervised main epochs. Adam lr0.01 wd0.0005; native mixup lr0.0001. First clean-validation maximum over200 supervised epochs. Raw main features,20/10/70 split,train-only uniform0.4; split/noise/downstream seeds equal actual0..4. Matched deployment/epoch budget is not matched compute.
Hypothesis: evaluate the approved adaptation across five paired main seeds. Granular-ball role:none. No test-driven changes/exclusions. Success:25 verified cells,seed0 reused exactly,200 supervised epochs each,historical split/noise equality,exact checkpoint replay. Failure:identity mismatch,leakage,nonfinite loss,OOM or replay mismatch; preserve artifacts and fail visibly.

## Tasks

- [x] Added tests; missing-runner assertion failed before implementation.
- [x] Seed-aware runner reuses frozen training function without modifying historical sources/globals. Actual training sources verified separately from orchestration sources.
- [x] Cora seed0 verified and copied without retraining; same guarded reuse applies to other datasets.
- [x] Three focused tests and real-Cora seed1 GPU smoke passed (checkpoint replay exact). Initial import-path error fixed before training artifacts were created.
- [ ] Run `D:/codex_temp/noisygl-rocm721/Scripts/python.exe -B -X utf8 -u experiments/gssc_matched_5seed_20261011/run_grid.py --grid`:25 cells serially,5 reuses +20 new trainings.
- [x] Added report.py; final25-cell re-audit/export pending until all cells complete.
- [x] Recorded partial results and user-requested pause in PAUSED_REPORT.md and99_session_handoff.md; no manuscript changes or other-method reruns.
