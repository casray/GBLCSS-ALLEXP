# GCN + SUPPORT/D1/C1: active five-graph, three-seed expansion

## Final completion: 2026-10-27 16:21:59 Beijing

COMPLETE_AUDITED; launcher stage COMPLETE. All45 cells/six teacher and student routes complete, 270 logical student endpoints. New1620 teacher candidate trajectories/197 physical students/173925 successful scientific updates; interrupted63 extra updates separately archived. No training or audit worker remains to resume.

Five-graph U.4/U.6 seed-first macro accuracy: JOINT teacher76.676+/-.697%, student77.464+/-.449%; student macro-F1 70.862+/-1.401%. Paired student gains: CE+2.559+/-.315pp, OLD64+.832+/-.222, historical SUPPORT32+.551+/-.458, FIXED_D1+.174+/-.168. Clean student minusCE-.047+/-.304pp, minusOLD64+2.016+/-.260. CoraFull clean repair toSUPPORT32+3.051+/-.209. C1 extra gain is small; negative cells and variable Photo gain retained.

Full interpretation: `research/gcn_support_d1_c1_3seed_20261027/RESULT_INTERPRETATION_zh.md`. CSV/TeX original tables in runtime and thin summary files in this source folder.

Initial frozen audit had an unbound `relative_input` helper and stopped before full audit. Preserve original audit/source freeze/failure; standalone `audit_recovery_v1.py` binds the existing frozen analysis validator and reruns the original checks. No fit/source/selection change.29 tests pass. Full audit passes540 endpoint metrics/2745 prefix scores/2520 snapshots/2270 target checks/720 summaries/453 old refs/90 CPU model reloads, maximum probability error1.90735e-6 with identical validation/test argmax.270 target diagnostics saved. Explicit wrapper SHA and failure archive in runtime `audit_recovery_v1/receipt.json`. Do not rerun the broken original launcher/audit automatically; current final markers take precedence over historical failure/PID receipts below.

## Recovery receipt: 2026-10-27 15:34 Beijing

The first fit worker and continuation driver both exited without an exception record; no matching system reboot was found, and the exit cause remains unknown. The last complete teacher-cell count was 25/45. A read-only inventory verified 763 existing trajectories and found exactly one partial segment: `corafull/seed2/uniform0.4/S128_L0.5` logged epochs through 138 while its saved optimizer/model/RNG and result remained at epoch75. All frozen fitting/postprocessing sources, saved best artifacts and prefix snapshots verified.

New standalone `recover_interruption.py` preserves the full interrupted candidate, status/logs/manifests and stale locks at `D:/codex_temp/gcn_support_d1_c1_3seed_20261027/interruption_recovery/20261027T073427Z`. It aligned only this candidate's active log to its exact epoch75 prefix and archived stale locks after confirming all former workers absent. Original scientific sources, hyperparameters, selection rules, completed trajectories and immutable manifests were unchanged. The abandoned 63 updates are separately recorded in `recovery_latest.json`/`recovery_receipt.json` as interrupted overhead; normal trajectory cost tables retain successful scientific updates.

All 28 tests passed (`tests_recovery_verified.log`), including three new log-prefix recovery contracts. At 15:34:41, the original `launch.py` was restarted hidden: venv launcher46732, actual driver66220, actual fitting worker55948. The driver now directly owns fitting followed by analysis and audit; a second queue is unnecessary. Use live locks/status/process inventory rather than old PID receipts.

The interrupted candidate reached epoch200 after restart. `interruption_recovery/20261027T073427Z/replay_verified.json` verifies that all 138 original epoch log records, including the replayed 76--138 segment, match byte-for-byte in training losses and validation metrics. This confirms exact optimizer/RNG continuation for the interrupted segment. Current scope remains the approved 45 cells and six teacher/student routes, with no new test-driven selection.

## Scope and current state

Authorized 2026-10-27: original GCN teacher with new regional supervision, followed by the unchanged attribute-only DIST student. Five datasets (Cora, CiteSeer, CoraFull, Coauthor-CS, Amazon-Photo), protocol seeds 0/1/2, clean/Uniform0.4/Uniform0.6: 45 cells. The original three-graph seed0 pilot contributes nine immutable cells. Thirty-six new cells are being fitted. This is separate from the completed R2-teacher DIST experiment.

Keep all six teacher routes CE, OLD64, LEGACY_SUPPORT32, DEEP_S32, FIXED_D1 and JOINT, with paired students. There are 270 logical student endpoints; 54 are historical, 216 new. Exact teacher probability hashes and identical student initialization/config permit physical-fit aliases. JOINT has the unchanged 42 -> 14 -> 4 validation race and CE fallback. No new Pair/KM/kNN/R2 initialization, manuscript editing or upload.

## Verified before launch

- `tests.log`: six contract tests passed.
- `preflight.log` and runtime `preflight_complete.json`: all 45 cells passed split, noise, geometry and target checks; old nine cells verified against original selection/artifact hashes.
- `smoke.log` and runtime `smoke_complete.json`: five two-epoch GCN/DIST trajectories, exact resumed model/optimizer/RNG/best-state checks, and exact initialization/two-epoch bridges to old GCN and DIST passed.
- Runtime `sources.json` freezes fitting code/config. Postprocessing sources are frozen separately, preserving the fitting manifest.
- Before launch, the read-only Python process inventory was empty. No other job was stopped.

## Launch and recovery

Runtime: `D:/codex_temp/gcn_support_d1_c1_3seed_20261027`.
Python: `D:/codex_temp/noisygl-rocm721/Scripts/python.exe`, RX7700XT.

At 2026-10-27 14:34:55 Beijing time, `run.py --stage fit` was launched hidden in the background. Venv launcher PID 61784; actual training worker PID 64372. At 14:35 it entered `cora/seed1/clean`, after importing nine historical teacher/student selections. These are launch identifiers; always recheck current status and processes before recovery.

Progress: runtime/source `status.json`, `run.lock`, `events.jsonl`, runtime `background_stdout.log` and `background_stderr.log`. Per-fit `epochs.jsonl` and `result.json` hold the latest actual epoch. The full teacher-choice freeze precedes all new DIST students; all student choices freeze before new test scoring. `fits_complete.json` means fitting is finished and test analysis remains deferred. Only `complete.json` plus a passed audit denotes final completion.

The independent CPU-only `analysis.py`, `audit.py` and their tests are delivered. Main-session final verification passed 25/25 tests (`tests_final.log`); the original six fitting-contract tests are included. An additional bounded old Cora clean check verified 61 prefix scores, 56 snapshots, all teacher targets, two JOINT CPU reloads and 410 input hashes. Runtime `postprocessing_sources.json` now freezes the three delivered postprocessing files separately from fitting code. Earlier postprocessing test failures remain in `all_tests.log` and `all_tests_verified.log`; fixes did not change fitting or scientific settings.

`launch.py` is the one-shot continuation driver. It waits for the existing fitting process to exit, invokes the idempotent fitting entry point (which skips if fitting is complete), then explicitly invokes `analysis.py` followed by `audit.py`. It waits for other Python owners rather than terminating them. Check `launcher_status.json` and `launcher.lock` before starting a driver; do not duplicate it.

Continuation receipt: at 2026-10-27 14:42:26 Beijing time, hidden `launch.py` started with venv PID 50480 and actual standard-library queue PID 60548. Its `WAITING_GPU` status correctly lists existing fit PIDs 61784/64372. In this receipt `training_started=false` belongs to the continuation driver; the separate GPU fitting worker is already active. Its wrapper logs are `launcher_stdout.log` and `launcher_stderr.log`; scientific fitting logs remain `background_stdout.log` and `background_stderr.log`. This is a single continuation process, not a recurring automation. Both locks are intentional while fitting is active.

If a job fails, inspect `failure_latest.json`, `failures.jsonl` and stderr before any repair. Confirm both processes and locks. Preserve prefix snapshots, selection decisions and failure records. A frozen-source mismatch is an audit failure, not permission to replace its manifest. Existing science, selection rules and test isolation remain fixed.

Static recovery caveats are documented in `research/gcn_support_d1_c1_3seed_20261027/STATIC_REVIEW.md`: a crash between prefix-result and snapshot writes may require an explicit verified snapshot repair; a mid-stage interruption can leave logs ahead of the last resume checkpoint. Both fail closed during audit/recovery. Preserve the failed attempt before repair; do not trim or replace scientific logs silently. Continuous execution currently has no such failure.

At 2026-10-27 14:47:43 Beijing time, 24/45 teacher-cell selections existed (nine historical, 15 newly completed); active fitting was CoraFull seed2 clean. `failure_latest.json` did not exist. The single fitting worker and waiting continuation driver remained active. This is an interim receipt, not final accuracy or completion.

## Intended outputs and interpretation

Three-seed mean/sample SD, raw seed rows, accuracy/macro-F1/per-class recall, paired gains against each control, student-minus-teacher gaps, actual supports/strengths and logical/physical cost accounting; CSV, LaTeX tables and Chinese report. Historical pilot test results are explicitly marked already observed. No new scientific gain is claimed while fitting is active. Old pilot gains were small; preserve negative cells and clean-label costs in the final report.
