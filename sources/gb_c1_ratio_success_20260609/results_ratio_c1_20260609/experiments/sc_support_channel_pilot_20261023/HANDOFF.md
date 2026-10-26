# Completed scheduled experiment: E1/E2/E4

Final state: completed on 2026-10-23. No pending fit or automatic seed expansion. Read `REPORT_zh.md`, `RESULT_TABLES.md`, the three `reports/*/REPORT_zh.md` and `final_audit.json` before any next action.

- E1 offline: 360 target cells, 60 frozen support choices, five graphs/three seeds. Classification pilot: 144 R0 fits, three graphs, seed0, four conditions. SC-VAL uniform validation -2.058519pp; gate failed. No E1 GCN expansion or E5.
- E2: 15 historical logs, three 600-step trajectories, 72 head fits. Cora/CS/Photo historical100 checkpoints exactly reproduced. All choose100 by frozen validation criteria. No seed1/2 extension.
- E4: 120 students, 15 reused noisy ordinary GCN teachers, five newly fitted clean teachers. Overall validation-selected candidate B; no dataset met +1pp noisy gain versus its validation-selected comparator. Gate failed; no seed1/2 extension.
- Test release and all selections are locked. Analysis completed: 336 formal classifier fits plus 48 selector aliases; three alignments/five clean teachers are additional formal fits, four smoke fits excluded from tables.
- Eleven unit/regression tests passed; final audit passed for 15 source records, 115 inputs, 336 result/selection/budget and model/probability hashes, 20 paired E4 cells. Three historical100 encoders exact. Single-seed classification std is null.
- Full200 Cora R0 CE predictions exactly match history. GPU smoke KD first-three losses exact; GCN loss differed by at most 1.1920928955078125e-7 with identical validation predictions.
- Two posthoc-only failures were repaired, with original source and logs retained: duplicate arm keyword; Path/string type in final hash check. See ANALYSIS_AMENDMENT.md. The initial frozen source manifest intentionally points to the original analysis script; final_audit verifies its backup and separately records repaired analysis SHA256. Do not rerun the old fit entry to bypass this amendment.
- E3/E5 were not started. No manuscript edits, old-run overwrites, uploads or new recurring automation.
- If continuing research, first assess teacher-wrong correction versus teacher-correct retention using E4_teacher_strata.csv. Cora D corrects more teacher errors but loses correct predictions; CiteSeer local gains coexist with clean/macro-F1 tradeoffs. This is a diagnostic hypothesis, not authorization for a new module or relaxed gate.

The following preparation/execution notes are retained as history. They do not describe a currently active job.

Prepared: 2026-10-23 Asia/Shanghai.
Historical state at the intermediate update: E1 completed and E2/E4 were running. All have now finished as stated above.

## Execution update

- E1a: 360 target/score cells and 60 frozen selections completed, historical split/noise matches confirmed. All 15 geometry families have six distinct cuts.
- E2 historical audit: 15 alignment logs inspected.
- GPU: AMD Radeon RX 7700 XT, torch 2.9.1+rocm7.2.1, about 12.7 GB free before launch.
- Nine synthetic unit tests passed. GPU smoke passed historical R0 initialization/RNG and first-three-loss checks, frozen encoder/BN, X-only deployment, KD and GCN fitting.
- Manual history bridge: KD first-three losses exact; GCN max loss difference 1.1920928955078125e-7, same validation predictions. Numerical target construction paths are not asserted bitwise identical.
- Active command: verified Python, run_fits.py --stage all; serial GPU, stdout/stderr in fits_raw.log. Check fit.lock, fit_status.json, completed stage manifests and process before resuming. Never start a second pipeline while it runs.
- Fit adapters: fit_pipeline.py, run_fits.py; posthoc scorer analyze_fits.py. Run posthoc scorer only after all three stage-complete manifests exist. Selections must remain frozen before test release.
- E3/E5 remain out of the automatic queue.
- E1 validation decision: SC minus VAL -2.058519 pp on uniform conditions. SC minus fixed32 by dataset: Cora -3.888889, CoraFull +0.050531, Photo -0.522876 pp. Clean SC minus CE: -1.481481, +1.616978, -1.307190 pp. Gate failed; do not tune or expand E1. Test scores not released yet.
- E2 Cora seed0: historical100 encoder exactly reproduced. Prefix300/600 mean validation changes -0.694444/-2.592593 pp; neither passed. CS and Photo follow.

The remaining text below records preparation history, not the current execution status.

- User authorization: design first, start approximately two hours later.
- Earliest start and scheduled wall time: 2026-10-23 02:09:00 +08:00.
- App heartbeat automation ID: gb-lcss; ACTIVE at creation, one occurrence (COUNT=1), attached to this conversation.
- Existing paused LaTeX automation was not changed.
- Preparation files: PLAN.md, config.json, RUNBOOK.md, kernels.py, test_kernels.py, run_diagnostics.py.
- Validation: 5 synthetic unit tests passed; Python syntax compilation passed; torch/numpy/scipy/matplotlib/torch_geometric/sklearn import specifications found. No GPU initialization or real-data fitting was performed during preparation.
- Tests cover own-label exclusion, label-independent tie scoring, exhaustive symmetric-noise expected-score identity, class-renaming invariance, and candidate-selection ties/invalid inputs.
- The initial test run failed as expected before kernels.py existed (TDD red phase). No scientific run failed or generated results.

Next: follow RUNBOOK.md. The prepared executable implements E1a and E2 historical-log checks. GPU fit adapters for E1b/E2 extensions/E4 still need to be implemented after diagnostics, in this new directory, using the frozen plan. Do not describe them as already implemented or trained.

No active subprocess or training session was left running. The scheduled app continuation must perform the launch. If the device/app is unavailable at the scheduled time, do not report an experiment as started; inspect actual execution state when resumed.

E3/E5 remain conditional later phases; they are not in the initial launch queue. The submitted paper and historical sources/results are unchanged.
