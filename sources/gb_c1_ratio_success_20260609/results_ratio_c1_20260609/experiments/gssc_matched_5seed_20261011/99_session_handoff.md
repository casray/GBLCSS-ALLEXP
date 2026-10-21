# GSSC five-seed extension: paused by user

2026-10-11: User requested GPU release after the last Coauthor-CS seed. Coauthor-CS seed4 had completed and Amazon-Photo seed1 had started. Stopped only the verified grid/training processes (87180,82312,91196,89116); session69742 exited, and a process inventory confirmed no matching runner remained. No further GPU checks were run.

Completed:21/25 records, including five reused seed0 runs and16 new trainings. Cora, CiteSeer, CoraFull and Coauthor-CS have seeds0..4. Amazon-Photo has only seed0; seed1 partial artifacts are retained and excluded from statistics. Seeds2..4 have not started. This interruption is user-requested, not an algorithm failure.

CPU-only summary of completed result.json records (percentage points; sample std ddof1): Cora65.00+/-3.37; CiteSeer54.79+/-2.58; CoraFull40.39+/-1.31; Coauthor-CS84.37+/-1.87. Amazon-Photo seed0 alone82.35%, not a five-seed estimate. All21 stored audits report200 supervised epochs, first validation maximum and exact checkpoint replay. Final25-cell report is pending.

Do not automatically resume. The runner refuses partial-run overwrite: Amazon-Photo seed1 requires an explicit preserved-interruption/new-attempt strategy before restarting. Do not change the frozen runner or existing completed artifacts.

User authorized 2026-10-11: same adapted GSSC protocol, five main datasets, uniform0.4, seeds0..4. Only20 new GPU trainings; five previous seed0 records reused after exact checks. Original source/results/manuscript unchanged.

New runner imports frozen adapter from ../gssc_matched_seed0_20261009. Model/config/core/losses/budgets unchanged. New sources are frozen after GPU smoke passed; do not modify run_grid.py while grid runs.

Checks: three CPU tests passed; real Cora seed1 short GPU smoke passed with exact checkpoint replay. Independent reviewer found no blockers (see CODE_REVIEW.md). Initial missing adapter import path was corrected before creating training artifacts.

GPU grid command:
`D:/codex_temp/noisygl-rocm721/Scripts/python.exe -B -X utf8 -u experiments/gssc_matched_5seed_20261011/run_grid.py --grid`
Unified exec session69742 is closed. Completed records will be verified/skipped; partial/failed records refuse overwrite. Check console/epochs.jsonl/result.json by dataset/seed; do not infer completion from directory existence.

After grid exits0 run:
`D:/codex_temp/noisygl-rocm721/Scripts/python.exe -B -X utf8 experiments/gssc_matched_5seed_20261011/report.py`
Report requires25 cells and exactly5 seed0 reuses, audits all source/artifact/protocol identities, first validation max and exact deployment-probability replay, then writes per_seed_results.csv/summary.csv/table_row.tex/artifact_audit.json/REPORT.md. Sample std is ddof1 in percentage points. Retain all seeds and poor outcomes, do not tune.

Grid paused; only the four complete five-seed groups support mean/std reporting. No other methods rerun, no paper edits requested.
