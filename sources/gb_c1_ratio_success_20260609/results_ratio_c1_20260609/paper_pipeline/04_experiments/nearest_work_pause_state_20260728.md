# Recent-work baseline experiment execution state

## Status

- State: `COMPLETED_AFTER_EXPLICIT_RESUME`
- Completion date: 2026-07-29 (Asia/Shanghai)
- Scope: corrected GSDN smoke and the formal 500-row recent-work grid
- Resume authorization: the user explicitly released the GPU and requested
  continuation from the frozen recovery point.

The pause is closed. All frozen protocol, fail-fast, immutable-raw,
exact-resume, and claim-boundary requirements remained in force through
completion.

## Second pause checkpoint

- Pause requested at 2026-07-28 21:18 (Asia/Shanghai).
- Formal-grid manifest: 17/25 commands complete, 8 planned.
- Immutable completed evidence: 17 CSV files, 340 rows.
- Completed datasets/regimes:
  - Cora: all five regimes.
  - CiteSeer: all five regimes.
  - Coauthor-CS: all five regimes.
  - CoraFull: uniform 0.2 and uniform 0.4.
- Interrupted block: CoraFull uniform 0.6.
- The interrupted working CSV exists but is incomplete and has no immutable
  copy. It is not admissible evidence.
- All nearest-grid orchestrator and downstream Python processes were stopped;
  the related process count was verified as zero.
- On the next explicit resume, the runner must preserve the 17 exact-complete
  immutable blocks and rerun CoraFull uniform 0.6 from the beginning.

## Completed before the pause

- The tie-safe v2 four-method grid is complete and separately audited: 500 unique rows, 125 protocol groups, and zero failed rows.
- PosteL, ALS, and SALS implementations passed the targeted static review and CPU tests.
- A pre-pause ROCm functional smoke completed for `postel_mlp_noipl_rm`, `als_mlp_rm`, and `sals_mlp_adapted` on Cora, uniform 0.4, seed 0. These three values are functional diagnostics only and are not manuscript performance evidence.
- The manuscript’s broad novelty claim was withdrawn after verifying PosteL, GSDN, ALS, and SALS as directly relevant prior work.

## Closed static gates (no training executed)

1. **GSDN edge semantics**
   - PASS: paired directed arcs are canonicalized into deterministic unique undirected relations; self-loops are removed and one-way relations are retained.
   - PASS: Eq. (7) retains its two directional terms; Eq. (8) propagates from every labeled endpoint, including the higher-index endpoint and both directions when both endpoints are labeled.
   - PASS: independent paper-derived Eq. (7)/(8) oracles and raw-arc/unique-relation diagnostics are covered.
   - Identity is frozen as `equation-grounded common-protocol adaptation`, independent and not official code. The per-batch unique-relation normalizer is a project adaptation, so equation-faithful/exact wording is prohibited.
   - Verification: 31 CPU tests passed; `py_compile` passed.

2. **Formal-grid execution path**
   - PASS: the non-dry-run `NotImplementedError` was replaced with 25-command fail-fast orchestration, exact-complete resume, immutable raw capture, 375+125 evidence assembly, protocol audits, statistics and report generation.
   - PASS: 14 CPU mocked tests cover first run, exact resume, completeness, immutable input, failure paths and exact GSDN identity.
   - PASS: real-input CLI dry-run generated 25 planned commands with an explicit ROCm interpreter and zero subprocess calls.
   - Real subprocess training remains prohibited while paused.

## Resume gates

Before any experiment is started:

1. PASS: `py -3 -m unittest tests.test_gsdn_faithful tests.test_standard_runner_baselines` → 31 tests / OK after independent review.
2. PASS: `py -3 -m unittest paper_pipeline.tests.test_nearest_work_full_grid` → 14 tests / OK, including mocked first-run and exact-resume behavior.
3. PASS: the dry-run manifest contains exactly 25 dataset–noise commands, four recent-work methods per command, five seeds per method, and an explicit Python interpreter path.
4. PASS: input snapshots and the existing tie-safe v2 evidence are read-only inputs; the mocked exact-resume and byte-preservation checks pass.
5. PASS: the user released the GPU and explicitly resumed execution.

## Completed after the first resumption

- Re-ran `tests.test_gsdn_faithful` and
  `tests.test_standard_runner_baselines`: 31 tests passed.
- Re-ran `paper_pipeline.tests.test_nearest_work_full_grid`: 14 tests passed.
- Completed the corrected GSDN Cora/uniform-0.4/seed-0 ROCm smoke:
  `status=ok`, test accuracy `0.5398`.
- Smoke output:
  `external/NoisyGL/runs/nearest_work_resumed_20260728/gsdn_smoke_cora_u04_s0.csv`.

## Historical resume order

The following order governed the resumed run and is retained only as lineage:

1. Run a single Cora/uniform-0.4/seed-0 ROCm smoke for the corrected GSDN adaptation.
2. Audit output status, split/noise hashes, train-only contamination counts, graph-free inference diagnostics, parameter counts, and unique undirected relation diagnostics.
3. Freeze the formal-grid output directory and input snapshot manifest.
4. Start the 25-command recent-work grid with monitored subprocess execution.
5. Validate 500-row completeness and paired statistics before updating any manuscript result table or performance claim.

All five steps are complete.

## Historical claim boundary while paused

Before grid completion, performance claims were prohibited. That temporary
restriction is superseded by the audited formal results. The implementation
identity restriction remains active: the existing historical `GSDN-inspired`
results are a mechanism adaptation, and the new GSDN result is an
equation-grounded common-protocol adaptation rather than an official or exact
reproduction.

## Completion after the second resumption

- Preserved all 17 pre-pause immutable command blocks.
- Rebuilt the interrupted CoraFull uniform-0.6 block from the beginning; the
  incomplete working CSV was not admitted as evidence.
- Completed the remaining 8 command blocks, yielding 25/25 immutable CSVs and
  500/500 native recent-work rows.
- Verified all 25 immutable file SHA-256 values with zero mismatches.
- Verified 500/500 independent evidence rows and 125/125 protocol groups.
- Generated 625/625 five-method manuscript rows and 125 aggregate rows.
- Re-ran the orchestrator after completion; all 25 commands were marked
  `skipped_exact_complete` and no training subprocess was launched.
- Confirmed zero remaining Python training processes and zero nonempty stderr
  logs.

The final result and narrowed performance claims are recorded in
`nearest_work_full_grid_final_report_20260729.md`.
