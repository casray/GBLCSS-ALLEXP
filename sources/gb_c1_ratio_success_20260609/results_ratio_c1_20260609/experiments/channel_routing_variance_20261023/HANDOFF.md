# Completed: channel variance and conditional routing

2026-10-23 Asia/Shanghai. Forty formal GPU fits and two three-epoch smoke fits completed. No active process or pending automatic fit. Read REPORT_zh.md and RESULT_TABLES.md.

- Protocol seed0 fixed; new student seed1 only. A300 epochs300, C_head epochs200, same targets/teacher/R2 encoder. C_head changes only head initialization.
- Stage0 PASSED: A0C0 union minus per-cell max same-channel unions, noisy means: Cora4.080915/CiteSeer2.720504/CoraFull2.381812/CS1.109120/Photo1.929661 pp. Three primary graphs pass2pp. Four cross-pair means also positive at similar size.
- L1 FAILED: validation-selected global mixture minus validation-best single, noisy means: .123457/.903614/.134748/.272777/.740741 pp. Only one of the three primary graphs passes.5pp, need2. Clean gate met.
- Stop at L1 under frozen PLAN. L2 logistic routers, L3 graph, L4 GB, L5 substitutions not run. No fused-student training, E3/E5, manuscript edits or uploads.
- Test GLOBAL vs best single: +.193492/+1.789805/+.197282/+.267544/+.441954 pp. CiteSeer direct mean65.177549 vs C0 63.187285. Teacher-confidence threshold gains tiny.
- Stage0 is explicitly exploratory test-gated screening; L1 choices use validation. One protocol, two student initializations; report actual scope. Existing val checkpoint reuse is recorded.
- 9 tests; 2 exact smoke bridges; 40 pairing audits; 80 source/new fit-record audits; 40 independent checkpoint replays; 20 fusion replays; 11 train sources/405 inputs/5 L1 sources hashed, final_audit passed. No formal runtime failures.
- Source frozen files remain unchanged. analysis and reporting were added in separate files after training froze. Rerun entry scripts honor completion markers. Do not rerun to alter gates.
- Next research question remains direct observable routing; present state tested simple global mixing and confidence thresholds. Any alternative continuation needs a new plan preserving this stop.

Commands used: verified Python run_stage0.py, analyze_stage0.py, run_routing_l1.py, audit_and_report.py, unittest discover. Raw logs and all result tables live here. PLAN/config are the original frozen specification; this file gives final state.
