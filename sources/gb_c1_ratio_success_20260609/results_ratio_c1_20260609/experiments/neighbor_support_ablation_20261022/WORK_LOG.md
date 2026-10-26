# Execution log

2026-10-22: User approved support/neighbor-identity ablation and resumed after
an interruption during code preparation. No training had started at interruption.

- Read previous NP PLAN, generator, runner, analysis, configuration and results.
- Read project literature map, experiment history and latest handoff.
- Confirmed Python D:/codex_temp/noisygl-rocm721/Scripts/python.exe,
  torch2.9.1+rocm7.2.1, AMD Radeon RX7700XT, GPU available.
- Added isolated frozen-plan/config/runner. Kernels/tests and independent
  analysis are delegated to separate agents with disjoint file scopes.
- Existing dirty project files are unrelated and are retained unchanged.
- Kernels:18 CPU tests passed, including exhaustive tiny-tree DP and cost identity.
- Preparation completed9 dataset/seed geometries and288 independent target checks;
  maximum target difference1.03314717669889e-7. No new fits at this stage.
- Final freeze is deferred until independent analyze.py is present; config and
  all prepared inputs must remain exact. Review fixes and first metadata-check
  stop are documented in ENGINEERING.md.
- Final source freeze completed:1333 source/input hashes, analysis source included.
- GPU smoke24/24 passed;12 historical CE/OLD64 first-three-epoch loss and
  validation bridges passed. Manifest binding is in smoke_audit.json.
- Full576-cell grid started on RX7700XT;216 historical anchors and360 new fits.
- Independent read-only review of runner and analyzer found no remaining
  protocol/statistical implementation issue after recorded pre-fit fixes.
  Full checkpoint/target/reuse audit remains mandatory after the grid.
- Formal grid completed576/576,216 reused360new,586.7576464000158seconds.
  No training failures or fitting reruns.
- Full audit v1 stopped after384 cells: float32 normalization tolerance too tight
  on historical CoraFull CE as well as new models. See ENGINEERING.md.
- Recovery v2 preserved frozen analyzer and introduced hash-bound audit source;
 576 checkpoints/288 targets/9 geometries/1333 frozen files all passed.
- Independently recomputed32 paired macro mean/std rows from raw results.csv;
  all differences below1e-10. Detailed conclusion and final decisions written.
- Final verification after documentation updates:1333 frozen source/input hashes
  and9 result-artifact hashes still match. AST comparison confirms13 analysis
  functions unchanged and audit_model changed only its normalization check.
  Final audit passed; no active fitting/audit session remains.
- Updated workspace experiment history/handoff and project core context with
  positive repair evidence, failed increments, clean risks and no-auto-expansion.

## Commands

All commands use the workspace root as the working directory.

```powershell
& 'D:/codex_temp/noisygl-rocm721/Scripts/python.exe' -m unittest discover -s experiments/neighbor_support_ablation_20261022 -p test_kernels.py -v
& 'D:/codex_temp/noisygl-rocm721/Scripts/python.exe' -u experiments/neighbor_support_ablation_20261022/run.py prepare
& 'D:/codex_temp/noisygl-rocm721/Scripts/python.exe' -u experiments/neighbor_support_ablation_20261022/run.py freeze
& 'D:/codex_temp/noisygl-rocm721/Scripts/python.exe' -u experiments/neighbor_support_ablation_20261022/run.py smoke
& 'D:/codex_temp/noisygl-rocm721/Scripts/python.exe' -u experiments/neighbor_support_ablation_20261022/analyze.py smoke
& 'D:/codex_temp/noisygl-rocm721/Scripts/python.exe' -u experiments/neighbor_support_ablation_20261022/run.py run
& 'D:/codex_temp/noisygl-rocm721/Scripts/python.exe' -u experiments/neighbor_support_ablation_20261022/analysis_recovery_v2.py full
```

JSON progress, per-fit epoch logs and raw command output are retained in this
folder. Do not rerun preparation after frozen_inputs.json exists. The run phase
resumes completed cells and rejects unfinished/failed cells for explicit review.
