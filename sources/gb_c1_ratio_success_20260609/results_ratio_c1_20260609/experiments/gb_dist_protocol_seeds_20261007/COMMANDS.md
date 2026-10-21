# Reproduction commands

Run in results_ratio_c1_20260609 with PowerShell. No installation needed.

```powershell
& 'D:/codex_temp/noisygl-rocm721/Scripts/python.exe' -m unittest discover -s experiments/gb_dist_protocol_seeds_20261007 -p 'test_*.py'
& 'D:/codex_temp/noisygl-rocm721/Scripts/python.exe' experiments/gb_dist_protocol_seeds_20261007/run_protocol.py prepare
& 'D:/codex_temp/noisygl-rocm721/Scripts/python.exe' experiments/gb_dist_protocol_seeds_20261007/run_protocol.py replay
& 'D:/codex_temp/noisygl-rocm721/Scripts/python.exe' experiments/gb_dist_protocol_seeds_20261007/run_protocol.py grid *> experiments/gb_dist_protocol_seeds_20261007/grid.log
& 'D:/codex_temp/noisygl-rocm721/Scripts/python.exe' experiments/gb_dist_protocol_seeds_20261007/analyze.py
```

Preparation is single-use. Completed grid fits are reused, never overwritten.
Analysis is executed only after all 20 cells complete. The analysis script was
added during training; the criteria and statistics were already frozen in PLAN.md.
The tensor-to-scalar warning in scalar loss logging does not change gradients;
the new student loop passed bit-exact prediction and full validation-trajectory
replay against the frozen seed-0 DIST runs, with and without the regional term.
