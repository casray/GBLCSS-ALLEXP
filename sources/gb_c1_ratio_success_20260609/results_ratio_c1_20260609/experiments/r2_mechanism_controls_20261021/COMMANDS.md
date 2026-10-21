# Exact commands and purposes

Run from the workspace root with PowerShell. Runtime: D:/codex_temp/noisygl-rocm721/Scripts/python.exe.

1. Behavior tests (RED before kernels implemented; GREEN after implementation):
   `& 'D:/codex_temp/noisygl-rocm721/Scripts/python.exe' -m unittest discover -s experiments/r2_mechanism_controls_20261021 -p test_controls.py -v`
2. Match controls, pretrain shuffled encoders, freeze source/input/checkpoint hashes:
   `& 'D:/codex_temp/noisygl-rocm721/Scripts/python.exe' experiments/r2_mechanism_controls_20261021/run.py prepare`
3. Nine three-update GPU engineering fits, not formal results:
   `& 'D:/codex_temp/noisygl-rocm721/Scripts/python.exe' experiments/r2_mechanism_controls_20261021/run.py smoke`
4. 156 formal paired classifier fits, no held-out test scoring:
   `& 'D:/codex_temp/noisygl-rocm721/Scripts/python.exe' experiments/r2_mechanism_controls_20261021/run.py grid`
5. Post-grid scoring, gates and checkpoint/input verification:
   `& 'D:/codex_temp/noisygl-rocm721/Scripts/python.exe' experiments/r2_mechanism_controls_20261021/analyze.py`
6. Independent metric, geometry, loss-gradient and summary checks after scoring:
   `& 'D:/codex_temp/noisygl-rocm721/Scripts/python.exe' experiments/r2_mechanism_controls_20261021/audit_posthoc.py`

Do not rerun preparation after freezing. Successful runs resume by skipping completed fits; failed fits stop
and require documented review. No overwritten formal models, no manual test-based selection.
