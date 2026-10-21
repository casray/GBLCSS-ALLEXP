# Executed commands

PowerShell from workspace root; runtime D:/codex_temp/noisygl-rocm721/Scripts/python.exe.

1. RED then GREEN kernel tests:
   `& 'D:/codex_temp/noisygl-rocm721/Scripts/python.exe' -m unittest discover -s experiments/r2_boundary_diagnostic_20261021 -p test_diagnostic.py -v`
2. Initial analysis, completed12boundary protocols then stopped on missing historical optional arm:
   `& 'D:/codex_temp/noisygl-rocm721/Scripts/python.exe' experiments/r2_boundary_diagnostic_20261021/diagnose.py`
3. Recovery regression tests (RED before checker, then all11 GREEN):
   `& 'D:/codex_temp/noisygl-rocm721/Scripts/python.exe' -m unittest discover -s experiments/r2_boundary_diagnostic_20261021 -p 'test_*.py' -v`
4. Recovery-only final review, no target or model rerun:
   `& 'D:/codex_temp/noisygl-rocm721/Scripts/python.exe' experiments/r2_boundary_diagnostic_20261021/finish_review.py`

Do not rerun initial analysis or modify frozen source/input files. No new training commands were executed.
