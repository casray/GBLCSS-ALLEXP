# Commands

From the results workspace, using the existing GPU environment:

```powershell
& 'D:/codex_temp/noisygl-rocm721/Scripts/python.exe' -m unittest discover -s experiments/representation_supervision_20261021 -p 'test_*.py' -v
& 'D:/codex_temp/noisygl-rocm721/Scripts/python.exe' experiments/representation_supervision_20261021/run_experiment.py prepare
& 'D:/codex_temp/noisygl-rocm721/Scripts/python.exe' experiments/representation_supervision_20261021/run_experiment.py smoke
& 'D:/codex_temp/noisygl-rocm721/Scripts/python.exe' experiments/representation_supervision_20261021/run_experiment.py tune
& 'D:/codex_temp/noisygl-rocm721/Scripts/python.exe' experiments/representation_supervision_20261021/run_experiment.py grid
& 'D:/codex_temp/noisygl-rocm721/Scripts/python.exe' experiments/representation_supervision_20261021/analyze.py
```

Inputs and source files are frozen after prepare. Completed runs are skipped without overwriting. Failed runs stop execution and preserve their logs. Raw per-epoch logs and the execution journal are local. No clean test labels are used until post-grid scoring.
