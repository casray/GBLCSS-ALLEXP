# Commands and provenance

Run from the existing results workspace in PowerShell. No packages were installed.

```powershell
& 'D:\codex_temp\noisygl-rocm721\Scripts\python.exe' -m unittest discover -s 'experiments/gb_guided_distillation_seed0_20261006' -p 'test_guided_targets.py'
& 'D:\codex_temp\noisygl-rocm721\Scripts\python.exe' -u 'experiments/gb_guided_distillation_seed0_20261006/run_guided.py' --prepare
& 'D:\codex_temp\noisygl-rocm721\Scripts\python.exe' -u 'experiments/gb_guided_distillation_seed0_20261006/run_guided.py' --replay
& 'D:\codex_temp\noisygl-rocm721\Scripts\python.exe' -u 'experiments/gb_guided_distillation_seed0_20261006/run_guided.py' --grid
& 'D:\codex_temp\noisygl-rocm721\Scripts\python.exe' 'experiments/gb_guided_distillation_seed0_20261006/analyze.py'
```

`--prepare` is intentionally non-overwriting. Do not rerun it in this frozen experiment.
`--grid` resumes only missing arms and checks input hashes before each fit.
Every D1-D4 run has its console log, 200 epoch records, config, probabilities and checkpoint.
D0 references the unchanged prior experiment; the one new D0 replay is in `replay/`.

Target construction and all student fits use the existing AMD GPU. Probability auditing
and clean-label diagnostics run on the CPU after target freezing.
`seconds` for new fits measures loading, frozen-input checking and student fitting together;
the reused D0 `seconds` is the previous student-only measurement. These fields must not
be interpreted as a directly comparable training-time benchmark. Peak allocated memory
for new fits includes loaded targets and student training, not teacher training.

The first test run failed because the implementation module had not been created.
The implementation was then added; all four target tests passed before preparation.
Source and input SHA-256 fingerprints are recorded in `frozen_inputs.json`.
