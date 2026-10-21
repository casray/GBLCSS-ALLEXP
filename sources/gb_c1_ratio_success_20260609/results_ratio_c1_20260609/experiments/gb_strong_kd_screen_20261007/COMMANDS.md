# Commands

Run from the results workspace. Uses the existing ROCm environment; no installs.

```powershell
& 'D:\codex_temp\noisygl-rocm721\Scripts\python.exe' -m unittest discover -s experiments/gb_strong_kd_screen_20261007 -p 'test_*.py' -v
& 'D:\codex_temp\noisygl-rocm721\Scripts\python.exe' experiments/gb_strong_kd_screen_20261007/run_challenge.py prepare
& 'D:\codex_temp\noisygl-rocm721\Scripts\python.exe' experiments/gb_strong_kd_screen_20261007/run_challenge.py replay
& 'D:\codex_temp\noisygl-rocm721\Scripts\python.exe' experiments/gb_strong_kd_screen_20261007/run_challenge.py grid
& 'D:\codex_temp\noisygl-rocm721\Scripts\python.exe' experiments/gb_strong_kd_screen_20261007/analyze.py
```

Official source downloads required a network sandbox exception. Only public
GitHub GET requests were made; no local data was sent. Initial tests failed as
intended before implementation. GPU/CPU official loss and gradient tests then
passed. No upstream training driver is executed. The loss test emits the
upstream deprecated size_average warning; no behavior change is made to it.
