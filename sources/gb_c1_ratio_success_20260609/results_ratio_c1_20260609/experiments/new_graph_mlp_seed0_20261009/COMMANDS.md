# Commands

Runtime: `D:/codex_temp/noisygl-rocm721/Scripts/python.exe`.
Working directory: this experiment folder.

```powershell
& 'D:/codex_temp/noisygl-rocm721/Scripts/python.exe' -m unittest test_protocol -v
& 'D:/codex_temp/noisygl-rocm721/Scripts/python.exe' -m py_compile common.py run_screen.py gssc_adapter.py tined_adapter.py
& 'D:/codex_temp/noisygl-rocm721/Scripts/python.exe' run_screen.py --method gssc --dataset cora --smoke
& 'D:/codex_temp/noisygl-rocm721/Scripts/python.exe' run_screen.py --method tined --dataset cora --smoke
& 'D:/codex_temp/noisygl-rocm721/Scripts/python.exe' run_screen.py --grid --methods gssc tined
& 'D:/codex_temp/noisygl-rocm721/Scripts/python.exe' run_screen.py --aggregate
& 'D:/codex_temp/noisygl-rocm721/Scripts/python.exe' report_screen.py
```

The grid launches a separate GPU process for each method/dataset sequentially and skips successful completed results. Smoke outputs are kept separately and cannot enter the result table. DGD training must not be launched until its missing core implementation is resolved with paper/author evidence. Completed rows and failure logs must not be deleted or replaced to improve rankings.

Five protocol subtests passed before training: original split/noise hashes match and nontraining labels in the adapters are all -1. No test-label access is allowed until selected probabilities are saved.

Final frozen-code quick checks: `python -m unittest test_protocol test_gssc test_tined -v` discovered 25 tests: 23 passed and 2 optional synthetic GPU tests skipped. Real GPU smoke runs are separate. `report_screen.py` only audits existing artifacts; it does not retrain models.
