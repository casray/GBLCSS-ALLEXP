# Analysis-only engineering repair

After all E1/E2/E4 training and validation choices were completed and locked, the first posthoc analysis failed at selector-row construction: `dict(**available[k], arm=label, ...)` supplied `arm` twice. No training failed and no scientific decision changed.

- Original analysis source is preserved as `analyze_fits.before_fix.txt`.
- Original SHA256: `8f2888bcfa28dcc35bd7810df59dc53e1a6ed7ccf95b0d29ac169f4e8d7798f5`.
- Failure traceback is retained in `analysis_raw.log`.
- Repair: use a dictionary merge to override the output label without modifying the candidate row.
- Add a two-test regression file covering row relabeling and macro-F1. The row test failed before implementation; the metric test passed.
- On retry, preserve the original `test_release.json` timestamp and require identical choice hashes. Do not create a fresh release pretending test values were never processed.
- `fit_sources.json` remains immutable and still records the analysis version present before training. Final verification must allow this documented analysis-only change, verify the original backup against the frozen hash, and hash the repaired analysis source separately.
- Re-run only analysis; no new optimization, thresholds, targets, selections, or model fits.
- A second analysis-only type error occurred at the final choice-hash assertion: the reused hash helper expects a Path, but JSON dictionary keys are strings. Convert that key to Path. The traceback is retained in `analysis_retry_raw.log`; output tables had already been computed, but completion was not claimed. This repair changes no metric or choice.
