# Execution log

2026-10-21: read project guidance, literature/history entry, previous frozen R2 report/sources/config/audit.
Verified ROCm GPU runtime (PyTorch2.9.1+rocm7.2.1; RX7700XT; sklearn1.8.0). No previous files changed.
Process inventory through Win32_Process was denied by environment; this was not a training error.

TDD RED: new kernels absent, required-interface assertion and numeric tests failed as expected.
Initial GREEN attempt: 9 tests passed, stable argsort rejected positional dim in installed PyTorch overload.
Fixed dim=1 keyword before preparing/training; no experimental hyperparameter or protocol change.
No test metrics used. Formal execution has not yet begun at this log entry.

Completion: all10 CPU tests and nine GPU three-update smoke fits passed. Prepared three fixed permutations,
three shuffled encoders and random states; frozen172 source/input/checkpoint files. KMeans physical-core
detection warning fell back to logical cores, with actual compute threadpool still explicitly limited to4.
No protocol changes, training failures or post-freeze source edits.

run.py grid completed156/156, exit0, formal grid wall56.474s. CE/GB24anchors reproduce previous prediction
tensors and best epochs exactly. Test scoring started only after complete marker.
analyze.py completed, exit0: independent48target checks,156 deployment reloads, BN/encoder frozen,
same classifier initial states and first strict validation maximum, source hashes all passed.
audit_posthoc.py completed, exit0: independent156accuracy,144paired deltas and table means;
96 cosine neighbor boundary queries, shuffled checkpoint selection and BOOT gradient identity passed.
Main result: GB57.9214< GCE59.7102/KNN59.1339/KM59.2768/MASS58.0492; fullGBgatefalse.
Detailed report and handoff written. No additional branches launched; submitted manuscript unchanged.
