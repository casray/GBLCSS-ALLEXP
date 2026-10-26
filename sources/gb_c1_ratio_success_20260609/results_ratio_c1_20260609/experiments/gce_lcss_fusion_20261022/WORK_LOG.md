# Work log

## 2026-10-22 setup and execution

- [x] Read project literature/history and previous representation/R2 configs, kernels, protocols and results.
- [x] User-approved bounded scope: offline target diagnostics plus GCE fusion, R0 primary/R2 secondary. No soft-membership training or adaptive generator yet.
- [x] Runtime verified: Python from D:/codex_temp/noisygl-rocm721, PyTorch2.9.1+rocm7.2.1, GPU RX7700XT.
- [x] Nine unit tests: observed expected RED before implementation; GREEN after implementation.
- [x] Preparation engineering rounding mismatch investigated and corrected before source freeze; failure retained in ENGINEERING.md/failure_prepare.json.
- [x] All36 input protocols independently checked with NumPy LOO. Historical U.4 fingerprints/targets checked; seed0 targets match previous study exactly.
- [x] Source/input hashes frozen. Historical GRACE reused, not three independent GRACE runs.
- [x] All14 short GPU smoke fits completed, no test selection, X-only exports agree exactly.
- [x] Started all504 formal fits, no arm skipped by oracle target statistics; raw execution and epoch logs retained.
- [x] Interim Cora seed0 CE/LCSS R0/R2 across four conditions:16 historical probability anchors match exactly. No test metrics used in this interim engineering check.
- [x] Finish all504 before unified test evaluation. Numerical recovery preserved97 superseded successes and1 failed soft-GCE run; all final GCE arms use stable exp(q*log_p). Six new label-free R2 encoders, three old seed0 encoders reused.
- [x] Reload and audit504 checkpoints,48 historical seed0 anchors, initial state/RNG,252 frozen BN/encoder checkpoints,504 epoch selections and168 summaries/336 paired rows/six gate components. Both audits passed;161 frozen sources and three recovery sources unchanged;12 tests passed.
- [x] Write complete Chinese conclusion, decisions and experiment handoff. Primary R0 W_LCSS−LCSS+.4878±.4247pp (8 positive cells/1 tie), below1pp gate; R2 full gate also fails. No automatic new training.

No old source, result, submitted paper, or remote repository has been modified by this experiment. Mean-weight formula is fixed, detached and unnormalized; W_MASS uses the same weighting rule with its own MASS target, not identical W_LCSS per-node weights.
