# Engineering and audit notes

- Before freezing, the first preflight used an accuracy tolerance of `1e-8`; GPU-recorded and CPU-recomputed accuracy differed by about `6e-8`. The preflight tolerance was corrected to `1e-6` before `frozen_manifest.json` was created. No student fit had started.
- After all 135 student fits completed, the first checkpoint audit found the same scale of CPU/GPU reduction difference in one test-accuracy recomputation. Only the offline analyzer was amended to use `1e-6` for accuracy and macro-F1 comparisons. No training code, configuration, teacher output, student checkpoint, or saved prediction was changed or rerun.
- The original pre-fit analyzer SHA-256 is preserved in `analysis_amendment.json`; the frozen manifest's analyzer hash is updated to the amended audit code so later audits can run. All other frozen hashes remain unchanged.
