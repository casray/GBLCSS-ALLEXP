# Reviewer Repository Implementation Plan

**Goal:** A concise GitHub repository that runs the experiments behind the main paper and supplementary tables, then summarizes newly computed results. Exact bitwise replay is not required.

**Architecture:** Retain the existing tested `src/gblcss` algorithms and protocol adapters. Add one experiment-suite entry point and one results summarizer. Keep frozen representations, configurations, licenses and focused tests; omit manuscript graphics, archived tables and internal reports from the delivered repository. New run outputs are never populated from paper result CSVs.

**Tech stack:** Python, PyTorch, PyG, NumPy, pandas, SciPy, scikit-learn, YAML.

## Tasks
- [x] Extract the source ZIP to a new `GB-LCSS_code` directory, leaving the ZIP unchanged. Run the existing tests as a baseline.
- [x] Map the final manuscript tables to experiment suites. Preserve data splits, noise, seed pairing, model selection and method settings.
- [x] Complete the Graph-MLP adapter from the original experiment implementation and test it in isolation.
- [x] Add suites covering matched baselines, graph-assisted methods, target controls, grouping, representations, distillation, validation noise, sensitivity, extra datasets and target diagnostics.
- [x] Generate accuracy, paired comparisons, target-error decomposition and dataset-statistics tables from fresh outputs. Test aggregation with synthetic records.
- [x] Exclude superseded scripts and manuscript assets from delivery, write concise README commands and method-to-code guide, retain licenses.
- [x] Run unit tests, suite-plan validation and real-data smoke runs; package only source, config, tests and required representations. Delivery: `GB-LCSS_GitHub/` and `GB-LCSS_GitHub.zip`, 70 files, 112.5 MiB. Archive integrity and Python syntax checked; delivered entry point lists all 16 suites.

## Verification
Use `D:/codex_temp/noisygl-rocm721/Scripts/python.exe -m pytest -q` in the new repository. Test table generation independently from training. Run Cora with all runnable method families, then full-length GB-LCSS and MLP+CE paired runs. Store internal development logs outside the deliverable. Report verification accurately to the user, with no unexecuted full-grid claims.

User subsequently limited further testing to smoke experiments. Existing Cora GPU checks had already finished: GB-LCSS 57.4881 +/- 0.7598 percent and MLP+CE 44.2427 +/- 2.6201 percent. Do not start another full training run. The distillation/target/statistics/cost Cora seed-0 two-epoch smoke completed 21 records. Unit tests for reporting and CLI passed; the Graph-MLP repeatability test uses float32 tolerance (1e-6 absolute), not bitwise equality.

Graph-MLP provenance: the final table's 63.23/57.23/37.05/88.95/77.05 means match the old shared-MLP edge-contrastive column of `external/NoisyGL/runs/mlp27_graphmlp_fidelity_20260712/mlp27_report.md`, not the later dedicated-network protocol port. The delivered implementation therefore follows `GraphFreeFamily_Predictor.py::graphmlp_Predictor` and describes that adaptation accurately. No manuscript edits made.
