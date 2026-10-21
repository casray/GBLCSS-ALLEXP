# GSSC matched-backbone single-seed implementation plan

> Execute inline with executing-plans and focused test-driven checks.

Goal: screen GSSC on five main datasets, uniform 0.4, seed 0; no other methods rerun.
Architecture: reuse frozen sparse GSSC core and audited main-protocol loader. Use exact common two-FC MLP (hidden64, ReLU, BN, dropout0.5); retain native auxiliary head, mixup, sampler and all losses.
Approved budget: 100 label-free edge-pretraining, 100 classifier-warmup, 100 main epochs. First clean-validation maximum over all 200 supervised epochs (warmup eligible). Adam classifier lr0.01 wd0.0005; native mixup lr0.0001. Equal epoch count does not imply equal optimizer steps or compute.
Hypothesis: preserved GSSC mechanisms can be evaluated with the main deployed backbone and supervised-epoch budget. Granular-ball role: none (external baseline). Screening does not establish multi-seed stability.
Presets: Cora/CiteSeer/Coauthor-CS native; CoraFull/Amazon-Photo retain predeclared Cora fallback. No test-selected settings.

## Tasks

- [x] Add test_adapter.py; unittest failed for missing adapter before implementation.
- [x] Add adapter.py: common deployed forward/native auxiliary head; structurally reuse frozen run, replacing only first-tie comparison and two deployment-parameter counts. Assert source hash and transformation counts; no historical edits or global mutation.
- [x] Four adaptation tests and thirteen existing GSSC core tests passed.
- [x] Add run.py: audited protocol, label isolation, seed reset, source freeze, isolated artifacts and no partial overwrite.
- [x] Real Cora GPU smoke passed; first maximum and exact checkpoint replay verified.
- [x] Five formal seed0 runs completed serially on GPU, all exit0; no other methods rerun.
- [x] report.py completed; all five source/protocol/artifact identities, first validation maxima and exact deployed-checkpoint probabilities passed. Results in results.csv, artifact_audit.json and REPORT.md. No mean/std/significance claims.

Success: finite probabilities, historical split/noise equality,200 supervised epochs, exact checkpoint replay; no test labels in training.
Failure: leakage, nonfinite loss, OOM, altered core or checkpoint mismatch; never silently reduce data or tune hyperparameters.
Decision: report screening; five-seed extension needs user authorization.
