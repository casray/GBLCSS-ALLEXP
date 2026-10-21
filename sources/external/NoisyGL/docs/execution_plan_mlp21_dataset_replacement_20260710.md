# MLP-21 Dataset Extension and PubMed Repositioning

## Goal

Keep PubMed as a train-only LCSS boundary dataset and a useful distillation dataset, then evaluate Amazon-Computers and CoraFull as candidate replacements under the same protocol.

## Fixed Protocol

- Split: percent `0.2/0.1/0.7`
- Label noise scope: train only
- Gate setting: uniform noise rate `0.4`
- Seeds: `0,1,2,3,4`
- Device: ROCm Python on `cuda:0`
- LCSS parameters: `max_balls=64`, `min_ball_size=32`, `split_improvement=0.05`, `target_beta=0.5`
- No dataset-specific tuning of balls, MLP, or GRACE hyperparameters
- Output directory: `runs/mlp21_dataset_replacement_20260710/`

## Phase 21a: Amazon-Computers Gate

Run 13 train-only methods on Amazon-Computers under uniform 0.4:

- `frozengrace_gb_mlp_softdist_beta05`
- `frozengrace_gb_mlp_softdist_beta05_shuffled`
- `mlp_ce`
- `mlp_small_loss`
- `mlp_coteaching`
- `mlp_ct`
- `mlp_jocor`
- `mlp_elr`
- `mlp_sce`
- `mlp_logitclip`
- `mlp_anl`
- `mlp_fine`
- `mlp_dividemix_lite`

Pass conditions:

- LCSS minus shuffled at least `+1.0pp` and at least `4/5` seeds positive.
- LCSS minus best non-LCSS baseline at least `+1.0pp` and at least `4/5` paired seeds positive.
- Val/test labels remain unmodified, and `modified_count == modified_train_count`.
- No NaN, OOM, or single-class collapse.

If only the first condition passes, mark as boundary: real local structure exists, but the strongest baseline absorbs the gain.

## Phase 21b: CoraFull Gate

Add `corafull` as PyG `CitationFull(name="cora")`, distinct from Planetoid Cora. Use fixed GRACE embeddings and run the same 13-method gate. CoraFull must pass the same criteria as Amazon-Computers. No beta, ball number, or min-ball-size adjustment is allowed for its 70 classes.

## Phase 21c: Expansion for Passed Datasets

Only datasets that pass the gate receive the full expansion:

- uniform `0.2`
- uniform `0.6`
- pair `0.4`
- instance `0.4`

Uniform `0.4` is reused from the gate. Each passed dataset adds `13 methods x 4 settings x 5 seeds = 260` runs.

Then run the graph-free Table B family under uniform `0.4`:

- `graphmlp`
- `gsdn`
- `mlp_distill_gcn`
- `ffg2m_adapted`
- `krd_adapted`
- `vqgraph_adapted`
- `mlp_distill_softdist_xh`

## Reporting

Generate:

- `mlp21_raw.csv`
- `mlp21_aggregate.csv`
- `mlp21_paired_deltas.csv`
- `mlp21_gate_decisions.csv`
- `mlp21_completeness.csv`
- `mlp21_report.md`

Report mean/std, paired deltas, leakage audit, ball diagnostics, and a clear pass/boundary/fail decision. Passed datasets can enter the main table. Boundary and failed datasets remain in the applicability analysis to avoid favorable-only reporting.

