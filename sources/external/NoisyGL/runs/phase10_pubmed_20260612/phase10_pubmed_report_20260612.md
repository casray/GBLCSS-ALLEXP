# Phase 10 PubMed Dataset Expansion Report

Date: 2026-06-12

## Protocol

- Dataset: PubMed.
- Split: `percent`, train/val/test = 0.2/0.1/0.7.
- Noise: `uniform`, noise_rate = 0.4.
- Seeds: 0, 1, 2, 3, 4.
- Device: ROCm GPU via `.venv-rocm`.
- SSL H: fixed-tau GRACE, official PubMed config from `upstream/GRACE/config.yaml`.
- Downstream H use: `gb.embedding_path` points to `checkpoints/embeddings_v2/PubMed/grace/H.pt`.
- Mechanism: unchanged SoftDist [X||H], no ball/alfa/target formula changes.

## GRACE Pretraining

| Dataset | Method | Seed | H shape | Epochs | Final loss | Source |
|---|---|---:|---:|---:|---:|---|
| PubMed | GRACE fixed tau | 0 | 19717 x 256 | 1500 | 9.3335 | official GRACE PubMed config |

Notes:

- Training is label-free; labels and masks are not used in SSL pretraining.
- The pretraining script reads PubMed data from the local NoisyGL `data` directory to avoid network download.

## Downstream Results

| Method | Mean test acc | Std | Seeds |
|---|---:|---:|---|
| GCN+CE | 0.8392 | 0.0057 | 0.8309; 0.8391; 0.8378; 0.8460; 0.8425 |
| SoftDist [X||H] | 0.8320 | 0.0078 | 0.8189; 0.8366; 0.8344; 0.8386; 0.8317 |
| SoftDist [X||H] shuffled | 0.8303 | 0.0057 | 0.8252; 0.8239; 0.8310; 0.8369; 0.8348 |
| GCN+CE [X||H] | 0.8296 | 0.0052 | 0.8208; 0.8299; 0.8314; 0.8341; 0.8318 |
| NRGNN | 0.8080 | 0.0088 | 0.7986; 0.8091; 0.7994; 0.8171; 0.8158 |

Paired deltas:

- SoftDist [X||H] - GCN+CE: -0.72pp mean.
- SoftDist [X||H] - GCN+CE [X||H]: +0.24pp mean.
- SoftDist [X||H] - shuffled: +0.17pp mean.
- SoftDist [X||H] - NRGNN: +2.40pp mean.

## Ball Diagnostics

Diagnostics use the same five seeds and noisy train labels. Clean labels are used only for diagnosis.

| Metric | Mean | Std |
|---|---:|---:|
| num_balls | 64.0000 | 0.0000 |
| mean_ball_size | 308.0781 | 0.0000 |
| true_all_purity_weighted | 0.8074 | 0.0003 |
| true_train_purity_weighted | 0.8110 | 0.0054 |
| soft_pseudo_changed_train_count | 1871.8000 | 38.8034 |
| soft_pseudo_ncr | 0.7903 | 0.0123 |
| soft_pseudo_cdr | 0.1942 | 0.0109 |
| mean_alpha_train | 0.2873 | 0.0149 |

Interpretation:

- Ball geometry is not degenerate on PubMed: all seeds produce 64 balls with high weighted purity (~0.807).
- SoftDist has strong NCR (~0.790), but CDR is still non-trivial (~0.194).
- The direct GCN+CE baseline is already very strong on PubMed under this percent split, so the remaining CDR plus target smoothing cost outweighs the ball correction gain.
- The real-vs-shuffled delta is positive but small (+0.17pp), so PubMed supports only a weak ball-structure contribution.

## Decision

PubMed does not pass the Phase 10 minimum generalization gate:

- It beats NRGNN, but it does not beat plain GCN+CE.
- Real balls beat shuffled only marginally.

Therefore PubMed should be reported as a limitation/regime boundary, not as evidence of broad dataset-level success. The likely explanation is not ball collapse; it is that PubMed's percent-split GCN+CE baseline is already strong, and soft label correction introduces enough clean-label damage to lose against direct CE.

## Amazon-Photo Status

Amazon-Photo was not run in this batch. Reason: Phase 9 selected GRACE as the current main SSL H, but the official GRACE config in this workspace contains Cora/CiteSeer/PubMed/DBLP only and has no Amazon-Photo default. Running Amazon-Photo would require an unregistered fallback GRACE configuration or switching to another SSL method, both of which would weaken the "no downstream hyperparameter tuning" protocol.

Recommended next action: add a pre-registered Amazon-Photo GRACE fallback rule before running it, for example "reuse PubMed GRACE config unchanged" or "use CCA-SSG official Amazon-Photo config as a separate SSL-family ablation." This should be recorded before training, not chosen after seeing results.

## Research Record

- Hypothesis: the GRACE + SoftDist [X||H] pipeline generalizes beyond Cora/CiteSeer to PubMed.
- Granular-ball role: frozen GRACE H defines local semantic balls; balls produce train-node soft targets.
- Code changes: added fixed-tau `grace` export to upstream `scripts/pretrain_ssl_candidates.py`, added PubMed XH configs, added CSV runners and ball diagnostics script.
- Config changes: PubMed uses official GRACE PubMed pretraining config; NoisyGL split is explicitly overridden to percent 0.2/0.1/0.7.
- Success standard: SoftDist [X||H] >= GCN+CE and real > shuffled.
- Result: SoftDist [X||H] = 0.8320 ± 0.0078, GCN+CE = 0.8392 ± 0.0057, shuffled = 0.8303 ± 0.0057.
- Final decision: PubMed is a negative/limitation result for broad generalization under this protocol.
