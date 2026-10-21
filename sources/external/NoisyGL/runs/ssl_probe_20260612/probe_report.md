# SSL Linear Probe Report

## Protocol

- Split: percent 0.2/0.1/0.7, seeds 0-2.
- Probe: one linear layer on frozen H, selected by clean validation accuracy.
- Clean probe trains on clean train labels; noisy probe trains on noisy train labels and evaluates on clean val/test.
- SSL embeddings are treated as label-free upstream artifacts; labels are used only inside this probe.
- Upstream references cloned under `C:\Users\Admin\Desktop\实验8去噪5.19-\gb_graph_aware_minival_6_1\upstream`: GRACE, BGRL (`60f9f19`), GraphMAE (`b14f080`), CCA-SSG (`ce5b2c0`).
- BGRL caveat: the official BGRL repository provides transductive configs for WikiCS/Amazon/Coauthor/PPI but not Cora/CiteSeer, so this run uses the official transductive Amazon-Photos augmentation/optimizer pattern without downstream tuning.

## Aggregate

| Dataset | Method | Probe label | Test acc mean | Test acc std | Delta vs GRACE | Pass >=1pp |
|---|---|---|---:|---:|---:|---:|
| citeseer | bgrl | clean | 0.6783 | 0.0097 | -0.0326 | 0 |
| citeseer | bgrl | noisy | 0.6383 | 0.0093 | -0.0631 | 0 |
| citeseer | ccassg | clean | 0.7265 | 0.0039 | 0.0156 | 1 |
| citeseer | ccassg | noisy | 0.7131 | 0.0076 | 0.0116 | 1 |
| citeseer | grace | clean | 0.7109 | 0.0047 | 0.0000 | 0 |
| citeseer | grace | noisy | 0.7015 | 0.0011 | 0.0000 | 0 |
| citeseer | grace_learnable_tau | clean | 0.7000 | 0.0080 | -0.0109 | 0 |
| citeseer | grace_learnable_tau | noisy | 0.6801 | 0.0096 | -0.0213 | 0 |
| citeseer | graphmae | clean | 0.5299 | 0.0364 | -0.1810 | 0 |
| citeseer | graphmae | noisy | 0.5020 | 0.0067 | -0.1995 | 0 |
| cora | bgrl | clean | 0.8007 | 0.0134 | -0.0287 | 0 |
| cora | bgrl | noisy | 0.7449 | 0.0063 | -0.0500 | 0 |
| cora | ccassg | clean | 0.8566 | 0.0061 | 0.0273 | 1 |
| cora | ccassg | noisy | 0.8237 | 0.0088 | 0.0288 | 1 |
| cora | grace | clean | 0.8294 | 0.0103 | 0.0000 | 0 |
| cora | grace | noisy | 0.7949 | 0.0241 | 0.0000 | 0 |
| cora | grace_learnable_tau | clean | 0.7836 | 0.0183 | -0.0457 | 0 |
| cora | grace_learnable_tau | noisy | 0.7710 | 0.0242 | -0.0239 | 0 |
| cora | graphmae | clean | 0.8396 | 0.0038 | 0.0102 | 1 |
| cora | graphmae | noisy | 0.8035 | 0.0167 | 0.0086 | 0 |

## Gate Decision

Phase 8 pass criterion: clean or noisy probe exceeds GRACE by at least 1pp on any dataset.

| Candidate | Decision | Evidence |
|---|---|---|
| CCA-SSG | Enter Phase 9 | Cora clean +2.73pp, Cora noisy +2.88pp, CiteSeer clean +1.56pp, CiteSeer noisy +1.16pp |
| GraphMAE | Enter Phase 9 seed0 smoke only | Cora clean +1.02pp; Cora noisy +0.86pp and CiteSeer strongly negative |
| GRACE-learnable-tau | Stop | Negative on both datasets and both probe labels |
| BGRL | Stop | Negative on both datasets and both probe labels; also lacks official Cora/CiteSeer config |

## Research Record

- Hypothesis: stronger label-free SSL H improves both classifier input and ball geometry.
- Granular-ball role: unchanged downstream consumer of H; no ball mechanism is modified in Phase 8.
- Code change: added NoisyGL `scripts/linear_probe_embeddings.py` and upstream `scripts/pretrain_ssl_candidates.py`.
- Config change: none to SoftDist [X||H] anchor; new H files are exported under upstream `checkpoints/embeddings_v2/{dataset}/{method}/H.pt`.
- Success standard: any candidate exceeds GRACE by >=1pp in clean or noisy probe on either dataset.
- Result: CCA-SSG is the clear Phase 9 candidate; GraphMAE is included in seed0 smoke because it barely passes Cora clean probe.
