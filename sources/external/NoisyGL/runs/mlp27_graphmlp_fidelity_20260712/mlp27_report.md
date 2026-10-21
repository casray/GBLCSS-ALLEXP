# MLP-27 Graph-MLP Fidelity Report

## Identity and protocol

- Official objective and network port from `yanghu819/Graph-MLP`, commit `5448add06b6f95a580fe984b5e6e199ab8aeff18`.
- Preserved components: the Graph-MLP MLP architecture, in-batch cosine-similarity matrix, and `NContrast` objective based on the r-hop adjacency matrix.
- Protocol adaptation: NoisyGL percent split 0.2/0.1/0.7, uniform 0.4 train-only label noise, and seeds 0--4.
- Cora and CiteSeer use the official dataset-specific `alpha` and `tau`; the other datasets use repository defaults without downstream tuning.
- Training uses graph structure; inference uses node features only.
- This row is reported as `Graph-MLP (protocol port)`, not as an official tuned reproduction.

## Results

| Dataset | Graph-MLP protocol port | Old edge-contrastive proxy | LCSS train-only | LCSS distillation |
|---|---:|---:|---:|---:|
| Cora | 64.91+-2.04 | 63.23+-3.07 | 55.59+-1.50 | 80.60+-0.69 |
| CiteSeer | 59.87+-1.73 | 57.23+-1.85 | 53.83+-2.75 | 69.40+-1.08 |
| CoraFull | 44.51+-0.55 | 37.05+-0.91 | 37.33+-0.81 | 58.22+-0.69 |
| Coauthor-CS | 89.97+-0.39 | 88.95+-0.43 | 90.62+-0.57 | 93.75+-0.22 |
| Amazon-Photo | 80.44+-0.65 | 77.05+-1.24 | 81.64+-0.93 | 90.49+-1.05 |

## Conclusion

All 25 rows completed successfully. The prior edge-contrastive proxy underestimated Graph-MLP by 1.02--7.46 percentage points because it did not preserve the paper's dedicated network and `NContrast` objective. The protocol port exceeds train-only LCSS on the three citation datasets, but is 0.65--1.20 points below it on Coauthor-CS and Amazon-Photo. LCSS distillation remains higher on all five datasets by 3.78--15.69 points. Therefore the old proxy row must be replaced by `Graph-MLP (protocol port)`, while the manuscript must continue to distinguish protocol ports from official tuned reproductions.
