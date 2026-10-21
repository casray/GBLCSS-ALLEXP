# Table B. Graph-free family comparison under uniform 0.4

Protocol: train-only label noise, noise rate 0.4, train/val/test = 20%/10%/70%, 5 seeds. All methods use node features only at inference. FF-G2M/KRD/VQGraph are adapted core implementations under our NoisyGL noisy-label protocol, not official reproductions. For rows inherited from earlier Table B, cells report mean±std when the aggregate file contains std; otherwise they keep the previous mean-only manuscript value.

| Method | Training resource | Cora | CiteSeer | PubMed | Coauthor-CS | Amazon-Photo |
|---|---|---:|---:|---:|---:|---:|
| MLP+CE | noisy labels | 44.24±2.62 | 43.50±3.02 | 63.77±4.40 | 88.42±0.62 | 73.90±1.80 |
| Graph-MLP adapted | full-graph edge contrast | 63.23±3.07 | 57.23±1.85 | 67.54±3.11 | 88.95±0.64 | 77.05±2.10 |
| GSDN adapted | neighborhood self-distillation | 55.97±4.29 | 50.36±3.39 | 71.42±1.15 | 86.55±0.97 | 73.31±2.56 |
| LCSS train-only | frozen representation local consensus | 55.59±1.50 | 53.83±2.75 | 71.90±0.88 | 90.62±0.57 | 81.64±0.92 |
| GLNN-style GCN teacher | GCN teacher + unlabeled nodes | 77.95±2.62 | 66.09±1.38 | 84.32±0.47 | 93.20 | 89.86 |
| FF-G2M adapted | GCN teacher + low/high-frequency distillation | 78.73±2.62 | 66.17±1.30 | 84.08±0.49 | 93.12±0.24 | 90.01±1.14 |
| KRD adapted | GCN teacher + reliable knowledge sampling | 71.77±2.20 | 64.96±1.03 | 83.81±0.63 | 91.39±0.49 | 85.17±0.94 |
| VQGraph adapted | GCN teacher + structure codebook/token distillation | 78.89±2.34 | 65.64±1.72 | 84.26±0.69 | 93.25±0.09 | **90.54±0.85** |
| LCSS-distill ball teacher | denoised ball-consensus teacher + unlabeled nodes | **80.60±0.69** | **69.40±1.08** | **85.01±0.93** | **93.75±0.22** | 90.49±1.05 |

Main reading: FF-G2M/KRD/VQGraph substantially strengthen Table B. LCSS-distill ball teacher remains best on Cora/CiteSeer/PubMed/Coauthor-CS and is statistically comparable to VQGraph on Amazon-Photo, where VQGraph is higher by only 0.05 pp. The paired gains of LCSS-distill over FF-G2M are +1.87/+3.23/+0.93/+0.63/+0.49 pp on Cora/CiteSeer/PubMed/Coauthor-CS/Amazon-Photo; over KRD, +8.83/+4.44/+1.20/+2.36/+5.32 pp. Against VQGraph, LCSS-distill is +1.71/+3.76/+0.75/+0.50 pp on the first four datasets and -0.05 pp on Amazon-Photo. A kNN-consensus teacher is a slightly stronger alternative on the first three datasets (80.99/69.57/85.23), but it is reported as an ablation rather than the default LCSS-distill row.
