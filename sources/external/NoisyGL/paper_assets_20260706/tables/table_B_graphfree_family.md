# Table B. Graph-free family comparison (uniform 0.4)

| Method | Cora | CiteSeer | PubMed | Coauthor-CS | Amazon-Photo |
| --- | --- | --- | --- | --- | --- |
| Vanilla MLP | 44.24±2.62 | 43.50±3.02 | 63.77±4.40 | 88.42±0.62 | 73.90±1.80 |
| Graph-MLP (adapted) | 63.23±3.07 | 57.23±1.85 | 67.54±3.11 | 88.95±0.43 | 77.05±1.24 |
| GSDN (adapted) | 55.97±4.29 | 50.36±3.39 | 71.42±1.15 | 86.55±0.78 | 73.31±2.00 |
| LCSS train-only | 55.59±1.50 | 53.83±2.75 | 71.90±0.88 | 90.62±0.57 | 81.64±0.93 |
| GCN(X)+CE reference | NA | NA | NA | NA | NA |
| GCN[X\|\|H]+CE reference | 80.05±1.41 | 68.35±0.97 | 83.25±0.72 | 92.05±0.19 | 91.87±0.64 |
| GLNN-style GCN teacher | 77.95±2.62 | 66.09±1.38 | 84.32±0.47 | NA | NA |
| GLNN-style GCN[X\|\|H] teacher | 80.23±1.57 | 68.81±1.12 | 84.80±0.85 | NA | NA |
| LCSS-distill (T3 ball teacher) | 80.60±0.69 | 69.40±1.08 | 85.01±0.93 | NA | NA |
| kNN-consensus distill (T4) | 80.99±1.02 | 69.57±1.46 | 85.23±0.94 | NA | NA |
