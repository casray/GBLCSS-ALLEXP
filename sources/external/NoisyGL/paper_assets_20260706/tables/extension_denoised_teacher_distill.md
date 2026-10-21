# Extension table. Denoised-teacher distillation

| Dataset | Noise | Arm | Student | Teacher | n |
| --- | --- | --- | --- | --- | --- |
| CiteSeer | uniform0.4 | T1 GLNN teacher=GCN(X)+CE | 66.09±1.38 | 64.90±1.11 | 5 |
| CiteSeer | uniform0.4 | T2 teacher=GCN[X\|\|H]+CE | 68.81±1.12 | 68.35±0.97 | 5 |
| CiteSeer | uniform0.4 | T4 teacher=kNN-GCN[X\|\|H] | 69.57±1.46 | 69.04±1.04 | 5 |
| CiteSeer | uniform0.4 | T3 teacher=SoftDist-GCN[X\|\|H] | 69.40±1.08 | 69.18±0.76 | 5 |
| CiteSeer | uniform0.4 | T5 teacher=shuffled-SoftDist-GCN[X\|\|H] | 68.60±1.14 | 68.10±0.91 | 5 |
| CiteSeer | uniform0.6 | T1 GLNN teacher=GCN(X)+CE | 53.30±4.42 | 52.29±4.00 | 5 |
| CiteSeer | uniform0.6 | T2 teacher=GCN[X\|\|H]+CE | 61.04±3.08 | 60.34±2.42 | 5 |
| CiteSeer | uniform0.6 | T4 teacher=kNN-GCN[X\|\|H] | 61.96±2.98 | 61.15±3.24 | 5 |
| CiteSeer | uniform0.6 | T3 teacher=SoftDist-GCN[X\|\|H] | 62.28±2.30 | 61.60±1.97 | 5 |
| CiteSeer | uniform0.6 | T5 teacher=shuffled-SoftDist-GCN[X\|\|H] | 61.07±2.76 | 60.08±2.47 | 5 |
| Cora | uniform0.4 | T1 GLNN teacher=GCN(X)+CE | 77.95±2.62 | 78.77±1.87 | 5 |
| Cora | uniform0.4 | T2 teacher=GCN[X\|\|H]+CE | 80.23±1.57 | 80.05±1.41 | 5 |
| Cora | uniform0.4 | T4 teacher=kNN-GCN[X\|\|H] | 80.99±1.02 | 81.45±1.23 | 5 |
| Cora | uniform0.4 | T3 teacher=SoftDist-GCN[X\|\|H] | 80.60±0.69 | 80.84±1.07 | 5 |
| Cora | uniform0.4 | T5 teacher=shuffled-SoftDist-GCN[X\|\|H] | 78.81±1.63 | 79.62±1.65 | 5 |
| Cora | uniform0.6 | T1 GLNN teacher=GCN(X)+CE | 65.50±3.28 | 65.44±2.59 | 5 |
| Cora | uniform0.6 | T2 teacher=GCN[X\|\|H]+CE | 69.44±2.18 | 69.33±1.64 | 5 |
| Cora | uniform0.6 | T4 teacher=kNN-GCN[X\|\|H] | 71.07±2.29 | 71.87±2.16 | 5 |
| Cora | uniform0.6 | T3 teacher=SoftDist-GCN[X\|\|H] | 71.27±2.46 | 71.35±2.20 | 5 |
| Cora | uniform0.6 | T5 teacher=shuffled-SoftDist-GCN[X\|\|H] | 68.51±1.82 | 68.49±1.80 | 5 |
| PubMed | uniform0.4 | T1 GLNN teacher=GCN(X)+CE | 84.32±0.47 | 83.75±0.57 | 5 |
| PubMed | uniform0.4 | T2 teacher=GCN[X\|\|H]+CE | 84.80±0.85 | 83.25±0.72 | 5 |
| PubMed | uniform0.4 | T4 teacher=kNN-GCN[X\|\|H] | 85.23±0.94 | 83.53±0.83 | 5 |
| PubMed | uniform0.4 | T3 teacher=SoftDist-GCN[X\|\|H] | 85.01±0.93 | 83.27±0.88 | 5 |
| PubMed | uniform0.4 | T5 teacher=shuffled-SoftDist-GCN[X\|\|H] | 85.08±0.93 | 83.36±0.76 | 5 |
| PubMed | uniform0.6 | T1 GLNN teacher=GCN(X)+CE | 68.45±4.58 | 66.79±5.75 | 5 |
| PubMed | uniform0.6 | T2 teacher=GCN[X\|\|H]+CE | 74.74±2.50 | 73.24±3.57 | 5 |
| PubMed | uniform0.6 | T4 teacher=kNN-GCN[X\|\|H] | 75.76±2.70 | 73.64±3.70 | 5 |
| PubMed | uniform0.6 | T3 teacher=SoftDist-GCN[X\|\|H] | 75.84±2.66 | 73.89±3.54 | 5 |
| PubMed | uniform0.6 | T5 teacher=shuffled-SoftDist-GCN[X\|\|H] | 75.42±2.67 | 73.44±3.62 | 5 |

## Decomposition deltas

| noise_rate | comparison | dataset | n | mean_delta | std_delta | positive_count |
| --- | --- | --- | --- | --- | --- | --- |
| 0.4 | D1_h_input_increment | citeseer | 5 | 0.027147766323024 | 0.0152397604370235 | 5 |
| 0.4 | D1_h_input_increment | cora | 5 | 0.022796833773087 | 0.0156959931076881 | 5 |
| 0.4 | D1_h_input_increment | pubmed | 5 | 0.0048402289689152 | 0.0053012208306348 | 4 |
| 0.4 | D2_knn_denoising_increment | citeseer | 5 | 0.0076460481099655 | 0.0107919647748743 | 4 |
| 0.4 | D2_knn_denoising_increment | cora | 5 | 0.007598944591029 | 0.0074497969357597 | 4 |
| 0.4 | D2_knn_denoising_increment | pubmed | 5 | 0.0042460691254257 | 0.004431713626212 | 4 |
| 0.6 | D1_h_input_increment | citeseer | 5 | 0.0774054982817869 | 0.0191703033728238 | 5 |
| 0.6 | D1_h_input_increment | cora | 5 | 0.0393667546174142 | 0.0266911029935102 | 4 |
| 0.6 | D1_h_input_increment | pubmed | 5 | 0.0629809434098978 | 0.0269776109871778 | 5 |
| 0.6 | D2_knn_denoising_increment | citeseer | 5 | 0.0091924398625429 | 0.0086956690210104 | 4 |
| 0.6 | D2_knn_denoising_increment | cora | 5 | 0.0163588390501318 | 0.021226580856062 | 3 |
| 0.6 | D2_knn_denoising_increment | pubmed | 5 | 0.0101731758568219 | 0.0082163933271757 | 4 |
