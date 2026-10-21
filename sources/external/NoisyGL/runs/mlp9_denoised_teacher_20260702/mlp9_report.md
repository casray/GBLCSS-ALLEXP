# MLP-9 Denoised Teacher Distillation Report

## Material Passport
- Artifact type: 5-seed experiment result with seed0 reuse
- Protocol: train-only uniform label noise, percent split 0.2/0.1/0.7
- Student: X-only 2-layer weak MLP for every arm
- Teacher: GCN variants; T2-T5 use [X||H] only inside teacher
- Clean labels: evaluation only; teacher and target construction use noisy train labels

## Seed0 Snapshot
| Dataset | Noise | Arm | Teacher test | Student test | teacher target | matched k |
|---|---:|---|---:|---:|---|---:|
| citeseer | 0.4 | T1 GLNN teacher=GCN(X)+CE | 64.48 | 64.82 | ce |  |
| citeseer | 0.4 | T2 teacher=GCN[X||H]+CE | 67.74 | 69.46 | ce |  |
| citeseer | 0.4 | T4 teacher=kNN-GCN[X||H] | 69.42 | 69.93 | soft_target | 10.0 |
| citeseer | 0.4 | T3 teacher=SoftDist-GCN[X||H] | 69.54 | 69.12 | soft_target |  |
| citeseer | 0.4 | T5 teacher=shuffled-SoftDist-GCN[X||H] | 68.34 | 69.54 | soft_target |  |
| cora | 0.4 | T1 GLNN teacher=GCN(X)+CE | 77.04 | 74.35 | ce |  |
| cora | 0.4 | T2 teacher=GCN[X||H]+CE | 79.58 | 79.42 | ce |  |
| cora | 0.4 | T4 teacher=kNN-GCN[X||H] | 80.63 | 80.47 | soft_target | 10.0 |
| cora | 0.4 | T3 teacher=SoftDist-GCN[X||H] | 81.16 | 80.90 | soft_target |  |
| cora | 0.4 | T5 teacher=shuffled-SoftDist-GCN[X||H] | 79.16 | 77.94 | soft_target |  |
| pubmed | 0.4 | T1 GLNN teacher=GCN(X)+CE | 83.20 | 83.68 | ce |  |
| pubmed | 0.4 | T2 teacher=GCN[X||H]+CE | 82.06 | 83.81 | ce |  |
| pubmed | 0.4 | T4 teacher=kNN-GCN[X||H] | 82.17 | 83.69 | soft_target | 62.0 |
| pubmed | 0.4 | T3 teacher=SoftDist-GCN[X||H] | 82.10 | 84.04 | soft_target |  |
| pubmed | 0.4 | T5 teacher=shuffled-SoftDist-GCN[X||H] | 82.05 | 83.52 | soft_target |  |
| citeseer | 0.6 | T1 GLNN teacher=GCN(X)+CE | 47.90 | 47.68 | ce |  |
| citeseer | 0.6 | T2 teacher=GCN[X||H]+CE | 58.55 | 57.95 | ce |  |
| citeseer | 0.6 | T4 teacher=kNN-GCN[X||H] | 56.10 | 57.90 | soft_target | 10.0 |
| citeseer | 0.6 | T3 teacher=SoftDist-GCN[X||H] | 58.93 | 59.19 | soft_target |  |
| citeseer | 0.6 | T5 teacher=shuffled-SoftDist-GCN[X||H] | 59.02 | 58.81 | soft_target |  |
| cora | 0.6 | T1 GLNN teacher=GCN(X)+CE | 61.85 | 62.74 | ce |  |
| cora | 0.6 | T2 teacher=GCN[X||H]+CE | 67.18 | 66.97 | ce |  |
| cora | 0.6 | T4 teacher=kNN-GCN[X||H] | 69.92 | 68.65 | soft_target | 10.0 |
| cora | 0.6 | T3 teacher=SoftDist-GCN[X||H] | 69.18 | 68.65 | soft_target |  |
| cora | 0.6 | T5 teacher=shuffled-SoftDist-GCN[X||H] | 66.44 | 66.39 | soft_target |  |
| pubmed | 0.6 | T1 GLNN teacher=GCN(X)+CE | 66.86 | 68.32 | ce |  |
| pubmed | 0.6 | T2 teacher=GCN[X||H]+CE | 76.94 | 76.38 | ce |  |
| pubmed | 0.6 | T4 teacher=kNN-GCN[X||H] | 77.10 | 77.32 | soft_target | 62.0 |
| pubmed | 0.6 | T3 teacher=SoftDist-GCN[X||H] | 77.11 | 77.99 | soft_target |  |
| pubmed | 0.6 | T5 teacher=shuffled-SoftDist-GCN[X||H] | 76.56 | 76.94 | soft_target |  |

## 5-seed Aggregate
| Dataset | Noise | Arm | n | Student mean +/- std | Teacher mean +/- std | seeds |
|---|---:|---|---:|---:|---:|---|
| citeseer | 0.4 | T1 GLNN teacher=GCN(X)+CE | 5 | 66.09+/-1.38 | 64.90+/-1.11 | 0,1,2,3,4 |
| citeseer | 0.4 | T2 teacher=GCN[X||H]+CE | 5 | 68.81+/-1.12 | 68.35+/-0.97 | 0,1,2,3,4 |
| citeseer | 0.4 | T4 teacher=kNN-GCN[X||H] | 5 | 69.57+/-1.46 | 69.04+/-1.04 | 0,1,2,3,4 |
| citeseer | 0.4 | T3 teacher=SoftDist-GCN[X||H] | 5 | 69.40+/-1.08 | 69.18+/-0.76 | 0,1,2,3,4 |
| citeseer | 0.4 | T5 teacher=shuffled-SoftDist-GCN[X||H] | 5 | 68.60+/-1.14 | 68.10+/-0.91 | 0,1,2,3,4 |
| citeseer | 0.6 | T1 GLNN teacher=GCN(X)+CE | 5 | 53.30+/-4.42 | 52.29+/-4.00 | 0,1,2,3,4 |
| citeseer | 0.6 | T2 teacher=GCN[X||H]+CE | 5 | 61.04+/-3.08 | 60.34+/-2.42 | 0,1,2,3,4 |
| citeseer | 0.6 | T4 teacher=kNN-GCN[X||H] | 5 | 61.96+/-2.98 | 61.15+/-3.24 | 0,1,2,3,4 |
| citeseer | 0.6 | T3 teacher=SoftDist-GCN[X||H] | 5 | 62.28+/-2.30 | 61.60+/-1.97 | 0,1,2,3,4 |
| citeseer | 0.6 | T5 teacher=shuffled-SoftDist-GCN[X||H] | 5 | 61.07+/-2.76 | 60.08+/-2.47 | 0,1,2,3,4 |
| cora | 0.4 | T1 GLNN teacher=GCN(X)+CE | 5 | 77.95+/-2.62 | 78.77+/-1.87 | 0,1,2,3,4 |
| cora | 0.4 | T2 teacher=GCN[X||H]+CE | 5 | 80.23+/-1.57 | 80.05+/-1.41 | 0,1,2,3,4 |
| cora | 0.4 | T4 teacher=kNN-GCN[X||H] | 5 | 80.99+/-1.02 | 81.45+/-1.23 | 0,1,2,3,4 |
| cora | 0.4 | T3 teacher=SoftDist-GCN[X||H] | 5 | 80.60+/-0.69 | 80.84+/-1.07 | 0,1,2,3,4 |
| cora | 0.4 | T5 teacher=shuffled-SoftDist-GCN[X||H] | 5 | 78.81+/-1.63 | 79.62+/-1.65 | 0,1,2,3,4 |
| cora | 0.6 | T1 GLNN teacher=GCN(X)+CE | 5 | 65.50+/-3.28 | 65.44+/-2.59 | 0,1,2,3,4 |
| cora | 0.6 | T2 teacher=GCN[X||H]+CE | 5 | 69.44+/-2.18 | 69.33+/-1.64 | 0,1,2,3,4 |
| cora | 0.6 | T4 teacher=kNN-GCN[X||H] | 5 | 71.07+/-2.29 | 71.87+/-2.16 | 0,1,2,3,4 |
| cora | 0.6 | T3 teacher=SoftDist-GCN[X||H] | 5 | 71.27+/-2.46 | 71.35+/-2.20 | 0,1,2,3,4 |
| cora | 0.6 | T5 teacher=shuffled-SoftDist-GCN[X||H] | 5 | 68.51+/-1.82 | 68.49+/-1.80 | 0,1,2,3,4 |
| pubmed | 0.4 | T1 GLNN teacher=GCN(X)+CE | 5 | 84.32+/-0.47 | 83.75+/-0.57 | 0,1,2,3,4 |
| pubmed | 0.4 | T2 teacher=GCN[X||H]+CE | 5 | 84.80+/-0.85 | 83.25+/-0.72 | 0,1,2,3,4 |
| pubmed | 0.4 | T4 teacher=kNN-GCN[X||H] | 5 | 85.23+/-0.94 | 83.53+/-0.83 | 0,1,2,3,4 |
| pubmed | 0.4 | T3 teacher=SoftDist-GCN[X||H] | 5 | 85.01+/-0.93 | 83.27+/-0.88 | 0,1,2,3,4 |
| pubmed | 0.4 | T5 teacher=shuffled-SoftDist-GCN[X||H] | 5 | 85.08+/-0.93 | 83.36+/-0.76 | 0,1,2,3,4 |
| pubmed | 0.6 | T1 GLNN teacher=GCN(X)+CE | 5 | 68.45+/-4.58 | 66.79+/-5.75 | 0,1,2,3,4 |
| pubmed | 0.6 | T2 teacher=GCN[X||H]+CE | 5 | 74.74+/-2.50 | 73.24+/-3.57 | 0,1,2,3,4 |
| pubmed | 0.6 | T4 teacher=kNN-GCN[X||H] | 5 | 75.76+/-2.70 | 73.64+/-3.70 | 0,1,2,3,4 |
| pubmed | 0.6 | T3 teacher=SoftDist-GCN[X||H] | 5 | 75.84+/-2.66 | 73.89+/-3.54 | 0,1,2,3,4 |
| pubmed | 0.6 | T5 teacher=shuffled-SoftDist-GCN[X||H] | 5 | 75.42+/-2.67 | 73.44+/-3.62 | 0,1,2,3,4 |

## 5-seed Pre-registered Gates
| Noise | Scope | Dataset | Gate | Mean delta | Std | Seed pass | Dataset pass count | Decision | Details |
|---:|---|---|---|---:|---:|---:|---:|---|---|
| 0.4 | dataset | cora | G1_win_glnn | +2.65pp | +2.81pp | 4/5 |  | pass | 0,1,2,3,4 |
| 0.4 | dataset | citeseer | G1_win_glnn | +3.31pp | +1.85pp | 4/5 |  | pass | 0,1,2,3,4 |
| 0.4 | dataset | pubmed | G1_win_glnn | +0.69pp | +0.68pp | 2/5 |  | not-pass | 0,1,2,3,4 |
| 0.4 | overall | all | G1_win_glnn | +2.22pp | +2.16pp |  | 2 | pass | cora:+2.65pp 4/5; citeseer:+3.31pp 4/5; pubmed:+0.69pp 2/5 |
| 0.4 | dataset | cora | G2_denoising_attribution | +0.37pp | +1.74pp | 4/5 |  | pass | 0,1,2,3,4 |
| 0.4 | dataset | citeseer | G2_denoising_attribution | +0.59pp | +0.98pp | 3/5 |  | not-pass | 0,1,2,3,4 |
| 0.4 | dataset | pubmed | G2_denoising_attribution | +0.21pp | +0.19pp | 4/5 |  | pass | 0,1,2,3,4 |
| 0.4 | overall | all | G2_denoising_attribution | +0.39pp | +1.08pp |  | 2 | pass | cora:+0.37pp 4/5; citeseer:+0.59pp 3/5; pubmed:+0.21pp 4/5 |
| 0.4 | dataset | cora | G3_ball_attribution | -0.39pp | +1.03pp | 0/5 |  | not-pass | 0,1,2,3,4 |
| 0.4 | dataset | citeseer | G3_ball_attribution | -0.17pp | +0.52pp | 0/5 |  | not-pass | 0,1,2,3,4 |
| 0.4 | dataset | pubmed | G3_ball_attribution | -0.22pp | +0.52pp | 0/5 |  | not-pass | 0,1,2,3,4 |
| 0.4 | overall | all | G3_ball_attribution | -0.26pp | +0.68pp |  | 0 | not-pass | cora:-0.39pp 0/5; citeseer:-0.17pp 0/5; pubmed:-0.22pp 0/5 |
| 0.4 | dataset | cora | G4_structure_sanity | +1.79pp | +1.96pp | 4/5 |  | pass | 0,1,2,3,4 |
| 0.4 | dataset | citeseer | G4_structure_sanity | +0.80pp | +0.77pp | 4/5 |  | pass | 0,1,2,3,4 |
| 0.4 | dataset | pubmed | G4_structure_sanity | -0.07pp | +0.61pp | 3/5 |  | not-pass | 0,1,2,3,4 |
| 0.4 | overall | all | G4_structure_sanity | +0.84pp | +1.41pp |  | 2 | pass | cora:+1.79pp 4/5; citeseer:+0.80pp 4/5; pubmed:-0.07pp 3/5 |
| 0.6 | dataset | cora | G1_win_glnn | +5.77pp | +1.08pp | 5/5 |  | pass | 0,1,2,3,4 |
| 0.6 | dataset | citeseer | G1_win_glnn | +8.98pp | +2.23pp | 5/5 |  | pass | 0,1,2,3,4 |
| 0.6 | dataset | pubmed | G1_win_glnn | +7.39pp | +2.63pp | 5/5 |  | pass | 0,1,2,3,4 |
| 0.6 | overall | all | G1_win_glnn | +7.38pp | +2.36pp |  | 3 | pass | cora:+5.77pp 5/5; citeseer:+8.98pp 5/5; pubmed:+7.39pp 5/5 |
| 0.6 | dataset | cora | G2_denoising_attribution | +1.84pp | +1.64pp | 5/5 |  | pass | 0,1,2,3,4 |
| 0.6 | dataset | citeseer | G2_denoising_attribution | +1.24pp | +1.31pp | 4/5 |  | pass | 0,1,2,3,4 |
| 0.6 | dataset | pubmed | G2_denoising_attribution | +1.09pp | +0.45pp | 5/5 |  | pass | 0,1,2,3,4 |
| 0.6 | overall | all | G2_denoising_attribution | +1.39pp | +1.20pp |  | 3 | pass | cora:+1.84pp 5/5; citeseer:+1.24pp 4/5; pubmed:+1.09pp 5/5 |
| 0.6 | dataset | cora | G3_ball_attribution | +0.20pp | +0.92pp | 1/5 |  | not-pass | 0,1,2,3,4 |
| 0.6 | dataset | citeseer | G3_ball_attribution | +0.32pp | +0.96pp | 2/5 |  | not-pass | 0,1,2,3,4 |
| 0.6 | dataset | pubmed | G3_ball_attribution | +0.08pp | +0.94pp | 1/5 |  | not-pass | 0,1,2,3,4 |
| 0.6 | overall | all | G3_ball_attribution | +0.20pp | +0.88pp |  | 0 | not-pass | cora:+0.20pp 1/5; citeseer:+0.32pp 2/5; pubmed:+0.08pp 1/5 |
| 0.6 | dataset | cora | G4_structure_sanity | +2.77pp | +0.80pp | 5/5 |  | pass | 0,1,2,3,4 |
| 0.6 | dataset | citeseer | G4_structure_sanity | +1.21pp | +1.23pp | 4/5 |  | pass | 0,1,2,3,4 |
| 0.6 | dataset | pubmed | G4_structure_sanity | +0.42pp | +0.71pp | 3/5 |  | not-pass | 0,1,2,3,4 |
| 0.6 | overall | all | G4_structure_sanity | +1.46pp | +1.33pp |  | 2 | pass | cora:+2.77pp 5/5; citeseer:+1.21pp 4/5; pubmed:+0.42pp 3/5 |

## Decomposition Deltas
| Noise | Dataset | Comparison | Mean delta | Std | Positive |
|---:|---|---|---:|---:|---:|
| 0.4 | citeseer | D1_h_input_increment | +2.71pp | +1.52pp | 5/5 |
| 0.4 | cora | D1_h_input_increment | +2.28pp | +1.57pp | 5/5 |
| 0.4 | pubmed | D1_h_input_increment | +0.48pp | +0.53pp | 4/5 |
| 0.4 | citeseer | D2_knn_denoising_increment | +0.76pp | +1.08pp | 4/5 |
| 0.4 | cora | D2_knn_denoising_increment | +0.76pp | +0.74pp | 4/5 |
| 0.4 | pubmed | D2_knn_denoising_increment | +0.42pp | +0.44pp | 4/5 |
| 0.6 | citeseer | D1_h_input_increment | +7.74pp | +1.92pp | 5/5 |
| 0.6 | cora | D1_h_input_increment | +3.94pp | +2.67pp | 4/5 |
| 0.6 | pubmed | D1_h_input_increment | +6.30pp | +2.70pp | 5/5 |
| 0.6 | citeseer | D2_knn_denoising_increment | +0.92pp | +0.87pp | 4/5 |
| 0.6 | cora | D2_knn_denoising_increment | +1.64pp | +2.12pp | 3/5 |
| 0.6 | pubmed | D2_knn_denoising_increment | +1.02pp | +0.82pp | 4/5 |

## kNN Matched-k Audit
| Noise | Dataset | Seed | matched k | matched balls | Student test | Teacher test |
|---:|---|---:|---:|---:|---:|---:|
| 0.4 | citeseer | 0 | 10.0 | 64.0 | 69.93 | 69.42 |
| 0.4 | citeseer | 1 | 10.0 | 64.0 | 67.18 | 67.18 |
| 0.4 | citeseer | 2 | 10.0 | 64.0 | 69.63 | 69.46 |
| 0.4 | citeseer | 3 | 10.0 | 64.0 | 71.18 | 69.50 |
| 0.4 | citeseer | 4 | 11.0 | 63.0 | 69.93 | 69.63 |
| 0.4 | cora | 0 | 10.0 | 52.0 | 80.47 | 80.63 |
| 0.4 | cora | 1 | 10.0 | 53.0 | 79.95 | 80.32 |
| 0.4 | cora | 2 | 10.0 | 52.0 | 81.90 | 83.01 |
| 0.4 | cora | 3 | 10.0 | 53.0 | 82.27 | 82.53 |
| 0.4 | cora | 4 | 10.0 | 52.0 | 80.37 | 80.74 |
| 0.4 | pubmed | 0 | 62.0 | 64.0 | 83.69 | 82.17 |
| 0.4 | pubmed | 1 | 62.0 | 64.0 | 85.82 | 84.04 |
| 0.4 | pubmed | 2 | 62.0 | 64.0 | 85.02 | 83.46 |
| 0.4 | pubmed | 3 | 62.0 | 64.0 | 86.04 | 84.31 |
| 0.4 | pubmed | 4 | 62.0 | 64.0 | 85.57 | 83.65 |
| 0.6 | citeseer | 0 | 10.0 | 64.0 | 57.90 | 56.10 |
| 0.6 | citeseer | 1 | 10.0 | 64.0 | 65.68 | 64.86 |
| 0.6 | citeseer | 2 | 10.0 | 64.0 | 63.92 | 62.80 |
| 0.6 | citeseer | 3 | 10.0 | 64.0 | 61.43 | 61.04 |
| 0.6 | citeseer | 4 | 11.0 | 63.0 | 60.87 | 60.95 |
| 0.6 | cora | 0 | 10.0 | 52.0 | 68.65 | 69.92 |
| 0.6 | cora | 1 | 10.0 | 53.0 | 71.50 | 71.93 |
| 0.6 | cora | 2 | 10.0 | 52.0 | 72.77 | 73.98 |
| 0.6 | cora | 3 | 10.0 | 53.0 | 73.67 | 74.04 |
| 0.6 | cora | 4 | 10.0 | 52.0 | 68.76 | 69.50 |
| 0.6 | pubmed | 0 | 62.0 | 64.0 | 77.32 | 77.10 |
| 0.6 | pubmed | 1 | 62.0 | 64.0 | 72.44 | 70.44 |
| 0.6 | pubmed | 2 | 62.0 | 64.0 | 74.84 | 72.12 |
| 0.6 | pubmed | 3 | 62.0 | 64.0 | 79.49 | 78.10 |
| 0.6 | pubmed | 4 | 62.0 | 64.0 | 74.72 | 70.44 |

## Branch Decision
Decision: **denoised_teacher_possible_but_ball_replaced_by_knn**

## Research Record
- Hypothesis: a denoised GCN[X||H] teacher can transmit teacher-side label denoising gains to an X-only MLP student.
- Granular-ball role: train-label soft target for the teacher only; the student never consumes H or ball targets directly.
- Success standard: on uniform 0.4, each dataset passes a gate when at least 4/5 paired seeds satisfy the rule; a gate passes overall when at least 2/3 datasets pass.
- Failure standard: if G1 or G2 fails, the denoised-teacher probe is closed; if G3 fails, kNN replacement remains the teacher-layer explanation.
