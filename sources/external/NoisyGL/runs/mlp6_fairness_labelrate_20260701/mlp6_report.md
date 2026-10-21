# MLP-6 Fairness + Label-Rate Report

## Data Completeness
- mlp6_new: observed 265/265, missing=0
- combined_with_mlp5_reuse: observed 435/435, missing=0

## Part A: H-input Fairness Control

Verdict: **reframe_needed**
- cora: weakest beta05-vs-H delta is -0.0454 against mlp_jocor_xh (1/5 positive)
- citeseer: weakest beta05-vs-H delta is -0.0477 against mlp_jocor_xh (0/5 positive)

### Aggregates

#### cora
- SoftDist-MLP beta=0.5: 0.5559 +/- 0.0150 (n=5)
- MLP+CE (X): 0.4424 +/- 0.0262 (n=5)
- MLP+small-loss (X): 0.5195 +/- 0.0197 (n=5)
- MLP+JoCoR (X): 0.5172 +/- 0.0130 (n=5)
- MLP+CE [X||H]: 0.4854 +/- 0.0417 (n=5)
- MLP+small-loss [X||H]: 0.5770 +/- 0.0288 (n=5)
- MLP+JoCoR [X||H]: 0.6013 +/- 0.0343 (n=5)
#### citeseer
- SoftDist-MLP beta=0.5: 0.5383 +/- 0.0275 (n=5)
- MLP+CE (X): 0.4350 +/- 0.0302 (n=5)
- MLP+small-loss (X): 0.4970 +/- 0.0307 (n=5)
- MLP+JoCoR (X): 0.4991 +/- 0.0337 (n=5)
- MLP+CE [X||H]: 0.5166 +/- 0.0422 (n=5)
- MLP+small-loss [X||H]: 0.5630 +/- 0.0228 (n=5)
- MLP+JoCoR [X||H]: 0.5860 +/- 0.0230 (n=5)
#### pubmed
- SoftDist-MLP beta=0.5: 0.7190 +/- 0.0088 (n=5)
- MLP+CE (X): 0.6377 +/- 0.0440 (n=5)
- MLP+small-loss (X): 0.7730 +/- 0.0129 (n=5)
- MLP+JoCoR (X): 0.7784 +/- 0.0110 (n=5)
- MLP+CE [X||H]: 0.8368 +/- 0.0052 (n=5)
- MLP+small-loss [X||H]: 0.8483 +/- 0.0050 (n=5)
- MLP+JoCoR [X||H]: 0.8479 +/- 0.0064 (n=5)

### Paired Deltas: beta05 - H-input baselines

- cora vs MLP+CE [X||H]: +0.0705 +/- 0.0324, 5/5 positive
- cora vs MLP+small-loss [X||H]: -0.0211 +/- 0.0366, 2/5 positive
- cora vs MLP+JoCoR [X||H]: -0.0454 +/- 0.0353, 1/5 positive
- citeseer vs MLP+CE [X||H]: +0.0217 +/- 0.0242, 4/5 positive
- citeseer vs MLP+small-loss [X||H]: -0.0247 +/- 0.0410, 1/5 positive
- citeseer vs MLP+JoCoR [X||H]: -0.0477 +/- 0.0284, 0/5 positive
- pubmed vs MLP+CE [X||H]: -0.1177 +/- 0.0126, 0/5 positive
- pubmed vs MLP+small-loss [X||H]: -0.1293 +/- 0.0085, 0/5 positive
- pubmed vs MLP+JoCoR [X||H]: -0.1289 +/- 0.0100, 0/5 positive

### Low-label Fairness (train=0.05)

- cora vs MLP+small-loss [X||H]: -0.0172 +/- 0.0242, 2/5 positive
- cora vs MLP+JoCoR [X||H]: -0.0227 +/- 0.0286, 1/5 positive
- citeseer vs MLP+small-loss [X||H]: -0.0723 +/- 0.0249, 0/5 positive
- citeseer vs MLP+JoCoR [X||H]: -0.0657 +/- 0.0250, 0/5 positive

## Part B: 5-seed Label-rate Curve

### cora
- train=0.05: SoftDist-MLP beta=0.5=0.4370 +/- 0.0581; MLP+small-loss (X)=0.3809 +/- 0.0545; MLP+JoCoR (X)=0.3702 +/- 0.0539; MLP-FINE (X)=0.3509 +/- 0.0511; SoftDist beta=0.5 shuffled=0.3713 +/- 0.0458
- train=0.10: SoftDist-MLP beta=0.5=0.4947 +/- 0.0348; MLP+small-loss (X)=0.4458 +/- 0.0350; MLP+JoCoR (X)=0.4467 +/- 0.0369; MLP-FINE (X)=0.4041 +/- 0.0382; SoftDist beta=0.5 shuffled=0.3996 +/- 0.0385
- train=0.20: SoftDist-MLP beta=0.5=0.5559 +/- 0.0150; MLP+small-loss (X)=0.5195 +/- 0.0197; MLP+JoCoR (X)=0.5172 +/- 0.0130; MLP-FINE (X)=0.4354 +/- 0.0148; SoftDist beta=0.5 shuffled=0.4363 +/- 0.0162
- train=0.35: SoftDist-MLP beta=0.5=0.5880 +/- 0.0303; MLP+small-loss (X)=0.5659 +/- 0.0204; MLP+JoCoR (X)=0.5743 +/- 0.0207; MLP-FINE (X)=0.4903 +/- 0.0319; SoftDist beta=0.5 shuffled=0.4721 +/- 0.0262
- train=0.50: SoftDist-MLP beta=0.5=0.6299 +/- 0.0064; MLP+small-loss (X)=0.6105 +/- 0.0073; MLP+JoCoR (X)=0.6175 +/- 0.0123; MLP-FINE (X)=0.5176 +/- 0.0308; SoftDist beta=0.5 shuffled=0.4938 +/- 0.0384
- train=0.65: SoftDist-MLP beta=0.5=0.6402 +/- 0.0230; MLP+small-loss (X)=0.6278 +/- 0.0214; MLP+JoCoR (X)=0.6384 +/- 0.0295; MLP-FINE (X)=0.5199 +/- 0.0405; SoftDist beta=0.5 shuffled=0.5253 +/- 0.0326

### citeseer
- train=0.05: SoftDist-MLP beta=0.5=0.4334 +/- 0.0351; MLP+small-loss (X)=0.3754 +/- 0.0296; MLP+JoCoR (X)=0.3723 +/- 0.0282; MLP-FINE (X)=0.3582 +/- 0.0293; SoftDist beta=0.5 shuffled=0.3503 +/- 0.0324
- train=0.10: SoftDist-MLP beta=0.5=0.4984 +/- 0.0314; MLP+small-loss (X)=0.4191 +/- 0.0197; MLP+JoCoR (X)=0.4331 +/- 0.0288; MLP-FINE (X)=0.3922 +/- 0.0181; SoftDist beta=0.5 shuffled=0.3976 +/- 0.0117
- train=0.20: SoftDist-MLP beta=0.5=0.5383 +/- 0.0275; MLP+small-loss (X)=0.4970 +/- 0.0307; MLP+JoCoR (X)=0.4991 +/- 0.0337; MLP-FINE (X)=0.4343 +/- 0.0343; SoftDist beta=0.5 shuffled=0.4235 +/- 0.0274
- train=0.35: SoftDist-MLP beta=0.5=0.5855 +/- 0.0149; MLP+small-loss (X)=0.5520 +/- 0.0223; MLP+JoCoR (X)=0.5613 +/- 0.0249; MLP-FINE (X)=0.5078 +/- 0.0276; SoftDist beta=0.5 shuffled=0.4908 +/- 0.0214
- train=0.50: SoftDist-MLP beta=0.5=0.6159 +/- 0.0078; MLP+small-loss (X)=0.5802 +/- 0.0055; MLP+JoCoR (X)=0.5890 +/- 0.0039; MLP-FINE (X)=0.5529 +/- 0.0131; SoftDist beta=0.5 shuffled=0.5395 +/- 0.0064
- train=0.65: SoftDist-MLP beta=0.5=0.6262 +/- 0.0082; MLP+small-loss (X)=0.6118 +/- 0.0178; MLP+JoCoR (X)=0.6128 +/- 0.0139; MLP-FINE (X)=0.5711 +/- 0.0184; SoftDist beta=0.5 shuffled=0.5531 +/- 0.0329

### Label-rate Paired Deltas
- cora train=0.05 vs MLP+small-loss (X): +0.0561, 5/5 positive
- cora train=0.05 vs MLP+JoCoR (X): +0.0668, 5/5 positive
- cora train=0.05 vs MLP-FINE (X): +0.0861, 5/5 positive
- cora train=0.05 vs SoftDist beta=0.5 shuffled: +0.0657, 5/5 positive
- cora train=0.10 vs MLP+small-loss (X): +0.0489, 5/5 positive
- cora train=0.10 vs MLP+JoCoR (X): +0.0480, 5/5 positive
- cora train=0.10 vs MLP-FINE (X): +0.0907, 5/5 positive
- cora train=0.10 vs SoftDist beta=0.5 shuffled: +0.0951, 5/5 positive
- cora train=0.20 vs MLP+small-loss (X): +0.0364, 4/5 positive
- cora train=0.20 vs MLP+JoCoR (X): +0.0387, 5/5 positive
- cora train=0.20 vs MLP-FINE (X): +0.1205, 5/5 positive
- cora train=0.20 vs SoftDist beta=0.5 shuffled: +0.1196, 5/5 positive
- cora train=0.35 vs MLP+small-loss (X): +0.0222, 5/5 positive
- cora train=0.35 vs MLP+JoCoR (X): +0.0137, 3/5 positive
- cora train=0.35 vs MLP-FINE (X): +0.0978, 5/5 positive
- cora train=0.35 vs SoftDist beta=0.5 shuffled: +0.1159, 5/5 positive
- cora train=0.50 vs MLP+small-loss (X): +0.0194, 5/5 positive
- cora train=0.50 vs MLP+JoCoR (X): +0.0124, 4/5 positive
- cora train=0.50 vs MLP-FINE (X): +0.1123, 5/5 positive
- cora train=0.50 vs SoftDist beta=0.5 shuffled: +0.1361, 5/5 positive
- cora train=0.65 vs MLP+small-loss (X): +0.0124, 5/5 positive
- cora train=0.65 vs MLP+JoCoR (X): +0.0018, 2/5 positive
- cora train=0.65 vs MLP-FINE (X): +0.1202, 5/5 positive
- cora train=0.65 vs SoftDist beta=0.5 shuffled: +0.1149, 5/5 positive
- citeseer train=0.05 vs MLP+small-loss (X): +0.0580, 5/5 positive
- citeseer train=0.05 vs MLP+JoCoR (X): +0.0611, 5/5 positive
- citeseer train=0.05 vs MLP-FINE (X): +0.0752, 5/5 positive
- citeseer train=0.05 vs SoftDist beta=0.5 shuffled: +0.0831, 5/5 positive
- citeseer train=0.10 vs MLP+small-loss (X): +0.0793, 5/5 positive
- citeseer train=0.10 vs MLP+JoCoR (X): +0.0652, 5/5 positive
- citeseer train=0.10 vs MLP-FINE (X): +0.1062, 5/5 positive
- citeseer train=0.10 vs SoftDist beta=0.5 shuffled: +0.1008, 5/5 positive
- citeseer train=0.20 vs MLP+small-loss (X): +0.0413, 5/5 positive
- citeseer train=0.20 vs MLP+JoCoR (X): +0.0393, 5/5 positive
- citeseer train=0.20 vs MLP-FINE (X): +0.1040, 5/5 positive
- citeseer train=0.20 vs SoftDist beta=0.5 shuffled: +0.1148, 5/5 positive
- citeseer train=0.35 vs MLP+small-loss (X): +0.0335, 5/5 positive
- citeseer train=0.35 vs MLP+JoCoR (X): +0.0242, 5/5 positive
- citeseer train=0.35 vs MLP-FINE (X): +0.0776, 5/5 positive
- citeseer train=0.35 vs SoftDist beta=0.5 shuffled: +0.0947, 5/5 positive
- citeseer train=0.50 vs MLP+small-loss (X): +0.0358, 5/5 positive
- citeseer train=0.50 vs MLP+JoCoR (X): +0.0269, 5/5 positive
- citeseer train=0.50 vs MLP-FINE (X): +0.0630, 5/5 positive
- citeseer train=0.50 vs SoftDist beta=0.5 shuffled: +0.0764, 5/5 positive
- citeseer train=0.65 vs MLP+small-loss (X): +0.0144, 4/5 positive
- citeseer train=0.65 vs MLP+JoCoR (X): +0.0135, 4/5 positive
- citeseer train=0.65 vs MLP-FINE (X): +0.0551, 5/5 positive
- citeseer train=0.65 vs SoftDist beta=0.5 shuffled: +0.0732, 5/5 positive

## Interpretation Guardrails
- Part A is an H-access fairness control, not a replacement for the shuffled-ball control.
- Part B reuses MLP-5 endpoints train=0.05/0.65 and newly fills train=0.10/0.20/0.35/0.50.
- PubMed is reported as a boundary dataset; Cora/CiteSeer determine the main fairness claim.
