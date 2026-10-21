# MLP-6 Fairness + Label-Rate Report

## Data Completeness
- mlp6_new: observed 0/265, missing=265
- combined_with_mlp5_reuse: observed 170/435, missing=265

## Part A: H-input Fairness Control

Verdict: **incomplete**
- cora: missing H-input paired deltas
- citeseer: missing H-input paired deltas

### Aggregates

#### cora
- SoftDist-MLP beta=0.5: 0.5559 +/- 0.0150 (n=5)
- MLP+CE (X): 0.4424 +/- 0.0262 (n=5)
- MLP+small-loss (X): 0.5195 +/- 0.0197 (n=5)
- MLP+JoCoR (X): 0.5172 +/- 0.0130 (n=5)
#### citeseer
- SoftDist-MLP beta=0.5: 0.5383 +/- 0.0275 (n=5)
- MLP+CE (X): 0.4350 +/- 0.0302 (n=5)
- MLP+small-loss (X): 0.4970 +/- 0.0307 (n=5)
- MLP+JoCoR (X): 0.4991 +/- 0.0337 (n=5)
#### pubmed
- SoftDist-MLP beta=0.5: 0.7190 +/- 0.0088 (n=5)
- MLP+CE (X): 0.6377 +/- 0.0440 (n=5)
- MLP+small-loss (X): 0.7730 +/- 0.0129 (n=5)
- MLP+JoCoR (X): 0.7784 +/- 0.0110 (n=5)

### Paired Deltas: beta05 - H-input baselines


### Low-label Fairness (train=0.05)


## Part B: 5-seed Label-rate Curve

### cora
- train=0.05: SoftDist-MLP beta=0.5=0.4370 +/- 0.0581; MLP+small-loss (X)=0.3809 +/- 0.0545; MLP+JoCoR (X)=0.3702 +/- 0.0539; MLP-FINE (X)=0.3509 +/- 0.0511; SoftDist beta=0.5 shuffled=0.3713 +/- 0.0458
- train=0.10: 
- train=0.20: 
- train=0.35: 
- train=0.50: 
- train=0.65: SoftDist-MLP beta=0.5=0.6402 +/- 0.0230; MLP+small-loss (X)=0.6278 +/- 0.0214; MLP+JoCoR (X)=0.6384 +/- 0.0295; MLP-FINE (X)=0.5199 +/- 0.0405; SoftDist beta=0.5 shuffled=0.5253 +/- 0.0326

### citeseer
- train=0.05: SoftDist-MLP beta=0.5=0.4334 +/- 0.0351; MLP+small-loss (X)=0.3754 +/- 0.0296; MLP+JoCoR (X)=0.3723 +/- 0.0282; MLP-FINE (X)=0.3582 +/- 0.0293; SoftDist beta=0.5 shuffled=0.3503 +/- 0.0324
- train=0.10: 
- train=0.20: 
- train=0.35: 
- train=0.50: 
- train=0.65: SoftDist-MLP beta=0.5=0.6262 +/- 0.0082; MLP+small-loss (X)=0.6118 +/- 0.0178; MLP+JoCoR (X)=0.6128 +/- 0.0139; MLP-FINE (X)=0.5711 +/- 0.0184; SoftDist beta=0.5 shuffled=0.5531 +/- 0.0329

### Label-rate Paired Deltas
- cora train=0.05 vs MLP+small-loss (X): +0.0561, 5/5 positive
- cora train=0.05 vs MLP+JoCoR (X): +0.0668, 5/5 positive
- cora train=0.05 vs MLP-FINE (X): +0.0861, 5/5 positive
- cora train=0.05 vs SoftDist beta=0.5 shuffled: +0.0657, 5/5 positive
- cora train=0.65 vs MLP+small-loss (X): +0.0124, 5/5 positive
- cora train=0.65 vs MLP+JoCoR (X): +0.0018, 2/5 positive
- cora train=0.65 vs MLP-FINE (X): +0.1202, 5/5 positive
- cora train=0.65 vs SoftDist beta=0.5 shuffled: +0.1149, 5/5 positive
- citeseer train=0.05 vs MLP+small-loss (X): +0.0580, 5/5 positive
- citeseer train=0.05 vs MLP+JoCoR (X): +0.0611, 5/5 positive
- citeseer train=0.05 vs MLP-FINE (X): +0.0752, 5/5 positive
- citeseer train=0.05 vs SoftDist beta=0.5 shuffled: +0.0831, 5/5 positive
- citeseer train=0.65 vs MLP+small-loss (X): +0.0144, 4/5 positive
- citeseer train=0.65 vs MLP+JoCoR (X): +0.0135, 4/5 positive
- citeseer train=0.65 vs MLP-FINE (X): +0.0551, 5/5 positive
- citeseer train=0.65 vs SoftDist beta=0.5 shuffled: +0.0732, 5/5 positive

## Interpretation Guardrails
- Part A is an H-access fairness control, not a replacement for the shuffled-ball control.
- Part B reuses MLP-5 endpoints train=0.05/0.65 and newly fills train=0.10/0.20/0.35/0.50.
- PubMed is reported as a boundary dataset; Cora/CiteSeer determine the main fairness claim.
