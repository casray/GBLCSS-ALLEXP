# MLP-26 FF-G2M Fidelity Report

## Identity and protocol

- Official objective port from `LirongWu/FF-G2M`, commit `47edcab20551cfecddf045199613a4d0f8db9b97`.
- Preserved objectives: noisy hard-label CE, neighborhood low-frequency KL, edge-difference high-frequency KL.
- Common-backbone adaptation: 2-layer width-64 GCN teacher and X-only MLP student, 200 epochs, fixed lambda=0.4 and tau=1.5.
- Teacher checkpoint chosen only by validation; test evaluated once after checkpoint restoration.
- Uniform 0.4 train-only noise, percent 0.2/0.1/0.7, seeds 0--4.

## Results

| Dataset | FF-G2M reimplemented | Old frequency proxy | LCSS distillation |
|---|---:|---:|---:|
| Cora | 77.65+-2.97 | 78.73+-2.62 | 80.60+-0.69 |
| CiteSeer | 65.80+-0.49 | 66.17+-1.30 | 69.40+-1.08 |
| CoraFull | 56.58+-0.93 | 58.54+-1.00 | 58.22+-0.64 |
| Coauthor-CS | 91.58+-0.24 | 93.12+-0.24 | 93.75+-0.17 |
| Amazon-Photo | 84.82+-0.84 | 90.01+-1.14 | 90.49+-0.62 |

## Conclusion

All 25 rows completed successfully. The prior proxy overestimated FF-G2M by 0.37--5.19 pp because its edge-similarity MSE objectives were not the paper's LFD/HFD. The faithful common-backbone reimplementation remains below LCSS distillation on all five datasets by 1.64--5.67 pp. This row can replace the proxy as `FF-G2M (common-backbone reimplementation)`; the proxy must not remain in method rankings.
