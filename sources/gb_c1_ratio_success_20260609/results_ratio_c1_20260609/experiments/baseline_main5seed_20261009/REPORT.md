# Five-seed baseline results and audit

75/75 formal records passed protocol, source/artifact identity, checkpoint reload,
200-epoch budget and clean-validation first/last maximum checks.
Mean +/- sample standard deviation (ddof=1), in percentage points.

| Method | Noise | Dataset | Test accuracy |
|---|---|---|---|
| simmlp | 0.4 | cora | 75.46 +/- 1.41 |
| simmlp | 0.4 | citeseer | 65.31 +/- 1.78 |
| simmlp | 0.4 | corafull | 40.24 +/- 0.34 |
| simmlp | 0.4 | coauthorcs | 86.47 +/- 1.79 |
| simmlp | 0.4 | amazonpho | 88.69 +/- 0.61 |
| tined | 0.4 | cora | 78.92 +/- 2.83 |
| tined | 0.4 | citeseer | 64.18 +/- 1.16 |
| tined | 0.4 | corafull | 52.07 +/- 1.36 |
| tined | 0.4 | coauthorcs | 92.19 +/- 0.46 |
| tined | 0.4 | amazonpho | 85.46 +/- 1.47 |
| tined | 0.6 | cora | 66.36 +/- 2.50 |
| tined | 0.6 | citeseer | 51.32 +/- 5.07 |
| tined | 0.6 | corafull | 32.07 +/- 9.33 |
| tined | 0.6 | coauthorcs | 88.93 +/- 1.11 |
| tined | 0.6 | amazonpho | 64.64 +/- 6.36 |

## Scope and interpretation

SimMLP uses the exact common two-FC deployed backbone and a fixed label-free
seed0 encoder, with new noisy-label probe heads for seeds1..4. Existing seed0
runs were source/config/fingerprint checked and reused. This estimates
downstream variability conditional on that encoder, not pretraining variance.

TINED uses a common GCN teacher and its native four-FC transformation-mimicking
student. Adam/budgets/data/teacher match the common protocol; native student
depth, dropout, minibatching and objective are disclosed exceptions. Native
SAGE method-knob presets were fixed before this run; no test-guided tuning.
Its sorted CE minibatches correct an earlier port omission, so all student
seed0 runs were regenerated. Prior unsorted student scores must not be mixed.

A low score or late degradation is retained. Best clean-validation checkpoint
selection does not prove the optimization is stable; inspect final_valid_percent
and epochs.jsonl for degradation. Five seeds do not establish global superiority.

GSSC is withheld because the present port has unmatched depth and supervised
budget. DGD has no verified runnable adaptation. No fabricated/missing rows.

Final decision: retain these audited adaptation rows as candidate additions;
update manuscript method/backbone descriptions before treating TINED as Table8
data. No manuscript or existing table was changed in this experiment.
