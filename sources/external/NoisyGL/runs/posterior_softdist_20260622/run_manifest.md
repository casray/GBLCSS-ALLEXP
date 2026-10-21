# Posterior-SoftDist Stage A execution manifest

- Date: 2026-06-22
- Branch: `codex/posterior-softdist`
- Design commit: `0eeb529`
- Diagnostic implementation commit: `51ff77a`
- Status: completed successfully
- Decision: `STOP`

## Exact command

```powershell
& 'C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe' scripts/diagnose_posterior_softdist.py --datasets cora,citeseer,coauthorcs --seeds 0,1,2,3,4 --noise-type uniform --noise-rate 0.4 --device cuda:0 --split-type percent --train-percent 0.2 --val-percent 0.1 --test-percent 0.7 --output-dir runs/posterior_softdist_20260622
```

## Protocol

- Datasets: Cora, CiteSeer, Coauthor-CS
- Seeds: 0, 1, 2, 3, 4
- Noise: uniform, rate 0.4
- Split: percent, train/validation/test = 0.2/0.1/0.7
- Representation and granular-ball parameters: loaded unchanged from each dataset's `frozengrace_gb_gcn_softdist_xh` configuration
- Labels used by ball voting: training-mask labels only
- Low-support bin: 2–3 training votes
- Primary comparator: 8+ training votes

## Traceability

- `support_bin_raw.csv`: canonical raw numeric record, one row per available dataset/seed/support-bin combination
- `support_bin_aggregate.csv`: mean/std aggregation
- `support_bin_gate.csv`: pre-registered gate decision by dataset
- `support_bin_report.md`: concise decision report

There is no optimization or epoch loop in Stage A, so no model-training log is expected. The raw CSV contains all diagnostic outputs used by the gate. It contains 41 rows covering all 15 dataset-seed combinations; fewer than 45 rows occur because the 2–3 vote bin is absent in four combinations.
