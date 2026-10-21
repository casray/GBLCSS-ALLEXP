# Phase 17 Efficiency Report

Date: 2026-06-15

## Research Record

- Hypothesis: SoftDist [X||H] should be cheaper than online graph-denoising methods because ball targets are built once and the classifier is a standard GCN.
- Granular-ball role: one-shot preprocessing for soft target construction. Balls are not iteratively refreshed during classifier training.
- Code changes: added `scripts/measure_efficiency.py` and `tests/test_measure_efficiency.py`; no predictor or mechanism behavior changed.
- Config: Cora, uniform noise 0.4, percent split 0.2/0.1/0.7, seed 0, GPU `cuda:0`.
- Success standard: provide reproducible wall-clock and parameter-count evidence. This is a support table, not a performance gate.
- Raw outputs: `efficiency_table.csv`, `efficiency_dryrun.csv`, `efficiency_20260615.log`.

## Results

| Method | Iterative denoising? | Trainable params | Wall time | Predictor time | Status |
|---|---:|---:|---:|---:|---|
| GCN [X||H] | no | 100,423 | 2.17s | 0.07s | ok |
| SoftDist [X||H] | no, one-shot ball target | 100,423 | 7.89s | 6.08s | ok |
| NRGNN | yes | 464,846 | 24.74s | 12.02s | ok |
| UnionNET | yes / per-epoch label aggregation | 92,231 | >1200s | n/a | timeout |

Interpretation:

- SoftDist has the same trainable parameter count as GCN [X||H]; the extra cost is one-shot ball target construction and target bookkeeping.
- NRGNN has about 4.6x the trainable parameters of SoftDist and about 3.1x the single-run wall time in this measurement.
- UnionNET timed out after 20 minutes on this Cora seed0 timing run; the parameter count came from dry-run initialization and should be reported separately from completed timings.
- H pretraining cost is not included in the downstream wall time. It must be disclosed as an amortizable label-free preprocessing cost reused across noise types, seeds, and downstream splits.

## Decision

The efficiency claim is supported if phrased narrowly: SoftDist [X||H] is a non-iterative downstream denoising method with standard-GCN trainable parameters and lower downstream runtime than NRGNN in this controlled seed0 Cora measurement. Do not claim zero preprocessing cost; explicitly separate amortized SSL pretraining from downstream denoising.
