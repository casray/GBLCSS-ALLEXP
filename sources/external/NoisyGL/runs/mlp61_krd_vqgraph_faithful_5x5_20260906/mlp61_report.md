# MLP-61 report: KRD and VQGraph faithful protocol ports

## Question

Do the official-core KRD and VQGraph protocol ports remain competitive with
the granular-ball consensus teacher when evaluated with paired splits, paired
noise realizations, five seeds, and the manuscript's attribute-only student
inference protocol?

## Execution

- Device: AMD Radeon RX 7700 XT through PyTorch ROCm (`cuda:0`).
- KRD: 50/50 planned runs completed.
- VQGraph: 30/50 planned runs completed. CoraFull and Coauthor-CS were not
  retained because the full-batch port did not complete within the current
  GPU environment.
- No CPU fallback was used.
- KRD source commit: `32be7e42ab65b1d246cf1591d8bc48dfe7fb6b21`.
- VQGraph source commit: `60aa1c99782d542c316a37f0394eba934bcf97b0`.

KRD uses official transductive GCN settings where available. CoraFull uses the
pre-registered Coauthor-CS mapping. VQGraph uses the paper settings for Cora,
CiteSeer, and Amazon-Photo. These are official-core protocol ports under the
manuscript's split/noise protocol, not official clean-label reproductions.

## Results

Each cell is five-seed test accuracy in percent (mean +/- sample standard
deviation). Delta is method minus granular-ball consensus teacher under the
same split and noise realization.

### KRD

| Noise | Dataset | KRD | Ball teacher | Delta (pp) |
|---|---|---:|---:|---:|
| 0.4 | Cora | 70.47 +/- 4.10 | 81.35 +/- 1.05 | -10.88 |
| 0.4 | CiteSeer | 68.40 +/- 1.22 | 69.27 +/- 1.57 | -0.87 |
| 0.4 | CoraFull | 56.92 +/- 1.25 | 57.29 +/- 0.24 | -0.37 |
| 0.4 | Coauthor-CS | 92.34 +/- 0.14 | 93.83 +/- 0.21 | -1.49 |
| 0.4 | Amazon-Photo | 88.55 +/- 1.00 | 90.40 +/- 0.77 | -1.86 |
| 0.6 | Cora | 54.99 +/- 6.71 | 71.34 +/- 3.91 | -16.35 |
| 0.6 | CiteSeer | 61.44 +/- 2.21 | 58.93 +/- 4.62 | +2.51 |
| 0.6 | CoraFull | 49.98 +/- 0.70 | 52.32 +/- 1.26 | -2.34 |
| 0.6 | Coauthor-CS | 88.50 +/- 0.71 | 92.40 +/- 0.65 | -3.89 |
| 0.6 | Amazon-Photo | 80.18 +/- 1.14 | 86.81 +/- 1.94 | -6.63 |

KRD is higher in one of ten aggregate cells (CiteSeer at uniform 0.6). Across
paired seeds, it is higher in 6/50 runs. Its reliability update is active in
all runs and reports no update failure.

### VQGraph

| Noise | Dataset | VQGraph | Ball teacher | Delta (pp) |
|---|---|---:|---:|---:|
| 0.4 | Cora | 69.53 +/- 5.30 | 81.35 +/- 1.05 | -11.82 |
| 0.4 | CiteSeer | 59.15 +/- 4.97 | 69.27 +/- 1.57 | -10.12 |
| 0.4 | Amazon-Photo | 65.02 +/- 30.72 | 90.40 +/- 0.77 | -25.38 |
| 0.6 | Cora | 50.70 +/- 4.45 | 71.34 +/- 3.91 | -20.63 |
| 0.6 | CiteSeer | 39.14 +/- 4.60 | 58.93 +/- 4.62 | -19.79 |
| 0.6 | Amazon-Photo | 66.05 +/- 27.24 | 86.81 +/- 1.94 | -20.77 |

VQGraph is lower in all six completed aggregate cells and is higher in 1/30
paired runs. Amazon-Photo has severe seed instability. The two collapsed
uniform-0.4 runs use only one or two active codes and obtain about 31.4%
accuracy. Low active-code counts also accompany the weakest uniform-0.6 runs.
This is diagnostic association, not a causal conclusion.

## Protocol audit

- Retained rows: 80/80 finite and successful.
- Seed coverage: five seeds in every retained method/dataset/noise cell.
- Split hash: 80/80 matches to the granular-ball teacher.
- Noise hash: 80/80 matches to the granular-ball teacher.
- Training graph use: 80/80 true.
- Student inference graph use: 80/80 false.
- Checkpoint selection: 80/80 uses clean validation.
- KRD reliability updates: positive in every run.
- VQGraph active codes: positive in every retained run.

## VQGraph scale limitation

The retained VQGraph implementation preserves its graph tokenizer,
vector-quantization objectives, and paper-level parameter settings, but uses a
full-batch PyG protocol port. Its feature-dependent layers and graph
reconstruction tensors make CoraFull and Coauthor-CS infeasible in this form.
The official repository relies on a DGL neighbor-sampled runtime that is not
compatible with the installed Windows PyTorch 2.9/ROCm environment. Reducing
feature dimensionality, codebook size, or architecture would answer a
different question, so no such substitute was introduced.

## Decision

1. The five-dataset KRD row can be reported as an official-core protocol port,
   with the CoraFull parameter mapping disclosed.
2. VQGraph does not support a complete five-dataset row from this experiment.
   It should not be presented as a faithful five-dataset reproduction. The
   defensible options are a three-dataset supplementary result with the scale
   limitation stated, or removal of its quantitative main-table row.
3. The completed evidence does not weaken the granular-ball consensus teacher
   on average. KRD is stronger only on CiteSeer at uniform 0.6; VQGraph is both
   lower and less stable in the retained cells.
4. No manuscript result should be changed until the author decides how to
   handle the incomplete VQGraph row.

## Files

- `mlp61_raw_completed.csv`: 80 retained per-seed rows with method diagnostics.
- `mlp61_aggregate.csv`: 16 method/dataset/noise summaries.
- `mlp61_paired_to_ball_teacher.csv`: 80 paired seed comparisons.
- `mlp61_protocol_audit.csv`: protocol and diagnostic checks.
- `krd_u04_seeds0_4.csv`, `krd_u06_seeds0_4.csv`: complete KRD outputs.
- `vqgraph_u04_seeds0_4_runnable.csv`,
  `vqgraph_u06_seeds0_4_runnable.csv`: retained VQGraph outputs.

## Verification

- `pytest tests/test_mlp60_krd_vqgraph_faithful.py tests/test_standard_runner_baselines.py -q`:
  26 tests and 4 subtests passed.
- `py_compile` passed for `predictor/G2MFaithful_Predictor.py` and
  `scripts/run_standard_downstream.py`.
- Final CSV audit passed all row-count, status, pairing, hash, graph-use,
  checkpoint-selection, and method-diagnostic checks.
- No MLP-61 experiment process remained active after verification.
