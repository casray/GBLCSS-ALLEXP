# Phase 14 NoisyGL Baseline Completion Report

Date: 2026-06-12

## Protocol

- Datasets: Cora, CiteSeer.
- Split: `percent`, train/val/test = 0.2/0.1/0.7.
- Noise: `uniform`, noise_rate = 0.4.
- Seeds: 0, 1, 2, 3, 4.
- Device: ROCm GPU via `.venv-rocm`.
- Methods: `unionnet`, `tss`, `cgnn`, `crgnn`, `r2lp`.
- Configs: NoisyGL built-in configs, no hyperparameter tuning.

## Results

Existing anchors:

| Method | Cora | CiteSeer |
|---|---:|---:|
| SoftDist [X||H] | 0.8137 ± 0.0067 | 0.6891 ± 0.0159 |
| NRGNN | 0.8107 ± 0.0230 | 0.7027 ± 0.0108 |
| GCN+CE | 0.7842 ± 0.0151 | 0.6530 ± 0.0121 |

New baseline completion:

| Method | Cora | Delta vs SoftDist | CiteSeer | Delta vs SoftDist | Status |
|---|---:|---:|---:|---:|---|
| UnionNET | 0.7837 ± 0.0161 | -3.00pp | 0.6867 ± 0.0136 | -0.24pp | ok |
| TSS | 0.6385 ± 0.0250 | -17.52pp | failed | n/a | CiteSeer HIPBLAS alloc failure |
| CGNN | 0.7389 ± 0.0380 | -7.48pp | 0.6135 ± 0.0322 | -7.56pp | ok |
| CRGNN | 0.7690 ± 0.0101 | -4.47pp | 0.6476 ± 0.0188 | -4.15pp | ok |
| R2LP | 0.5745 ± 0.0266 | -23.92pp | 0.4653 ± 0.0574 | -22.38pp | ok |

Raw rows:

- `baseline_completion_all_rows.csv`
- `baseline_completion_aggregate.csv`

## UnionNET vs SoftDist

UnionNET is the closest related baseline in this batch because it also uses a support set around each train node to estimate label reliability/correction. The empirical comparison separates the two mechanisms:

- UnionNET constructs node-level support sets directly from feature/neighborhood similarity and uses attention-style aggregation for reweighting/correction during GCN training.
- SoftDist [X||H] first builds label-blind semantic balls in frozen GRACE space, then uses only train noisy-label distributions inside each ball to produce soft targets for a fresh GCN with `[X||H]` input.
- UnionNET has higher node-level resolution, but its support construction is expensive and unstable in this protocol: CiteSeer UnionNET took much longer than other methods, and still only reached 0.6867 ± 0.0136.
- SoftDist is coarser but more stable and stronger on Cora, while matching/slightly exceeding UnionNET on CiteSeer.

This supports the related-work positioning: SoftDist is not a direct reimplementation of UnionNET. It trades per-node support attention for a frozen SSL semantic partition plus simple soft-label preprocessing.

## Failure Record

TSS on CiteSeer failed for all five seeds with:

```text
RuntimeError: CUDA error: HIPBLAS_STATUS_ALLOC_FAILED when calling hipblasStrsm(...)
```

This is recorded as an execution failure under the mandated ROCm GPU protocol. The method code was not modified.

## Decision

The completed baselines do not weaken the current main claim:

- None of the completed methods beats SoftDist [X||H] on Cora.
- None of the completed methods beats SoftDist [X||H] on CiteSeer.
- UnionNET, the closest mechanism-level neighbor, is below SoftDist on both datasets, though CiteSeer is close.

PubMed was not run for these five baselines in this pass. The mandatory Cora/CiteSeer completion already consumed multiple long ROCm runs, and PubMed was optional in the plan.

## Research Record

- Hypothesis: additional NoisyGL baselines may reveal a closer or stronger competitor than the already-run methods.
- Granular-ball role: none in these baselines; this is comparison-only.
- Code change: extended `scripts/run_standard_downstream.py` to register existing NoisyGL predictors and record failures without stopping the batch.
- Config change: none to baseline configs.
- Success/failure standard: report all methods under the same percent split/noise/seed protocol; failed methods must record error reason.
- Result: UnionNET/CGNN/CRGNN/R2LP completed; TSS completed on Cora but failed on CiteSeer under ROCm.
- Final decision: add these rows to the comparison table; keep SoftDist [X||H] as the main method.
