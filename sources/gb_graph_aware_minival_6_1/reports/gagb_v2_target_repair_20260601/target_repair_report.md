# GAGB v2 Target Update Repair Minival - 2026-06-01

## Research Hypothesis

The previous gated-node diagnostic showed that `gagb_qb_conflict_gate` already had a strong pre-alpha correction signal, but the old convex update

```text
target_i = (1 - alpha_i) * one_hot(noisy_y_i) + alpha_i * q_mix
```

kept `alpha_i` too small to change the target argmax. This run tests whether repairing only the target update can turn the pre-alpha signal into positive post-target noise correction.

## Ball Role

The graph-aware granular ball remains a frozen local region used to build `q_mix`. This run does not change GRACE, GAGB construction, teacher training, graph augmentation, `Q(B)`, prediction consistency, split, or noisy-label generation.

## Code Changes

- Added target-repair variants:
  - `gagb_qb_conflict_gate_alpha03`: old current version, kept as baseline.
  - `gagb_qb_conflict_gate_alpha06`
  - `gagb_qb_conflict_gate_alpha07`
  - `gagb_qb_conflict_gate_alpha09`
  - `gagb_qb_conflict_gate_replace_qmix`
  - `gagb_qb_conflict_gate_sharpen_qmix_T05`
  - `gagb_qb_conflict_gate_margin_replace`
- `alpha06/07/09` use fixed alpha on gated nodes. This is intentional: a smoke run showed that leaving the old geometry scaling active reduced actual `alpha_max_actual` below 0.5 and failed to test the intended repair.
- Added diagnostics:
  - `gated_train_ratio`
  - `gated_noisy_ratio`
  - `possible_correction_before_alpha`
  - `lost_by_alpha_rate`
  - `target_argmax_change_rate`
  - `teacher_gated_noisy_clean_acc`
  - `qmix_argmax_eq_clean_rate`
  - `target_argmax_eq_clean_rate`
- Clean labels are used only in diagnostics and evaluation, not in teacher training, target construction, classifier training, ball construction, or embedding extraction.

## Experimental Protocol

Command:

```powershell
py -3 scripts\run_gate_experiment.py --allow-external-python --python C:\Users\Admin\AppData\Local\Programs\Python\Python312\python.exe --datasets Cora CiteSeer --split-mode ratio --train-ratios 0.2 --val-ratio 0.1 --noise-rates 0.2 0.4 --seeds 0 1 2 3 4 --mode frozen --alphas 0.3 --methods ce gb gagb khop_only gagb_qb_conflict_gate_alpha03 gagb_qb_conflict_gate_alpha06 gagb_qb_conflict_gate_alpha07 gagb_qb_conflict_gate_alpha09 gagb_qb_conflict_gate_replace_qmix gagb_qb_conflict_gate_sharpen_qmix_T05 gagb_qb_conflict_gate_margin_replace --save-diagnostics
py -3 scripts\summarize_gate_results.py --results-root results --output-dir reports\gagb_v2_target_repair_20260601
```

Settings:

- datasets: Cora, CiteSeer
- split: ratio train 0.2, validation 0.1, split seed 0
- noise: symmetric 0.2 and 0.4
- seeds: 0, 1, 2, 3, 4
- embedding: frozen GRACE
- teacher: noisy-label-only linear teacher
- beta for `q_mix`: 0.5
- conflict gate: `q_mix[noisy_y_i] <= 0.35` and `max(q_mix) >= 0.55`
- margin for `margin_replace`: 0.2

Output files:

- `reports/gagb_v2_target_repair_20260601/gate_summary.csv`
- `reports/gagb_v2_target_repair_20260601/gate_paired_deltas.csv`
- raw logs: `logs/gate/20260601_171423`

## Main Results

### Accuracy, NCR, CDR

| Dataset | Noise | Method | Test Acc Mean | Test Acc Std | NCR | CDR | Possible Before Alpha | Lost By Alpha |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| Cora | 0.2 | `gb` | 0.8118 | 0.0051 | 0.0000 | 0.0000 | - | - |
| Cora | 0.2 | `gagb` | 0.8119 | 0.0053 | 0.0000 | 0.0000 | - | - |
| Cora | 0.2 | `alpha03` | 0.8117 | 0.0059 | 0.0000 | 0.0000 | 0.9722 | 0.9722 |
| Cora | 0.2 | `alpha07` | 0.8080 | 0.0081 | 0.2074 | 0.0115 | 0.9722 | 0.1657 |
| Cora | 0.2 | `alpha09` | 0.8120 | 0.0081 | 0.2519 | 0.0143 | 0.9722 | 0.0000 |
| Cora | 0.2 | `replace_qmix` | 0.8136 | 0.0049 | 0.2519 | 0.0143 | 0.9722 | 0.0000 |
| Cora | 0.2 | `margin_replace` | 0.8136 | 0.0049 | 0.2519 | 0.0143 | 0.9722 | 0.0000 |
| Cora | 0.4 | `gb` | 0.7974 | 0.0054 | 0.0000 | 0.0000 | - | - |
| Cora | 0.4 | `gagb` | 0.7950 | 0.0062 | 0.0000 | 0.0000 | - | - |
| Cora | 0.4 | `alpha03` | 0.7957 | 0.0079 | 0.0000 | 0.0000 | 0.9875 | 0.9875 |
| Cora | 0.4 | `alpha07` | 0.7968 | 0.0038 | 0.0664 | 0.0006 | 0.9875 | 0.2037 |
| Cora | 0.4 | `alpha09` | 0.7966 | 0.0073 | 0.0811 | 0.0025 | 0.9875 | 0.0000 |
| Cora | 0.4 | `replace_qmix` | 0.7961 | 0.0069 | 0.0811 | 0.0025 | 0.9875 | 0.0000 |
| Cora | 0.4 | `margin_replace` | 0.7961 | 0.0069 | 0.0811 | 0.0025 | 0.9875 | 0.0000 |
| CiteSeer | 0.2 | `gb` | 0.7009 | 0.0127 | 0.0000 | 0.0000 | - | - |
| CiteSeer | 0.2 | `gagb` | 0.7012 | 0.0127 | 0.0000 | 0.0000 | - | - |
| CiteSeer | 0.2 | `alpha03` | 0.7083 | 0.0095 | 0.0000 | 0.0000 | 0.8426 | 0.8426 |
| CiteSeer | 0.2 | `alpha07` | 0.7097 | 0.0045 | 0.1203 | 0.0188 | 0.8426 | 0.1451 |
| CiteSeer | 0.2 | `alpha09` | 0.7101 | 0.0051 | 0.1444 | 0.0255 | 0.8426 | 0.0000 |
| CiteSeer | 0.2 | `replace_qmix` | 0.7077 | 0.0074 | 0.1444 | 0.0255 | 0.8426 | 0.0000 |
| CiteSeer | 0.2 | `margin_replace` | 0.7077 | 0.0074 | 0.1444 | 0.0255 | 0.8426 | 0.0000 |
| CiteSeer | 0.4 | `gb` | 0.6976 | 0.0097 | 0.0000 | 0.0000 | - | - |
| CiteSeer | 0.4 | `gagb` | 0.6980 | 0.0099 | 0.0000 | 0.0000 | - | - |
| CiteSeer | 0.4 | `alpha03` | 0.6979 | 0.0090 | 0.0000 | 0.0000 | 0.9032 | 0.9032 |
| CiteSeer | 0.4 | `alpha07` | 0.6991 | 0.0091 | 0.0459 | 0.0040 | 0.9032 | 0.2743 |
| CiteSeer | 0.4 | `alpha09` | 0.6991 | 0.0101 | 0.0617 | 0.0075 | 0.9032 | 0.0000 |
| CiteSeer | 0.4 | `replace_qmix` | 0.7005 | 0.0079 | 0.0617 | 0.0075 | 0.9032 | 0.0000 |
| CiteSeer | 0.4 | `margin_replace` | 0.7005 | 0.0079 | 0.0617 | 0.0075 | 0.9032 | 0.0000 |

### Paired Accuracy Deltas

| Method | vs `alpha03` | vs `gb` | vs `gagb` | vs `khop_only` |
|---|---:|---:|---:|---:|
| `alpha06` | 3/4 settings positive | 2/4 | 3/4 | 2/4 |
| `alpha07` | 3/4 settings positive | 2/4 | 3/4 | 3/4 |
| `alpha09` | 4/4 settings positive | 3/4 | 4/4 | 4/4 |
| `replace_qmix` | 3/4 settings positive | 3/4 | 4/4 | 3/4 |
| `sharpen_qmix_T05` | 0/4 settings positive | 1/4 | 1/4 | 0/4 |
| `margin_replace` | 3/4 settings positive | 3/4 | 4/4 | 3/4 |

Important paired deltas:

- `alpha09` vs `alpha03`: positive in all four settings, but many deltas are small.
- `replace_qmix` and `margin_replace` beat `gagb` in all four settings and beat `gb` in three settings.
- Cora 0.4 remains the weak setting: `replace_qmix` / `margin_replace` are still below `gb` by about 0.0013 mean accuracy.
- `sharpen_qmix_T05` is consistently harmful or weak and should not be the next main variant.

## Success Criteria Check

Hard criteria:

| Criterion | Result |
|---|---|
| `NCR > 0` | pass for `alpha07`, `alpha09`, `replace_qmix`, `margin_replace` in all four settings |
| `CDR <= 0.05` | pass; max CDR is 0.0255 |
| at least 2/4 settings beat `alpha03` | pass for `alpha06`, `alpha07`, `alpha09`, `replace_qmix`, `margin_replace`; fail for `sharpen_qmix_T05` |
| at least 2/4 settings beat v1 `gagb` | pass for all except `sharpen_qmix_T05` |

Strong criteria:

| Criterion | Result |
|---|---|
| at least 3/4 settings beat `alpha03` | pass for `alpha06`, `alpha07`, `alpha09`, `replace_qmix`, `margin_replace`; fail for `sharpen_qmix_T05` |
| at least 3/4 settings beat `gb` | pass for `alpha09`, `replace_qmix`, `margin_replace`; fail for `alpha06`, `alpha07`, `sharpen_qmix_T05` |
| at least 2/4 settings beat shuffled matched | not tested in this repair run |
| `CDR <= 0.03` | pass |

## Interpretation

Supported conclusions:

1. The previous `NCR=0` was not evidence that GAGB had no correction signal. The target update was blocking that signal.
2. Repairing the target update can make post-target NCR positive while keeping CDR below 0.03 on this minival.
3. `alpha09`, `replace_qmix`, and `margin_replace` are the only repair variants that pass the main accuracy gate against `gb` in at least 3/4 settings.
4. `sharpen_qmix_T05` should be dropped for now.

Hypotheses, not yet conclusions:

1. `replace_qmix` / `margin_replace` may be better than large-alpha mixing because they remove the residual noisy one-hot mass.
2. `margin_replace` did not differ from `replace_qmix` in this run, likely because most replaced gated nodes already had a large q_mix margin. This needs margin distribution diagnostics before claiming the margin gate is useful.
3. Accuracy gains remain small relative to seed std in several settings, so the result is mechanism-positive and method-promising, not yet paper-ready.

Unsupported claims:

1. This does not prove GAGB has independent value over all fair controls. The repair run did not include `gagb_shuffled_alpha_matched`, `kmeans_gagb_matched`, or `topk_gagb_train_matched`.
2. This does not justify changing GRACE, GAGB construction, teacher architecture, or graph augmentation yet.
3. This does not support using hard same-ball labels or prototype pulling.

## Decision

This repair run should continue, but only as a controlled target-update direction.

Recommended next minimal check:

1. Add matched shuffled controls for the strongest variants:
   - `gagb_shuffled_alpha09_matched`
   - `gagb_shuffled_replace_matched`
   - optionally `gagb_shuffled_margin_replace_matched`
2. Add margin distribution diagnostics for `margin_replace`.
3. Keep `replace_qmix` and `margin_replace`; drop `sharpen_qmix_T05`.
4. Do not expand datasets or noise types until the shuffled/size/topk controls are restored.

Current status:

- Mechanism-positive: yes.
- Main-method-ready: no.
- Next action: fair-control repair v2.1, not ball-quality v2 yet.
