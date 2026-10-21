# GAGB v2 Gated Nodes Diagnostic Report - 2026-06-01

## Purpose

This diagnostic checks why `gagb_qb_conflict_gate` keeps `noise_correction_rate` at 0 without changing GAGB construction, GRACE embeddings, target training, or hyperparameters.

Clean labels are used only for diagnostics in this report.

## Per-setting Summary

| Dataset | Noise | Gated Ratio | Gated Noisy Ratio | Lift | Possible Before Alpha | Recomputed NCR | Lost By Alpha | Teacher Gated Noisy Clean Acc | Ball Clean Purity | Diagnosis |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| CiteSeer | 0.2000 | 0.0547 | 0.6262 | 3.1355 | 0.8426 | 0.0000 | 0.8426 | 0.8346 | 0.8525 | C. q_mix can point to clean label, but alpha mixing keeps target argmax noisy. |
| CiteSeer | 0.4000 | 0.0324 | 0.8812 | 2.2063 | 0.9032 | 0.0000 | 0.9032 | 0.9032 | 0.8956 | C. q_mix can point to clean label, but alpha mixing keeps target argmax noisy. |
| Cora | 0.2000 | 0.0631 | 0.8101 | 4.0655 | 0.9722 | 0.0000 | 0.9722 | 0.9663 | 0.9335 | C. q_mix can point to clean label, but alpha mixing keeps target argmax noisy. |
| Cora | 0.4000 | 0.0343 | 0.9664 | 2.4139 | 0.9875 | 0.0000 | 0.9875 | 0.9875 | 0.9861 | C. q_mix can point to clean label, but alpha mixing keeps target argmax noisy. |

## Required Questions

### CiteSeer noise=0.2000

1. Gate selects noisy nodes? `gated_noisy_ratio=0.6262`, base train noise rate `0.1997`, lift `3.1355`.
2. q_mix correction direction? `gated_noisy_qmix_argmax_eq_clean_rate=0.8426`, `gated_noisy_qmix_clean_minus_noisy_mean=0.4545`, possible before alpha `0.8426`.
3. Alpha too weak? possible before alpha `0.8426` vs recomputed NCR `0.0000`, lost by alpha `0.8426`.
4. Teacher independent correction? `teacher_gated_noisy_clean_acc=0.8346`, `teacher_gated_argmax_eq_clean_rate=0.5499`, `teacher_gated_argmax_eq_noisy_rate=0.0268`.
5. Gated ball quality? `clean_purity=0.8525`, `noisy_entropy=0.8154`, `train_count=14.2579`.
6. Main NCR=0 reason: C. q_mix can point to clean label, but alpha mixing keeps target argmax noisy.

### CiteSeer noise=0.4000

1. Gate selects noisy nodes? `gated_noisy_ratio=0.8812`, base train noise rate `0.3994`, lift `2.2063`.
2. q_mix correction direction? `gated_noisy_qmix_argmax_eq_clean_rate=0.9032`, `gated_noisy_qmix_clean_minus_noisy_mean=0.4077`, possible before alpha `0.9032`.
3. Alpha too weak? possible before alpha `0.9032` vs recomputed NCR `0.0000`, lost by alpha `0.9032`.
4. Teacher independent correction? `teacher_gated_noisy_clean_acc=0.9032`, `teacher_gated_argmax_eq_clean_rate=0.8103`, `teacher_gated_argmax_eq_noisy_rate=0.0128`.
5. Gated ball quality? `clean_purity=0.8956`, `noisy_entropy=0.9031`, `train_count=17.5939`.
6. Main NCR=0 reason: C. q_mix can point to clean label, but alpha mixing keeps target argmax noisy.

### Cora noise=0.2000

1. Gate selects noisy nodes? `gated_noisy_ratio=0.8101`, base train noise rate `0.1993`, lift `4.0655`.
2. q_mix correction direction? `gated_noisy_qmix_argmax_eq_clean_rate=0.9722`, `gated_noisy_qmix_clean_minus_noisy_mean=0.5212`, possible before alpha `0.9722`.
3. Alpha too weak? possible before alpha `0.9722` vs recomputed NCR `0.0000`, lost by alpha `0.9722`.
4. Teacher independent correction? `teacher_gated_noisy_clean_acc=0.9663`, `teacher_gated_argmax_eq_clean_rate=0.8119`, `teacher_gated_argmax_eq_noisy_rate=0.0297`.
5. Gated ball quality? `clean_purity=0.9335`, `noisy_entropy=0.8060`, `train_count=15.4362`.
6. Main NCR=0 reason: C. q_mix can point to clean label, but alpha mixing keeps target argmax noisy.

### Cora noise=0.4000

1. Gate selects noisy nodes? `gated_noisy_ratio=0.9664`, base train noise rate `0.4004`, lift `2.4139`.
2. q_mix correction direction? `gated_noisy_qmix_argmax_eq_clean_rate=0.9875`, `gated_noisy_qmix_clean_minus_noisy_mean=0.4871`, possible before alpha `0.9875`.
3. Alpha too weak? possible before alpha `0.9875` vs recomputed NCR `0.0000`, lost by alpha `0.9875`.
4. Teacher independent correction? `teacher_gated_noisy_clean_acc=0.9875`, `teacher_gated_argmax_eq_clean_rate=0.9539`, `teacher_gated_argmax_eq_noisy_rate=0.0000`.
5. Gated ball quality? `clean_purity=0.9861`, `noisy_entropy=0.8800`, `train_count=15.9141`.
6. Main NCR=0 reason: C. q_mix can point to clean label, but alpha mixing keeps target argmax noisy.

## Reproduction Command

```powershell
py -3 scripts\diagnose_gagb_v2_gated_nodes.py ^
--allow-external-python ^
--python C:\Users\Admin\AppData\Local\Programs\Python\Python312\python.exe ^
--datasets Cora CiteSeer ^
--split-mode ratio ^
--train-ratios 0.2 ^
--val-ratio 0.1 ^
--noise-rates 0.2 0.4 ^
--seeds 0 1 2 3 4 ^
--mode frozen ^
--alpha 0.3 ^
--method gagb_qb_conflict_gate ^
--output-dir reports\gagb_v2_gated_diagnostic_20260601
```

## Output Files

- `gated_nodes_per_seed.csv`
- `gated_nodes_summary.csv`
- `gated_nodes_diagnostic_report.md`