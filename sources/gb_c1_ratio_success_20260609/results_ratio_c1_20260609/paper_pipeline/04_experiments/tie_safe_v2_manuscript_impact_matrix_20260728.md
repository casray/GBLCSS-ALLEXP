# Tie-safe v2 manuscript impact matrix

## Frozen method change

The paper method uses `gb.tie_mode=distribution_only`. For a tied maximum vote,
the target retains the full local class distribution and removes only the
class-index-dependent majority one-hot term. The change affects results only
when a training target contains a positive-weight majority component.

## Manuscript-level impact

| v29 asset | Tie-safe impact | Required action for v31 |
|---|---|---|
| Table 0 dataset statistics | none | reuse |
| Table 1, GB-LCSS and shuffled rows | direct | use the completed immutable 500-row v2 grid |
| Table 1, MLP+CE and JoCoR rows | none | use the same-round v2 grid |
| Table 1, Small-loss/Co-teaching/SCE/ELR/LogitClip | none algorithmically, but current hash lineage is incomplete | rerun under the frozen protocol or exclude from paired current-round claims |
| Table 2 breadth datasets, GB-LCSS and shuffled | direct | rerun uniform 0.4 with `distribution_only` |
| Table 2 robust baselines | none | reuse only after split/noise lineage audit |
| Table 3 base GB-LCSS row | direct | inherit updated Table 1 values |
| Table 3 teacher-distillation rows | no training-target impact when `target_beta=0` | reuse after config audit |
| Section 5.3.1 default/LOO/shuffled targets | direct | rerun six-dataset uniform 0.4 table |
| Uniform/prior label smoothing controls | none | reuse after split/noise lineage audit |
| Section 5.3.2 kNN/kmeans comparisons | possible for `target_beta>0` | audit builder; rerun only the comparisons retained in v31 |
| Section 5.3.3 representation-space comparisons | direct for any GB-LCSS target with `target_beta>0` | retain only after tie-safe rerun or rewrite as historical evidence |
| Section 5.3.4 linear GB-LCSS | direct | rerun the retained linear GB-LCSS rows or remove exact values |
| Section 5.3.5 H-access controls | GB-LCSS row only | inherit updated v2 value; reuse H-access controls |
| Section 5.3.6 label-rate / clean-label results | direct for GB-LCSS rows | rerun only if exact quantitative claims remain in v31 |
| Section 5.4 GRACE-seed experiment | direct | rerun or state as historical min-index evidence; do not relabel |
| Table 4 runtime | negligible numerical target effect but method identity changed | refresh GB-LCSS seed-0 timing if table remains |
| Table 5 ball/random/kNN teachers | none when `target_beta=0` | reuse; verify every retained teacher config |
| Table 6 graph-method references and beta-0 distilled student | none | reuse after protocol audit |
| Figures generated from target distributions or predictions | possible | regenerate if retained |
| Closed exploratory directions absent from v31 | irrelevant | do not rerun |

## Execution priority

1. Current main performance and same-backbone baseline completeness.
2. Breadth Table 2 and the LOO/smoothing mechanism table.
3. Nearest-work formal grid.
4. Only those secondary ablations whose numerical claims remain in v31.
5. Manuscript rewrite and a final stale-number search.

## Claim rules

- Historical `min_index` results remain valid only as historical evidence and
  must not be relabeled as tie-safe results.
- No tie-safe accuracy-improvement claim is allowed without a direct paired
  comparison. The representative experiment supports only the statement that
  the correction removes class-index dependence while the updated main
  conclusions remain qualitatively stable.
- Any adapted nearest-work method must retain its explicit implementation
  identity in tables, captions, and prose.

## Execution status (2026-07-28)

| Asset | Closure |
|---|---|
| Table 1 GB-LCSS/CE/JoCoR/shuffled | complete: immutable tie-safe v2 500-row grid |
| Table 1 remaining five generic baselines | complete: 625/625 rows in `v31_baseline_refresh_20260728` |
| Table 2 GB-LCSS/shuffled breadth rows | complete: 60/60 rows in `v31_tiesafe_affected_20260728` |
| LOO and LOO-shuffled mechanism table | complete: 60/60 rows in `v31_tiesafe_affected_20260728` |
| Uniform/prior smoothing controls | complete: 60/60 rows in `v31_tiesafe_affected_20260728` |
| kNN/kmeans and representation-space diagnostics | retained only as explicitly labeled historical diagnostics; no current tie-safe ranking claim |
| Linear-capacity and label-rate paragraphs | retained as historical scope diagnostics; they are not used in the current tie-safe main ranking |
| Table 3 base GB-LCSS row | pending replacement by the formal nearest-work table |
| Table 4 runtime | pending seed-0 refresh after the formal nearest-work grid releases the GPU |
| Table 5 teacher-distillation rows | audited as `target_beta=0`; tie-safe majority branch is inactive and results are reusable |
| Table 6 graph-method references | no direct tie-safe target dependency; retained after protocol audit |

All completed reruns passed train-only label-noise checks:
`modified_val_count=modified_test_count=0` and
`modified_count=modified_train_count`.
