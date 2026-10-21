# GAGB v2 Target-Only Minival Result - 2026-06-01

## Hypothesis

GAGB v1 failed the independent-value gate partly because `q_B` relied too directly on noisy local majority. This v2 round keeps GRACE embeddings and GAGB grouping fixed, and only changes the ball-wise soft target so that GAGB behaves as a noise-aware soft-label generator.

## Ball Role

The ball remains a local region. The new role is to produce a more conservative target distribution using noisy labels plus a noisy-label-trained teacher. No view generation, no GAGB construction change, no `Q(B)`, and no parameter sweep over `lambda_h/lambda_t/tau_add` were added.

## Code Changes

- Added a noisy-label-only linear teacher on frozen embeddings.
- Added target-only v2 methods:
  - `gagb_qb_pred_mix`
  - `gagb_qb_teacher_prior`
  - `gagb_qb_conflict_gate`
- Added `topk_gagb_train_matched` as the formal alias for the old train-node matched top-k control.
- Added `topk_gagb_allnode_matched`.
- Added target-level sanity diagnostics for `gagb_shuffled_alpha_matched`.
- Updated method choices, routing, summary formal methods, paired deltas, and unit tests.

## Config

Command:

```powershell
py -3 scripts\run_gate_experiment.py --allow-external-python --python C:\Users\Admin\AppData\Local\Programs\Python\Python312\python.exe --datasets Cora CiteSeer --split-mode ratio --train-ratios 0.2 --val-ratio 0.1 --noise-rates 0.2 0.4 --seeds 0 1 2 3 4 --mode frozen --alphas 0.3 --methods ce global gb gagb gagb_ca_ce emb_only_gagb_local emb_only_gagb_global khop_only kmeans_gagb_matched topk_gagb_train_matched topk_gagb_allnode_matched gagb_shuffled_alpha_matched gagb_qb_pred_mix gagb_qb_teacher_prior gagb_qb_conflict_gate --save-diagnostics
py -3 scripts\summarize_gate_results.py --results-root results --output-dir reports\gagb_v2_target_minival_20260601
```

Protocol:

- datasets: Cora, CiteSeer
- split: ratio train 0.2, validation 0.1, split seed 0
- noise: symmetric 0.2 and 0.4
- seeds: 0, 1, 2, 3, 4
- embedding: frozen GRACE
- alpha: 0.3
- teacher: noisy-label-only linear classifier, 100 epochs, lr 0.01, weight decay 5e-4, temperature 1.0

## Results

Summary files:

- `reports/gagb_v2_target_minival_20260601/gate_summary.csv`
- `reports/gagb_v2_target_minival_20260601/gate_paired_deltas.csv`
- raw logs: `logs/gate/20260601_152739`

Note: the summary CSV may include older formal rows from `results` for other train ratios. The decision below uses only `train_ratio=0.2`, `val_ratio=0.1`, `alpha=0.3`.

### Main method: `gagb_qb_conflict_gate`

Paired accuracy deltas for `gagb_qb_conflict_gate`:

| Dataset | Noise | Baseline | Delta | Positive seeds |
|---|---:|---|---:|---:|
| Cora | 0.2 | gb | -0.00011 | 2/5 |
| Cora | 0.2 | gagb | -0.00021 | 2/5 |
| Cora | 0.2 | khop_only | -0.00032 | 2/5 |
| Cora | 0.2 | gagb_shuffled_alpha_matched | -0.00042 | 2/5 |
| Cora | 0.4 | gb | -0.00169 | 2/5 |
| Cora | 0.4 | gagb | +0.00063 | 2/5 |
| Cora | 0.4 | khop_only | +0.00201 | 4/5 |
| Cora | 0.4 | gagb_shuffled_alpha_matched | +0.00327 | 3/5 |
| CiteSeer | 0.2 | gb | +0.00747 | 4/5 |
| CiteSeer | 0.2 | gagb | +0.00713 | 4/5 |
| CiteSeer | 0.2 | khop_only | +0.00043 | 3/5 |
| CiteSeer | 0.2 | gagb_shuffled_alpha_matched | +0.00816 | 3/5 |
| CiteSeer | 0.4 | gb | +0.00034 | 2/5 |
| CiteSeer | 0.4 | gagb | -0.00009 | 2/5 |
| CiteSeer | 0.4 | khop_only | -0.00077 | 1/5 |
| CiteSeer | 0.4 | gagb_shuffled_alpha_matched | -0.00043 | 2/5 |

Success criteria check:

| Criterion | Result |
|---|---|
| beat `gb` in at least 3/4 settings | fail, 2/4 |
| beat v1 `gagb` in at least 3/4 settings | fail, 2/4 |
| beat `khop_only` in at least 2/4 settings | pass, 2/4 |
| beat `gagb_shuffled_alpha_matched` in at least 2/4 settings | pass, 2/4 |
| `noise_correction_rate > 0` and above v1 `gagb` | fail, NCR remains 0.0 |
| clean damage controlled | pass, clean damage remains 0.0 |

### Mechanism diagnostics

`gagb_qb_conflict_gate` was conservative:

| Dataset | Noise | Coverage | Mean gated train nodes | NCR | CDR | AUROC |
|---|---:|---:|---:|---:|---:|---:|
| Cora | 0.2 | 0.0631 | 34.2 | 0.0 | 0.0 | 0.6237 |
| Cora | 0.4 | 0.0343 | 18.6 | 0.0 | 0.0 | 0.5399 |
| CiteSeer | 0.2 | 0.0547 | 36.4 | 0.0 | 0.0 | 0.5738 |
| CiteSeer | 0.4 | 0.0324 | 21.6 | 0.0 | 0.0 | 0.5314 |

The gate changed only a small subset of train nodes, but those changes did not produce measurable correction of flipped labels. `gagb_qb_pred_mix` and `gagb_qb_teacher_prior` also failed to improve NCR/CDR.

`gagb_shuffled_alpha_matched` sanity remained clean:

| Diagnostic | Result |
|---|---:|
| `alpha_match_max_abs_diff_mean` | 0.0 |
| `coverage_match_abs_diff_mean` | 0.0 |
| `target_changed_mask_mismatch_rate_mean` | 0.0 |

## Decision

Target-only v2 does not satisfy the success criteria. The conflict gate gives a clear positive accuracy signal on CiteSeer 0.2 and a smaller signal on Cora 0.4, but it fails the main requirement against `gb` and v1 `gagb` across 3/4 settings, and it does not improve noise correction rate.

Supported:

- The v2 target pipeline works and is auditable.
- Teacher-aware target construction can be run without clean labels.
- The conflict gate is conservative and avoids clean damage.
- CiteSeer 0.2 shows a real paired positive signal worth noting.

Not supported:

- Target-only GAGB v2 is not yet a main method.
- Mixing teacher prediction into `q_B` does not by itself solve noisy-label self-confirmation.
- The current conflict gate does not generate actual correction signal under the measured NCR.

Next decision:

Stop target-only v2 as a main route unless the next change directly explains why NCR is zero. The most reasonable next experiment is ball-quality v2: add prediction consistency and an explicit `Q(B)` before using a ball to generate correction targets.
