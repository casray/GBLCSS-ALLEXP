# GAGB Fair-Comparison Minival Result - 2026-06-01

## Hypothesis

The previous GAGB minival was not enough to judge independent value because several controls were not matched to GAGB. This patch asks whether the same GAGB v1 still has value after adding GAGB-matched kmeans/top-k controls, embedding-only greedy controls, topology-only control, and an alpha-matched shuffled control.

## Ball Role

The ball remains a local soft-label and reliability region. No prediction consistency, Q(B), RTGNN integration, PubMed, pair noise, or alpha tuning was added.

## Code Changes

- Added `candidate_scope="khop" | "global"` to the graph-aware greedy builder.
- Added `emb_only_gagb_local`, `emb_only_gagb_global`, `kmeans_gagb_matched`, `topk_gagb_matched`, `gagb_shuffled_size_only`, and `gagb_shuffled_alpha_matched`.
- Added alpha distribution diagnostics and alpha-match sanity diagnostics.
- Updated method choices and summary pairings.

## Config

Command:

```powershell
py -3 scripts\run_gate_experiment.py --allow-external-python --python C:\Users\Admin\AppData\Local\Programs\Python\Python312\python.exe --datasets Cora CiteSeer --split-mode ratio --train-ratios 0.2 --val-ratio 0.1 --noise-rates 0.2 0.4 --seeds 0 1 2 3 4 --mode frozen --alphas 0.3 --methods ce global gb gb_ca_ce gagb gagb_ca_ce emb_only_gagb_local emb_only_gagb_global khop_only kmeans_gagb_matched topk_gagb_matched gagb_shuffled_size_only gagb_shuffled_alpha_matched --save-diagnostics
py -3 scripts\summarize_gate_results.py --results-root results --output-dir reports\gagb_fair_minival_20260601
```

Protocol:

- datasets: Cora, CiteSeer
- split: ratio train 0.2, validation 0.1, split seed 0
- noise: symmetric 0.2 and 0.4
- seeds: 0, 1, 2, 3, 4
- embedding: frozen GRACE

## Results

Summary files:

- `reports/gagb_fair_minival_20260601/gate_summary.csv`
- `reports/gagb_fair_minival_20260601/gate_paired_deltas.csv`
- raw logs: `logs/gate/20260601_144045`

Alpha-matched shuffled sanity:

| Dataset | Noise | alpha diff | coverage diff |
|---|---:|---:|---:|
| Cora | 0.2 | 0.0 | 0.0 |
| Cora | 0.4 | 0.0 | 0.0 |
| CiteSeer | 0.2 | 0.0 | 0.0 |
| CiteSeer | 0.4 | 0.0 | 0.0 |

Paired accuracy deltas for `gagb`:

| Dataset | Noise | Baseline | Delta | Positive seeds |
|---|---:|---|---:|---:|
| Cora | 0.2 | gb | +0.00011 | 2/5 |
| Cora | 0.2 | emb_only_gagb_local | -0.00053 | 1/5 |
| Cora | 0.2 | emb_only_gagb_global | +0.00063 | 3/5 |
| Cora | 0.2 | khop_only | -0.00011 | 3/5 |
| Cora | 0.2 | kmeans_gagb_matched | +0.00063 | 2/5 |
| Cora | 0.2 | topk_gagb_matched | +0.00127 | 3/5 |
| Cora | 0.2 | gagb_shuffled_alpha_matched | -0.00021 | 2/5 |
| Cora | 0.4 | gb | -0.00232 | 1/5 |
| Cora | 0.4 | emb_only_gagb_local | +0.00084 | 3/5 |
| Cora | 0.4 | emb_only_gagb_global | +0.00211 | 3/5 |
| Cora | 0.4 | khop_only | +0.00137 | 3/5 |
| Cora | 0.4 | kmeans_gagb_matched | -0.00116 | 2/5 |
| Cora | 0.4 | topk_gagb_matched | +0.00665 | 4/5 |
| Cora | 0.4 | gagb_shuffled_alpha_matched | +0.00264 | 2/5 |
| CiteSeer | 0.2 | gb | +0.00034 | 3/5 |
| CiteSeer | 0.2 | emb_only_gagb_local | +0.00026 | 3/5 |
| CiteSeer | 0.2 | emb_only_gagb_global | -0.00043 | 2/5 |
| CiteSeer | 0.2 | khop_only | -0.00670 | 1/5 |
| CiteSeer | 0.2 | kmeans_gagb_matched | -0.00490 | 1/5 |
| CiteSeer | 0.2 | topk_gagb_matched | -0.00129 | 3/5 |
| CiteSeer | 0.2 | gagb_shuffled_alpha_matched | +0.00103 | 2/5 |
| CiteSeer | 0.4 | gb | +0.00043 | 3/5 |
| CiteSeer | 0.4 | emb_only_gagb_local | -0.00034 | 2/5 |
| CiteSeer | 0.4 | emb_only_gagb_global | -0.00034 | 2/5 |
| CiteSeer | 0.4 | khop_only | -0.00069 | 1/5 |
| CiteSeer | 0.4 | kmeans_gagb_matched | -0.00034 | 2/5 |
| CiteSeer | 0.4 | topk_gagb_matched | +0.00447 | 4/5 |
| CiteSeer | 0.4 | gagb_shuffled_alpha_matched | -0.00034 | 2/5 |

## Decision

Current GAGB v1 does not satisfy the independent-value criterion. It beats ordinary `gb` on 3 of 4 settings by mean, but it does not consistently beat topology-only, GAGB-matched kmeans/top-k, embedding-only greedy controls, or alpha-matched shuffled grouping.

Supported:

- The fair-control patch is working: alpha-matched shuffled has exact alpha and coverage match.
- GAGB has small positive signals against ordinary GB in some settings.

Not supported:

- GAGB-specific value is not proven.
- A main-paper claim that topology plus embedding greedy balls are better than topology-only or alpha-matched shuffled controls is not supported by this run.

Next decision:

Treat v1 as a diagnostic baseline. If continuing, move to v2 only after explicitly changing the mechanism, such as prediction consistency, Q(B), or homophily-aware weighting.
