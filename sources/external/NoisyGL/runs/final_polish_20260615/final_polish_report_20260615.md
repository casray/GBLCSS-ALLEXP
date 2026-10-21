# Final Polish Report: Phase 21-23

Date: 2026-06-15  
Plan: `external/NoisyGL/docs/execution_plan_final_polish_20260615.md`  
Scope: P1/P2 paper polish only. SoftDist [X||H] mechanism and anchor configs were not changed.

## Phase 21: Ball-Count Sensitivity

- Research hypothesis: the default `max_balls=64` should lie on a stable performance plateau, not an isolated cherry-picked optimum.
- Granular-ball role: same as main SoftDist [X||H], local semantic unit for train-only soft target construction.
- Method motivation: answer the reviewer concern that ball count may have been tuned to Cora/CiteSeer.
- Code changes: added `scripts/run_ball_count_sensitivity.py`.
- Config changes: only `gb.max_balls` is overridden in memory; YAML anchor files are unchanged. `min_ball_size=32` is fixed to isolate the `max_balls` effect.
- Expected mechanism: too few balls may be coarse, too many balls may saturate under `min_ball_size`; the default should remain competitive.
- Success standard: real SoftDist remains stable across `{16,32,64,128}` and stays above shuffled.
- Failure standard: default 64 is an isolated peak, or real-vs-shuffled advantage disappears.
- Result files:
  - `runs/ball_count_sensitivity_20260615/ball_count_sensitivity_raw.csv`
  - `runs/ball_count_sensitivity_20260615/ball_count_sensitivity_summary.csv`
  - `runs/ball_count_sensitivity_20260615/ball_count_sensitivity.png`
  - `runs/ball_count_sensitivity_20260615/ball_count_sensitivity_report.md`
- Raw logs:
  - `runs/ball_count_sensitivity_20260615/ball_count_sensitivity_20260615.log`
  - `runs/ball_count_sensitivity_20260615/ball_count_sensitivity_20260615_rerun.log`

| Dataset | max_balls | Real SoftDist | Shuffled | Actual balls |
|---|---:|---:|---:|---:|
| Cora | 16 | 0.8132 ± 0.0065 | 0.7987 ± 0.0042 | 16.0 |
| Cora | 32 | 0.8132 ± 0.0109 | 0.7954 ± 0.0087 | 32.0 |
| Cora | 64 | 0.8137 ± 0.0067 | 0.7921 ± 0.0097 | 52.4 |
| Cora | 128 | 0.8137 ± 0.0067 | 0.7921 ± 0.0097 | 52.4 |
| CiteSeer | 16 | 0.6892 ± 0.0142 | 0.6723 ± 0.0184 | 16.0 |
| CiteSeer | 32 | 0.6923 ± 0.0129 | 0.6688 ± 0.0200 | 32.0 |
| CiteSeer | 64 | 0.6891 ± 0.0160 | 0.6696 ± 0.0234 | 63.8 |
| CiteSeer | 128 | 0.6901 ± 0.0150 | 0.6698 ± 0.0235 | 64.6 |

Final decision: sensitivity passed. Use as a paper appendix/figure. The default `64` is defensible as a stable plateau point.

## Phase 22: Analysis Figures

### 22a CDR Distance-Bin Figure

- Research hypothesis: clean-label damage should be more concentrated near ball boundaries.
- Granular-ball role: diagnostic of where soft/hard correction errors arise.
- Method motivation: visualize why hard replacement is risky and why soft correction is safer.
- Code changes: added `scripts/plot_cdr_distance.py`.
- Config changes: none; reads existing diagnostic CSV.
- Expected mechanism: CDR rises in outer distance bins.
- Success standard: non-empty figure and summary table; no training or new target construction.
- Failure standard: empty data or no interpretable trend.
- Result files:
  - `runs/figures_20260615/cdr_distance_bins.png`
  - `runs/figures_20260615/cdr_distance_bins.pdf`
  - `runs/figures_20260615/cdr_distance_bins_summary.csv`
  - `runs/figures_20260615/cdr_distance_bins_report.md`

Observed trend: Cora CDR rises from `0.0000` in the center bin to `0.0153` at the boundary bin. CiteSeer CDR rises from `0.0057` to `0.0203`.

### 22b GRACE vs CCA-SSG t-SNE

- Research hypothesis: visual and quantitative geometry should support the claim that linear separability is not the same as ball clusterability.
- Granular-ball role: downstream clusterability requirement.
- Method motivation: make the Section 6 representation-quality argument concrete.
- Code changes: added `scripts/plot_tsne_embeddings.py`.
- Config changes: none; uses existing `H.pt` files.
- Expected mechanism: GRACE should be more suitable for local ball construction than CCA-SSG despite lower probe.
- Success standard: t-SNE figure plus quantitative backup table.
- Failure standard: figure is visually ambiguous and no quantitative backup exists.
- Result files:
  - `runs/figures_20260615/tsne_cora_grace_vs_ccassg.png`
  - `runs/figures_20260615/tsne_cora_grace_vs_ccassg.pdf`
  - `runs/figures_20260615/tsne_class_distance_summary.csv`

| Method | Within-class distance | Between-class distance | Between/within |
|---|---:|---:|---:|
| GRACE | 0.539 | 0.740 | 1.372 |
| CCA-SSG | 1.059 | 1.378 | 1.301 |

Final decision: figures completed and inserted into `SoftDist_XH_论文初稿雏形.html`.

## Phase 23: TSS CiteSeer Completion

- Research hypothesis: the missing TSS CiteSeer cell is an environment/device issue rather than a conceptual missing baseline.
- Granular-ball role: none; this is a baseline completion run.
- Method motivation: remove the only blank cell from the main table.
- Code changes: `predictor/module/TSS.py` now respects the adjacency device in `Personalized_PageRank` instead of hard-coding `.cuda()`.
- Config changes: no hyperparameter change; only device changed to CPU.
- Expected mechanism: CPU avoids ROCm HIPBLAS allocation failure for dense PPR inverse.
- Success standard: TSS CiteSeer runs for seeds 0-4 and writes test accuracies.
- Failure standard: CPU also fails, in which case the table keeps an honest footnote.
- Result files:
  - `runs/tss_citeseer_completion_20260615/tss_citeseer_cpu_summary.csv`
  - `runs/tss_citeseer_completion_20260615/tss_citeseer_completion_report_20260615.md`
- Raw log: `runs/tss_citeseer_completion_20260615/tss_citeseer_cpu_5seed.log`

Result: `0.5055 ± 0.0510`.

Final decision: table cell filled. No TSS tuning is recommended.

## HTML Backfill

Updated: `C:\Users\Admin\Desktop\实验8去噪5.19-\SoftDist_XH_论文初稿雏形.html`

- Section 3.1: TSS CiteSeer cell filled.
- Section 4.2: ball-count sensitivity table and figure added.
- Section 6: t-SNE figure and quantitative class-distance table added.
- Section 7.2: CDR distance-bin figure added.
- Section 10: final TODO list updated; Phase 21-23 marked complete.

