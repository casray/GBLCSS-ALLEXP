# Phase 15 Instance Noise Report

Date: 2026-06-15

## Research Record

- Hypothesis: SoftDist [X||H] should remain useful under instance-dependent label noise, especially at higher noise where local semantic consensus may be more reliable than individual noisy labels.
- Granular-ball role: frozen GRACE embedding space provides label-free local semantic neighborhoods; train labels inside balls define soft targets. The mechanism is unchanged from the SoftDist [X||H] anchor.
- Code changes: no mechanism changes. Runners now record `actual_noise_rate` and `actual_train_noise_rate` because instance noise can differ from nominal rate.
- Config: `split_type=percent`, `train_percent=0.2`, `val_percent=0.1`, `test_percent=0.7`, `noise_type=instance`, `noise_rate in {0.2, 0.4}`, seeds `0-4`, GPU `cuda:0`.
- Success standard: this is a robustness-completion phase, not a go/no-go mechanism test. Win label follows the preregistered rule: mean delta > 0 and at least 4/5 paired seeds positive.
- Raw outputs: `instance_xh_rate0p2_summary.csv`, `instance_xh_rate0p4_summary.csv`, `instance_standard_rate0p2_summary.csv`, `instance_standard_rate0p4_summary.csv`.
- Aggregates: `phase15_instance_aggregate.csv`, `phase15_instance_paired_deltas.csv`.

## Results

| Dataset | Instance rate | GCN | GCN [X||H] | SoftDist [X||H] | Shuffled | NRGNN |
|---|---:|---:|---:|---:|---:|---:|
| Cora | 0.2 | 0.8347±0.0052 | 0.8409±0.0112 | 0.8412±0.0125 | 0.8387±0.0100 | 0.8357±0.0084 |
| Cora | 0.4 | 0.7893±0.0198 | 0.7961±0.0250 | 0.8132±0.0143 | 0.7938±0.0223 | 0.8108±0.0119 |
| CiteSeer | 0.2 | 0.6976±0.0123 | 0.7089±0.0145 | 0.7046±0.0255 | 0.7127±0.0119 | 0.7241±0.0038 |
| CiteSeer | 0.4 | 0.6391±0.0135 | 0.6601±0.0304 | 0.6753±0.0312 | 0.6526±0.0300 | 0.7074±0.0129 |

Actual noise rates matched the nominal rates closely: Cora 0.2016/0.3965, CiteSeer 0.2075/0.3986 for nominal 0.2/0.4.

Key paired deltas for SoftDist [X||H]:

| Dataset | Rate | vs GCN [X||H] | vs Shuffled | vs NRGNN | Decision |
|---|---:|---:|---:|---:|---|
| Cora | 0.2 | +0.02pp, 1/5 | +0.24pp, 2/5 | +0.55pp, 3/5 | weak, not a preregistered win |
| Cora | 0.4 | +1.71pp, 5/5 | +1.94pp, 5/5 | +0.24pp, 2/5 | clear ball contribution, NRGNN tie only |
| CiteSeer | 0.2 | -0.43pp, 1/5 | -0.81pp, 2/5 | -1.94pp, 0/5 | negative |
| CiteSeer | 0.4 | +1.53pp, 4/5 | +2.28pp, 5/5 | -3.20pp, 0/5 | ball helps over XH/shuffled, still below NRGNN |

## Decision

Instance noise strengthens the regime claim rather than producing a universal win. At 0.4 instance noise, SoftDist [X||H] has real ball contribution on both datasets over GCN [X||H] and shuffled balls. At 0.2 instance noise, the ball signal is weak or negative, especially on CiteSeer. NRGNN remains clearly stronger on CiteSeer instance noise.

Use in paper: include as robustness spectrum. Claim only that ball soft targets are most useful in higher-noise regimes and that CiteSeer instance noise remains a boundary where online NRGNN is stronger.
