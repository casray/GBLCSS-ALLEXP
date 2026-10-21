# MLP-40 Cora_ML / Coauthor-Physics Missing-Spectrum Seed0 Report

Protocol: percent 0.2/0.1/0.7, train-only noise, seed 0, frozen LCSS and baseline configs.

| Dataset | Setting | LCSS | Best formal baseline | Shuffled | LCSS-best | LCSS-shuffled |
|---|---|---:|---|---:|---:|---:|
| coauthorph | instance0.4 | 93.40 | mlp_sce 92.71 | 87.49 | +0.69pp | +5.91pp |
| coauthorph | pair0.4 | 89.39 | mlp_jocor 89.31 | 83.62 | +0.09pp | +5.78pp |
| coraml | instance0.4 | 52.91 | mlp_small_loss 55.53 | 46.56 | -2.62pp | +6.35pp |
| coraml | pair0.4 | 51.91 | mlp_small_loss 50.33 | 46.37 | +1.57pp | +5.53pp |
| coraml | uniform0.6 | 37.88 | mlp_jocor 42.08 | 31.73 | -4.20pp | +6.15pp |

## Frozen Seed0 Gates

### coraml
- Pass: 0
- Positive vs best: 1/3 (required >= 2)
- Mean LCSS-best: -1.75pp
- Minimum LCSS-best: -4.20pp
- Structure deltas >=1pp: 3/3
- Reasons: fewer_than_2_of_3_positive_vs_best;mean_delta_below_1pp;setting_regression_below_minus_1pp

### coauthorph
- Pass: 0
- Positive vs best: 2/2 (required >= 2)
- Mean LCSS-best: +0.39pp
- Minimum LCSS-best: +0.09pp
- Structure deltas >=1pp: 2/2
- Reasons: mean_delta_below_0p5pp

## Decision

Neither dataset passes its frozen seed0 expansion gate. Seeds 1-4 are not run. Cora_ML remains a breadth/boundary dataset; Coauthor-Physics remains a uniform-spectrum boundary dataset.
