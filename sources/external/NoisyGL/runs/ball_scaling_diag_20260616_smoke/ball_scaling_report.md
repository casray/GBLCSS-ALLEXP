# Phase 32a: Ball Construction Scaling Diagnostic

## Protocol

- Datasets: `dblp`
- Seeds: `0`
- max_balls: `64`
- min_ball_size: `32`
- split_improvement: `0.05`
- Noise: `uniform` `0.4`
- Split: percent 0.2 / 0.1 / 0.7
- Training: none. This is a geometry-only diagnostic on frozen H.
- Anchor configs are not modified.

## Small-Graph Reference

| Dataset | Params | num_balls | true_all_purity_weighted | mean_train_per_ball |
|---|---|---:|---:|---:|

Reference threshold for 'near small-graph purity': `0.6500` (min default Cora/CiteSeer purity minus 0.05).

## Best Non-Collapsed Points

| Dataset | Best params | num_balls | true_all_purity_weighted | mean_train_per_ball | Min non-collapse setting | Decision |
|---|---|---:|---:|---:|---|---|
| dblp | none | 0.00 | 0.0000 | 0.00 | not reached | A: collapsed for all scanned parameters |

## Full Summary

| Dataset | max_balls | min_ball_size | split_improvement | num_balls | true_all_purity | noisy_train_purity | mean_train_per_ball |
|---|---:|---:|---:|---:|---:|---:|---:|
| dblp | 64 | 32 | 0.05 | 1.00 | 0.4471 | 0.3506 | 3543.00 |
