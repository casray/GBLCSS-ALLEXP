# Smoke Report: NCI1 symmetric_0.4 seed0

## Test Accuracy
- clean_oracle: 0.6691
- sce: 0.6350
- ce: 0.6302
- sel_gb: 0.6034
- sel_topk: 0.6010
- sel_shuffled: 0.6010
- sel_noball: 0.6010
- sel_kmeans: 0.5961
- sel_random: 0.5912

## Deltas
- clean_oracle - ce: 0.0389
- sel_gb - sel_topk: 0.0024
- sel_gb - sel_kmeans: 0.0073
- sel_gb - sel_shuffled: 0.0024

## GB Stats
- num_balls: 780
- mean_size: 4.2153846153846155
- min_size: 1
- max_size: 47
- mean_purity: 0.7401086439125792