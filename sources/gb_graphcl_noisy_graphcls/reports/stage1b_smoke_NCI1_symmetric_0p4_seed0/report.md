# Stage1b Smoke Report: NCI1 symmetric_0.4 seed0

## Ranked Test Accuracy
- clean_oracle: test=0.6691, precision=1.0000, noise_exclusion=0.0000
- rw_kmeans: test=0.6545, precision=0.6604, noise_exclusion=0.3232
- rw_noball: test=0.6545, precision=0.6264, noise_exclusion=0.1463
- rw_random: test=0.6472, precision=0.6356, noise_exclusion=0.2370
- rw_gb: test=0.6448, precision=0.6456, noise_exclusion=0.3026
- rw_shuffled: test=0.6448, precision=0.6355, noise_exclusion=0.2149
- rw_topk: test=0.6399, precision=0.6506, noise_exclusion=0.2942
- sce: test=0.6350, precision=0.6010, noise_exclusion=0.0000
- sm_topk: test=0.6350, precision=0.0000, noise_exclusion=0.0000
- ce: test=0.6302, precision=0.6010, noise_exclusion=0.0000
- sm_noball: test=0.6302, precision=0.0000, noise_exclusion=0.0000
- sm_gb: test=0.6277, precision=0.0000, noise_exclusion=0.0000
- sm_kmeans: test=0.6204, precision=0.0000, noise_exclusion=0.0000
- sm_shuffled: test=0.6156, precision=0.0000, noise_exclusion=0.0000
- sm_random: test=0.6107, precision=0.0000, noise_exclusion=0.0000

## Baselines
- headroom clean_oracle - ce: 0.0389

## Route A: Reweight
- best: rw_kmeans (0.6545)
- route_a_best - ce: 0.0243
- rw_gb - rw_topk: 0.0049
- rw_gb - rw_kmeans: -0.0097
- rw_gb - rw_shuffled: 0.0000
- rw_gb - rw_random: -0.0024

## Route B: Label Smoothing
- best: sm_topk (0.6350)
- route_b_best - ce: 0.0049
- sm_gb - sm_topk: -0.0073
- sm_gb - sm_kmeans: 0.0073
- sm_gb - sm_shuffled: 0.0122
- sm_gb - sm_random: 0.0170

## GB Stats
- num_balls: 780
- mean_size: 4.2153846153846155
- min_size: 1
- max_size: 47
- mean_purity: 0.7401086439125792