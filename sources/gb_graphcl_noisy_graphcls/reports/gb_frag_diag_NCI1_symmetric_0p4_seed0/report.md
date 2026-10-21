# GB Fragmentation Diagnostic: NCI1 symmetric 0.4 seed0

## GB Statistics
### gb_orig (purity=0.7, min=5, max=100)
- num_balls: 780
- mean_size: 4.2153846153846155
- min_size: 1
- max_size: 47
- mean_purity: 0.7401086439125792

### gb_relaxed (purity=0.6, min=10, max=200)
- num_balls: 240
- mean_size: 13.7
- min_size: 1
- max_size: 175
- mean_purity: 0.6655743571051885

## Ranked Test Accuracy
- rw_noball: test_acc=0.6545, precision=0.6264, noise_exclusion=0.1463
- rw_gb_orig: test_acc=0.6448, precision=0.6456, noise_exclusion=0.3026
- rw_gb_relaxed: test_acc=0.6423, precision=0.6735, noise_exclusion=0.4421
- rw_shuffled_relaxed: test_acc=0.6399, precision=0.6515, noise_exclusion=0.3110
- rw_kmeans: test_acc=0.6350, precision=0.6665, noise_exclusion=0.3476
- rw_random_relaxed: test_acc=0.6326, precision=0.6786, noise_exclusion=0.4992

## Pre-registered Deltas
- rw_gb_relaxed - rw_noball: -0.0122 (FAIL; PASS if > 0)
- rw_gb_relaxed - rw_shuffled_relaxed: 0.0024 (FAIL; PASS if > 0.005)
- rw_gb_relaxed - rw_random_relaxed: 0.0097 (PASS; PASS if > 0)
- rw_gb_relaxed - rw_kmeans: 0.0073 (PASS; PASS if >= -0.005)

## Fragmentation Repair Delta
- rw_gb_relaxed - rw_gb_orig: -0.0024 (positive = repair helped)

## Verdict
FAIL: rw_gb_relaxed - rw_noball; rw_gb_relaxed - rw_shuffled_relaxed