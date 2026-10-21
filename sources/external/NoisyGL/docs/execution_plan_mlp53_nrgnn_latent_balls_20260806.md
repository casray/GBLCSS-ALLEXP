# MLP-53 NRGNN latent-space granular-ball smoke

## Research hypothesis

The previous two rounds showed that combining several label-free GRACE partitions improves target calibration, but the same gain is available to kNN. The remaining high-priority hypothesis is that a robust graph teacher may reshape the representation around noisy-label decision boundaries. Granular balls built in this denoised teacher space may therefore provide a better regional consensus than balls built in frozen GRACE H.

## Role of granular balls

Granular balls remain a fixed-budget, mutually exclusive local-consensus interface. They do not alter the teacher, use clean labels, or add a new downstream loss. The only changed component is the representation in which the formal leave-one-out balls are constructed.

## Frozen protocol

- Datasets: Cora, CiteSeer, CoraFull.
- Noise: train-only uniform 0.4 and 0.6.
- Split: percent 0.2/0.1/0.7.
- Seed: 0.
- Teacher: the existing NRGNN implementation and dataset configs, trained independently in every cell.
- Teacher representation: the penultimate hidden layer of the validation-selected NRGNN classifier on its validation-selected reformed graph, row L2-normalized before grouping.
- Ball target: `max_balls=64`, `min_ball_size=32`, `split_improvement=0.05`, `min_train_per_ball=2`, `beta=0.5`, `tie_mode=distribution_only`, `leave_one_out=true`.
- kNN target: exact cosine kNN in the same normalized representation, `exclude_self=true`; k is matched to the mean number of training votes per non-empty teacher-space ball.

## Experiment arms

1. `formal_grace_h_gb`: the current formal GRACE-H LOO GB target.
2. `nrgnn_direct_teacher`: NRGNN class probabilities without regional compression.
3. `nrgnn_z_gb`: formal LOO GB target in the NRGNN hidden space.
4. `nrgnn_z_knn`: exact vote-matched LOO kNN target in the same NRGNN hidden space.
5. `nrgnn_z_gb_shuffled`: size-matched shuffled teacher-space ball assignment.

Clean labels are used only after target construction to calculate diagnostic target accuracy, Brier score, NCR, and CDR.

## Expected mechanism

If NRGNN moves semantically compatible nodes together while separating corrupted local relations, regional aggregation in its hidden space should improve target calibration relative to GRACE-H balls. A genuine granular-ball contribution additionally requires the teacher-space balls to outperform exact teacher-space kNN, not merely shuffled groups.

## Phase-A success criteria

All criteria are frozen before execution:

1. `nrgnn_z_gb` has lower Brier score than `formal_grace_h_gb` in at least 4/6 cells.
2. `nrgnn_z_gb` has lower Brier score than `nrgnn_z_knn` in at least 4/6 cells, while its target accuracy is no more than 0.5 percentage points lower in each winning cell.
3. `nrgnn_z_gb` has lower Brier score than `nrgnn_z_gb_shuffled` in all 6 cells.
4. `nrgnn_direct_teacher` has lower Brier score than `formal_grace_h_gb` in at least 4/6 cells, confirming that the selected teacher contains useful denoising information.
5. All rows pass split/noise hash, train-only corruption, finite-value, target-simplex, and representation-provenance checks.

## Failure and stopping rule

Failure of any criterion closes this direction. Because MLP-51 and MLP-52 already rejected two multi-partition variants, a Phase-A failure is the third consecutive non-positive round and terminates this autonomous search. No threshold, ball parameter, k, teacher hyperparameter, or dataset is changed after seeing results.

## Outputs

- `runs/mlp53_nrgnn_latent_balls_20260806/phase_a_raw.csv`
- `runs/mlp53_nrgnn_latent_balls_20260806/teacher_artifacts/*.pt`
- `runs/mlp53_nrgnn_latent_balls_20260806/phase_a_gate.json`
- `runs/mlp53_nrgnn_latent_balls_20260806/mlp53_report.md`

