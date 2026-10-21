# MLP-53 NRGNN latent-space granular-ball report

## Frozen protocol

- NRGNN is retrained independently per dataset/noise cell; the validation-selected model and graph are used.
- The candidate changes only the grouping representation. GB parameters and the tie-safe LOO target are unchanged.
- The same normalized NRGNN representation is used by exact self-excluding, vote-matched kNN.
- Cora/CiteSeer/CoraFull, uniform 0.4/0.6, train-only noise, seed 0.

## Target quality

| Dataset | Noise | Method | Acc | Brier | NCR | CDR | Balls | k | Teacher test |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| citeseer | 0.4 | formal_grace_h_gb | 0.5805 | 0.5733 | 0.0612 | 0.0081 | 64 |  | 0.7101 |
| citeseer | 0.4 | nrgnn_direct_teacher | 0.7098 | 0.6000 | 0.6667 | 0.2561 |  |  | 0.7101 |
| citeseer | 0.4 | nrgnn_z_gb | 0.5534 | 0.5387 | 0.0204 | 0.0243 | 19 | 35 | 0.7101 |
| citeseer | 0.4 | nrgnn_z_gb_shuffled | 0.5579 | 0.8081 | 0.0000 | 0.0000 | 19 |  | 0.7101 |
| citeseer | 0.4 | nrgnn_z_knn | 0.5609 | 0.5495 | 0.0068 | 0.0000 | 19 | 35 | 0.7101 |
| citeseer | 0.6 | formal_grace_h_gb | 0.4045 | 0.8901 | 0.0205 | 0.0509 | 64 |  | 0.6808 |
| citeseer | 0.6 | nrgnn_direct_teacher | 0.6902 | 0.6604 | 0.6385 | 0.2364 |  |  | 0.6808 |
| citeseer | 0.6 | nrgnn_z_gb | 0.4135 | 0.8140 | 0.0000 | 0.0000 | 10 | 66 | 0.6808 |
| citeseer | 0.6 | nrgnn_z_gb_shuffled | 0.4135 | 1.0884 | 0.0000 | 0.0000 | 10 |  | 0.6808 |
| citeseer | 0.6 | nrgnn_z_knn | 0.4135 | 0.8613 | 0.0000 | 0.0000 | 10 | 66 | 0.6808 |
| cora | 0.4 | formal_grace_h_gb | 0.6396 | 0.4778 | 0.1957 | 0.0322 | 52 |  | 0.8385 |
| cora | 0.4 | nrgnn_direct_teacher | 0.8355 | 0.4743 | 0.8130 | 0.1479 |  |  | 0.8385 |
| cora | 0.4 | nrgnn_z_gb | 0.6525 | 0.4420 | 0.2174 | 0.0257 | 44 | 12 | 0.8385 |
| cora | 0.4 | nrgnn_z_gb_shuffled | 0.5712 | 0.7235 | 0.0000 | 0.0064 | 44 |  | 0.8385 |
| cora | 0.4 | nrgnn_z_knn | 0.6396 | 0.4372 | 0.1565 | 0.0032 | 44 | 12 | 0.8385 |
| cora | 0.6 | formal_grace_h_gb | 0.3734 | 0.8945 | 0.0118 | 0.0149 | 52 |  | 0.7499 |
| cora | 0.6 | nrgnn_direct_teacher | 0.7320 | 0.6273 | 0.6971 | 0.2090 |  |  | 0.7499 |
| cora | 0.6 | nrgnn_z_gb | 0.4159 | 0.8965 | 0.0765 | 0.0100 | 37 | 15 | 0.7499 |
| cora | 0.6 | nrgnn_z_gb_shuffled | 0.3715 | 1.0798 | 0.0000 | 0.0000 | 37 |  | 0.7499 |
| cora | 0.6 | nrgnn_z_knn | 0.3956 | 0.8789 | 0.0412 | 0.0050 | 37 | 15 | 0.7499 |
| corafull | 0.4 | formal_grace_h_gb | 0.5978 | 0.6335 | 0.0000 | 0.0000 | 64 |  | 0.6222 |
| corafull | 0.4 | nrgnn_direct_teacher | 0.6781 | 0.6325 | 0.6074 | 0.2743 |  |  | 0.6222 |
| corafull | 0.4 | nrgnn_z_gb | 0.5978 | 0.7889 | 0.0000 | 0.0000 | 1 | 3958 | 0.6222 |
| corafull | 0.4 | nrgnn_z_gb_shuffled | 0.5978 | 0.7889 | 0.0000 | 0.0000 | 1 |  | 0.6222 |
| corafull | 0.4 | nrgnn_z_knn | 0.5978 | 0.7889 | 0.0000 | 0.0000 | 1 | 3958 | 0.6222 |
| corafull | 0.6 | formal_grace_h_gb | 0.3982 | 1.0048 | 0.0000 | 0.0000 | 64 |  | 0.5844 |
| corafull | 0.6 | nrgnn_direct_teacher | 0.6349 | 0.7580 | 0.5873 | 0.2931 |  |  | 0.5844 |
| corafull | 0.6 | nrgnn_z_gb | 0.3982 | 1.1844 | 0.0000 | 0.0000 | 1 | 3958 | 0.5844 |
| corafull | 0.6 | nrgnn_z_gb_shuffled | 0.3982 | 1.1844 | 0.0000 | 0.0000 | 1 |  | 0.5844 |
| corafull | 0.6 | nrgnn_z_knn | 0.3982 | 1.1844 | 0.0000 | 0.0000 | 1 | 3958 | 0.5844 |

## Preregistered gate

- PASS: `0`.
- Reason: `formal_grace_h_gate_failed`.
- Z-GB Brier wins vs formal GRACE-H GB: `3/6`.
- Z-GB Pareto wins vs same-space kNN: `1/6`.
- Z-GB Brier wins vs shuffled Z-GB: `4/6`.
- Direct-teacher Brier wins vs formal GRACE-H GB: `5/6`.
- Integrity failures: `[]`.

## Decision

Phase A failed. This is the third non-positive round, so the autonomous search stops without downstream training or another mechanism variant.

## Detailed interpretation

The experiment separates teacher quality from grouping-space quality. Direct NRGNN probabilities have lower Brier score than the formal GRACE-H GB target in 5/6 cells. The robust teacher is therefore informative, especially at uniform 0.6, and its previously observed distillation advantage is not an artifact of student training alone.

Compressing that teacher into regional vote targets loses most of the benefit. NRGNN-Z balls improve Brier over formal GRACE-H balls in only 3/6 cells and achieve only 1/6 preregistered Pareto wins against exact, self-excluding, vote-matched kNN in the same Z space. They beat shuffled Z-space balls in only 4/6 cells rather than the required 6/6.

CoraFull identifies the failure mode directly. The fixed geometry rule produces one NRGNN-Z ball at both noise rates. Consequently GB, kNN with matched k=3958, and shuffled GB all collapse to the same global-label distribution. The NRGNN hidden layer is optimized for node prediction on the reformed graph, but it is not guaranteed to preserve the variance-reduction geometry required by recursive 2-means. A representation can be an effective classifier input while being unsuitable for granular-ball consensus.

The final decision is to stop this search after three rounds. Multi-H target averaging, cross-partition vote pooling, and robust-teacher latent balls all reveal useful signals, but every signal is either mirrored by kNN or destroyed by regional compression. The current paper should retain the formal GRACE-H GB interface and the direct robust-teacher distillation comparison; it should not add these seed0 diagnostics as positive method components.
