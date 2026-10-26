# SUPPORT-LCSS teacher to MLP distillation

## Material Passport

- ID: `teacher_support_distillation_20261022`
- Type: paired teacher-to-student experiment
- Status: planned; teacher outputs are frozen from `teacher_support_lcss_20261022`
- Scope: five datasets, three protocol seeds, three noisy-label conditions, three teacher arms

## Question and hypothesis

Does replacing the old LCSS teacher with the SUPPORT-LCSS teacher improve the same graph-free MLP student's test performance after distillation? The primary paired contrast is `SUPPORT_LCSS` teacher versus `OLD_LCSS` teacher. `CE` teacher distillation is a reference arm.

The teacher-only experiment found a modest SUPPORT-LCSS accuracy gain over old LCSS, below its predeclared continuation gate. Distillation is a separate question: student training may preserve, reduce, or reverse teacher differences, so it is tested directly rather than inferred from teacher scores.

## Granular-ball role

The ball-generation algorithm is not changed here. SUPPORT-LCSS replaces OLD_LCSS only while constructing the graph teacher's training targets. The student's input, architecture, loss, checkpoint selection, and inference remain fixed. No kNN, k-means, clean-label, or loss-function comparison is added in this experiment.

## Frozen protocol

- Reuse the completed teacher probability matrices for `CE`, `OLD_LCSS`, and `SUPPORT_LCSS`; do not retrain teachers.
- Reuse the corresponding split/noise caches for seeds 0, 1, and 2. The conditions are uniform noise 0.4, uniform noise 0.6, and pair noise 0.4, applied only to training labels.
- Train the same two-layer, 64-hidden-unit attribute-only MLP for 200 epochs with dropout 0.5, Adam learning rate 0.01, and weight decay 0.0005.
- Use temperature 1.0 soft cross-entropy against the teacher's full-node probability matrix. This is the established transductive DIST student protocol; the student reads node attributes only and does not read graph edges.
- Use all nodes in the soft-target loss, as in the existing DIST screen. Select the first epoch attaining the maximum clean validation accuracy. Test labels are used only for final metrics.
- Reset the student RNG to the protocol seed before each arm, so all three teachers in a dataset/seed/noise cell share the same initial student parameters and dropout RNG sequence.
- Freeze all teacher outputs, source code, data files, and protocol caches by SHA-256 before fitting.

## Outcomes and decision rule

Primary outcome: paired test-accuracy difference in percentage points, SUPPORT-distilled student minus OLD_LCSS-distilled student, averaged equally over the 45 dataset/seed/noise cells. Test macro-F1 and teacher-to-student transfer gaps are secondary descriptive outcomes.

The practical continuation gate is predeclared as all of:

1. Mean paired accuracy gain is at least 0.5 percentage points.
2. The mean paired gain is positive for all three protocol seeds.
3. The mean paired gain is positive for at least three of five datasets.

This is a screening gate, not a significance test. Three seeds do not establish statistical significance. No clean-label condition is included, so this experiment cannot support a claim about clean-data degradation.

## Required outputs

`config.json`, `frozen_manifest.json`, `preflight.csv`, environment metadata, per-fit epoch logs/checkpoints/probabilities/results, `results.csv`, paired and grouped summaries, reload audit, and a Chinese report with the final continue/modify/pause decision.
