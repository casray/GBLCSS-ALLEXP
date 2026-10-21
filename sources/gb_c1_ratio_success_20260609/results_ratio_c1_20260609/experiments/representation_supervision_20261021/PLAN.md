# Representation Transfer and Robust Supervision Implementation Plan

> Execute this approved plan inline with the experiment-agent and executing-plans workflows. No manuscript changes are included.

**Goal:** Test whether leave-one-out LCSS remains useful after an attribute-only student learns frozen GRACE representations, and whether it complements small-loss selection.

**Architecture:** Reuse the historical raw features, ordered 20/10/70 splits, train-only noise, frozen GRACE H and seed0 ball assignments. Keep the deployed two-layer MLP identical. A training-only linear projection aligns its 64-dimensional hidden representation to normalized H.

**Tech Stack:** Existing Python 3.12 / PyTorch 2.9.1 ROCm 7.2.1 environment, NumPy, existing gblcss utilities. Single GPU, four CPU threads. No installs or downloads.

## Material Passport

- Mode: code experiment, approved design and execution.
- Sources: current conversation; heterophily_mechanism_20261014; main_protocol_replay_20260922; MLP-6, MLP-31 and MLP-39 records.
- Status at creation: planned, no new results inspected.
- Evidence class: exploratory one-protocol-seed pilot; no significance or stability claim.

## Research hypothesis and mechanism

H1: Direct feature alignment preserves information that label-histogram targets alone do not transfer.
H2: With the same representation training, LCSS improves on noisy CE or adds value beyond small-loss.
H3: Simultaneous feature alignment and supervised classification may conflict; a budget-matched two-stage route tests that possibility.

Granular balls provide fixed local voting regions only. They do not create hard positive pairs. Main targets exclude the current node, use beta=0.5 and s_min=2, and keep the empirical distribution on a tie.

## Frozen scope

- Datasets: Cora, CiteSeer, CoraFull.
- Protocol/student seed: 0. One historical SSL pretraining seed. These are not independent SSL repetitions.
- Conditions: clean, uniform0.4, uniform0.6, pair0.4.
- Main matrix: 3 representation routes x 4 supervision rules x 3 datasets x 4 conditions = 144 logical fits.
- R0: no representation alignment, 200 supervised updates.
- R1: 200 joint updates, soft classification loss plus lambda times mean cosine alignment loss over all nodes.
- R2: 100 label-free alignment updates followed by 100 classifier-only updates. Freeze encoder parameters AND batch-normalization state in stage 2. Choose the stage-1 checkpoint by minimum label-free evaluation alignment loss. Share the same stage-1 checkpoint across conditions and supervision rules within a dataset.
- The alignment uses the hidden vector after ReLU and BN, before dropout. Classification retains original dropout=0.5. H is normalized and detached. Projection is discarded at deployment.
- Supervision T0: noisy one-hot CE; T1: LOO-LCSS soft CE; T2: small-loss on noisy CE; T3: small-loss on LOO soft CE, retaining the soft targets for updates.
- T2/T3 use the existing remember_count schedule: known synthetic noise rate, 10-epoch gradual increase, exponent 1. Report this information assumption explicitly. Noisy CE and soft CE rank different samples; this is part of T3 and is not claimed as isolated target-only causality.
- Backbone: raw X -> Linear64 -> ReLU -> BN -> dropout0.5 -> LinearC; Adam lr0.01, wd5e-4. Same initial deployed-model parameters and initial training RNG across T0..T3 and R0/R1.
- R1 lambda: {0.1, 1.0}, selected once by the average best clean validation accuracy for Cora and CiteSeer, uniform0.4, T0, seed0. Tie selects 0.1. Four development fits; no test metrics are used. Freeze selection before the main grid.
- R2 alignment has coefficient 1 because there is no simultaneous supervised loss. R2 has 200 nominal updates total; its shared pretraining time must be included in end-to-end costs.
- Legacy bridge: R0/R1 x T1/T3 x 3 datasets x uniform0.4/uniform0.6 = 24 fits, using inclusive votes and old min-index tie handling. These are separately identified and never pooled with the main LOO results.
- H probes: linear and two-layer64 classifiers directly on normalized H, all four conditions, 24 fits. They are information-access references, not theoretical upper bounds and not X-only deployment methods.
- Existing SimMLP seed0 uniform0.4 results may be referenced only after exact split/noise checks. No five-seed mean is compared as a paired seed0 result.
- Ball budgets and construction are frozen from main_protocol_replay; actual counts may be below 64. No adaptive CoraFull generator is introduced in this pilot.

## Data and model integrity

Prepare one input cache per dataset/condition. Model-facing cache contains only raw X, training observed labels, frozen targets, normalized H, indices, assignment and a separate clean validation vector. No clean training/test labels enter loss or representation checkpoint selection. Read final test labels only in post-grid analysis.

Hash raw data files, H, split/noise vectors, assignments, target arrays, source files, configs and checkpoints. Check historical uniform0.4 split/noise and target agreement. Cross-check noise and targets against existing reference functions. Record all failures and do not silently change an arm after seeing its results.

Save per-epoch logs, selected epoch/state/probabilities, final epoch validation, alignment loss, selected sample counts, initial hashes, and runtime. For R2 verify the encoder state including BN buffers is unchanged during stage 2. Verify exported deployment checkpoint accepts X alone and contains no projection or graph cache.

## Pilot progression rules, frozen before training

These are resource-allocation rules, not statistical significance tests.

1. Transfer signal: using T1 as the primary supervision comparison, at least one representation route gains >=1pp over matched R0 in at least 2/3 datasets on the average of the three noisy conditions. Other supervision comparisons are descriptive. Report clean differences and every condition, not just successful cells.
2. LCSS value: separately report T1-T0 and T3-T2 within each representation route. Candidate progression requires mean >=0.5pp in one comparison, positive in at least 2/3 datasets and at least 6/9 noisy cells, with no clean loss exceeding 1pp for the corresponding comparison. A pilot failure does not prove universal uselessness.
3. Combination: report T3-max(T1,T2) for the same representation route. Never claim complementarity from improvement over R0 alone.
4. No test-based selection of lambda, representation route, or per-dataset settings. Validation-selected route is recorded separately from the descriptive test matrix. Do not expand seeds or launch dynamic/ball-graph branches automatically in this task.
5. Report mean across conditions as a descriptive macro average, with seed std=null because n=1. A useful signal justifies a later independently frozen multi-seed confirmation; a weak signal leaves the branch deferred.

## Implementation and execution checklist

- [ ] Create behavioral tests for LOO/ties, legacy difference, training-only labels, reference MLP parity, projection RNG isolation, detached targets, small-loss schedule, frozen BN, and selection ties.
- [ ] Run `python -m unittest discover -s experiments/representation_supervision_20261021 -p test_*.py -v` before implementation; confirm missing implementation fails.
- [ ] Implement isolated core and runner; repeat the tests and GPU engineering smoke.
- [ ] Run `python experiments/representation_supervision_20261021/run_experiment.py prepare`; freeze inputs and source hashes.
- [ ] Run `python experiments/representation_supervision_20261021/run_experiment.py tune`; freeze validation-only lambda selection.
- [ ] Run `python experiments/representation_supervision_20261021/run_experiment.py grid`; complete main, bridge, and probe fits with a progress journal.
- [ ] Run `python experiments/representation_supervision_20261021/analyze.py`; verify checkpoints/metrics and produce CSVs, gates, Chinese conclusion and handoff.

Runtime executable: `D:/codex_temp/noisygl-rocm721/Scripts/python.exe`. Run from the results workspace. Commands and execution state are saved locally. A completed fit is never overwritten; a previously failed fit requires an explicit recorded engineering fix before retry.

## Results and decision

Pending execution. The final conclusion must distinguish measured effects, plausible explanations, untested hypotheses, and directions weakened by the results. Preserve all negative cells.
