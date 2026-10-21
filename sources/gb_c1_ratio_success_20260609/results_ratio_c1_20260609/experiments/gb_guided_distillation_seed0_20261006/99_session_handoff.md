# Handoff: fixed ERASE teacher, GB-guided student targets

Date: 2026-10-06. All GPU fits completed; no experiment process remains running.

## Objective and approved design

User approved the design in `../gb_guided_distillation_proposal_20261006/PLAN.md`.
Five datasets, uniform training-label noise .4/.6, seed 0. Reuse fixed ERASE teacher
probabilities and verified original GRACE assignments. D0 ordinary KD; D1 noisy-label
anchor; D2 constant-weight GB; D3 adaptive GB; D4 shuffled eligible training assignments.
lambda=.5, beta=.5, min_votes=2, shuffle seed=137. Same 2-layer attribute-only MLP,
hidden64, BN, dropout.5, Adam .01/.0005, 200 epochs, full-node softCE, strict clean-val selection.
All targets frozen before fitting; no teacher retraining or tuning.

## Required context read

Project AGENTS and original literature map/history were read before design.
Relevant negative history: MLP51 multi-target averaging, MLP52 vote pooling,
MLP53 classifier-space ball collapse. Latest histories 33 and 38 and the previous
robust-graph experiment handoff were reviewed. Do not revive these without a new hypothesis.

## Result

40/40 new students complete, 10 D0 rows reused, one D0 replay bit exact.
Mean accuracy: teacher 76.99285, D0 77.56402, D1 77.75034, D2 77.58211,
D3 77.60596, D4 77.66997 percent.
D3-D0 +.04194 pp (7 wins/3 losses); D3-D1 -.14438 (3/7);
D3-D2 +.02384 (4 wins/2 ties/4 losses); D3-D4 -.06401 (3/4/3).
D3-teacher +.61311 mean, worst -1.37135 pp.
Only better-than-constant and retain-teacher screening gates pass.
Decision: stop this fixed candidate; do not launch seeds1-4 without a new approved plan.

Original GB target has worse training Brier than ERASE in all ten settings.
Mixed D3 target improves Brier in nine, but D1 and D4 have lower mean Brier than D3.
Target improvements alone therefore do not establish geometry-specific or classification gains.
This does not disprove all GB distillation or the attribute-only transfer goal.
Ordinary KD already retains this ERASE teacher within 1.37856 pp in every cell.
One seed, two synthetic noises, fixed transductive graph: no stable/SOTA/generalization claims.

## Files and verification

`实验结论.md` is the detailed Chinese debrief. `paired_results.csv`, `diagnostics.csv`,
`summary.json`, `output_audit.json` contain traceable results. `runs/` holds checkpoints,
probabilities and 200-epoch logs. `targets/` holds frozen inputs and configs.
`provenance/` contains the approved proposal and reused source snapshots.
`run_guided.py`, `guided_targets.py`, `test_guided_targets.py`, `analyze.py` are new only.
Commands are in `COMMANDS.md`. Previous results and submitted manuscripts were untouched.

The audit compares integer correct counts because CPU/GPU float32 means differ by one ULP.
Do not interpret newly saved fit wall times and reused D0 student-only times as matched timing.
Four target unit tests, 50 target/output checks, 150 metrics, 40 checkpoint selections,
20 independent GB recomputations, all frozen/source hashes passed.

## Next-window first step

Read `实验结论.md` and `summary.json`. Explain why this candidate stopped before proposing
a new direction. Preserve the negative controls; do not raise lambda based on test results.
New experiments require user approval, a new folder, paired protocol and a preregistered decision.
Parent context/history files were outside this run's write scope and were not updated.
