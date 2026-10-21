# Realistic supervision recovery E/F implementation plan

Date: 2026-10-12. Registered before this experiment's training/test outcomes.
User approved the E/F design discussed in chat; implementation is inline.

## Material passport and research question

Type: paired three-protocol-seed exploratory confirmation, not external validation.
Sources: literature map; histories 41/42/43; oracle_teacher_transfer_20261012;
gb_dist_protocol_seeds_20261007; existing ROCm ERASE and DIST adapters.
Question: does correcting suspected corrupted labels outperform removing supervision
on exactly the same suspected nodes, and does any improvement transfer to an X-only MLP?
Granular balls have no method role. Existing protocol loader reads a historical partition
to verify provenance only; no partition, embedding or region loss enters training.

## Fixed design

Cora, CiteSeer, CoraFull; uniform .4/.6; original 20/10/70 protocol; seeds 1/2/3.
Each seed controls split, noise, teacher and student initialization. Seed 3 is an additional
protocol replicate, not a held-out dataset. Existing datasets have been used for development.

Arms:
- A_noisy: full original training set with observed labels.
- B_all_clean: full original training set with clean labels (oracle reference only).
- C_oracle_filter: only originally uncorrupted training labels (oracle reference only).
- E_correct: full training set; suspected labels replaced by the frozen A teacher argmax.
- F_delete: remove supervision on exactly the nodes suspected in E; retain other observed labels.

Suspicion score is 1 minus A's probability assigned to the observed class. Mark exactly
ceil(.20 * original training count) nodes. Stable descending sorting breaks score ties by
original training-index order. All marked E labels are replaced; unchanged argmax is recorded.
No clean labels, known noise indicators, validation labels or test labels enter this rule.
No threshold/coverage search, iteration, class-dependent rule or post-result adjustment.

ERASE retraining starts from the same seed, not the A checkpoint. Graph nodes/edges stay fixed.
ERASE supervised propagation, prototypes and classification probe follow each arm's mask/labels;
E-F is a pipeline intervention, not isolation of label count alone. Missing prototype classes
are reported failures, not fixed by oracle additions or silent resampling.

Teacher max400/patience150 and author presets unchanged. Student raw-X two-layer MLP,
hidden64, BN/dropout .5, Adam .01/wd .0005, 200 epochs, all-node DIST T1 beta/gamma1,
no CE or regional loss. Strict first maximum clean validation selection for both stages.
Other true labels are hidden from fit; oracle access is restricted to original training nodes.

Reuse seed1 A/B/C from the just-completed oracle experiment only after input/config/hash checks.
All other A/B/C and all E/F are new fits through its unchanged teacher/student functions.
B is fit once per dataset/seed because noise .4/.6 effective inputs match exactly.
90 logical arm-cells; 81 distinct pairs, 15 reused and 66 new teacher/student pairs.
Historical seed2 teacher predictions are NOT used for current E/F selection.

## Endpoints and investment gates

Primary: student E-F; secondary teacher E-F, E-A, F-A, B-A, C-A, B-C, B-E.
Report each seed/cell, per-cell mean/sample std, seed-balanced macro effects and seed macro std.
Also report test balanced accuracy/NLL/Brier and training-only detection precision/recall,
correction accuracy on corrupted nodes, correctly repaired/newly corrupted counts, residual noise,
per-class available supervision and predicted class coverage. Training truth is used solely for
diagnostic reporting, never for E/F selection or confidence weighting.

Predefined practical gates, not significance tests:
- Oracle recovery confirmed: mean student B-C >=1pp, >=4/6 cell means positive,
  and positive seed-macro B-C on all three seeds.
- Realistic recovery supported: mean student E-F >=.5pp, >=4/6 cell means positive,
  and positive seed-macro E-F on all three seeds.
- Net improvement over original pipeline: same conditions for student E-A.
- Teacher/student effects are reported separately; no inference from teacher alone.
- Gate failure means this fixed one-pass correction baseline lacks practical evidence;
  it does not refute all label recovery methods or justify adding GB mechanisms.
- If oracle gap vanishes across seeds, pause this motivation under the current protocol.
- If E improves but B-E remains, diagnose correction/coverage; no causal or novelty claim yet.

Test outcomes read only after all registered fits finish. No dropping negative cells.
Budget: 120 minutes for the grid, checked between pairs; failures are recorded and not silently
retried. No automatic expansion, additional methods or submitted-paper edits.
Teacher retraining is not bit-exact on this GPU, as established in the previous experiment.
Checkpoint reload must agree within1e-6 with identical argmax; compare initialization hashes.

## Files and execution steps

- [ ] test_recovery.py: test ranking, ties, identical E/F marked sets, label hiding and truth independence.
- [ ] run_recovery.py: freeze inputs/settings, save protocols, reuse validation, ranking and GPU grid.
- [ ] analyze.py: reload checkpoints, independently recompute metrics/diagnostics and paired summaries.
- [ ] PLAN.md, raw logs, configs, hashes, checkpoints, tables, decision/audit, Chinese report and handoff.
- [ ] Update project context/history only after results.

Runtime: D:/codex_temp/noisygl-rocm721/Scripts/python.exe (existing ROCm environment).

```powershell
& 'D:/codex_temp/noisygl-rocm721/Scripts/python.exe' -m pytest experiments/supervision_recovery_EF_20261012/test_recovery.py -q -p no:cacheprovider
& 'D:/codex_temp/noisygl-rocm721/Scripts/python.exe' experiments/supervision_recovery_EF_20261012/run_recovery.py prepare
& 'D:/codex_temp/noisygl-rocm721/Scripts/python.exe' experiments/supervision_recovery_EF_20261012/run_recovery.py grid
& 'D:/codex_temp/noisygl-rocm721/Scripts/python.exe' experiments/supervision_recovery_EF_20261012/analyze.py
```
