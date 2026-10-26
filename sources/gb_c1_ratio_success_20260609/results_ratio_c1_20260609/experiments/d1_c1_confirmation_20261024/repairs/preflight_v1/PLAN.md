# D1 and C1 five-graph three-seed confirmation

**Goal:** Execute the user's approved continuation of the existing D1 group-strength and C1 short-training support/strength experiments, on five graphs with three paired seeds.
**Architecture:** New isolated adapter and outputs; historical inputs and source read-only. One previously authorized gpt-6.1-sol/high worker implements and runs serial GPU work. Parent audits independently.
**Tech Stack:** Existing PyTorch ROCm environment `D:/codex_temp/noisygl-rocm721/Scripts/python.exe`.

## Scope and hypotheses

- D1 hypothesis: agreement/conflict-specific strength improves supervision beyond validation-selected uniform strength and identical-total-mass controls.
- C1 hypothesis: real 25/75/200 racing selects a support/strength candidate close to the finite full validation-selected grid at lower candidate update cost.
- Granular balls remain shared LOO evidence regions. D1 uses legacy SUPPORT32; C1 uses historical E1 deep-tree cuts. No encoder, loss-family, or joint-module redesign.
- Datasets: cora, citeseer, corafull, coauthorcs, amazonpho.
- Protocol and student seeds: 0,1,2, paired within each cell. Reuse the corresponding historical split, noise, geometry and R2 encoder. Fixed historical GRACE SSL; no new SSL pretraining. Seed0 is development replay; separately summarize new seed1/2 evidence.
- Conditions: clean, uniform0.4, uniform0.6, pair0.4. R0 and frozen R2 on all five graphs. Total120 dataset/backbone/condition/seed cells.
- Output root: `experiments/d1_c1_confirmation_20261024/`. All old scientific files remain unchanged. No manuscript edits, uploads, automation, extra seeds or combination of D1 and C1.

## Fixed training and paired anchors

Use 20261024 pilot R0/R2 training: AlignedMLP64, dropout.5, R0 encoder/BN trained normally; R2 historical100-epoch encoder/BN frozen, linear head only, no head-stage dropout. Adam lr.01 wd.0005,200epochs, first strictly highest validation-correct checkpoint. Classification initialization and RNG shared among arms within each seed and graph. Final prediction uses attributes only.

Per cell six anchors: CE; legacy SUPPORT32 lambda .25,.5,1; GCE q=.7; same-count KM LCSS lambda1. CE is lambda0. KM uses fixed GRACE, number of clusters equal to legacy SUPPORT32, KMeans n_init10 random_state=protocol seed; seed0 must exactly bridge historical cache. Seed1/2 use verified historical KM if available and identical, otherwise build once in new directory. Hash partitions, protocol, targets and initialization.

LCSS beta.5, min_votes2, LOO, ties unchanged. m=alpha*(1-q[z]), d[z]=0 and d[c]=q[c]/(1-q[z]) for alternatives; target=(1-lambda*m)e_z+lambda*m*d. Degenerate m=0 retains label. Float64 construction, float32 targets, historical numerical bridge.

## D1: same four candidates and controls

- Groups: observed label agrees with unique raw-vote winner; conflicts with unique winner; tied highest vote.
- Four (agree,conflict) strength pairs: (0,1),(.25,1),(1,.25),(1,0).
- Ties use the cell's best global lambda in {0,.25,.5,1}, validation accuracy descending then smaller lambda.
- Select group rule by same pilot rank: validation accuracy descending, selected-checkpoint NLL ascending, earlier best epoch, deterministic existing arm order for remaining exact ties (the pilot's D1 support/strength ranking fields are all zero). No condition-dependent rule hand-coding.
- Freeze choice before fitting EXACT_MASS_GLOBAL and MASS_MATCHED_SHUFFLE, preserving exact sum(m*lambda) using pilot formula and deterministic permutation seed20261024.
- All four rules and both controls saved. Compare selected group vs CE, original lambda1, best global, both mass controls, fixed GCE and fixed KM. KM is a baseline, not an extra group-selection grid.
- 6anchors+4group+2controls=12full fits per cell,1440 upper bound before exact reuse.

## C1: same real racing and full audit

- Reuse six deep-tree cuts from E1 for each protocol seed: support4,8,16,32,64,128. Strength .25,.5,1 gives18 candidates.
- All18 to25 epochs; keep6 to75; keep2 to200. Force deep-tree(k32,lambda1) survivor at both elimination rounds. Rank by best validation accuracy, checkpoint NLL, smaller lambda, larger support, earlier best epoch.
- Each arm resumes its own model/Adam/BN/CPU+GPU+numpy+Python RNG states. Same initialization within cell.
- Compare final2 to shared CE200 and freeze PROBE JSON/hash. Only then finish eliminated arms to200; compute FULL_VAL using same ranking. Entire selection record is frozen before test scoring.
- Additional references from the complete grid: fixed deepS32, support-only selection(lambda1), strength-only selection(k32), historical SC choice, legacy SUPPORT32, fixed GCE and KM.
- 18*120=2160candidate full fits. Actual PROBE candidate updates1000 plus sharedCE200; full-grid candidates3600. The current confirmation actually pays the full audit cost, report it separately.
- Exact historical seed0 reuse is allowed only for entire completed causal experiment cells whose inputs/initialization/optimizer/checkpoint rules and frozen decisions match. Never substitute pre-finished candidate scores for a fresh race. If provenance cannot prove equality, rerun. Unique new fits cannot exceed3600; reused files stay read-only with aliases.

## Inputs already inspected

- `experiments/sc_support_channel_pilot_20261023/diagnostics/e1/<dataset>/seed{0,1,2}/geometry.pt` and condition packs exist for all15 graph/seeds.
- Graph inputs read from existing common.DATA_FILES; expose only x/edges and training observed/validation labels to fitting.
- R2 first3 graph seed0: `experiments/representation_supervision_20261021/pretraining/<dataset>/model.pt`; seed1/2: `experiments/gce_lcss_fusion_20261022/encoders/<dataset>/seedN/model.pt`.
- CS/Photo allseeds: `experiments/r2_support_5dataset_20261022/pretraining/<dataset>/seedN/model.pt`.
- Pilot source: `experiments/lcss_strength_support_20261024/` plus its postprocess path fixes. Read before porting; do not import helpers that write pilot outputs. D2 and C2 are not run.
- Seed0 historical anchors, group choices and results available for three graphs; C1 pilot covers five R0 and three R2. CiteSeer/CS R2 and allnewseed counterparts need execution.

## Implementation checklist

- [ ] Read root literature map/history and pilot PLAN/config/report/audits. Freeze this plan and resolved configuration before training.
- [ ] New protocol/targets/fit/race/run/analyze/tests with output-confined paths, platform-normalized keys, explicit seed arguments (remove pilot seed0 hardcoding), per-cell input hashes and exclusive run lock.
- [ ] Verify all15 historical split/noise/encoder/geometry passports and fixed H node alignment. Record missing input as engineering failure; use no speculative paths or fresh protocol.
- [ ] Unit tests: lambda0/1, LOO/ties, agreement groups, exact mass, seeded init changes across seeds and equal across arms, resume RNG equality, causal prefix ranking/forcedanchor, label whitelist, no test selectors, mean/std ddof1 and dataset-wise aggregation, read-only historical paths.
- [ ] At most8 bounded2epoch smoke fits, plus checkpoint-forward reloading, before full GPU grid. No scientific tuning from smoke scores.
- [ ] Run all D1 cells and anchors, then all C1 cells and full-grid audit. Resume completed fits only with manifest checks. GPU busy: wait and record, never kill other jobs.
- [ ] Freeze selections first; score test probabilities separately afterward. Per-arm raw logs, source/input/config hashes, targets, checkpoints, allnode probabilities, support profiles and costs retained.
- [ ] Audit counts, full200epoch logs, strict first-best checkpoint, init pairing, frozenR2, temporalPROBE order, target bridge, exactmass and independent checkpoint samples.
- [ ] Report tables and Chinese mechanism interpretation, allnegative outcomes and failures; no automatic further training.

## Analysis and decision criteria

Per dataset/condition/backbone report seed0/1/2, mean and sample std(ddof1); paired differences computed within matching seeds, then their mean/std. Macro across conditions is computed per seed first. Show seed1/2 separately from development seed0. Independent SSL variation is unmeasured.

D1 primary: paired change vs best per-cell global strength, with matched-mass and shuffled controls; secondary originalLCSS/GCE/KM. Show clean costs, macro-F1 and per-class recall, frequency groups, observed agreement/conflict proportions, chosen strengths, corrupted/clean posthoc target effects. Noise truth only used for diagnostic strata.

C1 primary: validation/test gap to FULL_VAL, 0.5/1/2pp closeness, worstcell, selected support/strength, real wall time/update cost. Compare fixed32 and legacy32. Complete race can win test accuracy over validation-selected FULL_VAL; preserve sign.

Continue recommendation if new seeds support positive mean with broad dataset coverage and manageable clean/F1 costs; use +.5pp as descriptive useful-size guide, not a hard automatic gate. Differences near +/-.3pp remain uncertain. Prominently disclose clean loss>2pp or F1loss>1pp; preserve all results. Three seed units, not nodes or conditions, underlie uncertainty. A small node/group grid needs ordinary strong controls, not a claim of unique granular-ball benefit.

## Deliverables and final decision

PLAN/config, provenance and read-only alias manifests, execution state/HANDOFF, tests, raw logs, results.csv, summary.csv, paired_differences.csv, seeds12_summary.csv, selections, c1_regret/cost tables, D1 strata, failures, audit, REPORT_zh.md. Final decision per experiment: continue/modify/defer with evidence. Current results: NOT_RUN.

Parent writes only `research/d1_c1_confirmation_20261024/` and performs independent read-only recomputation; worker owns the new experiment directory. No changes to the archived 20261024 pilot.
