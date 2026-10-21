# R2 controls completed: 2026-10-21

Read 综合结论与接续.md first, then PLAN/config/results/decision and both audits.
No running training, no unexecuted formal cells. Do not restart prepare or add more seeds automatically.
Old representation_supervision_20261021 and submitted manuscript files were not changed.

156 new classifier fits: REAL nine losses/targets x3graphs x4conditions=108;
RANDOM_ENCODER and SHUFFLED_H each CE/GB x12conditions=48. Three shuffled pretrains100 updates;
three zero-update random encoder checkpoints; nine engineering three-update fits excluded.
Same protocol/student seed0, 20/10/70, clean validation, old H and GB partition. Same64 two-layer X-only MLP,
100 frozen-encoder/BN head updates, real R2 pretraining reused, test scored only after full grid.

Nine-noise macro CE56.6018,GB57.9214,LS56.4438,GCE59.7102,BOOT56.4658,MASS58.0492,
KNN59.1339,KM59.2768,RANDOM55.7196%. GB−CE+1.3196pp(8/9),−GCE−1.7889(1/9),
−MASS−.1278(4/9),−KNN−1.2125(0/9),−KM−1.3554(1/9),−RANDOM+2.2018(8/9).
GCE selected globally by noisy VALIDATION average among LS/GCE/BOOT, not per-cell test.
MASS matches each node's GB observed-class probability, uniformly redistributes the remainder;
retains GB-derived strength, only isolates class direction. KNN precisely matches each query's LOO votes.
KM matches actual52/64/64 regions, not support; random preserves both group sizes and training counts.

Real CE exceeds random CE21.6602 by34.9416pp and shuffle CE17.3037 by39.2981pp, both9/9positive.
Random lacks alignment/BN calibration, not compute matched; shuffled100+100 budget is matched.
Both control GB targets still use real GRACE geometry. No pure-X SSL control, so not uniquely causal topology proof.
BOOT current detached prediction gradient equals0.8CE; verified. Not a strong EMA self-correction baseline.

Clean GB−CE−1.4776/−1.2457/−2.3602pp. Clean GCE CoraFull−3.7604pp; KNN−1.0393pp;
do not promote alternatives as universally safe. Representation gates pass, GB only beats matched random;
local direction/generic/KM/KNN/clean gates fail. Full GB continuation false. Stop current defaultGB-onR2 branch.
Keep R2 backbone, no c-group/deeperhead/fine-tune, no new combos/seed expansion without authorization.

10 tests,9 GPU smokes,172 frozenfile hashes,48 independent targets(maxdiff1.034e−7),
156 reloads,24 exactoldR2 anchors,156 independentaccuracy(maxdiff1.421e−14pp),144paired diffs,
96 independently checked neighbor queries pass. Formal grid56.474s, excludes oldSSL/R2cost.
Raw preparation warning joblib physical core detection only; stable argsort dim keyword fixed before freeze.
All fitting GPU ROCm PyTorch2.9.1 RX7700XT via D:/codex_temp/noisygl-rocm721/Scripts/python.exe.
KM CPU4threads. Mean/std: one protocol/student seed, std=null; no significance claim.
