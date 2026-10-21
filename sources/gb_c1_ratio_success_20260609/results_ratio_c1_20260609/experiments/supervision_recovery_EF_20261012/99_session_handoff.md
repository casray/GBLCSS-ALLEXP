# 2026-10-12 E/F recovery experiment handoff

Status: completed. No remaining training/analysis sessions. All test outcomes evaluated after fits.
User authorized execution after accepting the E/F design. Submitted manuscript unchanged.

Directory: experiments/supervision_recovery_EF_20261012.
Environment: D:/codex_temp/noisygl-rocm721/Scripts/python.exe.
Former foreground exec session17780 completed successfully; do not resume or rerun it.
Log: raw_log.txt; configs/protocols/checkpoints under this experiment only.

Three datasets Cora/CiteSeer/CoraFull, uniform .4/.6, full protocol seeds1/2/3.
A original noisy; B all clean; C oracle removal; E relabel suspected nodes using A argmax;
F remove supervision from same suspects. Rank 1-P(observed label); ceil20%; stable ties.
No truth/test-guided threshold search. Original ERASE and DIST functions reused unchanged.
Teacher max400/patience150; student200, rawX hidden64, DIST T1. No GB/CE/region objective.
Seed1 A/B/C input fingerprints matched previous oracle caches and are reused explicitly.
B rate .6 shares rate .4 effective inputs; 90 logical rows, 81 distinct pairs,66 new pairs.
6 tests passed; preflight18 protocols and54 oracle-intervention checks passed.
Frozen source/hash/settings manifest saved before grid. Do not edit frozen sources during training.

All66 new pairs completed without failure/retry in3357.456seconds.90 logical rows exported.
analyze.py and independent_audit.py finished successfully; six tests passed (tests.xml).
Teacher reload maxdiff2.086e-7, identical argmax; student90/90 bit-exact.
Strict-first-validation and effective input/initialization checks passed. Independent intervention,
accuracy/balanced and paired/macro recomputation maxdiff1.421e-14. Frozen old sources unchanged.

Teacher/student macro effects(pp): B-A4.0620/4.0914; C-A2.6236/2.4534; B-C1.4384/1.6380;
E-F.1317/-.0819; E-A.1438/-.0391; F-A.0121/.0428; B-E3.9182/4.1305.
Oracle recovery gate passed; realistic recovery/net improvement gates failed.
Student E-F7wins/11losses, only1/6 cell means positive; E-A7wins/1tie/10losses,2/6 positive.
CoraFull B-C means2.9881/4.8166, but E-F six runs all negative, means-.5365/-1.0658.
Cora .6 E-F allnegative, mean-1.4424. CiteSeer .6 is the sole consistent positive cell:
E-F2.6632 and E-A2.2194, allthree seeds positive. Do not claim general benefit.
E reduced training corruption in all18 cells: repaired4286 and newly damaged279 node instances;
15498 wrong instances became11491. Counts are not distinct nodes across runs.

Detailed Chinese report:实验结论.md. Direction history44 and root项目核心总结_新窗口上下文.md updated.
Decision: stop fixed20% one-pass hard correction as a general accuracy route; no sweeps/GB modules.
If authorized to continue, first test oracle relabeling G on exactly the existing marked sets to
separate replacement-label errors from low-value selected nodes/pipeline effects. Not started.

Primary gate: student E-F mean>=.5pp, >=4/6 cell means positive, all3 seed macros positive.
Net improvement gate uses E-A identically; oracle recovery B-C threshold1pp.
Budget7200s checked between pairs. Teacher retraining non-bit-exact known from earlier audit;
checkpoint reload requires maxdiff<=1e-6 and identical argmax.
Three existing-dataset seeds are not external confirmation or evidence of novel SOTA.
