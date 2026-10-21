# Representation-supervision pilot handoff, 2026-10-21

Status: COMPLETE. No running jobs or automatic expansion. User approved design and execution after the integrated GPT/Claude review.

Read 综合结论与接续.md first; exact numbers in results.csv, paired_deltas.csv, decision.json. Frozen protocol and commands in PLAN.md, config.json, COMMANDS.md. Existing Python: D:/codex_temp/noisygl-rocm721/Scripts/python.exe, RX7700XT ROCm.

Scope: Cora/CiteSeer/CoraFull, clean/u0.4/u0.6/pair0.4, full protocol/student seed0, frozen old H and historical assignments. 144 main +24 legacy +24 H references; four validation-only tuning fits, three reusable pretrains. 192 grid fits completed in 511.259 seconds. One seed means std=null, not 9 independent replications.

R0 no alignment200; R1 joint200 lambda1; R2 alignment100 then frozen encoder/BN head100. Same raw-X MLP64 BN/dropout.5 at deployment; training projection discarded. T0 noisyCE, T1 LOO-LCSS, T2 small-loss, T3 soft-small-loss LCSS. Known synthetic rate used by filtering.

Noisy 9-cell macro accuracy T0/T1/T2/T3: R0 35.515/43.420/39.451/47.356; R1 38.022/47.699/44.375/51.321; R2 56.602/57.921/50.658/53.896. R2 T1-R0T1+14.501pp; R2 T1-T0+1.320pp (8/9 positive). R2 combo-best(T1,T2) -4.062pp (3/9 positive); R0/R1 combo increments +3.936/+3.217 (9/9 positive each).

Clean R2 T1-T0 Cora/CiteSeer/CoraFull -1.478/-1.246/-2.360pp. Therefore full LCSS-value gates failed due to preregistered clean damage cap, despite noisy gains. Both representation gates passed. Do not flip gate decisions or hide clean failures.

Global validation-selected R2/T1 uniform.4 test75.673/71.005/47.333; same-seed SimMLP75.620/67.139/40.166. Cora difference is one test node. Not a five-seed/5-dataset/SOTA claim. H-MLP CoraFull54.782 remains above student47.333.

14 tests passed; 12 input protocols; 52 frozen source/input hashes; 192 checkpoint reloads exact; independent192 accuracy recomputation exact. Frozen R2 weights/BN and first clean-validation maximum verified. Preparation format error retained in ENGINEERING.md/failure_prepare.json; no formal-fit failure. audit_independent.py was added separately after freeze and is not imported by training.

Next research proposal, not started: matched random-frozen/H-row-shuffle controls for R2, then frozen-config multiseed confirmation if appropriate. R2 changes alignment, ordering and encoder freezing together, so no proven gradient-conflict cause. kNN/kmeans target replacements are needed before any GB-specific benefit claim. Do not launch dynamic targets, ball graph, support shrinkage, new seeds or manuscript edits automatically.
