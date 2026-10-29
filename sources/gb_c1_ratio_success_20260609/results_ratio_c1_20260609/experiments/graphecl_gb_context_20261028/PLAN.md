# GB-ContextECL revised implementation plan

Goal: implement and start the user-authorized context-weighting experiment, retaining all results and engineering evidence.
Architecture: independent experiment directory, immutable old sources, CPU context statistics, exact chunked GraphECL gradients, shared warmup and frozen native-head evaluation. Python/PyTorch/NumPy/SciPy.

## Decisions after external review

Accept Cos control, offline diagnostics, eta=.25/1, and Roman mechanism extension. Both doses receive matched Cos and Shuffle controls: ECL plus six variants. Cos uses (1+cos)/2 to ensure the same score range. Correlation>.8 and small edge perturbations are descriptive warnings; gradient magnitude is also measured. Only invalid inputs, degenerate zero intervention, or failed engineering checks block training. Label diagnostics run after all fitting, not to select a specification.

No intermediate legacy checkpoints exist. Offline preflight uses ten existing terminal H artifacts, clearly marked terminal. Warmup replay is new training and its cost is recorded; repeat diagnostics at actual midpoint. Five main datasets retain original GraphECL configurations. Roman uses official splits0/1 and the existing undirected graph/noise generator; GraphECL configuration is a project adaptation (1024 dimensions,2 GNN layers,100 epochs,lr.0005,wd0,tau.5,lam.001,one projector). No official Roman GraphECL configuration was found.

Primary comparison keeps native normalized H, same initial linear head, CE/fixed ECL SUPPORT32; additional own SUPPORT32 isolates changed geometry. Five main graphs are the primary aggregate, Roman is a separate mechanism result. Two SSL/protocol seeds, U.4/U.6/Pair.4, no clean veto or new clean training. No R2/KNN/KMeans comparison/D1C1 search.

84 logical SSL endpoints:6 graphs×2 seeds×7 arms. Baseline is freshly replayed to audit original five-graph endpoints, and Roman baseline is new. Shared midpoint saves duplicate warmup. 720 logical heads:12×[6 baseline CE/fixed heads +6 variants×9 heads]. Existing60 native baseline heads may be referenced only after exact replay; this implementation refits all heads to keep one kernel, with legacy baseline validation checks. Both doses reported separately; no test-based dose choice. Validation screening reference: primary five-graph fixed-target SUPPORT mean gain>=.5pp, both seeds positive,>=3/5 graphs positive and GB>matched Cos and Shuffle mean. Roman evidence separate. Test results reported regardless of research gate.

## Tasks

- [x] Write test_context.py; run pytest to observe missing-module failure. Test explicit leave-one-out reference, range/mass/self-loop, isolated/degree1, cosine range, shuffle multiset, zero-dose and dense/chunk gradients.
- [x] Implement context.py; pass those numerical tests.
- [x] Implement runtime.py/config.json for confined atomic outputs, source hashes, exact inherited configurations, five-graph cached supervision and Roman official data.
- [x] Implement run.py: prepare signatures; terminal preflight; midpoint replay; matched interventions; native heads; post-fit scoring and reports. Preserve stage-specific checkpoints and failures. No test labels in preflight or fitting.
- [x] Run real GPU smoke with resume consistency, input replay, nonzero intervention and linear head validation; source audit after tests.
- [x] Implement launch.py with hidden process and OS single-instance lock, then launch all. Do not terminate unrelated processes.
- [x] Update design/handoff with actual preflight, startup status and paths. Completion report follows scoring and independent checks.

Expected source files: context.py, runtime.py, run.py, analyze.py, launch.py, test_context.py, config.json, PLAN.md, REVIEW_RESPONSE_zh.md, START_HERE.md. Runtime D:/codex_temp/graphecl_gb_context_20261028; old source files remain read-only.

Research hypothesis: granular conditional neighbor distributions improve noisy supervision utility. Role: label-free regional context estimator. Failure interpretations include weak intervention, representation/target disagreement, similarity-control equivalence, and graph-specific reversal. Record raw logs/config/seeds/means/sample-SD/paired deltas/cost/audit. Status: implementation in progress, no performance claim.


Execution: source receipt frozen (570 entries),9 tests and repaired GPU smoke passed; ten terminal offline preflights complete. Formal runner started, Cora seed0 baseline exact model replay passed. All720 heads are refitted to one kernel; old60 provide engineering comparisons. No final performance results yet.
