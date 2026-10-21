# Adaptive sqrt-N granular-ball experiment plan

> For agentic workers: use superpowers:subagent-driven-development or superpowers:executing-plans. Frozen before outcome inspection; execution state belongs in separate files.

2026-10-14。用户明确授权修改粒球生成算法，先改善不同数据集的适配性，再推进算法。使用sqrt(N)经验尺度及粒球几何分裂思想，不采用类别纯度或干净标签控制构球。

**Goal:** Separate budget, splitting rule, and voting support effects; determine whether adaptive balls improve the existing LCSS pipeline.

**Architecture:** Add a versioned label-free generator alongside the historical generator. Reuse frozen GRACE, official/prior splits, original LCSS and original attribute MLP. Every group/target is frozen before clean-label evaluation.

**Tech stack:** NumPy, SciPy, scikit-learn KMeans, PyTorch/PyG, existing GB-LCSS training/target code.

## Research hypothesis and ball role

Old min_child32 and fixed 5% average-distance gain impede fine partitions on small/high-dimensional datasets. A sqrt-N budget plus size-adaptive, best-first geometric splitting may improve local semantic evidence. Balls remain supervision neighborhoods; no hard positives/negatives/prototype pulling.

Number is a scale prior, not a proven optimum. N is the full node/embedding count, including unlabeled nodes. Use B0=ceil(sqrt(N)): Texas14, Wisconsin16, CoraFull141, Coauthor-CS136, Roman151. A separate fixed64 adaptive control identifies splitting effects independent of number.

## Generator spec

- Input normalized embeddings and optional training indices; API has no labels/classes.
- Requested target B; default ceil(sqrt(N)). Geometric min leaf m=max(1,ceil(N/(4B))), replacing the dataset-insensitive 32.
- Best-first split the leaf with largest total SSE=sum(||z_i-center||²), preventing tiny diffuse leaves from outranking large diffuse regions solely by mean radius. Cache completed leaf proposals; deterministic IDs/seeds.
- Propose two-means with n_init10. Accept only nonempty children meeting size constraints and positive geometric SSE reduction >max(1e-12,1e-8*parentSSE), replacing the 5% average-distance gate. This is an experimental variance criterion, not a claimed verbatim reproduction of any cited method.
- If two-means violates size/support constraints, try a deterministic constrained projection split along center difference (or a nonzero fallback direction). Select a feasible boundary by within-child SSE, reject cuts between identical projected coordinates, reject zero-gain or identical geometry. No arbitrary node-ID splitting to force count.
- Geometric mode uses no training positions. Support mode additionally requires >=4 training nodes in every split child: original LCSS needs two other votes, so four provides one extra support node. Budget is capped by floor(n_train/4), with at least one root allowed if insufficient total training support. Preserve explicit requested/feasible/actual counts and stop reasons; never report budget as actual count.
- Report sizes/radii/SSE/train counts and split trace. Uniform/pair/clean labels must yield the identical geometry for fixed embeddings/training indices/seed.

## Controlled variants

1. GB64_old: frozen old actual assignments.
2. GBsqrt_old: only replace old max_balls by B0; keep32/.05.
3. GB64_adaptive: geometric new generator at requested64.
4. GBsqrt_geo: geometric new generator at B0.
5. GBsqrt_support: new generator at B0 with training-support constraint (primary candidate).
6. KM_match: global k-means with the actual GBsqrt_support count, n_init10, same seed.
7. KNN_match: unique other training voters per root exactly match GBsqrt_support count, same frozen representation.
8. CE: observed one-hot labels.

No new alpha/weighting/loss or representation; LCSS min_votes2/beta0.5 unchanged. All groups label-free; support uses only fixed split membership, not noisy/clean labels. This preserves independence of candidates from artificial label flips.

## Data, protocol, phases

Five datasets: Texas/Wisconsin/Roman use existing official split0–4; CoraFull/Coauthor-CS original20/10/70 protocol seed0–4. Same frozen GRACE per dataset (only SSL seed0); all clean/Uniform.4/Pair.4, original noise function. Existing input files are read only and hash-checked. No split or test-label tuning.

Phase A: 25 dataset/seed partitions; 75 dataset/seed/noise cells, 600 arm rows. Report requested/feasible/actual ball counts, min/max/median sizes, support coverage/effective support, clean voting margin, q/target Brier and original-clean/corrupted harm/recovery. Unique voters only, exclude root.

Mechanism screening: primary vs old64, pooled Uniform/Pair per paired seed, mean Brier delta<=-.005 with >=4/5 improvements marks a useful signal. Compare KM/KNN and report clean harm and every graph. This is exploratory, not significance. Quantity success is actual target or a documented geometric/support obstruction, not alone algorithm success.

Phase B: bounded seed0 pilot on all five datasets/three conditions with CE, old64, primary, KM_match, KNN_match =75 fits, original200-epoch MLP/config/clean-validation checkpoint, matched init/seed. Pilot is run once to test transfer because prior experiments showed target Brier need not rank final accuracy. Do not claim it passed a mechanism gate if that gate fails. No post-test parameter search. Full-five student confirmation is a later decision after paired diagnostic evidence, not automatic.

## Success/failure and final decisions

Split modification successful structurally if targets are reached without unsupported/degenerate splits; useful scientifically only with target/downstream evidence. Prefer new mechanism improving at least two datasets, with matched ordinary-neighborhood controls. Failure includes arbitrary duplicate splitting, low voting coverage, worse Brier, major clean accuracy loss, or gains only from count/ordinary clustering. Final decision: continue, modify, pause research direction, or abandon the claim (not goal status).

## Files and validation tasks

- [ ] Add GB-LCSS_code/src/gblcss/granular_ball_adaptive.py; preserve old generator for frozen-source audits.
- [ ] Add test_adaptive_ball.py: complete/disjoint coverage, reproducibility, sqrt counts on distinct toy data, support floors/cap, degenerate stop, invalid input, SSE monotonicity, label-free API.
- [ ] Add run_screen.py: root-cause probe, frozen sources/config/splits/partitions/relations/targets, raw log, metrics and paired mean/std; no clean-label partition selection.
- [ ] Add run_pilot.py: original targets/trainer, hashes and saved logits, matching prior baseline results.
- [ ] Independent spec/quality review; tests before GPU work; single GPU process.
- [ ] Add analysis/audit scripts and experiment conclusion; update handoff/literature notes with evidence status and negative outcomes.

Each execution records config, raw log, result tables, mean/std, all seeds/baselines, stop reasons, timing and success/failure. No timing claim against cached baselines.
