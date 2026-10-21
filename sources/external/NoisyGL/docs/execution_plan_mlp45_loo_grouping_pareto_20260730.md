# MLP-45 LOO Grouping Pareto Plan

## Research question

After promoting tie-safe LOO-GB-LCSS, determine whether granular balls retain a
defensible advantage over alternative local-consensus groupings. The comparison
must not weaken kNN or tune any method on test accuracy.

## Fixed protocol

- Datasets: Cora, CiteSeer, CoraFull, Coauthor-CS, Amazon-Photo.
- Noise: uniform 0.4, train-only.
- Split: percent 0.2/0.1/0.7.
- Seeds: 0-4 for accuracy; seed 0 for isolated construction benchmarks.
- Representation: the same frozen GRACE H used by the main experiment.
- Target: beta=0.5, multiclass alpha, `tie_mode=distribution_only`.
- Self exclusion:
  - GB and k-means use leave-one-out votes.
  - kNN excludes the query node from its neighbor set.
- kNN neighborhood size is matched to the seed-specific mean number of train
  votes per GB, as in MLP-43. It is not selected by test accuracy.
- k-means cluster count is matched to the seed-specific GB count.

## Accuracy inputs

- LOO-GB: MLP-44 audited uniform 0.4 rows.
- Exact kNN: MLP-43 corrected rows with `exclude_self=true`.
- LOO-k-means: new five-seed rows. This is rerun because the existing MLP-43
  k-means target includes each node's own vote.

## Cost benchmark

Each structure is built in a fresh process to avoid allocator carry-over.
Report:

- wall-clock construction time;
- peak RSS increase above the loaded-data baseline;
- grouping-structure storage excluding the common dense soft-target tensor;
- repeated target-reconstruction time after the structure is available;
- asymptotic complexity and the actual matched k / cluster count.

The measured implementations are:

- recursive 2-means GB;
- exact dense-similarity kNN;
- sklearn k-means with `n_init=10`.

Approximate kNN is reported as unavailable unless a maintained ANN package is
already present in the experiment environment. No hand-written approximate
index may be substituted.

## Preregistered interpretation

The experiment is descriptive and cannot be used to select a favorable k.

- `fixed_budget_pareto_supported`: mean LOO-GB accuracy is within 1pp of the best
  alternative and at least 3/5 datasets are within 1pp, while GB has lower
  measured peak memory or construction time than exact kNN on at least 4/5
  datasets.
- `implementation_profile_only`: accuracy is competitive but the measured cost
  advantage is not stable. GB remains one fixed-budget implementation, without
  an efficiency-superiority claim.
- `alternative_dominates`: mean accuracy deficit exceeds 1.5pp and no stable
  measured cost advantage is observed. The paper must foreground local
  consensus rather than granular balls.

Regardless of outcome, kNN/k-means results remain visible in the paper.

