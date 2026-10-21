# DCGB v2.8 — Dual-view Consistency Granular Ball (ball-construction redesign)

## 1. Research Hypothesis

DIR-07 reached a wall: `grace_gagb` has high noise-detection AUROC but low clean purity (it mixes
classes), `Q(B)` filtering and boundary-safe repair only act as safety valves, and a matched
embedding top-k control beats `grace_gagb` on AUROC — so the current builder has no value beyond
embedding kNN. The recorded decision was to change the **ball construction / reliability model**,
not tune more target-repair formulas.

Hypothesis: if a ball admits a node only when **two independent views agree** (GRACE embedding
*and* graph topology), the ball should (a) keep `gb`-level clean purity instead of collapsing like
`grace_gagb`, and (b) carry a reliability signal a single-view top-k cannot replicate.

## 2. Granular Ball Role

Weak local repair unit / reliability signal — never a hard label, view generator, or prototype.
Clean labels are used only for diagnostics; grouping uses embeddings + topology + noisy train labels.

## 3. Method

New builder `build_dcgb_balls(...)` in `src/noisy_label_gb_grace/balls/graph_aware_ball.py`
(payload via the existing `_payload_from_groups`; reuses `_ppr_lite_matrix` / k-hop fallback):

- **Anchor = dual-view consensus interior node** — highest `|EmbTopM(i) ∩ TopoTopM(i)|` (replaces the
  degree·density hub anchor).
- **Generous pool, corroboration-gated membership** — candidate pool is `EmbTopM(c) ∪ k-hop(c)`
  (keeps balls substantive); node `j` joins anchor `c` only if `A[c,j] ≥ tau_h` **and** corroborated
  by a second view: reciprocal embedding kNN (`c ∈ EmbTopM(j)`) **or** topology support
  (`T[c,j] ≥ tau_topology`). Greedy growth with the monotone `delta_q` quality acceptance.
- Modes `emb_only` (reciprocity corroboration only) / `topo_only` (topology only) for ablation.
- Repair reuses the **unchanged** boundary-safe path; `Q(B)` reuses `compute_group_quality_scores`.

Minimal method set: `dcgb`, `dcgb_emb_only`, `dcgb_topo_only`, controls
`topk_dcgb_train_matched` / `kmeans_dcgb_matched` / `dcgb_shuffled_alpha_matched`, and one repair
method `dcgb_qsafe_boundary_alpha09_q50`. No alpha / q-quantile / class-safe sweeps.

> Note: the first parameterization (strict reciprocal ∩ topology *intersection*) collapsed to
> singletons (avg ball size 1.86, 10 eligible balls, purity trivially 0.99, AUROC 0.51). The
> corroboration-gate (OR within an embedding floor, generous pool) fixes the collapse. This change
> is part of building the new reliability model, not formula-shopping on `grace_gagb`.

## 4. Experiment Config

- Datasets `Cora`, `CiteSeer`; symmetric noise `0.2`, `0.4`; split `ratio_train0.2_val0.1_split0`;
  seeds `0 1 2 3 4`; frozen GRACE embeddings; noisy-train-only linear teacher.
- Classifier-free grouping diagnostic (`scripts/diagnose_grace_first_gagb_grouping.py`, the gate used
  for v2.5) + a seed-0 classifier smoke (`run_noisy_classifier.py`, Cora 0.4) for plumbing.
- Outputs: `reports/dcgb_v2_8_grouping_diag_20260605/{grouping_summary,per_seed_grouping_diagnostics}.csv`.

## 5. Main Results (5-seed mean)

Clean purity / noise-detection AUROC (classifier-free):

| setting | gb purity / AUROC | grace_gagb purity / AUROC | **dcgb** purity / AUROC | topk_dcgb AUROC |
|---|---|---|---|---|
| Cora 0.2 | 0.893 / 0.682 | 0.699 / 0.890 | **0.892 / 0.787** | 0.949 |
| Cora 0.4 | 0.893 / 0.645 | 0.699 / 0.850 | **0.892 / 0.733** | 0.902 |
| CiteSeer 0.2 | 0.797 / 0.606 | 0.635 / 0.811 | **0.751 / 0.705** | 0.845 |
| CiteSeer 0.4 | 0.797 / 0.582 | 0.635 / 0.777 | **0.751 / 0.669** | 0.797 |

DCGB AUROC vs gb (mean ± std, non-overlapping in all four settings):
`Cora 0.2` 0.787±0.035 vs 0.682±0.020 · `Cora 0.4` 0.733±0.013 vs 0.645±0.011 ·
`CiteSeer 0.2` 0.705±0.022 vs 0.606±0.013 · `CiteSeer 0.4` 0.669±0.013 vs 0.582±0.024.

`dcgb_qsafe_boundary_alpha09_q50` repair (NCR / CDR): `Cora 0.2` 0.087 / 0.000 · `Cora 0.4`
0.028 / 0.000 · `CiteSeer 0.2` 0.081 / 0.012 · `CiteSeer 0.4` 0.016 / 0.001 — positive correction
with bounded damage. Seed-0 Cora 0.4 classifier smoke (200 ep): `gb` 0.7951, `dcgb` 0.7878,
`dcgb_qsafe` 0.7893 — runs clean, no accuracy win on this single hard setting.

## 6. Mechanism Findings

**Supported:**
- DCGB **fixes the purity collapse**: it recovers `gb`-level purity (Cora 0.892 vs 0.893) with full
  coverage and substantive balls (≈72–100 eligible), unlike `grace_gagb` (purity 0.699, 24 balls).
- At matched purity, DCGB **beats ordinary `gb` on noise-detection AUROC in all 4 settings with
  non-overlapping std** — a genuine, statistically clean grouping-quality gain over the GB baseline.
- The repair mechanism is positive and conservative (NCR > 0, CDR ≤ 0.012).

**Not supported (the graph-aware claim is still not isolated):**
- **Topology corroboration adds nothing.** `dcgb` ≡ `dcgb_emb_only` to three decimals in every
  setting → on these GRACE embeddings the topology OR-branch never changes membership; the entire
  gain comes from *reciprocal (mutual) embedding kNN*, not graph structure. This reproduces the core
  DIR-07 finding with a cleaner builder.
- **The matched control still wins AUROC.** `topk_dcgb_train_matched` beats `dcgb` in all 4 settings
  (e.g. Cora 0.2 0.949 vs 0.787). This is structural — per-node top-k gives every train node a
  tailored full-coverage neighborhood, so it out-detects *any* partition (`gb`, `gagb`, `grace_gagb`
  all lose to their top-k equivalents). DCGB has therefore not shown value beyond embedding kNN.

## 7. Decision

**modify / mechanism-positive, not method-ready — do NOT promote to a full 5-seed classifier gate yet.**

DCGB is a strictly better *partition* than both `grace_gagb` (fixes purity collapse) and `gb` (higher
AUROC at matched purity, non-overlapping std), and it is a clean, falsifiable redesign. But its value
is from reciprocal-embedding partitioning, **not** from graph topology: the topology view is inert on
Cora/CiteSeer GRACE embeddings, and the per-node top-k control still detects noise better. So the
"graph-aware granular ball" claim remains unsupported.

Recommended next step (one of):
1. Reframe DCGB as a **reciprocal-embedding** reliability partition (drop the graph-aware framing) and
   test downstream repair vs `gb` on a full 5-seed gate — the AUROC gain over `gb` is real and may
   transfer.
2. If the graph-aware claim is to be kept, test where embedding and topology genuinely **disagree**
   (weaker/early embeddings, heterophilous graphs) so topology has room to add separable value;
   on strong GRACE embeddings it does not.

Do not claim accuracy success: the single-seed smoke shows no win on Cora 0.4, and DCGB has not
passed the matched-control falsification test.

## 8. Guardrails honored

No change to GRACE, the teacher, target-repair formulas, splits, or evaluation. Clean labels are
diagnostics-only. Success judged by purity + AUROC + matched controls (`topk_dcgb_train_matched`),
never purity alone; within-std deltas reported as inconclusive.
