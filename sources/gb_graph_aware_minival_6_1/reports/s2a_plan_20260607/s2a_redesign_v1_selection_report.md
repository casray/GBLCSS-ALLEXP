# S2-redesign-v1 — ball-guided sample selection (commit-style): result

> **single-cell smoke** (Cora / `ratio_train0.2_val0.1` / split0 / pretrain0 / symmetric 0.4 / seed 0 /
> frozen GRACE). **NOT planetoid, NOT NoisyGL.** One narrow shot with a pre-registered gate.
> Evidence: `reports/s2a_plan_20260607/s2a_redesign_v1_selection.{csv,json}`,
> code `scripts/run_selection_experiment.py` + `src/noisy_label_gb_grace/evaluation/selection.py`.

## Mechanism (fixed, identical across method + controls)

Frozen-GRACE linear head, **hard noisy** labels, per-epoch selection: keep top-`k` train nodes by
`z(−loss) + z(rel)`; schedule warmup `W=20` → ramp `20` → hold `ρ=1−noise=0.6` (fixed a priori, **not**
swept). Only the reliability `rel` source differs between rows. Clean labels eval-only.

## Results (test_acc_at_best_val; ce/gb_soft from Phase 1 smoke)

| row | rel source | test | sel_precision | noise_excl_recall | coverage |
|---|---|---|---|---|---|
| ce (smoke) | — | 0.7878 | – | – | – |
| gb soft-target (smoke) | — | 0.7962 | – | – | – |
| clean-oracle CE (headroom) | — | **0.8221** | – | – | – |
| **sel_dcgb_emb (METHOD)** | reciprocal-kNN reliable | **0.7788** | 0.8738 | 0.811 | 0.600 |
| sel_noball | loss only | 0.6642 | 0.6493 | 0.194 | 0.921 |
| sel_gb | ordinary GB consensus | **0.8094** | 0.9046 | 0.857 | 0.600 |
| sel_shuffled_matched | shuffle(dcgb) | 0.6705 | 0.6557 | 0.226 | 0.900 |
| sel_random_matched | shuffle(dcgb,2) | 0.6774 | 0.7785 | 0.668 | 0.600 |
| sel_topk_matched | kNN-train consensus | 0.7930 | 0.9138 | 0.871 | 0.600 |
| sel_clean_oracle (diag) | noisy==clean | 0.8126 | 1.0000 | 1.000 | 0.600 |

## Pre-registered gate → **STOP**

| # | criterion | result |
|---|---|---|
| 1 | sel_dcgb_emb > gb_soft 0.7962 | **FAIL** (0.7788; also < ce 0.7878) |
| 2 | sel_dcgb_emb > sel_noball | pass (0.7788 > 0.6642) |
| 3 | sel_dcgb_emb > random & shuffled | pass (> 0.6774 / 0.6705) |
| 4 | sel_prec(dcgb) > noball & random | pass (0.874 > 0.649 / 0.779) |
| 5 | fixed keep-ratio | pass (by construction) |

**Criterion 1 fails → per the agreed rule, STOP the reliable-mutual-kNN method line; do not expand
seeds, do not tune ρ, do not open the NoisyGL pair lane. Pivot to S3.**

## Honest secondary findings (single-cell, hypothesis-generating — NOT promoted to claims)

1. **The commit mechanism converts the ball signal — soft-target was the bottleneck.** Unlike Phase 1
   (where dcgb tied shuffled), here sel_dcgb_emb **crushes** no-ball/shuffled/random on both test
   (0.779 vs 0.66–0.68) and selection precision (0.874 vs 0.65–0.78). Criteria 2–4 all pass. The
   "conversion failure was the soft-target form" hypothesis is **confirmed**.
2. **The reciprocal-kNN "reliable" ball is NOT the best selection signal.** Within the *same* mechanism,
   **`sel_gb` 0.8094 > `sel_topk` 0.7930 > `sel_dcgb_emb` 0.7788.** Ordinary GB consensus and a plain
   kNN-train consensus both select better than the mutual-kNN reliable ball — even though dcgb's *ball
   purity* (0.889) far exceeds gb's. **Ball purity does not predict selection/downstream quality.**
3. **`sel_gb` captures ~63% of the 3.43pp headroom** (0.8094 vs oracle-CE 0.8221, vs ce 0.7878) and
   beats gb_soft — but this is **single-cell and NOT the pre-registered method**; promoting it now would
   be exactly the cherry-picking the protocol forbids. It is an **S3 hypothesis**, not a result.
4. **Loss-only selection is harmful here** (no-ball 0.6642 < ce 0.7878): the small-loss trick alone
   picks confidently-wrong noisy points on the frozen head; the ball consensus is what rescues it.

## Decision & S3 reframing

The reliable-mutual-kNN-ball *method line* is closed on this task (fails its pre-registered gate twice
over: < gb_soft and < ce, and beaten by ordinary-GB selection). **Pivot to S3**, now with a sharp,
evidence-backed question:

> **Why does ball *purity/AUROC* (dcgb ≫ gb) NOT predict *selection/downstream* quality (sel_gb >
> sel_dcgb)?** i.e. representation × ball-construction × which reliability signal actually converts.

S3 should study (multi-seed, with controls — not as a declared winner): ordinary-GB / kNN-consensus /
reliable-ball selection signals vs their ball-quality metrics, to characterize the quality→conversion
gap. The `sel_gb` thread is a candidate to investigate properly, **not** a method to announce from one
cell.

## Guardrails honored

Pre-registered single-cell gate; fixed schedule identical across method + controls (no ρ tuning, no
threshold shopping); clean labels eval/diagnostic only; ratio-split symmetric, not merged with NoisyGL.
No new method is claimed from this smoke.
