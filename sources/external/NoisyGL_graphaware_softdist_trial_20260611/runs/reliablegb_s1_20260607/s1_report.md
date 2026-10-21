# S1 — Reliable Granular-Ball Denoising on GCN (smoke result, controls corrected)

## Hypothesis

Replace ordinary-GB with reciprocal-kNN **reliable** granular balls and a conservative reweight
denoiser on a GCN host; fix OrginGB's `best_phase=warmup` failure via periodic ball rebuild + EMA, and
check whether reliable-ball denoising can enter the optimization trajectory and beat GCN + controls.
Granular ball role: weak local reliability signal driving conservative sample reweighting. Clean labels
are diagnostics/eval only.

## What was built

- `utils/reliable_gb.py` — reciprocal-kNN reliable balls (mutual k-NN membership, full coverage),
  same payload shape as `ordinary_gb`. Unit-tested (purity, coverage, determinism, random-match): 4/4 pass.
- `predictor/ReliableGB_GCN_Predictor.py` — GCN host with **periodic rebuild** (`gb_refresh_interval`)
  + **EMA hidden** (`gb_ema_decay`); methods `reliablegb_gcn_reweight` / `_random` / `_noball`.
- Registered in `single_exp.py` / `total_exp.py`; configs for Cora/CiteSeer.

## Control-bug correction (peer-review finding)

The first run's no-ball control was inert: its config set `suspicion_mode: gb`, which `_conf_value`
used to override the predictor's `suspicion_mode="no_ball"` class attribute, so the "no-ball" run
silently re-ran real-GB (CSV showed `suspicion_mode=gb`, precision identical to real). Fixed by setting
`suspicion_mode: no_ball` in `config/reliablegb_gcn_reweight_noball/*_{cora,citeseer}.yaml` and
re-running. The size-matched **random** control was unaffected (it acts via `random_control=True`,
which no config key overrides). The buggy CSV is archived at
`reliablegb_gcn_diagnostics_BUGGY_noball.csv`; numbers below are from the corrected re-run.

## Smoke result — Cora uniform 0.4, seed 3000 (corrected controls)

| method | warmup | test acc | best_phase | gated precision | suspicion_mode |
|---|---|---|---|---|---|
| gcn (baseline) | – | 0.7050 | – | – | – |
| reliablegb_gcn_reweight | 30 | 0.7050 | warmup | 0.5000 | gb |
| reliablegb_gcn_reweight_noball | 30 | 0.7050 | warmup | 0.5000 | no_ball |
| reliablegb_gcn_reweight_random | 30 | 0.7050 | warmup | 0.4286 | gb (random_control) |
| reliablegb_gcn_reweight | 3 / 5 | 0.6970 | **gb** | – | gb |
| reliablegb_gcn_reweight_noball | 3 / 5 | 0.6970 | **gb** | – | no_ball |

Balls were well-formed (549 reciprocal balls, mean size 4.9).

## Decision: **No-go on GCN host → proceed to S2 (per the agreed rule).**

Supported by the corrected evidence:

1. All reliable-GB variants tie GCN on test (0.7050) at warmup=30, and `best_phase` stays `warmup`.
2. With corrected controls, real-GB gated precision (0.50) **ties the now-valid no-ball control (0.50)**
   and only marginally beats random (0.43) — on the GCN warmup representation the reliable ball's
   gating gives no advantage over a no-ball confidence ranking.
3. Moving warmup earlier (3/5) does put the GB phase into the trajectory (`best_phase=gb`), but test is
   no better (0.697).

**Observed risk (not a universal theorem):** on Cora uniform 0.4 seed 3000, GCN val-acc peaks around
epoch 6 (0.442) and then declines as the model memorizes the 40% noisy labels (by epoch 30, train-acc
≈0.99, val-acc ≈0.37). So balls built after warmup sit on an already noise-memorized representation.

**Conclusion (supported wording):** in the Cora uniform 0.4 seed-3000 smoke, reliable-GB reweight on
the supervised GCN hidden state gave no training gain (test ties or falls below GCN, and the gating does
not beat a valid no-ball control); we record S1 as no-go and move to a label-free contrastive
representation. We do NOT claim that every "warm-up GCN → build balls → denoise" scheme is structurally
impossible — only that this host/protocol shows no gain and a clear noise-memorization risk that a
label-free representation is designed to avoid. Full multi-seed / multi-setting confirmation is deferred
(this is a single-seed smoke, not a formal result).

## What carries forward to S2

- `utils/reliable_gb.py` (validated construction) — reuse on the frozen contrastive representation.
- No-ball (now correct) + random-size-matched control discipline.
- The val-trajectory observation as S2 motivation: a label-free contrastive encoder never sees labels,
  so it cannot memorize noise the way the supervised GCN does here.
- S2a (next step, not run this session): build the SAME reliable balls on a frozen contrastive
  embedding (reuse `gb_graph_aware_minival_6_1` classifier-gate pipeline; `build_dcgb_balls(mode="emb_only")`
  is this construction) and test whether the contrastive representation is the missing ingredient.

## Guardrails honored

Reweight only (no target change, no hard positives); clean labels diagnostics-only; controls at matched
gate size (no-ball now actually active after the fix); did NOT threshold-shop after the tie — moved to
S2. Configs restored to warmup=30 after the timing probe; single-seed smoke is not reported as a formal
multi-seed result.
