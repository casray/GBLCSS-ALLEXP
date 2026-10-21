# GB-GCL Experiment History Addendum 12: Phase 20 Custom SSL Uniformity

Date: 2026-06-15  
Scope: SoftDist [X||H] / NoisyGL downstream line, custom label-free SSL upgrade attempt.

# 1. Current Overall Judgment

- Confirmed: Phase 20 tested the safest custom SSL change: GRACE plus a purely geometric, label-free uniformity regularizer.
- Confirmed: All tested weights `{0.1, 0.5, 1.0}` failed the pre-registered geometry gate on both Cora and CiteSeer.
- Confirmed: Stronger measured uniformity did not produce better granular balls; it collapsed the downstream ball construction to one ball.
- Confirmed: Phase 20b downstream training was not run, because Phase 20a failed exactly under the pre-registered stop rule.
- Likely: For SoftDist [X||H], the useful GRACE property is not maximal uniformity but a balance between alignment, uniformity, and local clusterability.
- Risk: Linear probe can remain acceptable while ball geometry collapses. Probe accuracy must not be used as the gate for SSL selection.
- Next priority: Keep GRACE as the representation anchor. Do not reopen custom SSL unless a candidate first passes a label-free geometry gate.

# 2. Direction Index

| Direction ID | Direction Name | GB Role | Versions Included | Datasets | Best Trend | Main Failure | Decision | Paper Value |
|---|---|---|---|---|---|---|---|---|
| DIR-12 | Label-free custom SSL for ball-clusterable representations | GB only as downstream geometry diagnostic; no labels or balls in pretraining | GRACE + uniformity regularizer, lambda `{0.1,0.5,1.0}` | Cora, CiteSeer | Uniformity metric improves but ball geometry collapses | All candidates form 1 ball and true-purity drops sharply | abandon | negative result / motivation |

# 3. Direction-Level Detailed Record

## DIR-12: Label-Free Custom SSL for Ball-Clusterable Representations

### 3.1 Direction Summary

- Direction ID: DIR-12
- Direction name: Label-free custom SSL for ball-clusterable representations
- Time period: 2026-06-15
- Target problem: Find a better `H` for SoftDist [X||H] without breaking the label-free representation-stage claim.
- Core hypothesis: Since GRACE's InfoNCE uniformity appears useful for ball construction, explicitly strengthening uniformity may yield better clusterable embeddings.
- Baseline: GRACE frozen embedding used by SoftDist [X||H].
- Main related files:
  - `gb_graph_aware_minival_6_1/scripts/pretrain_ssl_candidates.py`
  - `gb_graph_aware_minival_6_1/tests/test_ssl_candidates.py`
  - `external/NoisyGL/scripts/diagnose_embedding_geometry.py`
  - `external/NoisyGL/tests/test_embedding_geometry_diagnostic.py`
- Result files:
  - `external/NoisyGL/runs/phase20_custom_ssl_20260615/geometry_summary.csv`
  - `external/NoisyGL/runs/phase20_custom_ssl_20260615/geometry_gate_decisions.csv`
  - `external/NoisyGL/runs/phase20_custom_ssl_20260615/geometry_report.md`
  - `external/NoisyGL/runs/phase20_custom_ssl_20260615/phase20_custom_ssl_report_20260615.md`
- Raw logs:
  - `gb_graph_aware_minival_6_1/logs/phase20_grace_uniformity_pretrain_20260615_rerun.log`
  - `external/NoisyGL/runs/phase20_custom_ssl_20260615_geometry.log`
- Current decision: abandon

### 3.2 Why This Direction Was Tried

- GRACE/GCL weakness targeted: The SoftDist pipeline depends on an `H` space that is both linearly separable and geometrically clusterable.
- Why GB might help: GB does not enter pretraining; instead, ball diagnostics provide a direct test of whether the new SSL representation is useful for local semantic grouping.
- Expected improvement: More uniform encoder embeddings should avoid crowding and produce better balls.
- Literature or previous experiment inspiration: GRACE / InfoNCE uniformity and Phase 9's finding that SSL replacement can have high probe but poor ball geometry.
- Expected paper claim: Ball-aware representation selection requires clusterability diagnostics beyond linear probe; GRACE is a balanced local optimum for the current SoftDist pipeline.

### 3.3 Versions / Attempts Included

| Version | Main Change | Dataset | Result Trend | What It Tested | What It Showed |
|---|---|---|---|---|---|
| grace_unif0p1 | `L_total = L_GRACE + 0.1 * L_unif` | Cora, CiteSeer | uniformity improves; num_balls collapses to 1 | Small explicit uniformity | Already too strong for downstream ball splitting |
| grace_unif0p5 | `L_total = L_GRACE + 0.5 * L_unif` | Cora, CiteSeer | uniformity improves more; num_balls remains 1 | Medium explicit uniformity | No recovery of ball purity |
| grace_unif1 | `L_total = L_GRACE + 1.0 * L_unif` | Cora, CiteSeer | strongest uniformity; num_balls remains 1 | Larger explicit uniformity | Further confirms collapse |

### 3.4 Role of Granular Ball

- GB role: Diagnostic-only downstream geometry probe.
- Where GB is generated: NoisyGL diagnostic script after pretraining, using the same SoftDist ball construction protocol.
- Representation space: frozen `H` from GRACE or `grace_uniformity`.
- Affects loss: no.
- Affects augmentation: no.
- Affects graph structure: no.
- Affects sampling: no.
- Affects evaluation: yes, as a geometry gate for whether downstream training should be attempted.

### 3.5 Result Summary

| Dataset | Method | Num balls | Uniformity | True all purity | Clean probe | Noisy probe |
|---|---|---:|---:|---:|---:|---:|
| Cora | GRACE | 52.3333 | -1.0107 | 0.7871 | 0.8294 | 0.7949 |
| Cora | grace_unif0p1 | 1.0000 | -3.3627 | 0.3021 | 0.8303 | 0.7840 |
| Cora | grace_unif0p5 | 1.0000 | -3.5755 | 0.3021 | 0.8218 | 0.7692 |
| Cora | grace_unif1 | 1.0000 | -3.6397 | 0.3021 | 0.8218 | 0.7602 |
| CiteSeer | GRACE | 64.0000 | -1.5480 | 0.6991 | 0.7109 | 0.7015 |
| CiteSeer | grace_unif0p1 | 1.0000 | -3.6773 | 0.2107 | 0.6957 | 0.6534 |
| CiteSeer | grace_unif0p5 | 1.0000 | -3.7903 | 0.2107 | 0.7006 | 0.6143 |
| CiteSeer | grace_unif1 | 1.0000 | -3.8045 | 0.2107 | 0.6927 | 0.5968 |

- Best result: none passes the gate. Cora `grace_unif0p1` has slightly higher clean probe than GRACE, but it still collapses to one ball and fails.
- Worst result: all uniformity variants produce one ball on both datasets.
- Average trend: stronger uniformity metric, worse ball clusterability.
- Mean/std: geometry diagnostic over seeds 0-2. Uniformity variants are deterministic at `num_balls=1`.
- Baseline mean/std: GRACE produces 52.33 balls on Cora and 64 balls on CiteSeer with much higher true weighted purity.
- Delta: true purity drops from `0.7871` to `0.3021` on Cora and from `0.6991` to `0.2107` on CiteSeer.
- Std overlap: not relevant for gate because collapse is exact across seeds.
- Datasets improved: none by geometry gate.
- Datasets worsened: Cora and CiteSeer.
- Per-seed stability: failure is stable.
- Runtime/memory cost: first full-pairwise implementation OOMed on CiteSeer; final version uses deterministic `uniformity_max_nodes=2048`.
- Raw evidence: Phase 20 report and geometry CSVs listed above.

### 3.6 What Is Directly Supported

- Explicit uniformity regularization is label-free and does change the embedding uniformity metric.
- More negative uniformity is not sufficient for useful SoftDist ball construction.
- The tested uniformity grid collapses the ball splitter to one ball and sharply reduces true weighted ball purity.
- Probe accuracy cannot be used as a proxy for ball usability.

### 3.7 Reasonable Interpretation

The useful GRACE geometry is balanced. It is uniform enough to avoid representation crowding but not forced into a geometry that the downstream recursive 2-means splitter treats as one undifferentiated mass. For this pipeline, "more uniform" is not equivalent to "more clusterable".

### 3.8 What Cannot Be Concluded

- Cannot conclude that all possible custom SSL objectives are impossible.
- Cannot conclude that uniformity is irrelevant.
- Cannot report downstream gains or losses for `grace_uniformity`, because Phase 20b was intentionally skipped after Phase 20a failed.

### 3.9 Failure Analysis

- Main failure reason: explicit uniformity over-regularizes the encoder space for the downstream ball construction.
- Secondary failure reason: the diagnostic confirms the Phase 9 warning that probe can remain acceptable while ball geometry fails.
- Evidence: `geometry_gate_decisions.csv` shows all six dataset-candidate pairs fail `num_balls_ok`, `purity_ok`, and `pass`.
- Alternative explanation: the regularizer may need another scale or a different pairwise estimator, but tuning that would violate the pre-registered "safest one-shot" design unless new evidence is collected first.
- Minimum check needed: any revived custom SSL method must pass the same geometry gate before downstream training.

### 3.10 Lessons Learned

- Lesson: Do not optimize SSL candidates by probe alone. Evidence: Cora `grace_unif0p1` clean probe is slightly above GRACE, but ball geometry collapses. Future constraint: require ball count and true-purity gates.
- Lesson: More uniformity is not monotonically better for SoftDist. Evidence: all three lambda values improve uniformity metric but fail ball geometry. Future constraint: describe GRACE as a balanced representation, not as a target to maximize uniformity blindly.
- Lesson: Label-free pretraining remains necessary for the paper claim. Evidence: Phase 20 kept this constraint and still produced a useful negative result. Future constraint: no noisy labels, train masks, or ball-vote labels in pretraining.

### 3.11 What Is Still Useful

- Reusable code: `grace_uniformity` branch and geometry diagnostic script.
- Reusable paper evidence: supports the Section 6 claim that representation quality is two-dimensional: linear separability differs from granular-ball clusterability.
- Reusable stopping rule: Phase 20a geometry gate prevents wasting downstream runs on collapsed representations.

### 3.12 Decision

- Decision: abandon
- Reason: all candidates fail the pre-registered geometry gate on both datasets.
- What should not be tried again: do not run downstream SoftDist on `grace_unif0p1`, `grace_unif0p5`, or `grace_unif1`; do not tune lambda using downstream accuracy.
- Variant still worth trying: only a genuinely different label-free SSL method that first passes ball geometry diagnostics.
- Evidence required to revive this direction: noncollapsed ball count and true weighted purity at least matching GRACE on Cora and CiteSeer, followed by downstream seed0 smoke.

### Experimental Setup Summary

- Datasets: Cora, CiteSeer.
- Seeds: pretraining seed 0; geometry diagnostics seeds 0-2.
- Epochs: 200 pretraining epochs.
- Encoder: fixed GRACE encoder from the upstream candidate script.
- Base loss: GRACE InfoNCE.
- Added mechanism: label-free encoder uniformity regularizer.
- GB loss weight: none.
- Augmentation: unchanged GRACE augmentations.
- Ball generation method: SoftDist geometry protocol in NoisyGL.
- Evaluation protocol: geometry gate first; no downstream training after gate failure.
- Train/val/test split: percent split, 0.2 / 0.1 / 0.7.
- Hardware/runtime: ROCm GPU via `.venv-rocm`; deterministic node sampling added after CiteSeer OOM.

### Baseline Comparability Check

- Same dataset: yes.
- Same split: yes for downstream diagnostic.
- Same seeds: geometry diagnostic uses seeds 0-2 for all methods.
- Same encoder/projector: yes, except added uniformity regularizer.
- Same augmentation: yes.
- Same evaluation script: yes.
- Same linear evaluation: yes for reference probe.
- Baseline rerun: GRACE anchor included in diagnostic.
- Comparable conclusion: yes for geometry gate; downstream accuracy is not compared.

### Label Leakage Check

- Labels used in GB generation: no.
- Labels used in pretraining: no.
- Labels used in purity: clean labels are used only for post-hoc diagnostic.
- Labels used in sampling/coarsening: no.
- Validation/test labels used outside evaluation: no evidence.
- Supervised GB mechanism used in pretraining: no.
- Leakage risk: none for training; low for diagnostic because true labels are reporting-only and not used for selection beyond the pre-registered gate.

# 4. Cross-Direction Failed Pattern Ledger

## FP-12: Probe-First SSL Selection Is Unsafe

### Status
confirmed

### Related Directions
Phase 9 SSL replacement, Phase 20 custom SSL.

### Observation
CCA-SSG and GraphMAE had acceptable or higher probe but collapsed downstream balls. Phase 20 reproduces the same pattern with `grace_unif0p1`: probe can look acceptable while ball geometry is unusable.

### Possible Reason
Node classification probe measures linear separability, while SoftDist needs local geometric clusterability under a specific ball splitter.

### Evidence
Phase 20 `geometry_summary.csv`: Cora `grace_unif0p1` clean probe `0.8303` vs GRACE `0.8294`, but `num_balls=1` and true purity `0.3021`.

### Future Constraint
No SSL candidate enters downstream SoftDist without ball count and true-purity diagnostics.

## FP-13: Maximizing a Desirable Geometry Metric Can Destroy the Usable Geometry

### Status
confirmed for tested uniformity regularizer

### Related Directions
Phase 20 custom SSL.

### Observation
Uniformity metric improves strongly, but ball count and true purity collapse.

### Possible Reason
The useful embedding geometry is multi-objective. Over-emphasizing uniformity weakens the density/cluster structure used by recursive 2-means balls.

### Evidence
Cora uniformity moves from `-1.0107` to as low as `-3.6397`, while `num_balls` drops from `52.33` to `1`.

### Future Constraint
Do not optimize a single SSL geometry scalar without checking downstream ball diagnostics.

# 5. Cross-Direction Positive Signal Ledger

## PS-12: Geometry-Gated SSL Selection

### Status
confirmed as a useful diagnostic

### Related Directions
Phase 9 SSL replacement, Phase 20 custom SSL.

### Observation
The geometry gate stopped downstream runs when the representation was incompatible with SoftDist balls.

### Why It May Matter
It prevents over-interpreting high probe accuracy and saves experiment time.

### Required Verification
Future candidates should still include seed0 downstream smoke only after passing geometry gates.

### Possible Paper Use
diagnostic / motivation

# 6. Anti-Repetition Memory

## Do Not Retry Without New Evidence

| Pattern | Why It Failed | Evidence | Required New Evidence Before Retry |
|---|---|---|---|
| GRACE + explicit uniformity with lambda `{0.1,0.5,1.0}` | Collapses ball construction to one ball | Phase 20 geometry gate: all candidates fail on Cora/CiteSeer | A label-free variant that preserves at least half of GRACE ball count and matches GRACE true-purity |
| SSL candidate selection by clean/noisy probe | Probe does not measure ball clusterability | Phase 9 and Phase 20 both show probe/ball mismatch | Geometry gate plus downstream real-vs-shuffled ablation |
| Downstream training after geometry collapse | If `num_balls=1`, SoftDist ball signal degenerates | `geometry_summary.csv` all uniformity variants have `num_balls=1` | Noncollapsed ball construction |

## Failed Form, Not Necessarily Failed Idea

| Failed Form | Underlying Signal That May Remain Useful | Safer Future Form |
|---|---|---|
| Explicit uniformity regularizer over GRACE encoder outputs | Balanced uniformity helps GRACE | Treat GRACE as anchor; only test new SSL via geometry gate |
| Probe-led SSL replacement | SSL representation quality matters | Use two-dimensional selection: probe + ball clusterability |

## Minimum Bar Before a Direction Enters Paper Main Line

- Comparable baseline: same split and same NoisyGL downstream protocol.
- Multi-seed evidence: at least seed0 smoke plus 5-seed for main claims.
- Multi-dataset evidence: Cora and CiteSeer at minimum.
- Ablation removing GB signal: real-vs-shuffled or equivalent.
- Mechanism diagnostic: ball count, weighted true-purity, real-vs-shuffled contribution.
- No label leakage: pretraining must remain label-free.

# 7. Paper Claim Implications

## Claim-12

### Claim
For SoftDist [X||H], representation quality is two-dimensional: linear separability is not equivalent to granular-ball clusterability.

### Supported By Which Directions
Phase 9 SSL replacement and Phase 20 custom SSL.

### Weakened By Which Directions
None in the current evidence.

### Current Evidence
SSL candidates and uniformity-regularized GRACE can show acceptable probe scores while collapsing ball construction.

### Missing Evidence
Visualization of GRACE vs failed SSL representations would make the claim easier to communicate.

### Required Experiments
No more required for a diagnostic claim; optional t-SNE or UMAP figure.

### Risk
Do not overclaim that GRACE is globally optimal. Claim only that GRACE is the best tested anchor under the current SoftDist protocol.

### Current Readiness
moderate

# 8. TODO List

## Missing raw logs

- None for Phase 20.

## Missing baseline checks

- None for geometry-gate conclusion.

## Missing seeds/std

- Downstream 5-seed was intentionally not run after gate failure.

## Missing datasets

- PubMed/Amazon not tested for Phase 20 because Cora/CiteSeer gate failed.

## Missing ablations

- Not applicable; all candidates failed pre-downstream geometry gate.

## Missing GB diagnostics

- Optional visualization: t-SNE/UMAP of GRACE vs `grace_unif0p1` to show why one-ball collapse occurs.

## Missing paper-claim evidence

- Need a concise figure/table in Section 6 combining Phase 9 and Phase 20 into one "probe vs ball clusterability" story.

