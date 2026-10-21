# Oracle supervision and teacher-to-MLP transfer

Date: 2026-10-12. Status: registered before new training results.

## Engineering amendment before scientific training

The first Cora A replay reproduced ALL argmax predictions, validation accuracy and selected epoch,
but maximum teacher probability difference was .00997847, exceeding the initially planned 1e-6.
Repeating the UNCHANGED original ERASE function twice for three epochs also produced differences
(maximum probability difference 2.205e-6). Original/sanitized graphs, masks, features and label
preprocessing were exactly equal; specific nondeterministic GPU kernel has not been isolated.
Evidence is retained under engineering/short_repeat/ and the original failed replay log.

Before scientific outcomes were read, the design was amended to train SIX contemporary A baselines,
using the same current wrapper and initialization fingerprints as B/C/D. This yields 21 distinct
teacher/student pairs. Historical A results are retained as a reference, not reused for primary
intervention differences. The engineering teacher replay checks same argmax and validation choice;
student replay uses identical historical probability targets and still requires <=1e-6 and identical
validation trajectory. Teacher CHECKPOINT RELOAD remains <=1e-6 with identical argmax; teacher
RETRAINING is not claimed bit-exact. Numerical effect thresholds and intervention settings unchanged.
The original freeze manifest is archived; source revision hashes are logged before scientific fits.

## Material Passport

- User authorization: execute the proposed oracle decomposition; submitted paper work is deferred.
- Sources: literature map, experiment histories 13/18/41/42, gb_dist_protocol_seeds_20261007,
  gb_signal_increment_diagnostic_20261012, existing ROCm ERASE and official DIST adapters.
- Question: where does the current fixed graph-to-MLP pipeline retain a practically useful gap:
  imperfect labels, discarded supervision, or transfer to the attribute student?
- Type: one-seed intervention screen, NOT an external confirmation study or a new deployable method.

## Hypothesis, mechanism, and role of granular balls

Perfectly correcting training labels may improve the teacher and its DIST student. Keeping only
originally uncorrupted annotations may discard useful supervision; a matched clean random subset
distinguishes sample selection effects from reduced clean-label coverage. Teacher improvements may
or may not transfer through the fixed student objective.
Granular balls have no role in this intervention. Frozen historical partition files may be read by
the engineering replay because the unchanged historical student wrapper constructs an unused region
object; the actual DIST objective has region weight zero. No GB claim is made.

## Fixed cells and arms

Cora, CiteSeer, CoraFull; uniform .4/.6; protocol/teacher/student seed 1; original 20/10/70 split.
Raw X, original graph edges, clean validation selection. Original test labels are not passed to fit.

| Arm | Supervised nodes and labels | Purpose |
|---|---|---|
| A_noisy | All original training nodes, original corrupted annotations | Contemporary ERASE -> DIST baseline; historical version kept as reference |
| B_all_clean | All original training nodes, their true labels | Complete training-label correction reference |
| C_oracle_filter | Only original training nodes whose annotations were uncorrupted | Perfect exclusion without recovering discarded annotations |
| D_matched_clean | Random subset of original training nodes, true labels, same per-true-class counts as C | Control subset size/class composition and selection effects |

D uses NumPy RandomState(101); random subsets preserve original training-node order after selection.
All graph nodes and edges remain in representation learning and teacher prediction. Changing the
supervision mask also changes ERASE's supervised-subgraph propagation, semantic prototypes, and
classification probe. This is a supervised-set intervention, not merely a final classifier mask.
No extra clean training labels outside the original training split are introduced.

B has identical effective training inputs at the two noise rates (same split, labels, seed, graph).
It is fitted once per dataset and reused for both evaluations after checking input fingerprints.
This yields 15 distinct oracle teacher fits and 15 oracle student fits, representing 18 oracle arm-cells.
A contributes six contemporary baseline pairs, for 21 physical pairs in total. One extra Cora .4 baseline teacher and student are replayed
as an engineering check; they are excluded from scientific means.

## Teacher and student settings

- Existing ERASE GAT: hidden256, output512, heads8, layers2, dropout.5, Adam AMSGrad .001,
  weight decay .0005, author dataset preset, max400 epochs, patience150.
- Preserve train_erase, loss, semantic-label adapter, row-normalized adjacency, and GPU LBFGS
  probe exactly. Teacher selects the FIRST strict maximum clean validation accuracy.
- Reset all RNGs to seed1 before each teacher. Record initial state and RNG fingerprints,
  compare across interventions, and compare a label-sanitized A replay against cached predictions.
- Students: raw X, common two-layer width64 MLP with BN/dropout.5; Adam .01/wd.0005,
  200 epochs. DIST temperature1, beta/gamma1, all-node teacher probabilities, no noisy-label CE.
- Reset identical student RNG/state before each fit. Teacher probability target is the ONLY changed
  student input. Validation uses the same clean nodes and first strict maximum.
- No per-arm parameter search, no test-guided selection, no temperature adjustment.
- Full clean labels are a reference for THIS PIPELINE, not a theoretical accuracy ceiling.

## Label access and supervision checks

The source protocol archive contains all labels, but fit objects retain only intervention labels on
supervised training nodes and original clean validation labels. Other y values are -1; other noisy_y
values are zeroed. Public class count is fixed from the original teacher probability dimension.
ERASE's data.y.max() equals this class count because every selected set contains every class.
Before training, check original split/noise fingerprints; y and noisy_y outside permitted indices
are hidden; masks and indices agree; C and D class counts match; all nodes/edges are unchanged;
preprocessed prototype classes are supported. Missing classes are a reported failure, not repaired
by secretly adding supervised nodes.

## Outcomes, comparisons, and practical screen thresholds

After ALL fits are selected, evaluate A/B/C/D teacher and student test accuracy, balanced accuracy,
NLL, multiclass Brier, teacher-correct/student-wrong and teacher-wrong/student-correct rates,
prediction disagreement, and selected epochs. Save final-epoch probabilities as secondary outcomes.

Primary paired differences across the six cells: B-A, C-A, B-C, C-D for teacher and student.
Dataset/noise summaries are descriptive; n=1 per cell, no independent-seed std/significance claims.
Reuse of B across noise rates is explicit and does not create independent clean-teacher replicates.

Before results, fix these investment screens (percentage points, not significance tests):

- Teacher opportunity: mean B-A >=1.0pp and >=4/6 cells positive -> prioritize a multi-seed
  clean/noisy teacher confirmation before proposing an algorithm.
- Transferred opportunity: above teacher screen and mean student B-A >=0.5pp, >=4/6 positive.
- Recovering discarded supervision: mean B-C >=1.0pp, >=4/6 positive -> correction/label utilization
  is a candidate; C-D is used to check selection effects, not assumed zero.
- Transfer warning: teacher screen passes but mean student B-A <0.25pp, or B student trails B
  teacher by >=2.0pp in >=4/6 cells -> inspect student capacity/optimization/teacher errors.
- If teacher B-A is small/mixed, report a small/mixed gap for the fixed ERASE pipeline; do not
  conclude that all graph-label noise is solved. Further teachers/protocols need a separate question.
- The above screens can overlap; report each, do not force a single causal explanation.

## Success/failure, budget, and final decision

Engineering success: protocol checks pass, A teacher replay has identical argmax/validation choices,
A student replay max probability error <=1e-6 with identical validation trajectory,
all21+21 fits complete, checkpoint reload matches saved outputs,
all old input/source SHA-256 unchanged.
Scientific failure/negative outcome: report any B/C/D degradation or lack of transferable gain.
Do not drop cells, resample D, reinterpret n=1 as stable evidence, or start new methods automatically.
Run budget: 90 minutes for this screen; no overnight sweeps or automatic multi-seed expansion.
Engineering repairs are logged separately; thresholds and scientific settings cannot change after
observing test outcomes. Final action is confirmation, transfer diagnosis, or pause, not a paper claim.

## Code/config changes and records

New directory only: run_oracle.py, analyze.py, test_oracle.py, PLAN.md, preflight/manifest/environment,
protocols, runs, epochs/checkpoints/probabilities/configs, raw logs, paired tables, decision, audit,
Chinese report and handoff. Project history/context are updated only after conclusion.

Runtime: D:/codex_temp/noisygl-rocm721/Scripts/python.exe, AMD RX7700XT, ROCm PyTorch.

```powershell
& 'D:/codex_temp/noisygl-rocm721/Scripts/python.exe' -m pytest experiments/oracle_teacher_transfer_20261012/test_oracle.py -q -p no:cacheprovider
& 'D:/codex_temp/noisygl-rocm721/Scripts/python.exe' experiments/oracle_teacher_transfer_20261012/run_oracle.py --prepare --replay --grid
& 'D:/codex_temp/noisygl-rocm721/Scripts/python.exe' experiments/oracle_teacher_transfer_20261012/analyze.py
```
