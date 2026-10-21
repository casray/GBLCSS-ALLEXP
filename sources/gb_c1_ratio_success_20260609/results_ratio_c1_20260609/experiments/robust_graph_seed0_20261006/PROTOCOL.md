# Robust graph teachers: seed-0 screening, 2026-10-06

## Question and pre-registered decision

Under the submitted GB-LCSS split/noise/evaluation protocol, do TFR and ERASE
produce stronger graph teachers, and does their advantage transfer to the same
attribute-only MLP? Granular balls only construct the GB-LCSS reference targets;
they are not added to either robust baseline.

- Five datasets; uniform training-only noise 0.4 and 0.6; seed 0.
- Ordered NoisyGL percentage splits 0.2/0.1/0.7, clean validation checkpoint selection.
- Raw feature matrices and original graphs; no clean training labels enter a loss.
- Frozen main-experiment ball assignments are reused after fingerprint verification.
- TFR: official DBLP preset on all five datasets (no author preset for these five).
- ERASE: official Cora/CiteSeer/CoraFull presets; Cora preset transferred unchanged
  to Coauthor-CS and Amazon-Photo. No search using test labels or screening results.
- Teacher budgets: TFR 500; ERASE 400 with upstream patience 150; reference GCN 200.
- Shared student: submitted two-layer MLP, hidden 64, dropout .5, Adam .01,
  weight decay .0005, 200 epochs, all-node soft CE, temperature 1, no hard CE.
- For this controlled screening, student initialization and student training RNG
  are reset to seed 0 for every teacher. Historical paper seed-0 results are kept
  separate because this reset differs from the original teacher-consumed RNG.
- GPU mandatory: RX 7700 XT, D:/codex_temp/noisygl-rocm721/Scripts/python.exe.

## Implementation changes and evidence limits

Vendor code is unchanged. adapters.py extracts official class definitions with
Python AST, replaces torch_scatter mean/sum with native PyTorch index_add,
uses the author's exact many-class compression formula for all datasets (avoids
quadratic membership matrices), and implements semantic prototypes and adjacency
propagation on GPU. The logistic probe uses GPU LBFGS with the same multinomial
CE plus C=1 L2 objective as the author's sklearn probe. Equivalence checks are
required before full runs. Dataset split/noise routines come from latest local
GB-LCSS_GitHub; ordered arrays must match historical hashes.

ERASE's default Cora/CiteSeer loader uses row-normalized input features. This
screening instead uses the shared raw input features of the main experiment,
explicitly treating input preprocessing as part of the matched protocol.
TFR's upstream loop leaves the decoder in eval mode after its first validation;
the adapter preserves that behavior rather than silently correcting it.
The author DBLP dataset configuration disables graph-level self loops. TFR uses
the raw graph; self loops are added internally by its GCN and decoder. The first
eight TFR attempts mistakenly used pre-added graph loops. They are archived in
superseded_tfr_selfloops/ and excluded from all final comparisons; the corrected
adapter (revision 2) reruns those cases with unchanged seeds and hyperparameters.

This is official-core adaptation under a different evaluation protocol, not
reproduction of the original papers' reported SOTA numbers. Missing class
prototypes, numerical failures or OOM are recorded as failures, not repaired by
test-dependent heuristics or silent CPU fallback.

## Success, failure and final decision

- Stronger teacher and stronger student in multiple dataset/noise cells: expand
  to five paired seeds before any stable-performance claim.
- Stronger teacher but weaker student: investigate distillation/compression.
- Weak teacher with transferred presets: check preset sensitivity using validation
  data; do not declare the published method ineffective.
- One seed gives no standard deviation, confidence interval or significance claim.

Each run retains configuration, source commits/hashes, environment, epoch log,
best teacher/student checkpoints, probabilities, protocol hashes and results.
The final report must include negative results and a continue/modify/pause decision.

## Upstream repositories

- TFR: https://github.com/eaglelab-zju/TFR
- ERASE: https://github.com/eraseai/erase
- Exact downloaded commits and local GB-LCSS hashes are in environment.json.
- teacher_seconds includes input preparation and target construction; GB-LCSS
  timing excludes historical GRACE pretraining and ball generation. These times
  are diagnostic records, not an end-to-end cost comparison.
