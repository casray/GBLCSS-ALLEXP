# Phase 39: LastFM-Asia Diagnostic Result

Date: 2026-06-18

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run + validate
- Verification Status: ANALYZED
- Version Label: phase39_result_v1

## Research Hypothesis

LastFM-Asia may become a fourth controlled-success dataset because it combines
7,624 nodes with 18 classes. The preregistered mediator is not class count
itself, but whether frozen label-free GRACE `H` forms non-collapsed,
semantically coherent balls whose noise score remains informative inside the
GCN-clean subset.

## Granular-Ball Role

Granular balls remain frozen local semantic units used to construct soft label
distributions. They are not hard positive labels, hard negative filters, or
prototype-pulling constraints. Clean labels are used only for the Phase 39a
diagnostic metrics.

## Data Recovery and Provenance

PyG 2.7.0 points to
`https://graphmining.ai/datasets/ptg/lastfm_asia.npz`, but that file returned
HTTP 404 on 2026-06-18 and the HTTPS endpoint also presented a hostname
certificate mismatch. No successful Wayback snapshot was available.

The experiment therefore uses a deterministic reconstruction from the official
SNAP archive:

- Source: `https://snap.stanford.edu/data/lastfm_asia.zip`
- Source zip SHA256:
  `51acb78a923bb223ed6e61be88f91122fb29adca3f07beff7289cafd98601d47`
- Raw graph: 7,624 nodes, 27,806 undirected edges, 18 classes.
- Raw attributes: binary user-artist incidence matrix with 7,842 artist
  features and 3,014,361 non-zero entries.
- Reduction: Scikit-Learn `TruncatedSVD`, 128 components, randomized algorithm,
  `n_iter=5`, `random_state=0`, no pre-SVD feature normalization.
- Explained variance ratio sum: `0.3917775`.
- Labels used for feature construction: `False`.
- Reconstructed NPZ SHA256:
  `819a0abebd88ac18c816d706b2d10175514dfaf452f0faef7018e1c9723ff2c2`.

This follows the FEATHER paper's disclosed supervised-model preprocessing
family, but it is not claimed to be byte-identical to the unavailable original
PyG NPZ.

## Code and Configuration Changes

- Added native PyG `LastFMAsia` routing to NoisyGL and the GRACE pretraining
  loader.
- Added `lastfmasia` CLI/config support with percent split `0.2/0.1/0.7`.
- Added deterministic official-SNAP reconstruction script and tests.
- Added LastFM downstream configs for possible Phase 39b use.
- Added edge homophily reporting to the existing orthogonality diagnostic.
- GRACE used the preregistered unchanged PubMed fallback configuration:
  1,500 epochs, 256-dimensional `H`, seed 0, label-free `X,A` training.
- Ball parameters: `max_balls=256`, `min_ball_size=32`,
  `split_improvement=0.001`.
- Diagnostic: uniform noise 0.4, seeds `0,1,2`, warmup GCN epochs 10 with
  probability window 5-10.

## Phase 39a Results

GRACE completed 1,500/1,500 epochs with final loss `9.0956`. Exported `H` has
shape `[7624, 256]`, finite values, and unit row norms.

| Metric | Seed 0 | Seed 1 | Seed 2 | Mean ± std | Gate |
|---|---:|---:|---:|---:|---|
| Edge homophily | 0.8739 | 0.8739 | 0.8739 | 0.8739 ± 0.0000 | warning: >0.85 |
| Number of balls | 144 | 146 | 146 | 145.33 ± 1.15 | pass: ≥8 |
| True weighted purity | 0.2547 | 0.2535 | 0.2545 | 0.2542 ± 0.0006 | **fail: <0.65** |
| Ball AUC | 0.6337 | 0.6444 | 0.6417 | 0.6399 ± 0.0056 | diagnostic |
| Early GCN AUC | 0.9483 | 0.9548 | 0.9511 | 0.9514 ± 0.0032 | strong baseline |
| Ball AUC in GCN-clean50 | 0.5530 | 0.6521 | 0.5935 | 0.5995 ± 0.0498 | **fail: <0.70** |
| corr(ball, GCN) | 0.2320 | 0.2378 | 0.2095 | 0.2264 ± 0.0149 | weak overlap |

The partition does not collapse, but the balls are not semantically aligned
with the 18 country labels. Meanwhile, the early GCN already separates noisy
training labels extremely well. Both required information conditions for a
controlled SoftDist gain therefore fail.

## Evidence, Inference, and Boundary

Experiment-supported:

- Class count ≥6 is not sufficient for SoftDist success.
- LastFM-Asia forms many geometric balls, but their clean-label purity is only
  about 0.254.
- The conditional orthogonal AUC is about 0.600, below the preregistered 0.70
  gate.
- The graph is highly homophilous and the early GCN noise AUC is about 0.951.

Reasonable interpretation:

- Country labels and music-interest geometry are only weakly aligned, so
  artist-based `H` neighborhoods do not form country-pure balls.
- High graph homophily lets `GCN [X||H]` absorb most usable signal, leaving
  insufficient independent information for ball soft targets.

Not experimentally established:

- No downstream SoftDist result is available for LastFM-Asia, because the
  preregistered Phase 39a gate stopped Phase 39b.
- The reconstructed 128-dimensional features are protocol-compatible but not
  proven identical to the unavailable original PyG feature matrix.

## Final Decision

**Stop Phase 39b and record LastFM-Asia as a boundary dataset.**

The dataset fails two of three mandatory gates:

- `ball_auc_in_gcn_clean50 = 0.5995 < 0.70`
- `true_all_purity_weighted = 0.2542 < 0.65`
- `num_balls = 145.33 >= 8`

Decision: **abandon downstream training for this dataset under the current
SoftDist mechanism**. The main result remains three controlled-success
datasets: Cora, CiteSeer, and Coauthor-CS.

## Pair-Noise Future Work

Existing pair-noise results suggest a possible failure mechanism: systematic
class `A -> B` flips can create a false majority inside a ball, so collective
voting becomes consistently biased instead of averaging out independent
uniform errors. This is a hypothesis, not a verified conclusion.

A future mechanism may use label-free `H` geometry to validate the noisy-label
majority: compare each node and ball vote with nearby `H`-space modes or
centroids, flag geometrically inconsistent majorities, and reduce their target
weight. This requires a new preregistered diagnostic and experiment series; it
was not implemented in Phase 39.

## Artifacts

- `phase39_data_prepare.log`
- `phase39_data_load.log`
- `phase39_grace_pretrain.log`
- `phase39_grace_pretrain.err.log`
- `ssl_pretrain_summary.csv`
- `orthogonality/orthogonality_raw.csv`
- `orthogonality/orthogonality_summary.csv`
- `orthogonality/offline_usage_raw.csv`
- `orthogonality/offline_usage_summary.csv`
- `orthogonality/biggraph_orthogonality_report.md`
- GRACE embedding:
  `../../../../gb_graph_aware_minival_6_1/checkpoints/embeddings_v2/LastFMAsia/grace/H.pt`
