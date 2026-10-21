# Submission Integrity Audit - Phase 4: GRACE Provenance and Ball Construction

## Verdict

- GRACE embedding provenance: PASS for file existence, finite normalized tensors, and node-order alignment.
- Label-free training path: PASS by code inspection; GRACE losses read only features and edges. The v2 metadata also records `label_free=true` and `used_labels_or_masks_in_training=false`.
- Manuscript mechanism facts: **FAIL until corrected**. Two P0 factual statements contradict the formal artifacts and implementation.

## Embedding Manifest

| Dataset | Shape | SHA256 | Config source | Checkpoint recoverable |
|---|---:|---|---|---|
| Cora | 2708x128 | `5e72323b2dd82cb5...` | native/legacy payload | True |
| CiteSeer | 3327x256 | `0c76a21aef43f9b6...` | native/legacy payload | True |
| CoraFull | 19793x256 | `1940d2f55754ccc7...` | DBLP | False |
| CoauthorCS | 18333x256 | `de7520acae6d1049...` | DBLP | False |
| AmazonPhoto | 7650x256 | `596e8cf3ad57f820...` | PubMed | False |

## Node-order Audit

| Dataset | Nodes | Labels | Features | Canonical edges |
|---|---:|---:|---:|---:|
| Cora | True | True | True | True |
| CiteSeer | True | True | True | True |
| CoraFull | True | True | True | True |
| CoauthorCS | True | True | True | True |
| AmazonPhoto | True | True | True | True |

## Five-seed Ball Counts

| Dataset | Ball-count range | Seeds hitting B_max=64 |
|---|---:|---:|
| Cora | 52-53 | 0/5 |
| CiteSeer | 63-64 | 4/5 |
| CoraFull | 64-64 | 5/5 |
| CoauthorCS | 64-64 | 5/5 |
| AmazonPhoto | 64-64 | 5/5 |

Across the 25 formal reconstructions, the observed range is 52-64; 19/25 runs hit the hard cap.

## P0 Manuscript Corrections

1. Replace the universal `H in R^{N x 256}` statement with `H in R^{N x d_h}`. The formal Cora embedding is 128-dimensional; the other four core embeddings are 256-dimensional.
2. Remove `22-133 balls` and `the cap is not triggered on most datasets`. The implementation cannot exceed 64 balls. The formal five-seed result is 52-64 balls, with 19/25 runs hitting the cap.
3. Reframe large balls on CoraFull/CoauthorCS/AmazonPhoto as a consequence of the fixed 64-ball capacity as well as embedding geometry; do not claim purely geometry-driven stopping.

## Reproducibility Boundary

- Cora and CiteSeer retain GRACE model checkpoints and are numerically re-extractable. An independent CPU extraction differed from the archived H by at most 8.94e-8 (Cora) and 1.04e-7 (CiteSeer).
- The v2 pretraining path for CoraFull, CoauthorCS, and AmazonPhoto saves H, metadata, and losses, but not model weights. The reported H files are hashable artifacts, yet exact regeneration from a saved encoder checkpoint is unavailable. This is a P1 reproducibility weakness, not evidence that the reported H is fabricated.
