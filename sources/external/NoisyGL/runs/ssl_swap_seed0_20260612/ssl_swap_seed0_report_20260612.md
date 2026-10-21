# Phase 8-9 SSL Upgrade Report

Date: 2026-06-12

## Protocol

- Split: `percent`, train/val/test = 0.2/0.1/0.7.
- Noise: `uniform`, noise_rate = 0.4.
- Device: ROCm GPU via `.venv-rocm`.
- Phase 8 probe seeds: 0, 1, 2.
- Phase 9 downstream smoke seed: 0.
- Downstream mechanism: unchanged `frozengrace_gb_gcn_softdist_xh`; only `gb.embedding_path` is swapped in memory.
- Anchor behavior: GRACE + SoftDist [X||H] is not modified.

## Phase 8 Linear Probe

Phase 8 gate: a candidate enters downstream only if clean or noisy probe exceeds GRACE by at least 1pp on at least one dataset.

| Dataset | Method | Clean probe | Delta vs GRACE | Noisy probe | Delta vs GRACE | Decision |
|---|---:|---:|---:|---:|---:|---|
| Cora | GRACE | 0.8294 | 0.0000 | 0.7949 | 0.0000 | Anchor |
| Cora | CCA-SSG | 0.8566 | +0.0273 | 0.8237 | +0.0288 | Enter Phase 9 |
| Cora | GraphMAE | 0.8396 | +0.0102 | 0.8035 | +0.0086 | Phase 9 seed0 only |
| Cora | GRACE-learnable-tau | 0.7836 | -0.0457 | 0.7710 | -0.0239 | Stop |
| Cora | BGRL | 0.8007 | -0.0287 | 0.7449 | -0.0500 | Stop |
| CiteSeer | GRACE | 0.7109 | 0.0000 | 0.7015 | 0.0000 | Anchor |
| CiteSeer | CCA-SSG | 0.7265 | +0.0156 | 0.7131 | +0.0116 | Enter Phase 9 |
| CiteSeer | GraphMAE | 0.5299 | -0.1810 | 0.5020 | -0.1995 | Stop except Cora smoke |
| CiteSeer | GRACE-learnable-tau | 0.7000 | -0.0109 | 0.6801 | -0.0213 | Stop |
| CiteSeer | BGRL | 0.6783 | -0.0326 | 0.6383 | -0.0631 | Stop |

Interpretation: CCA-SSG is the only clear probe winner. GraphMAE barely passes on Cora clean probe, so it is included in Phase 9 seed0 smoke only.

## Phase 9 Downstream Seed0

Phase 9 expansion gate: `softdist_xh(H')` must be at least `softdist_xh(GRACE) - 0.5pp`; otherwise the candidate is not expanded to 5 seeds.

| Dataset | SSL H | SoftDist [X||H] | GCN [X||H] CE | SoftDist shuffled | SoftDist delta vs GRACE | Real minus shuffled | Decision |
|---|---|---:|---:|---:|---:|---:|---|
| Cora | GRACE | 0.8058 | 0.8106 | 0.7805 | 0.0000 | +0.0253 | Anchor |
| Cora | CCA-SSG | 0.8253 | 0.8248 | 0.8253 | +0.0195 | +0.0000 | No 5-seed: ball gain collapses |
| Cora | GraphMAE | 0.8227 | 0.8317 | 0.8227 | +0.0169 | +0.0000 | No 5-seed: ball gain collapses |
| CiteSeer | GRACE | 0.6959 | 0.6890 | 0.6753 | 0.0000 | +0.0207 | Anchor |
| CiteSeer | CCA-SSG | 0.6744 | 0.6718 | 0.6744 | -0.0215 | +0.0000 | Stop |
| CiteSeer | GraphMAE | 0.6460 | 0.6684 | 0.6581 | -0.0498 | -0.0120 | Stop |

Interpretation: CCA-SSG and GraphMAE improve Cora seed0 as classifier features, but they do not preserve useful ball geometry. On CiteSeer both candidates are clearly worse than GRACE for the main downstream method, so neither candidate enters 5-seed downstream.

## Ball Quality Diagnostics

Diagnostics are computed with the same seed0 split and noisy labels. Labels are used only for diagnosis.

| Dataset | SSL H | Embedding dim | Num balls | Mean ball size | All-node purity | Train purity | Noisy-train purity | Changed train | NCR | CDR | Mean alpha |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cora | GRACE | 128 | 52 | 52.1 | 0.7873 | 0.7893 | 0.5453 | 246 | 0.6878 | 0.2244 | 0.4695 |
| Cora | CCA-SSG | 512 | 1 | 2708.0 | 0.3021 | 0.3198 | 0.2403 | 411 | 0.3280 | 0.6847 | 0.1137 |
| Cora | GraphMAE | 512 | 1 | 2708.0 | 0.3021 | 0.3198 | 0.2403 | 411 | 0.3280 | 0.6847 | 0.1137 |
| CiteSeer | GRACE | 256 | 64 | 52.0 | 0.6991 | 0.7008 | 0.4617 | 358 | 0.5911 | 0.2980 | 0.3540 |
| CiteSeer | CCA-SSG | 512 | 1 | 3327.0 | 0.2107 | 0.2180 | 0.1955 | 535 | 0.2193 | 0.7828 | 0.0346 |
| CiteSeer | GraphMAE | 512 | 64 | 52.0 | 0.5266 | 0.5579 | 0.4120 | 391 | 0.3903 | 0.4495 | 0.2944 |

Key diagnosis: the stronger linear-probe embeddings do not automatically produce stronger granular balls. CCA-SSG collapses to one ball on both datasets; GraphMAE also collapses to one ball on Cora and has much lower ball purity on CiteSeer. This explains why the downstream shuffled control matches or exceeds the real-ball variant for these embeddings.

## Research Record

- Hypothesis: replacing GRACE with stronger label-free SSL embeddings may improve H quality, ball quality, and downstream noisy-label robustness.
- Granular-ball role: unchanged; balls consume frozen H to produce SoftDist targets.
- Code change: added `scripts/linear_probe_embeddings.py`, `scripts/run_ssl_swap_downstream.py`, and upstream `scripts/pretrain_ssl_candidates.py`.
- Config change: none to SoftDist/XH anchor; Phase 9 swaps `gb.embedding_path` in memory only.
- Success standard: candidate H must improve probe, then preserve or improve downstream SoftDist [X||H] on seed0 before 5-seed expansion.
- Result: CCA-SSG passes probe but fails downstream on CiteSeer and loses real-ball contribution; GraphMAE fails CiteSeer and also loses ball contribution.
- Decision: do not expand CCA-SSG or GraphMAE to 5 seeds. GRACE remains the main downstream H for now.

## Implementation Caveats

- BGRL official configs do not include Cora/CiteSeer; this run uses the official transductive Amazon-Photos augmentation/optimizer pattern without downstream tuning.
- GraphMAE and CCA-SSG were implemented as local PyG-compatible label-free exporters using official code/config references, not by directly running their original DGL training scripts.
- Because downstream uses fixed NoisyGL node order and split protocol, all candidate H files are exported to a shared `checkpoints/embeddings_v2/{dataset}/{method}/H.pt` layout before probing or downstream use.
