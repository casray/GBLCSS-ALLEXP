# Anonymous LCSS Reproducibility Artifact

This directory contains the code, fixed configurations, audited result tables, and statistical scripts needed to inspect the LCSS experiments. It is generated from an explicit allowlist and contains no repository history. Every copied file is recorded in `ARTIFACT_MANIFEST.csv` with its SHA256 digest.

## Environment

The versions used in the reported ROCm runs are listed in `docs/anonymous_requirements_20260716.txt`. Install the platform-appropriate PyTorch build.

## GRACE preprocessing

Frozen embeddings are identified by SHA256 in `docs/h_embedding_manifest_20260712.md`. The MLP-37 pretraining script accepts `--upstream-root` pointing to the archived GRACE checkout; the downstream runner only needs the resulting `H.pt` path. GRACE preprocessing is label-free.

## Scope

The copied configs cover the five full-spectrum datasets, LCSS, its size-matched shuffled control, and the seven same-backbone baselines used for the formal ranking.
