# Phase 19 Amazon-Photo Preregistered Protocol

Date: 2026-06-15

## Purpose

Evaluate whether the frozen-GRACE SoftDist [X||H] pipeline generalizes to Amazon-Photo, a co-purchase graph with different graph/domain structure from citation graphs.

## Preregistered SSL Fallback Rule

Amazon-Photo has no official GRACE entry in the workspace's `upstream/GRACE/config.yaml`.

Before any Amazon-Photo result is generated, the fallback rule is fixed as:

- Use the official PubMed GRACE pretraining config unchanged.
- Do not tune SSL hyperparameters for Amazon-Photo.
- Reuse PubMed values for epochs, learning rate, weight decay, hidden/projection dimensions, layer count, augmentation drop rates, activation, and fixed temperature tau.
- Train label-free only on `(X, A)`. Labels and split masks are not used in SSL pretraining.

This rule is intentionally conservative: it prioritizes protocol cleanliness over best possible Amazon-Photo performance.

## Downstream Protocol

- Dataset: `amazonpho`.
- Noise: `uniform`, noise_rate = 0.4.
- Split: `percent`, train/val/test = 0.2/0.1/0.7.
- Seeds: 0, 1, 2, 3, 4.
- Methods: SoftDist [X||H], CE [X||H], SoftDist shuffled, GCN+CE, NRGNN.
- Required diagnostics: ball count, purity, NCR/CDR, actual noise rate.

## Decision Rule

- SoftDist [X||H] >= GCN+CE and real > shuffled: third successful dataset.
- SoftDist [X||H] < GCN+CE but > NRGNN: PubMed-like regime boundary.
- Ball collapse (`num_balls < 8`): geometry mismatch limitation.
