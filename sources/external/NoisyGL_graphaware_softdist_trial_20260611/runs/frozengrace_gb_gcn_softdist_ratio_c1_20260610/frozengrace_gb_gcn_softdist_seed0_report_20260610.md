# Frozen-GRACE GB-GCN SoftDist Seed0 Smoke

- Date: 2026-06-10
- Protocol: NoisyGL percent split 0.2/0.1/0.7, uniform noise 0.4, seed 0, CPU.
- Variant: `alpha=(margin - 1/C) / (1 - 1/C)` and target blending with full ball noisy-label distribution.
- Boundary: single-seed smoke only; clean labels are used only for diagnostics, not training.

## Research Record

- Hypothesis: the previous soft target was too weak because it used a binary zero point `0.5` and one-hot majority targets. A multiclass-aware alpha and ball distribution target should recover useful correction strength while staying softer than hard replacement.
- Granular ball role: frozen-GRACE embedding space supplies local semantic neighborhoods; balls act as a soft label-distribution teacher for a fresh end-to-end GCN.
- Code/config change: added `alpha_baseline: multiclass`, `target_mode: distribution`, and `frozengrace_gb_gcn_softdist` / `frozengrace_gb_gcn_softdist_shuffled` configs.
- Expected mechanism: all eligible balls can contribute a calibrated soft signal; uncertain balls pass a weaker distributional target instead of a hard class overwrite.
- Success standard for this smoke: improve over GCN+CE on both Cora and CiteSeer, and beat the shuffled control.
- Failure standard: shuffled control matches the real-ball variant, or the method collapses to noisy CE / hard correction behavior.

## Test Accuracy

| Dataset | GCN+CE | Hard Real | Gated Real | Old Soft Real | SoftDist Real | SoftDist Shuffled |
|---|---:|---:|---:|---:|---:|---:|
| Cora | 0.8100 | 0.7947 | 0.8137 | 0.8100 | 0.8232 | 0.8047 |
| CiteSeer | 0.6327 | 0.6830 | 0.6314 | 0.6379 | 0.6761 | 0.6357 |

## SoftDist Diagnostics

| Dataset | Variant | Alpha+ Train | Mean Alpha | Soft Shift Mass | Pseudo Changed | Pseudo NCR | Pseudo CDR | Clean-Mass Delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Cora | softdist real | 541 | 0.4695 | 99.0472 | 246 | 0.6878 | 0.2244 | -35.8371 |
| Cora | softdist shuffled | 541 | 0.2086 | 72.3945 | 367 | 0.1376 | 0.6648 | -50.2571 |
| CiteSeer | softdist real | 665 | 0.3540 | 114.0526 | 358 | 0.5911 | 0.2980 | -46.4799 |
| CiteSeer | softdist shuffled | 665 | 0.2132 | 87.5551 | 436 | 0.1338 | 0.6641 | -54.2536 |

Notes:

- The multiclass alpha fixed the under-strength problem. Mean alpha increased from 0.1738 to 0.4695 on Cora and from 0.0707 to 0.3540 on CiteSeer.
- The pseudo NCR/CDR values match hard majority voting because the pseudo majority label is the same; the difference is that SoftDist trains against a probability distribution instead of replacing labels with a single class.
- Cora is now above GCN+CE by 1.32pp and above shuffled by 1.85pp. This is the first seed0 setting that gives a clear Cora gain without relying on an almost-no-op gate.
- CiteSeer is above GCN+CE by 4.34pp and above shuffled by 4.04pp, but still below hard correction by 0.69pp.
- Shuffled controls stay near or below GCN+CE, so the gain is not just from generic soft-label regularization.

## Decision

This is the first variant in the frozen-GRACE-ball plus end-to-end GCN line that has seed0 signal on both Cora and CiteSeer under the aligned ratio split.

Do not claim stability yet. The next step is either:

| Option | Purpose |
|---|---|
| expand `softdist` to 5 seeds | check whether the Cora +1.32pp and CiteSeer +4.34pp are stable |
| run one ablation first: multiclass alpha + one-hot majority | isolate whether the gain comes from alpha calibration or ball distribution target |
| run one ablation first: binary alpha + ball distribution | isolate whether distribution target alone helps |

Recommended order: run the two single-seed ablations before 5 seeds only if we need mechanism clarity immediately; otherwise expand `softdist` to 5 seeds because this is now the strongest balanced setting.
