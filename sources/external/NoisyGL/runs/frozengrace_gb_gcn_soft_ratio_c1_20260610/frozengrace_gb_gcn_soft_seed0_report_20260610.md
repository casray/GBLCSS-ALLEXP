# Frozen-GRACE GB-GCN Soft Target Seed0 Smoke

- Date: 2026-06-10
- Protocol: NoisyGL percent split 0.2/0.1/0.7, uniform noise 0.4, seed 0, CPU.
- Soft variant: frozen-GRACE balls define train-label majority; `alpha=max(0,(margin-0.5)/0.5)` blends noisy one-hot with majority one-hot.
- Boundary: single-seed smoke only; clean labels are used only for diagnostics, not training.

## Research Record

- Hypothesis: hard correction has high NCR but high CDR, while binary gate collapses both. A continuous soft target should reduce CDR without fully giving up NCR.
- Granular ball role: frozen-GRACE embedding space supplies local semantic neighborhoods; balls act as a soft label teacher for a fresh end-to-end GCN.
- Code/config change: added soft target construction, soft cross-entropy training, and `frozengrace_gb_gcn_soft` / `frozengrace_gb_gcn_soft_shuffled` configs.
- Expected mechanism: high-margin balls behave closer to hard correction; low-margin balls stay closer to noisy CE.
- Success standard for this smoke: improve over GCN+CE on both Cora and CiteSeer, with shuffled control near or below GCN+CE.
- Failure standard: alpha becomes too small and the method degenerates to noisy CE, or shuffled control performs similarly to the real balls.

## Test Accuracy

| Dataset | GCN+CE | Hard Real | Gated Real | Soft Real | Soft Shuffled |
|---|---:|---:|---:|---:|---:|
| Cora | 0.8100 | 0.7947 | 0.8137 | 0.8100 | 0.8111 |
| CiteSeer | 0.6327 | 0.6830 | 0.6314 | 0.6379 | 0.6323 |

## Soft Diagnostics

| Dataset | Variant | Alpha+ Train | Mean Alpha | Soft Shift Mass | Pseudo Changed | Pseudo NCR | Pseudo CDR | Clean-Mass Delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Cora | soft real | 258 | 0.1738 | 24.1174 | 82 | 0.3228 | 0.0312 | 14.5941 |
| Cora | soft shuffled | 3 | 0.0018 | 0.3333 | 1 | 0.0000 | 0.0028 | -0.3333 |
| CiteSeer | soft real | 175 | 0.0707 | 14.6331 | 64 | 0.1413 | 0.0404 | 2.9155 |
| CiteSeer | soft shuffled | 22 | 0.0060 | 1.5905 | 9 | 0.0000 | 0.0152 | -1.1619 |

Notes:

- Cora soft real did not beat GCN+CE. It lands exactly at 0.8100, while shuffled soft is 0.8111, so this setting does not isolate a useful ball signal on Cora.
- CiteSeer soft real improves over GCN+CE by 0.52pp and over shuffled by 0.56pp, but it is far below hard correction's 0.6830.
- Compared with hard correction, soft target cuts the effective pseudo CDR sharply, but it also cuts pseudo NCR from 0.6878 to 0.3228 on Cora and from 0.5911 to 0.1413 on CiteSeer.
- The main failure is under-strength correction: `mean_alpha_train` is only 0.1738 on Cora and 0.0707 on CiteSeer, so the model remains close to noisy CE.

## Decision

The soft-target direction is mechanically valid, but the exact alpha formula is too conservative for multiclass Cora/CiteSeer. It avoids the hard-replacement CDR problem but does not preserve enough NCR to beat strong GCN+CE reliably.

Do not expand this exact setting to 5 seeds.

Next candidate should keep the soft-target framework but make alpha multiclass-aware, for example:

| Setting | Rationale |
|---|---|
| `alpha=(margin - 1/C) / (1 - 1/C)` | uses random multiclass confidence as the zero point instead of 0.5 |
| blend with `ball_dist` instead of one-hot majority | keeps uncertainty among non-majority classes rather than forcing one class |
| combine soft alpha with a small floor only for high-purity balls | preserves more NCR without returning to hard replacement |
