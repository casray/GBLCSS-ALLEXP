# MLP-35 Same-Seed ROCm Repeatability Audit

> Ranges below are within-seed rerun spreads, not hardware variance estimates.

## Per-method spread

| Setting | Method | Mean | Std | Min | Max | Range |
|---|---|---:|---:|---:|---:|---:|
| cora_pair0p4 | mlp_small_loss | 48.34 | 0.00 | 48.34 | 48.34 | 0.00pp |
| cora_pair0p4 | frozengrace_gb_mlp_softdist_beta05_small_loss | 50.87 | 0.00 | 50.87 | 50.87 | 0.00pp |
| amazonpho_pair0p4 | mlp_small_loss | 66.37 | 0.00 | 66.37 | 66.37 | 0.00pp |
| amazonpho_pair0p4 | frozengrace_gb_mlp_softdist_beta05_small_loss | 71.65 | 0.00 | 71.65 | 71.65 | 0.00pp |
| pubmed_pair0p4 | mlp_jocor | 75.01 | 0.00 | 75.01 | 75.01 | 0.00pp |
| pubmed_pair0p4 | frozengrace_gb_mlp_softdist_beta05_small_loss | 78.08 | 0.05 | 78.02 | 78.12 | 0.10pp |

## Paired-delta spread

| Setting | Baseline | Mean delta | Min | Max | Range | Positive | Same sign |
|---|---|---:|---:|---:|---:|---:|---:|
| cora_pair0p4 | mlp_small_loss | +2.53pp | +2.53pp | +2.53pp | 0.00pp | 3/3 | 1 |
| amazonpho_pair0p4 | mlp_small_loss | +5.28pp | +5.28pp | +5.28pp | 0.00pp | 3/3 | 1 |
| pubmed_pair0p4 | mlp_jocor | +3.07pp | +3.01pp | +3.11pp | 0.10pp | 3/3 | 1 |

## Gate

```json
{
  "pass": true,
  "maximum_method_range": 0.0010144192449822453,
  "same_sign_settings": 3,
  "n_settings": 3,
  "reasons": []
}
```
