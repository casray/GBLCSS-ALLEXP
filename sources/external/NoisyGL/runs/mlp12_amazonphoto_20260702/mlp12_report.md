# MLP-12 Amazon-Photo Report

## Material Passport
- Dataset: Amazon-Photo (`amazonpho`), train-only label noise, percent split 0.2/0.1/0.7.
- Backbone: X-only 2-layer MLP unless method explicitly uses frozen GRACE targets.
- Clean labels: evaluation and diagnostics only.

## Gates
- D0 geometry gate: pass
- A1 smoke gate: pass (5/5 direction settings)
- Selected LogitClip tau: `0.5`

### D0 Details
| num balls | true all purity | true train purity | noisy train purity | ball AUC | ball AUC in GCN-clean50 | pass |
|---:|---:|---:|---:|---:|---:|---:|
| 133.0 | 0.8716339869281046 |  | 0.5431372549019607 | 0.9468104379209412 | 0.9406947039761573 | 1 |

### A1 Smoke Details
| Noise | Ours | CE | Shuffled | direction | train/scope ok |
|---|---:|---:|---:|---:|---:|
| uniform0.2 | 84.03 | 80.84 | 80.65 | 1 | 1 |
| uniform0.4 | 81.44 | 74.49 | 73.02 | 1 | 1 |
| uniform0.6 | 66.63 | 57.16 | 55.28 | 1 | 1 |
| pair0.4 | 71.11 | 60.35 | 62.32 | 1 | 1 |
| instance0.4 | 78.99 | 70.57 | 69.69 | 1 | 1 |

## Aggregate
| Noise | Method | n | mean±std |
|---|---|---:|---:|
| instance0.4 | Ours beta=0.5 | 5 | 79.59±1.16 |
| instance0.4 | Shuffled | 5 | 68.98±2.44 |
| instance0.4 | ANL | 5 | 73.18±2.36 |
| instance0.4 | MLP+CE | 5 | 69.13±3.00 |
| instance0.4 | Co-teaching | 5 | 73.59±1.19 |
| instance0.4 | CT | 5 | 73.57±2.03 |
| instance0.4 | DivideMix-lite | 5 | 70.82±1.52 |
| instance0.4 | ELR | 5 | 69.41±2.70 |
| instance0.4 | FINE | 5 | 66.38±2.68 |
| instance0.4 | JoCoR | 5 | 74.80±1.79 |
| instance0.4 | LogitClip | 5 | 73.06±1.66 |
| instance0.4 | SCE | 5 | 73.08±2.54 |
| instance0.4 | Small-loss | 5 | 73.00±1.25 |
| pair0.4 | Ours beta=0.5 | 5 | 72.51±4.02 |
| pair0.4 | Shuffled | 5 | 62.95±4.94 |
| pair0.4 | ANL | 5 | 64.90±6.57 |
| pair0.4 | MLP+CE | 5 | 62.61±4.31 |
| pair0.4 | Co-teaching | 5 | 67.24±6.97 |
| pair0.4 | CT | 5 | 66.86±6.94 |
| pair0.4 | DivideMix-lite | 5 | 66.01±7.84 |
| pair0.4 | ELR | 5 | 66.07±5.25 |
| pair0.4 | FINE | 5 | 61.54±4.53 |
| pair0.4 | JoCoR | 5 | 66.94±6.36 |
| pair0.4 | LogitClip | 5 | 68.88±3.40 |
| pair0.4 | SCE | 5 | 66.51±6.96 |
| pair0.4 | Small-loss | 5 | 66.70±5.73 |
| uniform0.2 | Ours beta=0.5 | 5 | 86.20±1.46 |
| uniform0.2 | Shuffled | 5 | 81.99±1.03 |
| uniform0.2 | ANL | 5 | 85.36±0.47 |
| uniform0.2 | MLP+CE | 5 | 82.34±1.39 |
| uniform0.2 | Co-teaching | 5 | 85.21±0.51 |
| uniform0.2 | CT | 5 | 84.40±1.12 |
| uniform0.2 | DivideMix-lite | 5 | 80.30±1.51 |
| uniform0.2 | ELR | 5 | 82.31±1.98 |
| uniform0.2 | FINE | 5 | 82.27±1.07 |
| uniform0.2 | JoCoR | 5 | 85.60±1.26 |
| uniform0.2 | LogitClip | 5 | 83.74±0.67 |
| uniform0.2 | SCE | 5 | 84.74±0.94 |
| uniform0.2 | Small-loss | 5 | 84.96±0.58 |
| uniform0.4 | Ours beta=0.5 | 5 | 81.64±0.93 |
| uniform0.4 | Shuffled | 5 | 72.06±2.97 |
| uniform0.4 | ANL | 5 | 76.94±2.80 |
| uniform0.4 | MLP+CE | 5 | 73.90±1.80 |
| uniform0.4 | Co-teaching | 5 | 78.21±1.00 |
| uniform0.4 | CT | 5 | 77.70±1.16 |
| uniform0.4 | DivideMix-lite | 5 | 72.53±3.69 |
| uniform0.4 | ELR | 5 | 73.93±2.12 |
| uniform0.4 | FINE | 5 | 73.02±2.69 |
| uniform0.4 | JoCoR | 5 | 79.57±0.53 |
| uniform0.4 | LogitClip | 5 | 76.29±2.00 |
| uniform0.4 | SCE | 5 | 76.08±3.11 |
| uniform0.4 | Small-loss | 5 | 77.25±1.62 |
| uniform0.6 | Ours beta=0.5 | 5 | 70.78±3.19 |
| uniform0.6 | Shuffled | 5 | 59.65±5.46 |
| uniform0.6 | ANL | 5 | 59.03±4.96 |
| uniform0.6 | MLP+CE | 5 | 58.60±3.77 |
| uniform0.6 | Co-teaching | 5 | 63.29±5.00 |
| uniform0.6 | CT | 5 | 60.81±3.08 |
| uniform0.6 | DivideMix-lite | 5 | 56.62±4.50 |
| uniform0.6 | ELR | 5 | 59.32±4.53 |
| uniform0.6 | FINE | 5 | 58.01±5.25 |
| uniform0.6 | JoCoR | 5 | 66.48±4.43 |
| uniform0.6 | LogitClip | 5 | 61.58±4.64 |
| uniform0.6 | SCE | 5 | 57.82±4.07 |
| uniform0.6 | Small-loss | 5 | 61.40±3.72 |

## Ours Paired Deltas
| Noise | Baseline | n | Δ | pos |
|---|---|---:|---:|---:|
| uniform0.2 | frozengrace_gb_mlp_softdist_beta05_shuffled | 5 | +4.21pp | 5/5 |
| uniform0.2 | mlp_ce | 5 | +3.86pp | 5/5 |
| uniform0.2 | mlp_small_loss | 5 | +1.24pp | 4/5 |
| uniform0.2 | mlp_coteaching | 5 | +0.99pp | 4/5 |
| uniform0.2 | mlp_jocor | 5 | +0.60pp | 4/5 |
| uniform0.2 | mlp_elr | 5 | +3.89pp | 5/5 |
| uniform0.2 | mlp_sce | 5 | +1.46pp | 4/5 |
| uniform0.2 | mlp_fine | 5 | +3.93pp | 5/5 |
| uniform0.2 | mlp_dividemix_lite | 5 | +5.90pp | 5/5 |
| uniform0.2 | mlp_ct | 5 | +1.80pp | 5/5 |
| uniform0.2 | mlp_logitclip | 5 | +2.46pp | 5/5 |
| uniform0.2 | mlp_anl | 5 | +0.84pp | 4/5 |
| uniform0.4 | frozengrace_gb_mlp_softdist_beta05_shuffled | 5 | +9.58pp | 5/5 |
| uniform0.4 | mlp_ce | 5 | +7.73pp | 5/5 |
| uniform0.4 | mlp_small_loss | 5 | +4.39pp | 5/5 |
| uniform0.4 | mlp_coteaching | 5 | +3.43pp | 5/5 |
| uniform0.4 | mlp_jocor | 5 | +2.07pp | 5/5 |
| uniform0.4 | mlp_elr | 5 | +7.70pp | 5/5 |
| uniform0.4 | mlp_sce | 5 | +5.56pp | 5/5 |
| uniform0.4 | mlp_fine | 5 | +8.62pp | 5/5 |
| uniform0.4 | mlp_dividemix_lite | 5 | +9.11pp | 5/5 |
| uniform0.4 | mlp_ct | 5 | +3.94pp | 5/5 |
| uniform0.4 | mlp_logitclip | 5 | +5.34pp | 5/5 |
| uniform0.4 | mlp_anl | 5 | +4.70pp | 5/5 |
| uniform0.6 | frozengrace_gb_mlp_softdist_beta05_shuffled | 5 | +11.14pp | 5/5 |
| uniform0.6 | mlp_ce | 5 | +12.19pp | 5/5 |
| uniform0.6 | mlp_small_loss | 5 | +9.39pp | 5/5 |
| uniform0.6 | mlp_coteaching | 5 | +7.49pp | 5/5 |
| uniform0.6 | mlp_jocor | 5 | +4.30pp | 5/5 |
| uniform0.6 | mlp_elr | 5 | +11.47pp | 5/5 |
| uniform0.6 | mlp_sce | 5 | +12.96pp | 5/5 |
| uniform0.6 | mlp_fine | 5 | +12.77pp | 5/5 |
| uniform0.6 | mlp_dividemix_lite | 5 | +14.16pp | 5/5 |
| uniform0.6 | mlp_ct | 5 | +9.97pp | 5/5 |
| uniform0.6 | mlp_logitclip | 5 | +9.20pp | 5/5 |
| uniform0.6 | mlp_anl | 5 | +11.75pp | 5/5 |
| pair0.4 | frozengrace_gb_mlp_softdist_beta05_shuffled | 5 | +9.56pp | 5/5 |
| pair0.4 | mlp_ce | 5 | +9.90pp | 5/5 |
| pair0.4 | mlp_small_loss | 5 | +5.80pp | 5/5 |
| pair0.4 | mlp_coteaching | 5 | +5.27pp | 4/5 |
| pair0.4 | mlp_jocor | 5 | +5.57pp | 4/5 |
| pair0.4 | mlp_elr | 5 | +6.44pp | 5/5 |
| pair0.4 | mlp_sce | 5 | +5.99pp | 5/5 |
| pair0.4 | mlp_fine | 5 | +10.97pp | 5/5 |
| pair0.4 | mlp_dividemix_lite | 5 | +6.50pp | 5/5 |
| pair0.4 | mlp_ct | 5 | +5.65pp | 5/5 |
| pair0.4 | mlp_logitclip | 5 | +3.63pp | 4/5 |
| pair0.4 | mlp_anl | 5 | +7.60pp | 5/5 |
| instance0.4 | frozengrace_gb_mlp_softdist_beta05_shuffled | 5 | +10.61pp | 5/5 |
| instance0.4 | mlp_ce | 5 | +10.46pp | 5/5 |
| instance0.4 | mlp_small_loss | 5 | +6.59pp | 5/5 |
| instance0.4 | mlp_coteaching | 5 | +6.01pp | 5/5 |
| instance0.4 | mlp_jocor | 5 | +4.79pp | 5/5 |
| instance0.4 | mlp_elr | 5 | +10.18pp | 5/5 |
| instance0.4 | mlp_sce | 5 | +6.51pp | 5/5 |
| instance0.4 | mlp_fine | 5 | +13.22pp | 5/5 |
| instance0.4 | mlp_dividemix_lite | 5 | +8.78pp | 5/5 |
| instance0.4 | mlp_ct | 5 | +6.02pp | 5/5 |
| instance0.4 | mlp_logitclip | 5 | +6.53pp | 5/5 |
| instance0.4 | mlp_anl | 5 | +6.41pp | 5/5 |

## Completeness
- Expected method-setting groups: 65
- Missing groups: 0
