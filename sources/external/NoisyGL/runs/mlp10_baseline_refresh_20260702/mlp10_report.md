# MLP-10 Baseline Refresh Report

## Material Passport
- Artifact type: baseline refresh and Coauthor-CS expansion
- Protocol: train-only label noise, percent split 0.2/0.1/0.7
- Backbone: X-only 2-layer MLP unless method name explicitly says otherwise
- Clean labels: evaluation only

## Selected LogitClip Tau
| Dataset | selected tau | mean seed0 noisy-valid | n settings |
|---|---:|---:|---:|
| citeseer | 0.5 | 47.17 | 5 |
| coauthorcs | 2.0 | 86.76 | 5 |
| cora | 0.5 | 50.22 | 5 |
| pubmed | 0.5 | 66.52 | 5 |

## Smoke Gates
- GA1 implementation health: not-pass
- GA2 Coauthor-CS feasibility: pass (5/5 settings)
- GA3 protocol reuse audit: pass
- Branch: **stop_after_smoke**

## Revised 7.3 Gate
- A4 ANL targeted audit: pass (3/4 seeds >= CE-2pp)
- Revised ANL decision: **include_anl_after_a4**

### A4 Details
| Seed | ANL test | CE same seed | ANL-CE | pass >= -2pp |
|---:|---:|---:|---:|---:|
| 1 | 41.45 | 41.92 | -0.47pp | 1 |
| 2 | 44.07 | 43.77 | +0.30pp | 1 |
| 3 | 45.19 | 46.48 | -1.29pp | 1 |
| 4 | 37.03 | 39.22 | -2.19pp | 0 |

### GA1 Details
| Dataset | Noise | Method | test | CE seed0 | pass CE-2pp | status |
|---|---|---|---:|---:|---:|---|
| cora | instance0.4 | mlp_anl | 44.64 | 44.17 | 1 | ok |
| citeseer | instance0.4 | mlp_anl | 41.71 | 43.64 | 1 | ok |
| pubmed | instance0.4 | mlp_anl | 66.66 | 60.95 | 1 | ok |
| cora | pair0.4 | mlp_anl | 43.48 | 41.85 | 1 | ok |
| citeseer | pair0.4 | mlp_anl | 42.70 | 44.42 | 1 | ok |
| pubmed | pair0.4 | mlp_anl | 61.29 | 58.80 | 1 | ok |
| cora | uniform0.2 | mlp_anl | 51.82 | 52.35 | 1 | ok |
| citeseer | uniform0.2 | mlp_anl | 52.84 | 53.44 | 1 | ok |
| pubmed | uniform0.2 | mlp_anl | 79.04 | 74.06 | 1 | ok |
| cora | uniform0.4 | mlp_anl | 42.27 | 41.74 | 1 | ok |
| citeseer | uniform0.4 | mlp_anl | 40.59 | 46.09 | 0 | ok |
| pubmed | uniform0.4 | mlp_anl | 68.07 | 62.62 | 1 | ok |
| cora | uniform0.6 | mlp_anl | 31.50 | 30.71 | 1 | ok |
| citeseer | uniform0.6 | mlp_anl | 31.49 | 32.13 | 1 | ok |
| pubmed | uniform0.6 | mlp_anl | 47.11 | 47.21 | 1 | ok |
| cora | instance0.4 | mlp_logitclip | 48.87 | 44.17 | 1 | ok |
| citeseer | instance0.4 | mlp_logitclip | 47.59 | 43.64 | 1 | ok |
| pubmed | instance0.4 | mlp_logitclip | 65.05 | 60.95 | 1 | ok |
| cora | pair0.4 | mlp_logitclip | 44.22 | 41.85 | 1 | ok |
| citeseer | pair0.4 | mlp_logitclip | 46.13 | 44.42 | 1 | ok |
| pubmed | pair0.4 | mlp_logitclip | 66.07 | 58.80 | 1 | ok |
| cora | uniform0.2 | mlp_logitclip | 54.14 | 52.35 | 1 | ok |
| citeseer | uniform0.2 | mlp_logitclip | 56.44 | 53.44 | 1 | ok |
| pubmed | uniform0.2 | mlp_logitclip | 78.65 | 74.06 | 1 | ok |
| cora | uniform0.4 | mlp_logitclip | 49.66 | 41.74 | 1 | ok |
| citeseer | uniform0.4 | mlp_logitclip | 48.32 | 46.09 | 1 | ok |
| pubmed | uniform0.4 | mlp_logitclip | 67.02 | 62.62 | 1 | ok |
| cora | uniform0.6 | mlp_logitclip | 34.09 | 30.71 | 1 | ok |
| citeseer | uniform0.6 | mlp_logitclip | 31.66 | 32.13 | 1 | ok |
| pubmed | uniform0.6 | mlp_logitclip | 46.74 | 47.21 | 1 | ok |

### GA2 Details
| Noise | Ours seed0 | CE seed0 | Shuffled seed0 | pass |
|---|---:|---:|---:|---:|
| uniform0.2 | 91.61 | 90.90 | 90.31 | 1 |
| uniform0.4 | 90.35 | 88.48 | 87.84 | 1 |
| uniform0.6 | 86.22 | 81.38 | 81.07 | 1 |
| pair0.4 | 85.67 | 75.07 | 76.08 | 1 |
| instance0.4 | 90.31 | 86.94 | 86.18 | 1 |

### GA3 Details
| Noise | Method | split match | noise match |
|---|---|---:|---:|
| instance0.4 | frozengrace_gb_mlp_softdist_beta05 | 1 | 1 |
| instance0.4 | frozengrace_gb_mlp_softdist_beta05_shuffled | 1 | 1 |
| instance0.4 | mlp_fine | 1 | 1 |
| instance0.4 | mlp_dividemix_lite | 1 | 1 |
| instance0.4 | mlp_ct | 1 | 1 |
| instance0.4 | mlp_anl | 1 | 1 |
| pair0.4 | frozengrace_gb_mlp_softdist_beta05 | 1 | 1 |
| pair0.4 | frozengrace_gb_mlp_softdist_beta05_shuffled | 1 | 1 |
| pair0.4 | mlp_fine | 1 | 1 |
| pair0.4 | mlp_dividemix_lite | 1 | 1 |
| pair0.4 | mlp_ct | 1 | 1 |
| pair0.4 | mlp_anl | 1 | 1 |
| uniform0.2 | frozengrace_gb_mlp_softdist_beta05 | 1 | 1 |
| uniform0.2 | frozengrace_gb_mlp_softdist_beta05_shuffled | 1 | 1 |
| uniform0.2 | mlp_fine | 1 | 1 |
| uniform0.2 | mlp_dividemix_lite | 1 | 1 |
| uniform0.2 | mlp_ct | 1 | 1 |
| uniform0.2 | mlp_anl | 1 | 1 |
| uniform0.4 | frozengrace_gb_mlp_softdist_beta05 | 1 | 1 |
| uniform0.4 | frozengrace_gb_mlp_softdist_beta05_shuffled | 1 | 1 |
| uniform0.4 | mlp_fine | 1 | 1 |
| uniform0.4 | mlp_dividemix_lite | 1 | 1 |
| uniform0.4 | mlp_ct | 1 | 1 |
| uniform0.4 | mlp_anl | 1 | 1 |
| uniform0.6 | frozengrace_gb_mlp_softdist_beta05 | 1 | 1 |
| uniform0.6 | frozengrace_gb_mlp_softdist_beta05_shuffled | 1 | 1 |
| uniform0.6 | mlp_fine | 1 | 1 |
| uniform0.6 | mlp_dividemix_lite | 1 | 1 |
| uniform0.6 | mlp_ct | 1 | 1 |
| uniform0.6 | mlp_anl | 1 | 1 |
| instance0.4 | mlp_logitclip | 1 | 1 |
| pair0.4 | mlp_logitclip | 1 | 1 |
| uniform0.2 | mlp_logitclip | 1 | 1 |
| uniform0.4 | mlp_logitclip | 1 | 1 |
| uniform0.6 | mlp_logitclip | 1 | 1 |
| instance0.4 | mlp_logitclip | 1 | 1 |
| pair0.4 | mlp_logitclip | 1 | 1 |
| uniform0.2 | mlp_logitclip | 1 | 1 |
| uniform0.4 | mlp_logitclip | 1 | 1 |
| uniform0.6 | mlp_logitclip | 1 | 1 |
| instance0.4 | mlp_logitclip | 1 | 1 |
| pair0.4 | mlp_logitclip | 1 | 1 |
| uniform0.2 | mlp_logitclip | 1 | 1 |
| uniform0.4 | mlp_logitclip | 1 | 1 |
| uniform0.6 | mlp_logitclip | 1 | 1 |

## Aggregate
| Dataset | Noise | Method | n | mean +/- std |
|---|---|---|---:|---:|
| citeseer | instance0.4 | Ours beta=0.5 | 5 | 54.62+/-2.25 |
| citeseer | instance0.4 | Shuffled | 5 | 42.81+/-2.34 |
| citeseer | instance0.4 | ANL | 5 | 42.21+/-4.40 |
| citeseer | instance0.4 | MLP+CE | 5 | 44.33+/-2.04 |
| citeseer | instance0.4 | Co-teaching | 5 | 46.81+/-1.68 |
| citeseer | instance0.4 | CT | 5 | 49.91+/-2.50 |
| citeseer | instance0.4 | DivideMix-lite | 5 | 44.19+/-2.86 |
| citeseer | instance0.4 | ELR | 5 | 41.94+/-3.27 |
| citeseer | instance0.4 | FINE | 5 | 44.88+/-2.40 |
| citeseer | instance0.4 | JoCoR | 5 | 50.86+/-2.45 |
| citeseer | instance0.4 | LogitClip | 5 | 47.53+/-2.72 |
| citeseer | instance0.4 | SCE | 5 | 40.92+/-4.23 |
| citeseer | instance0.4 | Small-loss | 5 | 50.75+/-3.33 |
| citeseer | pair0.4 | Ours beta=0.5 | 5 | 50.09+/-2.40 |
| citeseer | pair0.4 | Shuffled | 5 | 45.04+/-1.73 |
| citeseer | pair0.4 | ANL | 5 | 42.65+/-1.59 |
| citeseer | pair0.4 | MLP+CE | 5 | 45.15+/-1.67 |
| citeseer | pair0.4 | Co-teaching | 5 | 46.25+/-2.29 |
| citeseer | pair0.4 | CT | 5 | 48.81+/-2.50 |
| citeseer | pair0.4 | DivideMix-lite | 5 | 45.27+/-1.84 |
| citeseer | pair0.4 | ELR | 5 | 42.69+/-2.43 |
| citeseer | pair0.4 | FINE | 5 | 44.24+/-2.20 |
| citeseer | pair0.4 | JoCoR | 5 | 48.63+/-2.35 |
| citeseer | pair0.4 | LogitClip | 5 | 46.94+/-2.68 |
| citeseer | pair0.4 | SCE | 5 | 42.22+/-1.56 |
| citeseer | pair0.4 | Small-loss | 5 | 48.16+/-2.56 |
| citeseer | uniform0.2 | Ours beta=0.5 | 5 | 63.38+/-1.20 |
| citeseer | uniform0.2 | Shuffled | 5 | 54.53+/-1.83 |
| citeseer | uniform0.2 | ANL | 5 | 53.69+/-2.43 |
| citeseer | uniform0.2 | MLP+CE | 5 | 55.10+/-1.17 |
| citeseer | uniform0.2 | Co-teaching | 5 | 57.05+/-0.82 |
| citeseer | uniform0.2 | CT | 5 | 59.61+/-1.45 |
| citeseer | uniform0.2 | DivideMix-lite | 5 | 55.81+/-1.54 |
| citeseer | uniform0.2 | ELR | 5 | 53.62+/-1.48 |
| citeseer | uniform0.2 | FINE | 5 | 55.86+/-1.06 |
| citeseer | uniform0.2 | JoCoR | 5 | 59.36+/-1.43 |
| citeseer | uniform0.2 | LogitClip | 5 | 59.17+/-1.97 |
| citeseer | uniform0.2 | SCE | 5 | 52.12+/-3.28 |
| citeseer | uniform0.2 | Small-loss | 5 | 59.26+/-1.94 |
| citeseer | uniform0.4 | Ours beta=0.5 | 5 | 53.83+/-2.75 |
| citeseer | uniform0.4 | Shuffled | 5 | 42.35+/-2.74 |
| citeseer | uniform0.4 | ANL | 5 | 41.67+/-3.20 |
| citeseer | uniform0.4 | MLP+CE | 5 | 43.50+/-3.02 |
| citeseer | uniform0.4 | Co-teaching | 5 | 46.24+/-2.89 |
| citeseer | uniform0.4 | CT | 5 | 49.45+/-2.57 |
| citeseer | uniform0.4 | DivideMix-lite | 5 | 43.93+/-2.91 |
| citeseer | uniform0.4 | ELR | 5 | 43.17+/-3.95 |
| citeseer | uniform0.4 | FINE | 5 | 43.43+/-3.43 |
| citeseer | uniform0.4 | JoCoR | 5 | 49.91+/-3.37 |
| citeseer | uniform0.4 | LogitClip | 5 | 47.29+/-3.26 |
| citeseer | uniform0.4 | SCE | 5 | 42.09+/-3.47 |
| citeseer | uniform0.4 | Small-loss | 5 | 49.70+/-3.07 |
| citeseer | uniform0.6 | Ours beta=0.5 | 5 | 40.17+/-2.75 |
| citeseer | uniform0.6 | Shuffled | 5 | 31.28+/-1.31 |
| citeseer | uniform0.6 | ANL | 5 | 31.53+/-2.70 |
| citeseer | uniform0.6 | MLP+CE | 5 | 33.35+/-1.36 |
| citeseer | uniform0.6 | Co-teaching | 5 | 34.19+/-2.92 |
| citeseer | uniform0.6 | CT | 5 | 37.29+/-2.85 |
| citeseer | uniform0.6 | DivideMix-lite | 5 | 31.99+/-1.94 |
| citeseer | uniform0.6 | ELR | 5 | 30.80+/-2.70 |
| citeseer | uniform0.6 | FINE | 5 | 32.23+/-2.69 |
| citeseer | uniform0.6 | JoCoR | 5 | 39.07+/-2.47 |
| citeseer | uniform0.6 | LogitClip | 5 | 33.05+/-2.13 |
| citeseer | uniform0.6 | SCE | 5 | 30.59+/-2.83 |
| citeseer | uniform0.6 | Small-loss | 5 | 37.73+/-2.86 |
| coauthorcs | instance0.4 | Ours beta=0.5 | 5 | 90.50+/-0.49 |
| coauthorcs | instance0.4 | Shuffled | 5 | 86.89+/-1.53 |
| coauthorcs | instance0.4 | ANL | 5 | 88.33+/-0.70 |
| coauthorcs | instance0.4 | MLP+CE | 5 | 87.44+/-1.15 |
| coauthorcs | instance0.4 | Co-teaching | 5 | 88.95+/-0.97 |
| coauthorcs | instance0.4 | CT | 5 | 88.34+/-0.98 |
| coauthorcs | instance0.4 | DivideMix-lite | 5 | 87.42+/-1.28 |
| coauthorcs | instance0.4 | ELR | 5 | 87.78+/-0.94 |
| coauthorcs | instance0.4 | FINE | 5 | 87.35+/-1.16 |
| coauthorcs | instance0.4 | JoCoR | 5 | 89.17+/-0.85 |
| coauthorcs | instance0.4 | LogitClip | 5 | 89.20+/-0.76 |
| coauthorcs | instance0.4 | SCE | 5 | 87.99+/-0.56 |
| coauthorcs | instance0.4 | Small-loss | 5 | 88.97+/-1.02 |
| coauthorcs | pair0.4 | Ours beta=0.5 | 5 | 85.96+/-1.41 |
| coauthorcs | pair0.4 | Shuffled | 5 | 76.05+/-0.54 |
| coauthorcs | pair0.4 | ANL | 5 | 79.55+/-2.56 |
| coauthorcs | pair0.4 | MLP+CE | 5 | 75.91+/-1.34 |
| coauthorcs | pair0.4 | Co-teaching | 5 | 79.06+/-1.30 |
| coauthorcs | pair0.4 | CT | 5 | 77.71+/-2.02 |
| coauthorcs | pair0.4 | DivideMix-lite | 5 | 77.38+/-1.82 |
| coauthorcs | pair0.4 | ELR | 5 | 78.03+/-1.48 |
| coauthorcs | pair0.4 | FINE | 5 | 75.12+/-1.19 |
| coauthorcs | pair0.4 | JoCoR | 5 | 81.02+/-1.62 |
| coauthorcs | pair0.4 | LogitClip | 5 | 81.03+/-0.57 |
| coauthorcs | pair0.4 | SCE | 5 | 79.77+/-2.51 |
| coauthorcs | pair0.4 | Small-loss | 5 | 80.13+/-1.50 |
| coauthorcs | uniform0.2 | Ours beta=0.5 | 5 | 91.56+/-0.25 |
| coauthorcs | uniform0.2 | Shuffled | 5 | 90.40+/-0.69 |
| coauthorcs | uniform0.2 | ANL | 5 | 91.02+/-0.49 |
| coauthorcs | uniform0.2 | MLP+CE | 5 | 91.38+/-0.54 |
| coauthorcs | uniform0.2 | Co-teaching | 5 | 91.70+/-0.29 |
| coauthorcs | uniform0.2 | CT | 5 | 91.05+/-0.50 |
| coauthorcs | uniform0.2 | DivideMix-lite | 5 | 91.14+/-0.57 |
| coauthorcs | uniform0.2 | ELR | 5 | 91.21+/-0.60 |
| coauthorcs | uniform0.2 | FINE | 5 | 91.27+/-0.67 |
| coauthorcs | uniform0.2 | JoCoR | 5 | 91.81+/-0.25 |
| coauthorcs | uniform0.2 | LogitClip | 5 | 91.81+/-0.21 |
| coauthorcs | uniform0.2 | SCE | 5 | 90.63+/-0.47 |
| coauthorcs | uniform0.2 | Small-loss | 5 | 91.50+/-0.49 |
| coauthorcs | uniform0.4 | Ours beta=0.5 | 5 | 90.62+/-0.57 |
| coauthorcs | uniform0.4 | Shuffled | 5 | 87.60+/-1.03 |
| coauthorcs | uniform0.4 | ANL | 5 | 88.91+/-0.57 |
| coauthorcs | uniform0.4 | MLP+CE | 5 | 88.42+/-0.62 |
| coauthorcs | uniform0.4 | Co-teaching | 5 | 88.80+/-0.97 |
| coauthorcs | uniform0.4 | CT | 5 | 88.37+/-0.72 |
| coauthorcs | uniform0.4 | DivideMix-lite | 5 | 88.41+/-0.62 |
| coauthorcs | uniform0.4 | ELR | 5 | 88.29+/-0.73 |
| coauthorcs | uniform0.4 | FINE | 5 | 87.79+/-0.78 |
| coauthorcs | uniform0.4 | JoCoR | 5 | 89.38+/-0.61 |
| coauthorcs | uniform0.4 | LogitClip | 5 | 88.94+/-0.24 |
| coauthorcs | uniform0.4 | SCE | 5 | 88.37+/-0.53 |
| coauthorcs | uniform0.4 | Small-loss | 5 | 89.06+/-0.88 |
| coauthorcs | uniform0.6 | Ours beta=0.5 | 5 | 87.17+/-0.62 |
| coauthorcs | uniform0.6 | Shuffled | 5 | 78.83+/-1.93 |
| coauthorcs | uniform0.6 | ANL | 5 | 79.53+/-1.33 |
| coauthorcs | uniform0.6 | MLP+CE | 5 | 80.03+/-1.91 |
| coauthorcs | uniform0.6 | Co-teaching | 5 | 81.88+/-0.72 |
| coauthorcs | uniform0.6 | CT | 5 | 82.45+/-0.57 |
| coauthorcs | uniform0.6 | DivideMix-lite | 5 | 80.03+/-1.91 |
| coauthorcs | uniform0.6 | ELR | 5 | 80.54+/-1.28 |
| coauthorcs | uniform0.6 | FINE | 5 | 79.94+/-1.82 |
| coauthorcs | uniform0.6 | JoCoR | 5 | 82.58+/-0.41 |
| coauthorcs | uniform0.6 | LogitClip | 5 | 80.48+/-1.16 |
| coauthorcs | uniform0.6 | SCE | 5 | 79.43+/-1.45 |
| coauthorcs | uniform0.6 | Small-loss | 5 | 81.65+/-0.91 |
| cora | instance0.4 | Ours beta=0.5 | 5 | 55.67+/-0.81 |
| cora | instance0.4 | Shuffled | 5 | 44.03+/-1.75 |
| cora | instance0.4 | ANL | 5 | 44.14+/-2.39 |
| cora | instance0.4 | MLP+CE | 5 | 45.33+/-2.22 |
| cora | instance0.4 | Co-teaching | 5 | 47.35+/-2.24 |
| cora | instance0.4 | CT | 5 | 50.43+/-2.26 |
| cora | instance0.4 | DivideMix-lite | 5 | 44.75+/-3.03 |
| cora | instance0.4 | ELR | 5 | 45.93+/-2.26 |
| cora | instance0.4 | FINE | 5 | 44.76+/-2.74 |
| cora | instance0.4 | JoCoR | 5 | 52.26+/-1.01 |
| cora | instance0.4 | LogitClip | 5 | 47.35+/-1.27 |
| cora | instance0.4 | SCE | 5 | 43.29+/-4.35 |
| cora | instance0.4 | Small-loss | 5 | 51.82+/-0.94 |
| cora | pair0.4 | Ours beta=0.5 | 5 | 50.29+/-2.76 |
| cora | pair0.4 | Shuffled | 5 | 39.38+/-1.28 |
| cora | pair0.4 | ANL | 5 | 42.59+/-1.89 |
| cora | pair0.4 | MLP+CE | 5 | 42.83+/-2.19 |
| cora | pair0.4 | Co-teaching | 5 | 44.26+/-2.30 |
| cora | pair0.4 | CT | 5 | 45.69+/-2.83 |
| cora | pair0.4 | DivideMix-lite | 5 | 43.34+/-2.34 |
| cora | pair0.4 | ELR | 5 | 41.83+/-1.75 |
| cora | pair0.4 | FINE | 5 | 41.72+/-1.11 |
| cora | pair0.4 | JoCoR | 5 | 45.91+/-2.38 |
| cora | pair0.4 | LogitClip | 5 | 45.44+/-1.14 |
| cora | pair0.4 | SCE | 5 | 41.40+/-2.04 |
| cora | pair0.4 | Small-loss | 5 | 46.99+/-2.77 |
| cora | uniform0.2 | Ours beta=0.5 | 5 | 62.94+/-1.60 |
| cora | uniform0.2 | Shuffled | 5 | 53.70+/-2.06 |
| cora | uniform0.2 | ANL | 5 | 54.24+/-1.67 |
| cora | uniform0.2 | MLP+CE | 5 | 54.96+/-2.22 |
| cora | uniform0.2 | Co-teaching | 5 | 58.62+/-1.97 |
| cora | uniform0.2 | CT | 5 | 59.87+/-1.05 |
| cora | uniform0.2 | DivideMix-lite | 5 | 55.19+/-2.30 |
| cora | uniform0.2 | ELR | 5 | 52.95+/-2.38 |
| cora | uniform0.2 | FINE | 5 | 54.85+/-1.98 |
| cora | uniform0.2 | JoCoR | 5 | 60.12+/-2.25 |
| cora | uniform0.2 | LogitClip | 5 | 58.22+/-2.48 |
| cora | uniform0.2 | SCE | 5 | 53.57+/-2.17 |
| cora | uniform0.2 | Small-loss | 5 | 59.84+/-1.29 |
| cora | uniform0.4 | Ours beta=0.5 | 5 | 55.59+/-1.50 |
| cora | uniform0.4 | Shuffled | 5 | 43.63+/-1.62 |
| cora | uniform0.4 | ANL | 5 | 44.71+/-2.26 |
| cora | uniform0.4 | MLP+CE | 5 | 44.24+/-2.62 |
| cora | uniform0.4 | Co-teaching | 5 | 46.83+/-1.78 |
| cora | uniform0.4 | CT | 5 | 51.45+/-2.20 |
| cora | uniform0.4 | DivideMix-lite | 5 | 44.79+/-2.05 |
| cora | uniform0.4 | ELR | 5 | 44.28+/-1.38 |
| cora | uniform0.4 | FINE | 5 | 43.54+/-1.48 |
| cora | uniform0.4 | JoCoR | 5 | 51.72+/-1.30 |
| cora | uniform0.4 | LogitClip | 5 | 48.79+/-1.49 |
| cora | uniform0.4 | SCE | 5 | 43.60+/-1.86 |
| cora | uniform0.4 | Small-loss | 5 | 51.95+/-1.97 |
| cora | uniform0.6 | Ours beta=0.5 | 5 | 41.04+/-2.37 |
| cora | uniform0.6 | Shuffled | 5 | 31.70+/-2.69 |
| cora | uniform0.6 | ANL | 5 | 32.99+/-2.44 |
| cora | uniform0.6 | MLP+CE | 5 | 32.51+/-3.38 |
| cora | uniform0.6 | Co-teaching | 5 | 33.29+/-3.48 |
| cora | uniform0.6 | CT | 5 | 34.48+/-1.24 |
| cora | uniform0.6 | DivideMix-lite | 5 | 32.30+/-3.61 |
| cora | uniform0.6 | ELR | 5 | 32.82+/-3.51 |
| cora | uniform0.6 | FINE | 5 | 31.77+/-3.04 |
| cora | uniform0.6 | JoCoR | 5 | 36.92+/-2.14 |
| cora | uniform0.6 | LogitClip | 5 | 34.09+/-2.06 |
| cora | uniform0.6 | SCE | 5 | 32.66+/-3.78 |
| cora | uniform0.6 | Small-loss | 5 | 36.72+/-1.09 |
| pubmed | instance0.4 | Ours beta=0.5 | 5 | 72.01+/-1.23 |
| pubmed | instance0.4 | Shuffled | 5 | 61.05+/-3.37 |
| pubmed | instance0.4 | ANL | 5 | 68.73+/-1.80 |
| pubmed | instance0.4 | MLP+CE | 5 | 62.69+/-2.78 |
| pubmed | instance0.4 | Co-teaching | 5 | 74.89+/-0.87 |
| pubmed | instance0.4 | CT | 5 | 76.25+/-1.22 |
| pubmed | instance0.4 | DivideMix-lite | 5 | 76.69+/-1.51 |
| pubmed | instance0.4 | ELR | 5 | 62.80+/-1.74 |
| pubmed | instance0.4 | FINE | 5 | 60.05+/-2.43 |
| pubmed | instance0.4 | JoCoR | 5 | 76.89+/-0.80 |
| pubmed | instance0.4 | LogitClip | 5 | 67.08+/-3.43 |
| pubmed | instance0.4 | SCE | 5 | 67.75+/-2.45 |
| pubmed | instance0.4 | Small-loss | 5 | 76.47+/-1.25 |
| pubmed | pair0.4 | Ours beta=0.5 | 5 | 68.49+/-1.27 |
| pubmed | pair0.4 | Shuffled | 5 | 58.10+/-1.19 |
| pubmed | pair0.4 | ANL | 5 | 61.17+/-0.88 |
| pubmed | pair0.4 | MLP+CE | 5 | 59.48+/-1.38 |
| pubmed | pair0.4 | Co-teaching | 5 | 70.41+/-1.57 |
| pubmed | pair0.4 | CT | 5 | 70.97+/-2.34 |
| pubmed | pair0.4 | DivideMix-lite | 5 | 73.42+/-2.25 |
| pubmed | pair0.4 | ELR | 5 | 59.96+/-1.85 |
| pubmed | pair0.4 | FINE | 5 | 52.53+/-3.22 |
| pubmed | pair0.4 | JoCoR | 5 | 73.93+/-1.47 |
| pubmed | pair0.4 | LogitClip | 5 | 64.54+/-2.38 |
| pubmed | pair0.4 | SCE | 5 | 61.08+/-2.08 |
| pubmed | pair0.4 | Small-loss | 5 | 73.19+/-1.70 |
| pubmed | uniform0.2 | Ours beta=0.5 | 5 | 82.44+/-0.42 |
| pubmed | uniform0.2 | Shuffled | 5 | 75.31+/-0.97 |
| pubmed | uniform0.2 | ANL | 5 | 79.69+/-0.74 |
| pubmed | uniform0.2 | MLP+CE | 5 | 74.63+/-1.49 |
| pubmed | uniform0.2 | Co-teaching | 5 | 81.12+/-0.55 |
| pubmed | uniform0.2 | CT | 5 | 81.52+/-0.53 |
| pubmed | uniform0.2 | DivideMix-lite | 5 | 83.25+/-0.72 |
| pubmed | uniform0.2 | ELR | 5 | 76.71+/-0.53 |
| pubmed | uniform0.2 | FINE | 5 | 74.78+/-1.13 |
| pubmed | uniform0.2 | JoCoR | 5 | 81.70+/-0.71 |
| pubmed | uniform0.2 | LogitClip | 5 | 79.25+/-0.66 |
| pubmed | uniform0.2 | SCE | 5 | 79.45+/-0.62 |
| pubmed | uniform0.2 | Small-loss | 5 | 81.88+/-0.85 |
| pubmed | uniform0.4 | Ours beta=0.5 | 5 | 71.90+/-0.88 |
| pubmed | uniform0.4 | Shuffled | 5 | 62.72+/-4.49 |
| pubmed | uniform0.4 | ANL | 5 | 69.31+/-2.43 |
| pubmed | uniform0.4 | MLP+CE | 5 | 63.77+/-4.40 |
| pubmed | uniform0.4 | Co-teaching | 5 | 75.77+/-0.99 |
| pubmed | uniform0.4 | CT | 5 | 77.23+/-0.33 |
| pubmed | uniform0.4 | DivideMix-lite | 5 | 78.28+/-0.54 |
| pubmed | uniform0.4 | ELR | 5 | 64.03+/-2.35 |
| pubmed | uniform0.4 | FINE | 5 | 61.55+/-5.80 |
| pubmed | uniform0.4 | JoCoR | 5 | 77.84+/-1.10 |
| pubmed | uniform0.4 | LogitClip | 5 | 65.69+/-1.31 |
| pubmed | uniform0.4 | SCE | 5 | 68.37+/-1.27 |
| pubmed | uniform0.4 | Small-loss | 5 | 77.30+/-1.29 |
| pubmed | uniform0.6 | Ours beta=0.5 | 5 | 47.53+/-2.42 |
| pubmed | uniform0.6 | Shuffled | 5 | 44.27+/-2.72 |
| pubmed | uniform0.6 | ANL | 5 | 46.69+/-3.00 |
| pubmed | uniform0.6 | MLP+CE | 5 | 45.50+/-2.45 |
| pubmed | uniform0.6 | Co-teaching | 5 | 54.50+/-6.19 |
| pubmed | uniform0.6 | CT | 5 | 53.32+/-4.35 |
| pubmed | uniform0.6 | DivideMix-lite | 5 | 48.57+/-4.14 |
| pubmed | uniform0.6 | ELR | 5 | 45.74+/-2.13 |
| pubmed | uniform0.6 | FINE | 5 | 44.03+/-3.65 |
| pubmed | uniform0.6 | JoCoR | 5 | 58.12+/-5.83 |
| pubmed | uniform0.6 | LogitClip | 5 | 47.08+/-2.15 |
| pubmed | uniform0.6 | SCE | 5 | 45.49+/-3.02 |
| pubmed | uniform0.6 | Small-loss | 5 | 54.72+/-5.02 |

## Ours Paired Deltas
| Dataset | Noise | Baseline | n | mean delta | std | positive |
|---|---|---|---:|---:|---:|---:|
| cora | uniform0.2 | frozengrace_gb_mlp_softdist_beta05_shuffled | 5 | +9.25pp | +1.46pp | 5/5 |
| cora | uniform0.2 | mlp_anl | 5 | +8.71pp | +2.37pp | 5/5 |
| cora | uniform0.2 | mlp_ce | 5 | +7.99pp | +1.57pp | 5/5 |
| cora | uniform0.2 | mlp_coteaching | 5 | +4.33pp | +1.74pp | 5/5 |
| cora | uniform0.2 | mlp_ct | 5 | +3.07pp | +1.98pp | 5/5 |
| cora | uniform0.2 | mlp_dividemix_lite | 5 | +7.76pp | +0.82pp | 5/5 |
| cora | uniform0.2 | mlp_elr | 5 | +9.99pp | +2.27pp | 5/5 |
| cora | uniform0.2 | mlp_fine | 5 | +8.09pp | +2.25pp | 5/5 |
| cora | uniform0.2 | mlp_jocor | 5 | +2.83pp | +2.37pp | 4/5 |
| cora | uniform0.2 | mlp_logitclip | 5 | +4.73pp | +2.67pp | 5/5 |
| cora | uniform0.2 | mlp_sce | 5 | +9.37pp | +1.86pp | 5/5 |
| cora | uniform0.2 | mlp_small_loss | 5 | +3.10pp | +2.11pp | 5/5 |
| cora | uniform0.4 | frozengrace_gb_mlp_softdist_beta05_shuffled | 5 | +11.96pp | +0.29pp | 5/5 |
| cora | uniform0.4 | mlp_anl | 5 | +10.88pp | +1.53pp | 5/5 |
| cora | uniform0.4 | mlp_ce | 5 | +11.35pp | +1.71pp | 5/5 |
| cora | uniform0.4 | mlp_coteaching | 5 | +8.76pp | +1.64pp | 5/5 |
| cora | uniform0.4 | mlp_ct | 5 | +4.14pp | +2.40pp | 5/5 |
| cora | uniform0.4 | mlp_dividemix_lite | 5 | +10.80pp | +1.81pp | 5/5 |
| cora | uniform0.4 | mlp_elr | 5 | +11.30pp | +1.15pp | 5/5 |
| cora | uniform0.4 | mlp_fine | 5 | +12.05pp | +1.05pp | 5/5 |
| cora | uniform0.4 | mlp_jocor | 5 | +3.87pp | +1.81pp | 5/5 |
| cora | uniform0.4 | mlp_logitclip | 5 | +6.80pp | +1.09pp | 5/5 |
| cora | uniform0.4 | mlp_sce | 5 | +11.99pp | +2.88pp | 5/5 |
| cora | uniform0.4 | mlp_small_loss | 5 | +3.64pp | +2.28pp | 4/5 |
| cora | uniform0.6 | frozengrace_gb_mlp_softdist_beta05_shuffled | 5 | +9.34pp | +2.97pp | 5/5 |
| cora | uniform0.6 | mlp_anl | 5 | +8.05pp | +2.57pp | 5/5 |
| cora | uniform0.6 | mlp_ce | 5 | +8.54pp | +3.57pp | 5/5 |
| cora | uniform0.6 | mlp_coteaching | 5 | +7.76pp | +5.22pp | 5/5 |
| cora | uniform0.6 | mlp_ct | 5 | +6.56pp | +3.42pp | 5/5 |
| cora | uniform0.6 | mlp_dividemix_lite | 5 | +8.75pp | +3.85pp | 5/5 |
| cora | uniform0.6 | mlp_elr | 5 | +8.22pp | +3.16pp | 5/5 |
| cora | uniform0.6 | mlp_fine | 5 | +9.28pp | +1.80pp | 5/5 |
| cora | uniform0.6 | mlp_jocor | 5 | +4.13pp | +3.16pp | 5/5 |
| cora | uniform0.6 | mlp_logitclip | 5 | +6.96pp | +1.91pp | 5/5 |
| cora | uniform0.6 | mlp_sce | 5 | +8.38pp | +3.47pp | 5/5 |
| cora | uniform0.6 | mlp_small_loss | 5 | +4.33pp | +2.28pp | 5/5 |
| cora | pair0.4 | frozengrace_gb_mlp_softdist_beta05_shuffled | 5 | +10.91pp | +3.12pp | 5/5 |
| cora | pair0.4 | mlp_anl | 5 | +7.70pp | +2.04pp | 5/5 |
| cora | pair0.4 | mlp_ce | 5 | +7.46pp | +2.25pp | 5/5 |
| cora | pair0.4 | mlp_coteaching | 5 | +6.03pp | +2.43pp | 5/5 |
| cora | pair0.4 | mlp_ct | 5 | +4.60pp | +2.22pp | 5/5 |
| cora | pair0.4 | mlp_dividemix_lite | 5 | +6.96pp | +1.24pp | 5/5 |
| cora | pair0.4 | mlp_elr | 5 | +8.46pp | +3.34pp | 5/5 |
| cora | pair0.4 | mlp_fine | 5 | +8.57pp | +2.29pp | 5/5 |
| cora | pair0.4 | mlp_jocor | 5 | +4.38pp | +0.85pp | 5/5 |
| cora | pair0.4 | mlp_logitclip | 5 | +4.85pp | +2.33pp | 5/5 |
| cora | pair0.4 | mlp_sce | 5 | +8.89pp | +2.16pp | 5/5 |
| cora | pair0.4 | mlp_small_loss | 5 | +3.30pp | +1.03pp | 5/5 |
| cora | instance0.4 | frozengrace_gb_mlp_softdist_beta05_shuffled | 5 | +11.64pp | +1.23pp | 5/5 |
| cora | instance0.4 | mlp_anl | 5 | +11.54pp | +2.15pp | 5/5 |
| cora | instance0.4 | mlp_ce | 5 | +10.34pp | +1.79pp | 5/5 |
| cora | instance0.4 | mlp_coteaching | 5 | +8.33pp | +1.66pp | 5/5 |
| cora | instance0.4 | mlp_ct | 5 | +5.25pp | +2.76pp | 5/5 |
| cora | instance0.4 | mlp_dividemix_lite | 5 | +10.92pp | +2.27pp | 5/5 |
| cora | instance0.4 | mlp_elr | 5 | +9.74pp | +1.83pp | 5/5 |
| cora | instance0.4 | mlp_fine | 5 | +10.91pp | +2.27pp | 5/5 |
| cora | instance0.4 | mlp_jocor | 5 | +3.41pp | +1.19pp | 5/5 |
| cora | instance0.4 | mlp_logitclip | 5 | +8.33pp | +1.54pp | 5/5 |
| cora | instance0.4 | mlp_sce | 5 | +12.38pp | +3.88pp | 5/5 |
| cora | instance0.4 | mlp_small_loss | 5 | +3.85pp | +1.40pp | 5/5 |
| citeseer | uniform0.2 | frozengrace_gb_mlp_softdist_beta05_shuffled | 5 | +8.86pp | +0.92pp | 5/5 |
| citeseer | uniform0.2 | mlp_anl | 5 | +9.70pp | +2.93pp | 5/5 |
| citeseer | uniform0.2 | mlp_ce | 5 | +8.28pp | +1.21pp | 5/5 |
| citeseer | uniform0.2 | mlp_coteaching | 5 | +6.33pp | +1.44pp | 5/5 |
| citeseer | uniform0.2 | mlp_ct | 5 | +3.77pp | +1.24pp | 5/5 |
| citeseer | uniform0.2 | mlp_dividemix_lite | 5 | +7.58pp | +1.31pp | 5/5 |
| citeseer | uniform0.2 | mlp_elr | 5 | +9.77pp | +1.61pp | 5/5 |
| citeseer | uniform0.2 | mlp_fine | 5 | +7.53pp | +0.73pp | 5/5 |
| citeseer | uniform0.2 | mlp_jocor | 5 | +4.03pp | +1.43pp | 5/5 |
| citeseer | uniform0.2 | mlp_logitclip | 5 | +4.22pp | +0.88pp | 5/5 |
| citeseer | uniform0.2 | mlp_sce | 5 | +11.26pp | +2.79pp | 5/5 |
| citeseer | uniform0.2 | mlp_small_loss | 5 | +4.12pp | +1.60pp | 5/5 |
| citeseer | uniform0.4 | frozengrace_gb_mlp_softdist_beta05_shuffled | 5 | +11.48pp | +2.63pp | 5/5 |
| citeseer | uniform0.4 | mlp_anl | 5 | +12.16pp | +2.48pp | 5/5 |
| citeseer | uniform0.4 | mlp_ce | 5 | +10.34pp | +2.04pp | 5/5 |
| citeseer | uniform0.4 | mlp_coteaching | 5 | +7.59pp | +1.47pp | 5/5 |
| citeseer | uniform0.4 | mlp_ct | 5 | +4.38pp | +2.59pp | 5/5 |
| citeseer | uniform0.4 | mlp_dividemix_lite | 5 | +9.90pp | +1.35pp | 5/5 |
| citeseer | uniform0.4 | mlp_elr | 5 | +10.66pp | +3.01pp | 5/5 |
| citeseer | uniform0.4 | mlp_fine | 5 | +10.40pp | +2.64pp | 5/5 |
| citeseer | uniform0.4 | mlp_jocor | 5 | +3.93pp | +2.27pp | 5/5 |
| citeseer | uniform0.4 | mlp_logitclip | 5 | +6.55pp | +1.50pp | 5/5 |
| citeseer | uniform0.4 | mlp_sce | 5 | +11.74pp | +2.58pp | 5/5 |
| citeseer | uniform0.4 | mlp_small_loss | 5 | +4.13pp | +1.86pp | 5/5 |
| citeseer | uniform0.6 | frozengrace_gb_mlp_softdist_beta05_shuffled | 5 | +8.89pp | +1.51pp | 5/5 |
| citeseer | uniform0.6 | mlp_anl | 5 | +8.64pp | +0.85pp | 5/5 |
| citeseer | uniform0.6 | mlp_ce | 5 | +6.82pp | +2.83pp | 5/5 |
| citeseer | uniform0.6 | mlp_coteaching | 5 | +5.98pp | +1.76pp | 5/5 |
| citeseer | uniform0.6 | mlp_ct | 5 | +2.89pp | +0.63pp | 5/5 |
| citeseer | uniform0.6 | mlp_dividemix_lite | 5 | +8.18pp | +2.40pp | 5/5 |
| citeseer | uniform0.6 | mlp_elr | 5 | +9.37pp | +1.43pp | 5/5 |
| citeseer | uniform0.6 | mlp_fine | 5 | +7.94pp | +0.91pp | 5/5 |
| citeseer | uniform0.6 | mlp_jocor | 5 | +1.10pp | +1.63pp | 4/5 |
| citeseer | uniform0.6 | mlp_logitclip | 5 | +7.12pp | +2.04pp | 5/5 |
| citeseer | uniform0.6 | mlp_sce | 5 | +9.58pp | +1.19pp | 5/5 |
| citeseer | uniform0.6 | mlp_small_loss | 5 | +2.44pp | +1.32pp | 5/5 |
| citeseer | pair0.4 | frozengrace_gb_mlp_softdist_beta05_shuffled | 5 | +5.04pp | +2.47pp | 5/5 |
| citeseer | pair0.4 | mlp_anl | 5 | +7.43pp | +2.07pp | 5/5 |
| citeseer | pair0.4 | mlp_ce | 5 | +4.94pp | +2.07pp | 5/5 |
| citeseer | pair0.4 | mlp_coteaching | 5 | +3.83pp | +1.29pp | 5/5 |
| citeseer | pair0.4 | mlp_ct | 5 | +1.27pp | +0.66pp | 5/5 |
| citeseer | pair0.4 | mlp_dividemix_lite | 5 | +4.82pp | +1.81pp | 5/5 |
| citeseer | pair0.4 | mlp_elr | 5 | +7.40pp | +2.93pp | 5/5 |
| citeseer | pair0.4 | mlp_fine | 5 | +5.84pp | +2.33pp | 5/5 |
| citeseer | pair0.4 | mlp_jocor | 5 | +1.45pp | +1.17pp | 5/5 |
| citeseer | pair0.4 | mlp_logitclip | 5 | +3.14pp | +1.08pp | 5/5 |
| citeseer | pair0.4 | mlp_sce | 5 | +7.87pp | +1.36pp | 5/5 |
| citeseer | pair0.4 | mlp_small_loss | 5 | +1.92pp | +1.16pp | 5/5 |
| citeseer | instance0.4 | frozengrace_gb_mlp_softdist_beta05_shuffled | 5 | +11.81pp | +2.00pp | 5/5 |
| citeseer | instance0.4 | mlp_anl | 5 | +12.41pp | +4.16pp | 5/5 |
| citeseer | instance0.4 | mlp_ce | 5 | +10.29pp | +1.92pp | 5/5 |
| citeseer | instance0.4 | mlp_coteaching | 5 | +7.81pp | +1.84pp | 5/5 |
| citeseer | instance0.4 | mlp_ct | 5 | +4.71pp | +0.94pp | 5/5 |
| citeseer | instance0.4 | mlp_dividemix_lite | 5 | +10.43pp | +1.45pp | 5/5 |
| citeseer | instance0.4 | mlp_elr | 5 | +12.68pp | +3.49pp | 5/5 |
| citeseer | instance0.4 | mlp_fine | 5 | +9.74pp | +1.55pp | 5/5 |
| citeseer | instance0.4 | mlp_jocor | 5 | +3.76pp | +0.98pp | 5/5 |
| citeseer | instance0.4 | mlp_logitclip | 5 | +7.09pp | +1.71pp | 5/5 |
| citeseer | instance0.4 | mlp_sce | 5 | +13.70pp | +4.08pp | 5/5 |
| citeseer | instance0.4 | mlp_small_loss | 5 | +3.87pp | +2.20pp | 5/5 |
| pubmed | uniform0.2 | frozengrace_gb_mlp_softdist_beta05_shuffled | 5 | +7.12pp | +0.84pp | 5/5 |
| pubmed | uniform0.2 | mlp_anl | 5 | +2.74pp | +0.86pp | 5/5 |
| pubmed | uniform0.2 | mlp_ce | 5 | +7.80pp | +1.43pp | 5/5 |
| pubmed | uniform0.2 | mlp_coteaching | 5 | +1.32pp | +0.69pp | 5/5 |
| pubmed | uniform0.2 | mlp_ct | 5 | +0.91pp | +0.53pp | 5/5 |
| pubmed | uniform0.2 | mlp_dividemix_lite | 5 | -0.82pp | +0.79pp | 1/5 |
| pubmed | uniform0.2 | mlp_elr | 5 | +5.72pp | +0.61pp | 5/5 |
| pubmed | uniform0.2 | mlp_fine | 5 | +7.66pp | +1.12pp | 5/5 |
| pubmed | uniform0.2 | mlp_jocor | 5 | +0.74pp | +0.68pp | 5/5 |
| pubmed | uniform0.2 | mlp_logitclip | 5 | +3.18pp | +0.73pp | 5/5 |
| pubmed | uniform0.2 | mlp_sce | 5 | +2.98pp | +0.69pp | 5/5 |
| pubmed | uniform0.2 | mlp_small_loss | 5 | +0.55pp | +0.82pp | 4/5 |
| pubmed | uniform0.4 | frozengrace_gb_mlp_softdist_beta05_shuffled | 5 | +9.18pp | +4.36pp | 5/5 |
| pubmed | uniform0.4 | mlp_anl | 5 | +2.59pp | +2.56pp | 4/5 |
| pubmed | uniform0.4 | mlp_ce | 5 | +8.13pp | +4.24pp | 5/5 |
| pubmed | uniform0.4 | mlp_coteaching | 5 | -3.87pp | +0.31pp | 0/5 |
| pubmed | uniform0.4 | mlp_ct | 5 | -5.33pp | +0.73pp | 0/5 |
| pubmed | uniform0.4 | mlp_dividemix_lite | 5 | -6.38pp | +0.83pp | 0/5 |
| pubmed | uniform0.4 | mlp_elr | 5 | +7.87pp | +2.10pp | 5/5 |
| pubmed | uniform0.4 | mlp_fine | 5 | +10.36pp | +5.67pp | 5/5 |
| pubmed | uniform0.4 | mlp_jocor | 5 | -5.93pp | +0.69pp | 0/5 |
| pubmed | uniform0.4 | mlp_logitclip | 5 | +6.21pp | +1.55pp | 5/5 |
| pubmed | uniform0.4 | mlp_sce | 5 | +3.53pp | +1.75pp | 5/5 |
| pubmed | uniform0.4 | mlp_small_loss | 5 | -5.40pp | +0.72pp | 0/5 |
| pubmed | uniform0.6 | frozengrace_gb_mlp_softdist_beta05_shuffled | 5 | +3.26pp | +1.50pp | 5/5 |
| pubmed | uniform0.6 | mlp_anl | 5 | +0.85pp | +1.79pp | 3/5 |
| pubmed | uniform0.6 | mlp_ce | 5 | +2.03pp | +1.29pp | 5/5 |
| pubmed | uniform0.6 | mlp_coteaching | 5 | -6.97pp | +3.93pp | 0/5 |
| pubmed | uniform0.6 | mlp_ct | 5 | -5.79pp | +2.64pp | 0/5 |
| pubmed | uniform0.6 | mlp_dividemix_lite | 5 | -1.04pp | +2.26pp | 2/5 |
| pubmed | uniform0.6 | mlp_elr | 5 | +1.79pp | +1.18pp | 5/5 |
| pubmed | uniform0.6 | mlp_fine | 5 | +3.50pp | +1.92pp | 5/5 |
| pubmed | uniform0.6 | mlp_jocor | 5 | -10.58pp | +3.60pp | 0/5 |
| pubmed | uniform0.6 | mlp_logitclip | 5 | +0.45pp | +1.59pp | 3/5 |
| pubmed | uniform0.6 | mlp_sce | 5 | +2.04pp | +1.70pp | 4/5 |
| pubmed | uniform0.6 | mlp_small_loss | 5 | -7.19pp | +2.66pp | 0/5 |
| pubmed | pair0.4 | frozengrace_gb_mlp_softdist_beta05_shuffled | 5 | +10.39pp | +1.14pp | 5/5 |
| pubmed | pair0.4 | mlp_anl | 5 | +7.31pp | +0.92pp | 5/5 |
| pubmed | pair0.4 | mlp_ce | 5 | +9.01pp | +1.41pp | 5/5 |
| pubmed | pair0.4 | mlp_coteaching | 5 | -1.92pp | +1.36pp | 0/5 |
| pubmed | pair0.4 | mlp_ct | 5 | -2.48pp | +1.87pp | 0/5 |
| pubmed | pair0.4 | mlp_dividemix_lite | 5 | -4.93pp | +1.29pp | 0/5 |
| pubmed | pair0.4 | mlp_elr | 5 | +8.53pp | +1.59pp | 5/5 |
| pubmed | pair0.4 | mlp_fine | 5 | +15.96pp | +3.06pp | 5/5 |
| pubmed | pair0.4 | mlp_jocor | 5 | -5.44pp | +1.16pp | 0/5 |
| pubmed | pair0.4 | mlp_logitclip | 5 | +3.95pp | +2.31pp | 5/5 |
| pubmed | pair0.4 | mlp_sce | 5 | +7.41pp | +1.35pp | 5/5 |
| pubmed | pair0.4 | mlp_small_loss | 5 | -4.70pp | +1.30pp | 0/5 |
| pubmed | instance0.4 | frozengrace_gb_mlp_softdist_beta05_shuffled | 5 | +10.96pp | +3.11pp | 5/5 |
| pubmed | instance0.4 | mlp_anl | 5 | +3.28pp | +1.74pp | 5/5 |
| pubmed | instance0.4 | mlp_ce | 5 | +9.32pp | +2.99pp | 5/5 |
| pubmed | instance0.4 | mlp_coteaching | 5 | -2.88pp | +0.79pp | 0/5 |
| pubmed | instance0.4 | mlp_ct | 5 | -4.24pp | +0.67pp | 0/5 |
| pubmed | instance0.4 | mlp_dividemix_lite | 5 | -4.68pp | +0.68pp | 0/5 |
| pubmed | instance0.4 | mlp_elr | 5 | +9.21pp | +1.51pp | 5/5 |
| pubmed | instance0.4 | mlp_fine | 5 | +11.96pp | +2.68pp | 5/5 |
| pubmed | instance0.4 | mlp_jocor | 5 | -4.88pp | +0.77pp | 0/5 |
| pubmed | instance0.4 | mlp_logitclip | 5 | +4.93pp | +3.62pp | 4/5 |
| pubmed | instance0.4 | mlp_sce | 5 | +4.26pp | +2.77pp | 4/5 |
| pubmed | instance0.4 | mlp_small_loss | 5 | -4.46pp | +0.97pp | 0/5 |
| coauthorcs | uniform0.2 | frozengrace_gb_mlp_softdist_beta05_shuffled | 5 | +1.16pp | +0.46pp | 5/5 |
| coauthorcs | uniform0.2 | mlp_anl | 5 | +0.55pp | +0.35pp | 5/5 |
| coauthorcs | uniform0.2 | mlp_ce | 5 | +0.18pp | +0.39pp | 3/5 |
| coauthorcs | uniform0.2 | mlp_coteaching | 5 | -0.14pp | +0.18pp | 2/5 |
| coauthorcs | uniform0.2 | mlp_ct | 5 | +0.51pp | +0.37pp | 5/5 |
| coauthorcs | uniform0.2 | mlp_dividemix_lite | 5 | +0.43pp | +0.37pp | 4/5 |
| coauthorcs | uniform0.2 | mlp_elr | 5 | +0.36pp | +0.44pp | 4/5 |
| coauthorcs | uniform0.2 | mlp_fine | 5 | +0.29pp | +0.48pp | 3/5 |
| coauthorcs | uniform0.2 | mlp_jocor | 5 | -0.25pp | +0.17pp | 0/5 |
| coauthorcs | uniform0.2 | mlp_logitclip | 5 | -0.24pp | +0.27pp | 1/5 |
| coauthorcs | uniform0.2 | mlp_sce | 5 | +0.94pp | +0.42pp | 5/5 |
| coauthorcs | uniform0.2 | mlp_small_loss | 5 | +0.07pp | +0.33pp | 3/5 |
| coauthorcs | uniform0.4 | frozengrace_gb_mlp_softdist_beta05_shuffled | 5 | +3.02pp | +0.98pp | 5/5 |
| coauthorcs | uniform0.4 | mlp_anl | 5 | +1.71pp | +0.43pp | 5/5 |
| coauthorcs | uniform0.4 | mlp_ce | 5 | +2.19pp | +0.55pp | 5/5 |
| coauthorcs | uniform0.4 | mlp_coteaching | 5 | +1.81pp | +0.57pp | 5/5 |
| coauthorcs | uniform0.4 | mlp_ct | 5 | +2.24pp | +0.27pp | 5/5 |
| coauthorcs | uniform0.4 | mlp_dividemix_lite | 5 | +2.21pp | +0.57pp | 5/5 |
| coauthorcs | uniform0.4 | mlp_elr | 5 | +2.32pp | +0.48pp | 5/5 |
| coauthorcs | uniform0.4 | mlp_fine | 5 | +2.83pp | +0.73pp | 5/5 |
| coauthorcs | uniform0.4 | mlp_jocor | 5 | +1.23pp | +0.16pp | 5/5 |
| coauthorcs | uniform0.4 | mlp_logitclip | 5 | +1.68pp | +0.56pp | 5/5 |
| coauthorcs | uniform0.4 | mlp_sce | 5 | +2.25pp | +0.46pp | 5/5 |
| coauthorcs | uniform0.4 | mlp_small_loss | 5 | +1.55pp | +0.41pp | 5/5 |
| coauthorcs | uniform0.6 | frozengrace_gb_mlp_softdist_beta05_shuffled | 5 | +8.33pp | +2.25pp | 5/5 |
| coauthorcs | uniform0.6 | mlp_anl | 5 | +7.64pp | +1.71pp | 5/5 |
| coauthorcs | uniform0.6 | mlp_ce | 5 | +7.14pp | +2.10pp | 5/5 |
| coauthorcs | uniform0.6 | mlp_coteaching | 5 | +5.28pp | +1.05pp | 5/5 |
| coauthorcs | uniform0.6 | mlp_ct | 5 | +4.71pp | +0.99pp | 5/5 |
| coauthorcs | uniform0.6 | mlp_dividemix_lite | 5 | +7.14pp | +2.10pp | 5/5 |
| coauthorcs | uniform0.6 | mlp_elr | 5 | +6.62pp | +1.46pp | 5/5 |
| coauthorcs | uniform0.6 | mlp_fine | 5 | +7.23pp | +2.07pp | 5/5 |
| coauthorcs | uniform0.6 | mlp_jocor | 5 | +4.58pp | +0.96pp | 5/5 |
| coauthorcs | uniform0.6 | mlp_logitclip | 5 | +6.69pp | +1.54pp | 5/5 |
| coauthorcs | uniform0.6 | mlp_sce | 5 | +7.74pp | +1.88pp | 5/5 |
| coauthorcs | uniform0.6 | mlp_small_loss | 5 | +5.52pp | +0.90pp | 5/5 |
| coauthorcs | pair0.4 | frozengrace_gb_mlp_softdist_beta05_shuffled | 5 | +9.91pp | +1.27pp | 5/5 |
| coauthorcs | pair0.4 | mlp_anl | 5 | +6.41pp | +2.14pp | 5/5 |
| coauthorcs | pair0.4 | mlp_ce | 5 | +10.05pp | +1.21pp | 5/5 |
| coauthorcs | pair0.4 | mlp_coteaching | 5 | +6.90pp | +0.61pp | 5/5 |
| coauthorcs | pair0.4 | mlp_ct | 5 | +8.26pp | +1.11pp | 5/5 |
| coauthorcs | pair0.4 | mlp_dividemix_lite | 5 | +8.58pp | +1.97pp | 5/5 |
| coauthorcs | pair0.4 | mlp_elr | 5 | +7.93pp | +1.26pp | 5/5 |
| coauthorcs | pair0.4 | mlp_fine | 5 | +10.84pp | +1.63pp | 5/5 |
| coauthorcs | pair0.4 | mlp_jocor | 5 | +4.94pp | +1.05pp | 5/5 |
| coauthorcs | pair0.4 | mlp_logitclip | 5 | +4.93pp | +1.58pp | 5/5 |
| coauthorcs | pair0.4 | mlp_sce | 5 | +6.19pp | +2.22pp | 5/5 |
| coauthorcs | pair0.4 | mlp_small_loss | 5 | +5.83pp | +0.72pp | 5/5 |
| coauthorcs | instance0.4 | frozengrace_gb_mlp_softdist_beta05_shuffled | 5 | +3.61pp | +1.22pp | 5/5 |
| coauthorcs | instance0.4 | mlp_anl | 5 | +2.16pp | +0.48pp | 5/5 |
| coauthorcs | instance0.4 | mlp_ce | 5 | +3.06pp | +0.74pp | 5/5 |
| coauthorcs | instance0.4 | mlp_coteaching | 5 | +1.55pp | +0.55pp | 5/5 |
| coauthorcs | instance0.4 | mlp_ct | 5 | +2.16pp | +0.82pp | 5/5 |
| coauthorcs | instance0.4 | mlp_dividemix_lite | 5 | +3.07pp | +0.90pp | 5/5 |
| coauthorcs | instance0.4 | mlp_elr | 5 | +2.72pp | +0.59pp | 5/5 |
| coauthorcs | instance0.4 | mlp_fine | 5 | +3.14pp | +0.79pp | 5/5 |
| coauthorcs | instance0.4 | mlp_jocor | 5 | +1.32pp | +0.47pp | 5/5 |
| coauthorcs | instance0.4 | mlp_logitclip | 5 | +1.30pp | +0.33pp | 5/5 |
| coauthorcs | instance0.4 | mlp_sce | 5 | +2.50pp | +0.35pp | 5/5 |
| coauthorcs | instance0.4 | mlp_small_loss | 5 | +1.53pp | +0.81pp | 5/5 |

## Completeness
- observed method-setting groups: 260
