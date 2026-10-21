# MLP-21 Dataset Replacement Report

Protocol: percent 0.2/0.1/0.7, train-only noise, seeds 0-4.

## Gate Decisions
| Dataset | Decision | Best baseline | Ours-shuffled | Positive | Ours-best | Positive | Audit |
|---|---|---|---:|---:|---:|---:|---:|
| amazoncom | boundary | mlp_jocor | +5.64pp | 5/5 | -1.28pp | 0/5 | 1 |
| corafull | pass | mlp_jocor | +5.06pp | 5/5 | +1.63pp | 5/5 | 1 |

## Gate Aggregate
| Dataset | Method | mean+-std |
|---|---|---:|
| amazoncom | frozengrace_gb_mlp_softdist_beta05 | 75.53+-0.87 |
| amazoncom | frozengrace_gb_mlp_softdist_beta05_shuffled | 69.89+-1.19 |
| amazoncom | mlp_anl | 75.69+-1.95 |
| amazoncom | mlp_ce | 72.11+-0.88 |
| amazoncom | mlp_coteaching | 76.05+-0.86 |
| amazoncom | mlp_ct | 74.64+-1.65 |
| amazoncom | mlp_dividemix_lite | 69.70+-1.35 |
| amazoncom | mlp_elr | 72.26+-1.52 |
| amazoncom | mlp_fine | 69.21+-1.22 |
| amazoncom | mlp_jocor | 76.81+-0.85 |
| amazoncom | mlp_logitclip | 74.14+-0.98 |
| amazoncom | mlp_sce | 75.35+-1.46 |
| amazoncom | mlp_small_loss | 76.21+-1.18 |
| corafull | frozengrace_gb_mlp_softdist_beta05 | 37.33+-0.81 |
| corafull | frozengrace_gb_mlp_softdist_beta05_shuffled | 32.27+-0.81 |
| corafull | mlp_anl | 28.12+-0.61 |
| corafull | mlp_ce | 31.98+-0.98 |
| corafull | mlp_coteaching | 34.39+-0.24 |
| corafull | mlp_ct | 34.11+-0.24 |
| corafull | mlp_dividemix_lite | 32.01+-0.80 |
| corafull | mlp_elr | 32.14+-0.67 |
| corafull | mlp_fine | 32.00+-0.88 |
| corafull | mlp_jocor | 35.70+-0.32 |
| corafull | mlp_logitclip | 34.69+-1.17 |
| corafull | mlp_sce | 29.29+-0.45 |
| corafull | mlp_small_loss | 33.04+-0.50 |

## CoraFull Noise Spectrum
| Noise | Best baseline | Ours | Shuffled | Ours-best | Ours-shuffled |
|---|---|---:|---:|---:|---:|
| uniform0.2 | mlp_jocor 43.33+-0.70 | 43.51+-0.41 | 40.84 | +0.17pp | +2.67pp |
| uniform0.4 | mlp_jocor 35.70+-0.32 | 37.33+-0.81 | 32.27 | +1.63pp | +5.06pp |
| uniform0.6 | mlp_logitclip 24.62+-0.72 | 27.32+-0.75 | 21.40 | +2.70pp | +5.93pp |
| pair0.4 | mlp_jocor 31.45+-0.80 | 34.41+-1.30 | 28.43 | +2.96pp | +5.97pp |
| instance0.4 | mlp_jocor 35.17+-0.51 | 37.08+-0.69 | 31.19 | +1.91pp | +5.89pp |

## CoraFull Graph-Free Table B
| Method | Train graph | Inference graph | Note | mean+-std |
|---|---|---|---|---:|
| ffg2m_adapted | yes | no | adapted | 58.54+-1.00 |
| graphmlp | yes | no | adapted | 37.05+-0.91 |
| gsdn | yes | no | adapted | 35.80+-0.66 |
| krd_adapted | yes | no | adapted | 47.12+-0.95 |
| mlp_distill_gcn | yes | no | GLNN-style | 58.74+-0.82 |
| mlp_distill_softdist_xh | yes | no | LCSS teacher | 58.22+-0.69 |
| vqgraph_adapted | yes | no | adapted | 58.52+-0.77 |

## Completeness
- Missing groups: 0
