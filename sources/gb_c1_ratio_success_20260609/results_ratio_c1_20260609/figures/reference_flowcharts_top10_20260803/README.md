# Top 10 Local Flowchart References

The images in this folder are direct crops from local papers for layout study. They are not redrawings.

| No. | Paper and figure | Local source | Page | Most useful design lesson |
|---:|---|---|---:|---|
| 1 | GLNN, Fig. 2 | `SOFTdistmlp最相关/01_GLNN_ICLR2022_Graph-less_Neural_Networks.pdf` | 4 | Separates graph-assisted training from graph-free deployment with two large visual zones. This is the strongest reference for the right side of the GB-LCSS overview. |
| 2 | GRACE, Fig. 1 | `对比学习/GRACE.pdf` | 2 | Explains two graph augmentations with very few objects. Useful for showing label-free representation learning without overloading Stage 1. |
| 3 | Structure-aware granular-ball clustering (SAGBC), Fig. 1 | `导师相关/Structure-aware-granular-ball-clustering_2026_Information-Sciences.pdf` | 4 | Uses colored dashed macro-regions to organize a dense multi-stage granular-ball pipeline. Useful as an upper bound on how much detail an Elsevier overview can contain. |
| 4 | SGBGC, Fig. 2 | `深度学习+粒球/1.23Graph Coarsening via Supervised Granular-Ball for Scalable Graph Neural Network Training.pdf` | 3 | Shows coarse partition, refinement, granular-ball graph construction, and downstream learning as four readable stages. Best granular-ball process reference. |
| 5 | G-GBC, Fig. 3 | `粒球噪声学习/G-GBC--A-Gaussian-mixture-based-granular-ball-computing-me_2026_Knowledge-Ba.pdf` | 6 | Combines inputs/outputs, four numbered stages, and a restrained training loop in one full-width composition. Useful for stage hierarchy, not for copying its density. |
| 6 | MgCNL, Fig. 1 | `粒球噪声学习/MgCNL_A_Sample_Separation_Approach_via_Multi-Granularity_Balls_for_Fault_Diagnosis_With_the_Interference_of_Noisy_Labels.pdf` | 2 | Encodes noisy labels by both color and shape, then separates confident and uncertain granular balls. Closest visual reference to noisy-label granular-ball learning. |
| 7 | VQGraph, Fig. 2 | `SOFTdistmlp最相关/08_VQGraph_ICLR2024_Rethinking_Graph_Representation_Space.pdf` | 4 | Uses stacked lanes for tokenizer learning, shared codebook, and teacher-student transfer. Strong reference for visualizing different knowledge channels. |
| 8 | NOSMOG, Fig. 1 | `SOFTdistmlp最相关/03_NOSMOG_ICLR2023_Learning_MLPs_on_Graphs.pdf` | 4 | Pairs one large end-to-end framework with three compact mechanism callouts. Useful when one central path needs supporting detail without fragmenting the figure. |
| 9 | FF-G2M, Fig. 3 | `SOFTdistmlp最相关/05_FF-G2M_AAAI2023_Full_Frequency_GNN_to_MLP.pdf` | 5 | Contrasts ordinary node-to-node distillation with the proposed low/high-frequency mechanism in a single horizontal comparison. Useful for ablation or extension figures. |
| 10 | NRGNN, Fig. 1 | `粒球噪声学习/NRGNN.pdf` | 4 | Uses a compact legend to distinguish noisy, clean, pseudo-labeled, and unlabeled nodes while preserving a readable graph-repair flow. Best symbol reference for noisy-node states. |

## Suggested priority for the GB-LCSS overview

1. Use **GLNN** for the global training/deployment split.
2. Use **SGBGC** for the granular-ball construction sequence.
3. Use **MgCNL** and **NRGNN** for noisy-label symbols and correction semantics.
4. Use **VQGraph** or **NOSMOG** only if a zoomed mechanism inset is needed.
5. Treat **SAGBC** and **G-GBC** as density references rather than direct layout templates.
