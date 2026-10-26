# Recent experiments and next question

Updated 2026-10-26. This page summarizes completed local evidence; the proposed teacher experiment below has not run.

## Latest completed integration

Five graphs, clean / uniform0.4 / uniform0.6 / pair0.4, three protocol/student seeds0/1/2. Frozen historical GRACE and per-seed R2 encoder/BN; attribute-only deployed student. Same clean-validation selection and paired data/noise. Sources and all detailed results retain negative findings.

| R2 supervision | Mean noisy accuracy (%) |
|---|---:|
| CE | 67.630 |
| GCE | 71.151 |
| OLD64 LCSS | 69.940 |
| Legacy SUPPORT32 | 70.615 |
| Fixed support + D1 | 71.042 |
| C1 support/strength selection | 70.952 |
| Joint SUPPORT + D1 + C1 | 71.531 |

Full combination minus the validation-selected stronger single component: +0.307 +/-0.165pp accuracy and +0.474 +/-0.242pp macro-F1 (paired seed-level sample SD). All three seed macro differences positive. Clean mean accuracy versus R2 CE +0.135pp; CoraFull clean macro-F1 -0.286pp remains reported. Joint candidate search costs2450 updates including CE versus1200 for global C1 and1600 for fixed D1. Search-budget effects and fixed historical SSL remain relevant to interpretation.

- [Full report](sources/gb_c1_ratio_success_20260609/results_ratio_c1_20260609/experiments/r2_integrated_20261026/REPORT_zh.md)
- [Parent review](sources/gb_c1_ratio_success_20260609/results_ratio_c1_20260609/research/r2_integrated_20261026/PARENT_REVIEW.md)
- [Independent audit](sources/gb_c1_ratio_success_20260609/results_ratio_c1_20260609/research/r2_integrated_20261026/PARENT_AUDIT.json)

1500 new candidate trajectories, including407 full200-epoch fits; others stopped at25/75. Four smoke fits and123475 actual scientific updates. Independent audit passed120 races,5280 prefix scores,300 selectors,660 paired rows,1134 summary rows and30 CPU checkpoint reloads. KM/kNN were omitted in this integration at the owner's request; older matched alternatives and their stronger results are preserved elsewhere.

## Strong graph reference: exploratory common-cell comparison

This table joins the latest direct attribute student with the existing adapted ERASE teacher and its ordinary DIST student, using seeds1/2 and uniform0.4/0.6 only (four cells per graph). All use the historical20/10/70 training-noise-only protocol. Different original method/search budgets are retained; this is an archival comparison, not a newly trained budget-matched study or a state-of-the-art ranking.

| Graph | Latest direct R2 student (%) | Adapted ERASE teacher (%) | ERASE + DIST student (%) |
|---|---:|---:|---:|
| Cora | 74.62 | 79.01 | 79.29 |
| CiteSeer | 67.31 | 65.47 | 65.69 |
| CoraFull | 47.54 | 58.29 | 58.35 |
| Coauthor-CS | 91.54 | 91.00 | 94.07 |
| Amazon-Photo | 83.83 | 90.77 | 92.31 |

The current direct method is competitive with this graph teacher on CiteSeer and Coauthor-CS. Cora, CoraFull and Photo have substantial gaps. DIST already transfers a strong graph teacher effectively; the next experiment should isolate whether the updated regional supervision improves the teacher and the resulting student, beyond strong ordinary knowledge distillation.

Sources: latest selected_results.csv; historical experiments/gb_dist_protocol_seeds_20261007/paired_results.csv. Scores are reported across every graph, with seeds/rates aligned explicitly. The deployed student uses raw attributes; cached graph representations/predictions are a distinct deployment option and should be described separately when relevant.

## Proposed teacher experiment (NOT_RUN)

Question: does SUPPORT + D1 + teacher-specific C1 improve a fixed GCN teacher, and does the improvement survive the same DIST transfer to an attribute MLP?

1. Keep teacher architecture, preprocessing, splits/noise, initialization and training/selection contracts paired. Teacher arms: CE; OLD64-LCSS; SUPPORT32-LCSS; SUPPORT32+D1 with global/CE fallback; joint SUPPORT+D1+C1. For joint selection, measure validation trajectories of that GCN teacher; an R2-selected support is not substituted for teacher selection. Retain the new tie-lambda1 convention and disclose its difference from historical D1.
2. Transfer every selected teacher through the identical verified DIST implementation to a random-initialized two-layer64-hidden attribute student, with matched initialization and200 student epochs. Keep the direct integrated R2 student and the existing ERASE-to-DIST route as deployment references. Reuse historical fits only after exact protocol/artifact checks.
3. First pilot: Cora/CiteSeer/CoraFull, seed0, clean/uniform0.4/uniform0.6. If the pilot has a useful paired teacher/student signal with acceptable clean/macro-F1 trade-offs, freeze the specification before the five-graph three-seed continuation. This page authorizes no training by itself.
4. Primary comparison: joint teacher to same student versus SUPPORT32 teacher to same student, plus CE and OLD64 anchors. Report teacher gain, student gain, student-minus-teacher gap, macro-F1, all clean results and end-to-end training/inference costs. Attribute inference remains fixed across students; whole training budgets differ and are recorded.
5. Separate two possible contributions in interpretation: stronger teacher supervision and better knowledge transfer. A teacher gain that is preserved by unchanged DIST supports the former. Any additional transfer-specific claim needs its own intervention and matched evidence.

R2 initialization followed by ordinary knowledge distillation has already been studied with weak overall results; it is not automatically repeated. This proposed experiment upgrades the teacher targets and their selection, leaving the verified transfer engine unchanged.

## Archive update

The initial2026-10-21 archive and manifests remain as a historical snapshot. The incremental scoped inventory, changed-file SHA256 list and local-only artifact records for this update are under manifests/update_20261026/. Model weights, arrays and raw data stay local under the owner's existing policy.
