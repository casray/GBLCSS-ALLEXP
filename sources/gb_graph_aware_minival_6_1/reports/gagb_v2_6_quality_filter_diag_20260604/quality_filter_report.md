# GAGB v2.6 Q(B) Quality Filtering Diagnostic

Rows: 140
Summary rows: 28

This script does not train a classifier. It rebuilds groupings, trains the noisy-label-only teacher when needed, and uses build_method_targets for all target construction.

Clean labels are used only after target construction for diagnostics such as NCR, CDR, selected-ball clean purity, and rejected-ball clean purity.

Primary methods: grace_gagb_qb_conflict_gate_alpha07, grace_gagb_qfilter_alpha07_q50, grace_gagb_qfilter_alpha07_q70, grace_gagb_qweight_alpha07, gagb_qfilter_alpha07_q50, khop_only_qfilter_alpha07_q50, grace_gagb_shuffled_qfilter_alpha07_q50