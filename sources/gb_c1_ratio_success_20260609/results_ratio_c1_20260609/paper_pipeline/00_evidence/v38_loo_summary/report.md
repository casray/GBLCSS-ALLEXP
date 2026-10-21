# v38 Formal LOO Manuscript Summary

Formal method identity: `tie_mode=distribution_only`, `leave_one_out=true`.

- Best or tied among seven main baselines: 24/25.
- Versus `frozengrace_gb_mlp_softdist_beta05_loo_shuffled`: mean +8.69pp, positive 25/25, Holm-significant 25/25.
- Versus `mlp_ce`: mean +7.74pp, positive 25/25, Holm-significant 24/25.
- Versus `mlp_jocor`: mean +3.46pp, positive 24/25, Holm-significant 11/25.
- Versus `postel_mlp_noipl_rm`: mean +5.69pp, positive 24/25, Holm-significant 18/25.
- Versus `gsdn_faithful_core`: mean -0.21pp, positive 13/25, Holm-significant 4/25.
- Versus `als_mlp_rm`: mean +7.71pp, positive 25/25, Holm-significant 24/25.
- Versus `sals_mlp_adapted`: mean +4.69pp, positive 24/25, Holm-significant 13/25.

The clean-label control is reported separately and is not included in these noisy-label headline counts.
