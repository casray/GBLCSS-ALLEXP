# Fixed-teacher GB-guided distillation, seed 0

Approved by the user on 2026-10-06. The immutable design reference is
`../gb_guided_distillation_proposal_20261006/PLAN.md`.

- [x] Test target mixing, eligibility, fixed-mass control and shuffled memberships.
- [x] Freeze all ten settings and targets before fitting students.
- [x] Replay Cora uniform 0.4 standard distillation against the saved baseline (bit exact).
- [x] Fit D1-D4 for all five datasets and two noise rates on the existing GPU (40/40).
- [x] Audit cached inputs, probability outputs and results; report all screening gates.

Decision: the fixed candidate fails three of five gates; do not expand to five seeds.
See `实验结论.md`, `summary.json` and `output_audit.json`.

No teacher retraining, parameter search, changes to prior results or manuscript edits.
Only seed 0 is used. Standard deviations and statistical significance are not estimable.
