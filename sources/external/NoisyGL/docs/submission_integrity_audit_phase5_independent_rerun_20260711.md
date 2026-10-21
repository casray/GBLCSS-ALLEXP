# Submission Integrity Audit - Phase 5: Independent Seed Rerun

## Scope

The current code was used to rerun seed 0 under the formal protocol (`percent=0.2/0.1/0.7`, `uniform=0.4`, `noise_scope=train`) for three representative datasets and three core arms:

- LCSS / SoftDist-MLP beta=0.5;
- size-matched shuffled grouping;
- MLP+CE.

Cora and CiteSeer exercise the legacy checkpoint-backed H path; CoraFull exercises the v2 tensor-only H path, 70 classes, and the large-graph 64-ball cap.

## Result

| Dataset | Method | Archived test | Independent rerun | Delta | Split hash | Noise hash |
|---|---|---:|---:|---:|---:|---:|
| Cora | LCSS | 0.5546174142480211 | 0.5546174142480211 | 0 | match | match |
| Cora | shuffled | 0.4385224274406332 | 0.4385224274406332 | 0 | match | match |
| Cora | MLP+CE | 0.4174142480211081 | 0.4174142480211081 | 0 | match | match |
| CiteSeer | LCSS | 0.5532646048109966 | 0.5532646048109966 | 0 | match | match |
| CiteSeer | shuffled | 0.4222508591065292 | 0.4222508591065292 | 0 | match | match |
| CiteSeer | MLP+CE | 0.4609106529209621 | 0.4609106529209621 | 0 | match | match |
| CoraFull | LCSS | 0.3738722482858174 | 0.3738722482858174 | 0 | match | match |
| CoraFull | shuffled | 0.3220498015156983 | 0.3220498015156983 | 0 | match | match |
| CoraFull | MLP+CE | 0.3200288704438830 | 0.3200288704438830 | 0 | match | match |

## Verdict

- **PASS** for reproducibility of the current core LCSS, shuffled, and MLP+CE implementations on the audited seed.
- The equality includes the final test value and the split/noise fingerprints, not only rounded manuscript cells.
- This pass does not certify every baseline and does not override the Phase 3 method-fidelity failures. In particular, rows whose local implementation materially differs from the named paper still require faithful reruns or relabeling/removal.

## Artifacts

- `runs/submission_integrity_audit_20260710/independent_rerun_seed0.csv`
- `runs/submission_integrity_audit_20260710/independent_rerun_corafull_seed0.csv`
