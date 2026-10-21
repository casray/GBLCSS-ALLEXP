# DIST independent protocol seeds: handoff

Status: COMPLETE. execution.json confirms 20 teachers and 40 students.
See the detailed Chinese report and summary.json. No further training launched.
Mean increment .120901pp (12 wins/1 tie/7 losses); seed1 .210956pp,
seed2 .030846pp; 7/10 positive mean cells. Predefined >=.20pp gate FAILED.
Decision: pause this fixed regional-on-DIST branch, no added modules.
DIST alone already matches/exceeds this ERASE teacher in 17/20 runs;
its macro accuracy is 77.940659, versus teacher 76.910443 and DIST+GB 78.061560.

Goal: fixed DIST T=1 versus DIST + regional KL (coefficient 1), seeds 1/2,
five datasets x uniform .4/.6. Twenty new ERASE teachers and forty students.
Split/noise/teacher/student seeds change; existing corresponding-seed ball
assignments are used; label-free GRACE remains fixed per dataset.
No search, no paper edits, no new figures, no automatic follow-up experiment.

Three protocol tests pass, including all twenty historical split/noise and
partition hashes. Two seed-0 Cora engineering replays exactly match previous
DIST T=1 checkpoints and 200 validation epochs.

See PLAN.md (frozen before training), frozen_inputs.json, environment.json,
COMMANDS.md, and run_protocol.py. Do not modify frozen files or rerun prepare.
To resume after confirming the original process has exited, run the grid
command in COMMANDS.md. Completed teacher/student fits are reused. If a cell
failed, inspect its failure.json before retrying; never silently drop it.

Completed analysis is reproducible with analyze.py. It reads
test labels only at this final stage, audits checkpoint selection, reloads all
students at selected/fixed epochs and all teachers, and writes paired_results,
cell_summary, summary and reload_audit. The detailed Chinese report is complete.
All 80 student reloads are bit-exact. All 20 teacher reloads preserve ALL node
predictions, with maximum probability error 2.9802322387695312e-7. The initial
bit-exact teacher assertion failed; see preserved analysis.log and
verification_note.md. Teacher tolerance is <=1e-6 plus identical argmax and
validation/epoch checks. No checkpoints or saved predictions were replaced.

58.62% of net gain comes from CiteSeer .6 seed1; other 19 pairs average
.052658pp. CoraFull .4 declines for both seeds; .6 gains .238181/.079394pp.
Fixed-epoch200 average .134205pp also misses the practical threshold.
Do not convert these small two-seed effects into stable/significant improvement.
Next window: read the report before proposing mechanisms. The missing evidence
is a specific independently testable failure mechanism, not another module.

Frozen gate: >=.20pp macro paired gain, positive mean for each seed, >=7/10
positive two-seed cells. Report failure honestly, sample std with n=2, no claim
of statistical significance. Seed0 is development-only; fixed-T1 development
mean gain was .0157393253pp (not the .009201pp validation-temperature-selected
summary from the preceding screen).
