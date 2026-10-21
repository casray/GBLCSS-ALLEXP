# GPU screening handoff: finished with DGD blocked

Date: 2026-10-09. Authorized scope: five main datasets, train-only uniform label noise 0.4, seed 0; GPU only. No manuscript changes, hyperparameter search, further seeds, SimMLP rerun or wholesale reproduction of the original paper.

## Final status

GSSC and TINED completed all five datasets: 10 successful GPU cells, independently audited. DGD has five pre-training blocked cells and no scores. All parent exec sessions (2039, 54012, 5277) have exited. All subagents were closed. There is no remaining training queue.

| Dataset | GSSC test percent | TINED test percent |
|---|---:|---:|
| Cora | 59.42 | 53.51 |
| CiteSeer | 53.09 | 48.24 |
| CoraFull | 35.58 | 19.19 |
| Coauthor-CS | 83.29 | 83.33 |
| Amazon-Photo | 78.71 | 83.55 |

Primary artifacts: `REPORT.md`, `results.csv`, `screen_audit.json`, `runs/<dataset>/<method>/`. Each completed cell contains epoch/console logs, config, environment, protocol fingerprint, selected checkpoint, full probabilities and `artifact_audit.json`. Source/author provenance is preserved in notes, launch hashes and pinned unmodified `vendor/` repositories.

## Interpretation

- GSSC best validation models came from classifier pretraining on four graphs; Amazon-Photo selected main epoch 141. Do not describe only main-phase final checkpoints as the reported results.
- TINED teacher-to-student validation loss is particularly large on CoraFull (teacher 61.14%, student 18.70%). Cause is unverified. Keep the scores and investigate student settings/protocol adaptation before expanding to a formal multi-seed claim. No test-based tuning occurred.
- Single-seed results are screening evidence, not manuscript means or statistical significance. GSSC is graph-assisted MLP training; TINED is teacher distillation. Architectures/total budgets are not identical.

## Protocol and adaptations

`common.py` uses original raw attributes/edges, ordered historical 20/10/70 splits and seed-0 noise generation. All five split/noise SHA256 fingerprints match `../main_protocol_replay_20260922/matrix/<dataset>/default/seed0.json`. Only noisy training labels enter optimization; nontraining labels are -1, validation labels are separate, test labels enter saved-prediction evaluation only.

GSSC: hidden64/dropout0.5/common Adam; 100 edge-sampler pretrain + 100 classifier pretrain + 200 main epochs (300 classifier epochs). Preserves sampler, virtual-node mixing, dual heads, self-contrast and outer homophily updates, using sparse chunked Torch calculations.

TINED: hidden64/dropout0.5/common Adam; SAGE-gcn teacher200 + FT/GP four-FC student200; injection, response and energy distillation retained. Official sequence constructor/forward reused through AST; DGL teacher ported to sparse Torch full-neighbor aggregation. This differs from upstream neighbor-sampled training trajectories. Missing dataset presets use predeclared Cora mappings. See `GSSC_NOTES.md` and `TINED_NOTES.md`.

## Preserved integration failures

- The first Cora GSSC import overlapped the worker's last source edit. Record preserved under `excluded_runs/cora_gssc_before_freeze`; Cora alone was rerun with frozen source, same protocol/config. Replacement gives the same displayed score. Exclusion is source provenance, not best-of-restarts selection.
- TINED first smoke failed before training because its validator incorrectly required exhaustive split coverage. Original percentage sizes floor independently, leaving up to two unlabeled context nodes. Parent changed only the overlap check, not indices/masks/data/objectives. Regression reproduced the failure before repair and passed afterward; failed logs are preserved under `excluded_runs/cora_tined_split_validation_failure`.
- Frozen quick suite: 25 tests, 23 passed, two optional synthetic GPU tests skipped. Both real Cora GPU smokes passed. Final saved-artifact audit passed all 10 runs; no full original-paper testing was performed.

## DGD / SimMLP

DGD official repo `KangL-G/Dynamic-Gradient-Distillation`, commit `cdffc16861193ed7ea64054388b59f6ef6ef07d3`, calls undefined `dtw_dsasm`; generator feedback, fusion and alpha=18 objective also remain unresolved. Source/paper audit recorded in `DGD_NOTES.md`, `dgd_preflight.json` and each blocked cell. Public abstract/metadata cannot establish missing equations. Ask for the complete paper or author-fixed implementation; never replace core modules with guessed/plain KD to produce a score.

SimMLP read-only search found literature/discussion but no verified local implementation/config/log/result. The earlier `D:/papercode_final_v4.zip` was inspected (119 names, 91 text/source/config entries): no name/content matches. Other remote/unopened archives are not ruled out. See `SIMMLP_HISTORY.md`; no run was launched and no published scores were copied.

## Next authorized action

Report the screening to the user. Do not train more seeds, tune against test scores or edit paper tables without a new request. Re-running `report_screen.py` only audits existing files and regenerates the report, not training. Runtime: `D:/codex_temp/noisygl-rocm721/Scripts/python.exe`; experiment scripts are self-contained apart from existing project data and loader.
