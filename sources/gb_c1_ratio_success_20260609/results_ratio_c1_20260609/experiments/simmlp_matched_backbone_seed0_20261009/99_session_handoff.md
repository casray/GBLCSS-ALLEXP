# Session handoff: 2026-10-09

User asked for one seed of SimMLP adapted to the main experiment protocol.
Completed five datasets, uniform 0.4, seed 0, GPU. All processes finished.
No original scores or manuscript edits; no extra seeds, tuning or reruns.

Read PLAN.md and REPORT.md first; results.csv and artifact_audit.json are the
verified machine-readable result and audit sources. Four unit tests, Cora 2+2
GPU smoke and all five checkpoint reloads passed.

New deployed network exactly matches common MLP d→64→C, ReLU then BatchNorm then
dropout .5. Encoder is its first hidden block; classifier is its second Linear.
Author self-supervised alignment/reconstruction and frozen linear probe remain.
200 final-epoch pretraining plus 200 probe epochs; Adam .01/wd.0005 in both;
no scheduler; first clean-validation maximum. Encoder and head initialize together.
Pretraining receives x/edges only. Frozen encoder and BN state hashes are audited.

Test percentages: Cora75.6201, CiteSeer67.1392, CoraFull40.1660,
Coauthor-CS88.1711, Photo88.3287. Exact main split/noise fingerprints match.
All lower than previous three-FC AdamW version, but bundled changes prohibit
attribution to depth alone. Single seed: no std or stable superiority evidence.

Author modules/vendor are reused from ../simmlp_tined_audit_seed0_20261009;
main data protocol from ../new_graph_mlp_seed0_20261009/common.py.
All effective sources and presets hashed per run. CoraFull's Cora preset fallback
is explicit and unchanged. No TINED work was done this turn.

Runtime D:/codex_temp/noisygl-rocm721/Scripts/python.exe, workdir this folder:

```powershell
python -X utf8 -m unittest test_matched -v
python -X utf8 -u run_matched.py --dataset cora --smoke
python -X utf8 -u run_matched.py --grid
python -X utf8 -u report_results.py
```

Completed training outputs skip rerun. report_results.py only audits saved
artifacts. Formal paper insertion needs the remaining seeds after user authorization.
