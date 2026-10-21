# TINED common-GCN adaptation, 2026-10-09

## Request and hypothesis

Five main datasets, uniform noise 0.4, seed 0 only, GPU. Test a Table-8-style
common GCN teacher while preserving TINED's native transformation-mimicking student.
The previous CoraFull validation-only control showed that imposed student
dropout .5 was harmful; restore predeclared author student dropout presets.
This is a bundled adaptation, not a single-factor causal experiment.

## Method, motivation, granular-ball role

TINED baseline uses no granular balls. Keep teacher weight injection, learned
GP-imitation FCs, response KL and Dirichlet-energy ratio matching. A two-layer
teacher implies a four-FC student; reducing it to two FCs would remove its core.
The deployed student takes raw attributes only. This is TINED, not TINED+.

## Fixed configuration and changes

- Main raw attributes, historical 20/10/70 split and train-only uniform .4 noise.
  Exact historical split/noise SHA-256 verification mandatory.
- Exact common PyG GCN, raw X and coalesced native sparse adjacency as Table 8;
  hidden64, two layers, teacher dropout .5, Adam .01/wd.0005, 200 epochs.
- Match common CE route initialization order (common MLP then common GCN).
  Teacher selects the last validation maximum, as existing Table 8 CE routes.
- Teacher FT/GP traces use GCN symmetric normalization, not old SAGE-gcn means;
  copy PyG lin.weight and deferred bias into the official sequence student.
- Student four FCs, hidden64; keep method optimizer groups and learning-rate
  ratios, CE/KL/DE stages, identity GP initialization and original energy formula.
  Adam .01/wd.0005, 200 student epochs; first clean-validation maximum.
- Author best-record JSON contains only SAGE-transductive/inductive presets,
  not complete GCN-specific best presets. Reuse the previously selected SAGE
  algorithm knobs, explicitly label this as common-GCN adaptation.
- Author student dropout: Cora0, CiteSeer0, Photo.5. CoraFull and Coauthor-CS
  retain the previous Cora fallback, thus dropout0. No test-driven tuning.
- CoraFull70 classes exceed hidden64: correct order is FT-GP-GP-FT.
  Other four graphs use FT-GP-FT-GP.
- Sparse energy and sampled-edge adapter reused from previous verified port.
  Full official DGL training parity remains unverified, no silent CPU fallback.
- New folder only; old experiments and manuscript untouched.

## Success/failure criteria and expected mechanism

Success: common teacher forward and transformation trace agree; injected
weights/biases and official student order correct; energy value/gradient agree
with direct edge differences; finite runs; preserved main fingerprints;
200+200 logs; teacher/student reload and validation-selected metrics agree.
Failure: wrong normalized propagation, leakage, changed split/noise, nonfinite
objectives, failed checkpoint selection/reload. Stop and record rather than tune.
Expected mechanism: faithful common graph teacher plus method-specific student
regularization restores useful transfer, but numerical improvements are not
guaranteed and do not prove stable superiority from one seed.

## Outputs and commands

Runtime D:/codex_temp/noisygl-rocm721/Scripts/python.exe; workdir this folder:

```powershell
python -X utf8 -m unittest test_gcn_adapter -v
python -X utf8 -u run_gcn.py --dataset cora --smoke
python -X utf8 -u run_gcn.py --grid
python -X utf8 -u report_results.py
```

Config, protocol, environment, source hashes, raw logs, selected teacher/student
checkpoint and probabilities are saved per dataset. REPORT.md and results.csv
record the conclusion; artifact_audit.json records reload checks.
No mean/std inferred; final decision recorded after completion.
