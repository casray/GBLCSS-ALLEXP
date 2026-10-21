# SimMLP matched-backbone experiment, 2026-10-09

## Question and hypothesis

User-authorized five main datasets, uniform noise 0.4, seed 0 only.
Test whether SimMLP remains effective after matching the main two-FC deployed
MLP and Adam optimization protocol. This is not an ablation isolating depth:
backbone, optimizer, schedule, head initialization and tie handling change together.

## Method and granular-ball role

SimMLP is a baseline with no granular balls. Preserve the author's augmented
graph alignment, stop-gradient, nonparametric aggregation, reconstruction and
frozen-encoder linear probe. Do not replace the self-supervised objective with CE.
Auxiliary predictor, aggregator and decoder are training-only.

## Fixed changes and configuration

- Exact common MLP deployed structure: Linear(d,64), ReLU, BatchNorm,
  dropout 0.5, Linear(64,C); no residual, PReLU or final L2 normalization
  in the deployed encoder. PReLU inside auxiliary author modules is retained.
- Both optimizer stages: Adam, lr 0.01, weight decay 0.0005; no scheduler.
- 200 final-epoch label-free pretraining steps; 200 supervised head steps.
  Encoder parameters, BatchNorm statistics and dropout are frozen during probe.
- Initialize encoder and classification head together in common MLP order.
- First best clean-validation checkpoint; no test-based selection.
- Main raw attributes, 20/10/70 splits and noisy training labels reused via
  common.py; exact historical split/noise SHA-256 checks mandatory.
- Other method settings retain the previous predeclared author dataset presets.
  CoraFull has no author preset and retains the predeclared Cora fallback.
- New folder only; old scores, code, manuscript and original configurations untouched.

## Expected mechanism and boundaries

Self-supervised graph information is embedded into the attribute encoder before
noisy supervised classification. Same deployed architecture does not mean same
training flow as GB-LCSS; preserve the method-specific frozen encoder.
No tuning is authorized and a low score must not trigger test-driven adjustments.

## Success / failure criteria

Success: exact common-backbone train/eval parity and deployed parameter count;
finite training; all five completed runs; historical hashes match; saved
predictions reproduce validation/test metrics; selected epoch is the first
validation maximum; checkpoint reload reproduces saved probabilities.
Failure: architecture mismatch, leakage, changed split/noise, nonfinite loss,
wrong selection or failed checkpoint reload. Stop and record rather than invent scores.

## Commands

Runtime: D:/codex_temp/noisygl-rocm721/Scripts/python.exe; workdir: this folder.

```powershell
python -X utf8 -m unittest test_matched -v
python -X utf8 -u run_matched.py --dataset cora --smoke
python -X utf8 -u run_matched.py --grid
python -X utf8 -u report_results.py
```

## Results and final decision

Recorded after completion in REPORT.md, results.csv and artifact_audit.json.
Single-seed screening only: no mean/std and no stable superiority claim.
