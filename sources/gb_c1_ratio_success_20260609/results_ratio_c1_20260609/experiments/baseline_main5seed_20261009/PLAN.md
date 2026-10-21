# Main-table baseline audit and five-seed extension, 2026-10-09

## Request, hypothesis and scope

Run only after protocol/code checks. SimMLP: five main datasets, uniform .4,
seeds 0..4. TINED: same datasets/seeds, uniform .4 and .6. GPU required.
No test-based tuning, new noise types, or GRACE/GB-LCSS reruns. No manuscript edits.
These baselines use no granular balls. Hypothesis: the audited adaptations can
produce complete, reproducible rows under the existing split/noise/evaluation rules.
Accuracy need not improve; failure/negative seeds must not be excluded.

## Audit decisions and fixed settings

- Historical random-choice train/val/test index order and train-only noise retained.
  split=noise=downstream seed. Uniform .4 checked against historical hashes.
  Both rates checked against unchanged original NoisyGL noise functions.
  Raw feature and edge tensors/file bytes hashed without float-to-int conversion.
- SimMLP common two-FC d-64-C deployment; audited author stop-grad/alignment,
  reconstruction and auxiliary graph modules retained. Coauthor-CS author preset
  has reconstruction coefficient zero. Adam .01/wd.0005, fixed200 SSL +200 head.
  Reuse label-free seed0 encoder across downstream seeds, as fixed-representation
  main comparisons do. Each seed initializes the common MLP in its original order,
  then keeps its new head and loads the fixed encoder; BN/dropout remain frozen.
  Existing validated seed0 scores reused, not retrained. First clean-val maximum.
- TINED exact common two-layer GCN teacher d-64-C, dropout.5, Adam .01/wd.0005,
  200 epochs, last clean-val maximum. Keep native four-FC student, injected weights,
  CE/KL/DE stages, author dropout and predeclared algorithm presets, no scheduler,
  200 student epochs, first clean-val maximum. All three RNG resets now use seed.
  IMPORTANT: sort CE minibatch node indices only AFTER original-order noise
  injection, matching official distill_run_transductive. Prior adapter did not sort;
  therefore prior TINED student seed0 cannot be reused. No loss/preset tuning.
- TINED common-GCN adaptation is not the official SAGE experiment; its four-FC
  student and dropout are method-specific exceptions to common MLP deployment.
  Complete official DGL runtime parity is not claimed; sparse energy values/
  gradients and GCN operation traces are tested. Cora/Photo late degradation under
  the fixed optimizer was observed earlier and is NOT a reason to tune on test.
- GSSC withheld: current deployment has three FC, 100 warmup +200 supervised
  epochs, last-tie selection and mislabeled inference parameter count. These
  are protocol mismatches, not grounds to call the author's method incorrect.
  A new common-backbone/budget GSSC adaptation needs a separate explicit design.
- DGD has no verified runnable adaptation in this batch and is not run.

## Expected mechanism, success and failure criteria

SimMLP transfers unlabeled graph information into a frozen attribute encoder.
TINED mimics the trained GCN's transformation/propagation using graph-free FCs.
Success: complete required five seeds per cell; finite losses/probabilities;
correct hashes and seed mappings; clean-val-only selection; saved checkpoint
reload probabilities and first/last maximum agree; sample std ddof1 reported.
Failure: protocol/core error, leakage, numerical nonfinite or reload mismatch.
Stop on failure; preserve artifacts; never silently replace or drop a seed.

## Minimal changes and commands

New folder only: protocol.py, sim_probe.py, tined_formal.py (copy with seed/batch
ordering fixes), run_grid.py, tests and report script. Historical files unchanged.

Runtime: D:/codex_temp/noisygl-rocm721/Scripts/python.exe. In this folder:

```powershell
python -X utf8 -m unittest test_formal -v
python -X utf8 -u run_grid.py --method simmlp --dataset cora --seed 1 --smoke
python -X utf8 -u run_grid.py --method tined --dataset cora --rate .6 --seed 1 --smoke
python -X utf8 -u run_grid.py --grid
python -X utf8 -u report_results.py
```

No full screening repetition: 20 new SimMLP heads +50 TINED teacher/student pairs.
Config, raw log, assets/source hashes, checkpoints/probabilities preserved.
Final decision and detailed results to be recorded in REPORT.md after verification.
