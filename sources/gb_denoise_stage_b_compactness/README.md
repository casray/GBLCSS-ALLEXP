# GB Denoise Stage B

This experiment tests whether a label-free granular-ball compactness loss can shape a graph SSL encoder and improve granular-ball quality.

The project follows the Stage B design in `../docs/superpowers/specs/2026-05-19-gb-compactness-stage-b-design.md`.

Labels are used only for linear evaluation and diagnostic purity. Labels are never used for SSL training, granular-ball generation, reliable-ball gating, or compactness loss.

## Quick Smoke Run

```powershell
cd gb_denoise_stage_b_compactness
py -m pip install -e .[dev]
py -m pytest tests/test_config.py -q
```

## Minimal Gate

Use `py` on this Windows workspace because `python` may resolve to the Microsoft Store launcher.

```powershell
py -m pytest -q
py -m gb_stage_b.cli run --config configs/stage_b.yaml --dataset Synthetic --method B0 --seed 0 --epochs 4 --warmup-epochs 2
py -m gb_stage_b.cli run --config configs/stage_b.yaml --dataset Synthetic --method B1 --seed 0 --epochs 4 --warmup-epochs 2
py -m gb_stage_b.cli summarize --results results
```

The first real Stage B gate should run paired B0/B1/B3/B4 experiments with the same dataset and seeds:

```powershell
py -m gb_stage_b.cli run --config configs/stage_b.yaml --dataset Cora --method B0 --seed 0
py -m gb_stage_b.cli run --config configs/stage_b.yaml --dataset Cora --method B1 --seed 0
py -m gb_stage_b.cli run --config configs/stage_b.yaml --dataset Cora --method B3 --seed 0
py -m gb_stage_b.cli run --config configs/stage_b.yaml --dataset Cora --method B4 --seed 0
py -m gb_stage_b.cli summarize --results results
```

Pass condition: B1 must improve ball diagnostics over B0 and beat B3/B4 without collapse diagnostics worsening. Treat any change smaller than run-to-run variation as inconclusive.

## Stability Ablation Entry Points

Use `device auto` for real runs so this workspace uses the available CUDA/ROCm backend instead of CPU:

```powershell
py -m gb_stage_b.cli run --config configs/stage_b.yaml --dataset Cora --method B0 --ssl-objective infonce --experiment-group aug_stage_a --variant negfree_plain --device auto --seed 0 --results-dir results_stability
py -m gb_stage_b.cli run --config configs/stage_b.yaml --dataset Cora --method B0 --ssl-objective negfree --experiment-group aug_stage_a --variant negfree_plain --device auto --seed 0 --results-dir results_stability
```

Augmentation ablation values:

```powershell
# aug_weak
--edge-drop-1 0.1 --edge-drop-2 0.2 --feat-mask-1 0.1 --feat-mask-2 0.2

# aug_stage_a
--edge-drop-1 0.2 --edge-drop-2 0.4 --feat-mask-1 0.3 --feat-mask-2 0.4

# aug_strong
--edge-drop-1 0.3 --edge-drop-2 0.5 --feat-mask-1 0.4 --feat-mask-2 0.5
```

Encoder / stabilizer variants:

```powershell
--variant negfree_plain
--variant negfree_bn
--variant negfree_var
--variant negfree_ema
```

Rebuild granular balls from a saved embedding without retraining:

```powershell
py -m gb_stage_b.cli rebuild-balls --run-dir <run_dir> --min-ball-size 4 --max-depth 8 --min-split-gain 0.001
py -m gb_stage_b.cli summarize --results results_stability
```
