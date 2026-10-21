# GB Denoise Stage A

This experiment compares InfoNCE / GRACE and negative-free graph SSL representations by granular-ball quality.

Stage A does not train with granular-ball loss. It trains SSL encoders, extracts embeddings, generates label-free granular balls, and reports post-hoc diagnostics.

Labels are used only for linear evaluation and diagnostic purity. Labels are never used for SSL training or granular-ball generation.

## Current Scaffold Check

```powershell
cd gb_denoise_stage_a_negfree_vs_infonce
python -m pip install -e ".[dev]"
python -c "import gb_stage_a; print(gb_stage_a.__all__)"
python -m pytest -q
```

At this scaffold stage, `pytest` may report that no tests were collected.

## Planned Smoke Run

```powershell
cd gb_denoise_stage_a_negfree_vs_infonce
python -m gb_stage_a.cli run --config configs/stage_a.yaml --dataset Cora --method raw_features --seed 0 --epochs 0
python -m gb_stage_a.cli summarize --results results
```
