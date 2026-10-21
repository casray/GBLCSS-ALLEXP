# GSSC matched adaptation handoff

Completed 2026-10-09. User approved five datasets, uniform0.4, seed0 only. No five-seed authorization this turn.

Approved model: common two-FC MLP hidden64 ReLU-BN-dropout0.5; native separate auxiliary head retained. Edge sampler/mixup/objectives preserved from frozen gssc_adapter.py (SHA checked by adapter). Native presets and predeclared Cora fallback unchanged. Budget100 unlabeled edge-pretrain +100 classifier warmup +100 main; first clean-val max over200 supervised epochs. This is matched deployment/epoch budget, not identical training resources.

All five formal runs exit0 and report.py exit0. Four new CPU tests and thirteen historical core tests passed; old optional GPU test skipped, new real-Cora GPU smoke passed. Independent read-only reviewer found no blockers; residual test gaps recorded in CODE_REVIEW.md.

| Dataset | Test (%) | Selected supervised epoch |
|---|---:|---:|
| Cora |61.84696569920845|172|
| CiteSeer |54.51030927835051|3|
| CoraFull |39.96391194514616|152|
| Coauthor-CS |84.86713940621834|1|
| Amazon-Photo |82.35294117647058|65|

CiteSeer, Coauthor-CS and Amazon-Photo selected warmup checkpoints; all completed200 supervised epochs (not early stopping). No tuning or seed exclusion. This is single-seed screening, not main-table five-seed evidence.

Artifact layout: runs/<dataset>/uniform_0.4/seed0/{config.json, protocol.json, source_hashes.json, environment.json, console.log, epochs.jsonl, checkpoint.pt, probabilities.pt, adapted_training_core.py, result.json}. Separate smoke/cora. report.py audits all five identities and deployed checkpoint reload with max absolute probability difference0. Source hashes frozen; do not edit adapter.py or run.py in place after results, use a new version for changes.

Commands (no rerun needed):
`D:/codex_temp/noisygl-rocm721/Scripts/python.exe -B -X utf8 experiments/gssc_matched_seed0_20261009/report.py`
`D:/codex_temp/noisygl-rocm721/Scripts/python.exe -B -X utf8 experiments/gssc_matched_seed0_20261009/run.py --grid` verifies/skips completed results, refuses partial overwrite.

Original ports/results, prior SimMLP/TINED experiments and manuscript are unchanged. No active agent or training session remains.
