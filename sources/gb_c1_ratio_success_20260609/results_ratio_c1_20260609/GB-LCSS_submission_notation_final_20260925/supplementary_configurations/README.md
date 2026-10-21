# Supplementary configuration snapshots

This archive accompanies the GB-LCSS manuscript. It provides archived configuration files, not an executable software distribution or a replacement for per-run experimental logs.

- `configs/main/gblcss.yaml`: base method, model, optimization, and split settings.
- `configs/main/protocol_adaptations.yaml`: baseline adaptation settings.
- `configs/representations/grace.yaml`: GRACE settings for the five main datasets.
- `configs/datasets/`: dataset loader identifiers.
- `runtime_records/`: environment and runtime-setting snapshots for the matched GPU checks, five-dataset extension, and protocol replay.

Configuration files contain defaults that can be overridden by a run command. In particular, `training.device: cpu` in the base snapshot is not a claim that the matched GPU runs executed on CPU. Where recorded, the runtime `device` field identifies that run's device. The base noise setting is uniform 0.4; the manuscript's full comparison grid additionally includes uniform 0.2, uniform 0.6, pair 0.4, and instance-dependent 0.4. Consult the manuscript and its tables for the evaluated protocol and parameter variants.

These snapshots do not establish that all historical experiments ran on one machine. They do not include datasets, pretrained representations, membership caches, or every run's command-line arguments. Machine-specific code and data root paths were omitted from the JSON runtime records; numerical settings were retained.
