# GRACE + Granular-Ball Contrastive Learning

This repository contains project guidance and research notes for exploring Granular Ball Computing with node-level Graph Contrastive Learning, centered on GRACE-style methods.

## Main Documents

- `AGENTS.md`: project rules that Codex should read before proposing methods, editing code, running experiments, or summarizing results.
- `01_literature_and_research_map_ai_optimized.md`: AI-facing experiment decision layer. Read this first when designing the next experiment.
- `01_literature_and_research_map.md`: detailed literature map and evidence base. Use this to trace paper-level support, uncertainty, and detailed notes.

## Current Research Direction

The preferred direction is to use granular balls as auxiliary soft structure signals, not as hard labels:

- weak-view granular-ball relation teacher
- confidence-weighted soft relation
- boundary-distance ball-to-ball topology prior
- structure-aware augmentation control
- denoising or uncertainty weighting

Avoid using label purity, class-wise boundaries, supervised graph coarsening, or hard same-ball membership directly in unsupervised GRACE pretraining.

## Git Tracking Policy

Local literature PDFs under `文献文件/` are ignored by default because they are large and may have redistribution restrictions. Keep reusable research conclusions in Markdown files instead.
