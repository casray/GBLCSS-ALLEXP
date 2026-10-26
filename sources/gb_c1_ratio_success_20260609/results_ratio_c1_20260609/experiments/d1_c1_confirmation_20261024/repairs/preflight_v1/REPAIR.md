# Preflight-only source repair

Scientific fits and smoke fits at repair: 0.

The original preflight stopped at coauthorcs seed0 clean because the historical
r2_support input directory contains only noisy conditions. E1 clean caches exist.
The repaired adapter verifies E1 observed/validation labels against original
train/validation labels using the historical geometry indices. It generates no
new noise or labels. Existing historical noisy-condition bridges remain required.

The original source files, sources.json, test log, preflight log and failure are
archived in this directory. New regression tests first failed on the missing
verifier and missing legacy clean file. After they pass, the active source
manifest is re-frozen before any smoke or scientific training.
