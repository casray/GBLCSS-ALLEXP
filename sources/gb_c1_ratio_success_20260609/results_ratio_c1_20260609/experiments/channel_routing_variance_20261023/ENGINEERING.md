# Engineering record

2026-10-23. New directory only; original fitting implementation imported read-only. Runtime ROOT and cell lookup rebound: all new output/input registries point here, protocol lookup stays at source seed0 while fitting RNG uses student seed1.

Tests were introduced before implementations. Initial metric tests raised ModuleNotFoundError for metrics; after implementation four passed. Initial routing tests raised ModuleNotFoundError for routing_kernels; after implementation all nine passed. These are expected development red phases, not failed scientific fits.

The Win32_Process CIM inventory was denied by local permissions. Get-Process found no python process before launch; GPU free-memory check and exclusive fit.lock were used. No process was terminated.

Both three-epoch smoke paths reproduced old seed0 initialization/RNG/loss/validation exactly. Forty formal fits completed on GPU with the same input targets and encoder. No fit restart, scientific threshold change or runtime repair occurred.

analyze_stage0.py was created after fitting started and separately frozen in stage0_test_release.json. L1 sources were separately frozen before its selections. Later auditing/reporting added audit_and_report.py. fit_sources.json remains the original pre-fit source snapshot.

Stage0 computed both A200-independent A300 pairings directly from cached all-node probabilities and true labels only during posthoc scoring. The old R2_KD_oracle_union field was not reused.

Audits: 80 source/new fit metadata/hash/validation checks; 40 new checkpoint replays; 20 validation selection/fusion replays; input/source hash validation. Human-facing table weight display is rounded to two decimals; full floating-point selections remain in L1_selections.json.

After L1 gate failed, subsequent layers were recorded as not run. The region-feature helper is unit-tested but no graph/GB router was fitted or scored on scientific data. No router AUROC table is claimed.
