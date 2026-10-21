# SAMTD Repository Audit

## Source

- URL: `https://anonymous.4open.science/r/SAMTD`
- Retrieved: 2026-07-23
- Remote inventory: 130 files, 68,553,720 bytes
- Local inspection snapshot: `external/SAMTD_anonymous_snapshot_20260723`

The source is a paper-linked anonymous snapshot. No release tag, commit identifier, license file, dependency lock file, or substantive README was present.

## Static Checks

| Check | Result |
|---|---|
| Python source compiles | Pass |
| README contains run instructions | Fail, 7-byte title only |
| LICENSE/COPYING/NOTICE | Missing |
| requirements/environment file | Missing |
| BO-NNC implementation | Not found |
| Cora/CiteSeer/Photo teacher weights | Present |
| Main entry signature matches call | Fail |
| Required iteration and pseudo-label settings recoverable | Fail |
| Original split matches 20/10/70 | Fail |

AST evidence from `main.py`:

```text
DEF  line 30: arg_parse(dataset_name, noise_type, ites, pseudon)
CALL line 150: one positional argument + keyword noise_type
```

No other occurrence assigns `ites`, `pseudon`, `iteration`, or `pseudo_num`.

## Runtime Checks

ROCm environment:

```text
torch 2.9.1+rocm7.2.1
cuda_available True
torch_geometric 2.7.0
```

Missing dependencies:

```text
termcolor
torch_scatter
torch_sparse
```

Original entry smoke:

```text
ModuleNotFoundError: No module named 'termcolor'
```

After injecting a temporary `termcolor` stub:

```text
ModuleNotFoundError: No module named 'torch_scatter'
```

These dependency failures are technically repairable. They are not the decisive blocker. The decisive blockers are the missing algorithm hyperparameters and incompatible unlabeled-pool assumption.

## Protocol Compatibility

SAMTD's code selects pseudo-label candidates only from nodes outside train, validation, and test masks. Its original custom splits intentionally leave such nodes unmasked.

Under the target percent split:

| Dataset | Train | Validation | Test | Extra unmasked |
|---|---:|---:|---:|---:|
| Cora | 541 | 270 | 1895 | 2 |
| CiteSeer | 665 | 332 | 2328 | 2 |
| Amazon-Photo | 1530 | 765 | 5355 | 0 |

Therefore, the label-enhancement module would be inactive or nearly inactive. Reclassifying test nodes as pseudo-label candidates would be a new transductive adaptation not specified by the released code.

## Decision

`not_runnable_under_same_protocol`

No performance smoke was launched. A hand-built reduced SAMTD would not be a faithful baseline and would create a larger review risk than reporting the reproduction boundary.
