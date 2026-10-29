# Latest Update: 2026-10-26

Recent experiment sources and results were pushed successfully in commit
[cdbf847827aad749eedcf11b7a427ba91d848681](https://github.com/casray/GBLCSS-ALLEXP/commit/cdbf847827aad749eedcf11b7a427ba91d848681).
`git ls-remote` independently returned that commit for GitHub `main` after the push.

- Added41,104 source files; updated2 existing source files,423,227,664 bytes of changed source payload before Git compression.
- Includes R2+SUPPORT+D1+C1 integration, D1/C1 confirmation and combination experiments, earlier unsynced experiments and research notes.
- Experiment index now covers66 branches. See [recent results](RECENT_RESULTS_20261026.md).
- Scoped inventory110,115 files; local-only weights, arrays, datasets and duplicate packages retain per-file hashes in [the update manifests](manifests/update_20261026/summary.json).
- All41,106 changed source copies match source SHA256 and exact staged Git blob bytes. Seven policy tests and credential scans passed; quarantine count0.
- Original experiment files remain unchanged. Initial2026-10-21 manifests remain intact as a historical snapshot.
- See [local verification](manifests/update_20261026/local_verification.json) and [remote verification](manifests/update_20261026/remote_verification.json).

# Initial Upload Completed

Date: 2026-10-21. Repository: https://github.com/casray/GBLCSS-ALLEXP

## Published archive

- Archive commit: [9c829fde97739c9060ca53f1bac55cbf67daf00b](https://github.com/casray/GBLCSS-ALLEXP/commit/9c829fde97739c9060ca53f1bac55cbf67daf00b)
- Archive tree: `002118f20482ecdffcfc6ea2d006490aaefffcd1`
- Project-root source files actually published: **32,158**, totaling **2,363,451,505 bytes** before Git compression.
- Additional manuscript originals from Downloads: **9**, totaling **20,839,342 bytes**.
- Archive commit contains **32,187 tracked files**, including navigation, manifests and archive tooling.
  This count refers to the archive commit, before the small follow-up upload-verification documents.
- Main experiment index covers **52** historical branch directories. Earlier experiments outside that directory
  are preserved under their original project-relative paths as well.

## Verification completed

The normal non-force Git push succeeded. `git ls-remote` reported the archive commit on `main`.
GitHub's Git-commit API independently returned the same tree SHA as the local archive commit.
Every published project source file matched its source SHA-256 and its staged Git blob's exact bytes.
The nine supplemental originals also matched their SHA-256 and Git blob bytes.
Archive policy tests passed (7 tests); two credential-signature scans found no files requiring quarantine.
This is content-integrity verification, not a claim that GitHub marked the commit as cryptographically signed.

## Deliberately not uploaded

As selected by the owner, model weights, array caches and raw datasets remain on the local machine.
In particular, **18,146 model/array files (39,788,671,837 bytes)** and **267 raw-dataset files** are listed
with their paths, sizes and SHA-256 in the manifests, not silently represented as uploaded contents.
Duplicate packages, generated/compiler caches and unclassified binary files are also explicitly accounted for.
The supplemental manuscript packages were separately checked and contained no model/array payloads.

See [the complete source inventory](manifests/files.csv),
[local-only file records](manifests/retained_locally.jsonl),
[local byte verification](manifests/local_verification.json), and
[remote archive verification](manifests/remote_verification.json).

Original source files and the original project's `orgrin` remote were not modified.
The independent local archive checkout is `C:/Users/Admin/Desktop/GBLCSS-ALLEXP`.


## 2026-10-29 incremental archive

Recent experiment records through 2026-10-29 were added to `main` in content commit [576b6c900439f104dacde474ad05e8780ca70442](https://github.com/casray/GBLCSS-ALLEXP/commit/576b6c900439f104dacde474ad05e8780ca70442).

- Added 281 files across 9 experiment and supporting-code directories (4,414,849 source bytes).
- Updated 3 project context documents (181,843 source bytes); 284 source files are listed in [the SHA-256 manifest](manifests/update_20261029/files.csv).
- Added [the 2026-10-29 results summary](RECENT_RESULTS_20261029.md), updated the experiment index and README, and recorded the scoped archive summary.
- The included experiment files are text records, scripts, configurations, logs, result tables and reports. Model weights, array caches and raw datasets remain local.
- Source files were read back from GitHub and compared with local copies; 9 primary experiment artifacts, 3 updated context documents, README, index and both update manifests matched exactly. GitHub API refs for `main` and `codex/research-update-20261029` both resolved to the content commit before this status-only verification commit.
- The original experiment files were not modified.
