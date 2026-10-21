# GBLCSS historical archive upload plan

User destination: https://github.com/casray/GBLCSS-ALLEXP (public, empty, push authorized).
Original project remote orgrin must not be changed; source files are read-only.

1. Inventory all related code, experiments, configs, logs, results and research/history records.
2. Exclude credentials/secret stores, git internals, dependency caches, environments and unrelated domains.
   Review text for embedded credentials before publishing. Do not print secret values.
3. Report size/type counts, then resolve storage of large raw data/checkpoints/duplicate packages explicitly.
   No silent claim that a manifest equals actual uploaded experimental artifacts.
4. Prepare independent archive root on Desktop, preserve source-relative paths and source provenance.
   Include README, experiment directory index, inventory/exclusion status and file hashes.
5. Upload only reviewed files using authorized GitHub access. Preserve any pre-existing remote changes.
6. Verify published commit/tree and file counts; clearly report what was actually uploaded and what remains local.

Known access issue: git ls-remote fails TLS with both schannel/OpenSSL inside and outside sandbox.
GitHub connector get_repo succeeds; use connector as fallback, never extract credential-manager tokens.
