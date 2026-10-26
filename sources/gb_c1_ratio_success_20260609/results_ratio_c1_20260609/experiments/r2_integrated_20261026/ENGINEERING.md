# Implementation record

Authorized 2026-10-26. Isolated adapter; historical files read-only.
Test-first run failed on missing rules module before implementation.
Training core fit.py copied byte-for-byte from completed confirmation.
Protocol copied from completed followup; removed KM loading/computation entirely.
New D1 tie lambda is1 for every support including fixed32.
Global prefix replay is explicitly retrospective: archived epoch logs are
truncated at25/75/200 before ranking. New group arms train and restore their
own optimizer/RNG. Prefix resume snapshots are retained.
OLD64 will be refitted with the common core because its older artifact schema
does not prove the complete input/RNG hash contract. Existing old probabilities
are used only for a post-selection numerical bridge.
