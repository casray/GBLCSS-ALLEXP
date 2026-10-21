# MLP-59 manuscript amendment audit

## Scope

- Frozen predecessor: `paper_final_zh_v59_pretranslation_revised_20260814.md`.
- New manuscript: `paper_final_zh_v60_mlp59_extension_20260814.md`.
- Experiment source: `external/NoisyGL/runs/mlp59_final_extensions_20260814/`.

## Authorized changes

1. Extend the graph-teacher comparison from three to five datasets at uniform 0.4 and 0.6.
2. Add formal leave-one-out boundary diagnostics for six historical breadth datasets.
3. Add the complete CiteSeerFull spectrum because it alone passed the frozen uniform-0.4 entry rule.
4. Keep the original five-dataset main matrix because CiteSeerFull failed the full-spectrum worst-cell condition.

## Evidence checks

- Teacher phase: 60/60 rows, all protocol cells passed.
- GNN-to-MLP phase: 60/60 rows, all protocol cells passed.
- Boundary entry: 60/60 rows, all protocol cells passed.
- CiteSeerFull formal targets: 50/50 rows, all protocol cells passed.
- CiteSeerFull baseline controls: 175/175 rows, all protocol cells passed.
- Validation/test modified-label counts are zero in every new formal row.
- All paired comparisons use matching split and noise hashes.
- Teacher gate passed. Only CiteSeerFull passed the boundary entry gate. Its full-spectrum gate failed because uniform 0.6 was 4.20 pp below JoCoR.

## Manuscript checks

- `audit_v60_manuscript.py`: 0 errors.
- Elsevier human-writing linter: no configured AI-writing patterns found.
- Table 4 generated values for Coauthor-CS and Amazon-Photo match MLP-59 aggregates.
- Appendix I values match the boundary and CiteSeerFull full-spectrum sources.
- Equation tags are unique and contiguous.
- Reference numbering is contiguous and the reference section is unchanged from v59.
- Markdown table column counts are consistent.
- No NRGNN teacher row, stale six-cell Table-4 statement, or Chinese semicolon remains.
- Frozen v59 SHA256 remains `0A785668C3D02311C2C4BD4F46BC18242353F8799C5779E601E599472F23FAD8`.

## Decision

V60 is the paper version authorized by MLP-59. It strengthens the teacher extension but does not change the base method, the main five-dataset comparison, or the claim ceiling.
