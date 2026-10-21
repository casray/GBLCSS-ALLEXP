# Excluded pre-freeze run

`cora_gssc_before_freeze` preserves the first Cora GSSC run, including its original metrics, epoch log, checkpoint and probabilities. Its adapter import occurred at approximately 12:26:26 on 2026-10-09; the worker's last adapter edit was at 12:26:54. The imported version was not separately hashed, so it is not merged into the final frozen-source screening table.

The final version includes the verified Pyro-equivalent probability clamp and finalized diagnostics. The original record is excluded for source-version consistency, not because of its accuracy. Only Cora needs a frozen-source replacement; later GSSC launches occurred after the source freeze. No original records were deleted. Replacement seed, data, split, noise and configuration remain unchanged.
