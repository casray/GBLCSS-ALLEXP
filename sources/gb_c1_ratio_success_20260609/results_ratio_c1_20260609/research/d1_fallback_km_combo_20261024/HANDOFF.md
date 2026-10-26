# Execution handoff

Approved2026-10-24, sole executor Carson01a16bac-69f1-72c2-ba47-c6253cc99849, gpt-6.1-sol/high. Parent owns this research folder. Worker owns experiments/d1_fallback_km_combo_20261024. Scope/limits inPLAN.md; do not restart preceding completed experiments or duplicate agents.

Status: COMPLETE_PARENT_VERIFIED. 684new200epochfits,4twoepochsmoke,11tests; worker audit passed, no Python training process. Parent audit independently passed2040selectors/pairedrows,3672summaryrows,30CPU checkpoint reloads/sklearnmetrics; maxprobabilityerror7.7486e-7 and identical test predictions. No extra experiments authorized. Do not restart completed queue.

Parent review priorities: reuse provenance; fallback/global selector matched; balanced criterion at original best-accuracy checkpoint; KM group ties derive from KM's global pool; fixed/adaptive controls use identical deep-tree cuts; frozen C1 support and CE shortcut; per-seed standard deviation and newseed1/2; cost includes upstream C1; correct negative clean/F1 reporting. Parent audit will independently recompute comparisons and sample checkpoint predictions.

Current source baseline: experiments/d1_c1_confirmation_20261024/, complete3600fits, audited. Python D:/codex_temp/noisygl-rocm721/Scripts/python.exe. Cdrive around14GiB at dispatch; preserve3GiBreserve. All old files read-only; no uploads or manuscript changes.

Read PARENT_REVIEW.md and worker REPORT_zh.md for conclusions: fallback mitigates clean cost; KM+D1 beatsGB+D1 by.713pp; same-treecombo gains.249ppvsvalidationselectedsinglemodule but still.362ppbelowKM+D1. These use primaryaccuracyselector; balanced is predetermined secondary analysis. Three protocol/student seeds, historicalSSL fixed; retrospectivedevelopmentreuse.
