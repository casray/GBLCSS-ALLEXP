# Approved D1 fallback, matched KM, and same-tree C1+D1 continuation

User approved execution on2026-10-24 using gpt-6.1-sol. Reuse the high-effort executor. Parent owns this research folder; worker owns experiments/d1_fallback_km_combo_20261024. No historical edits, manuscript edits, uploads, new seeds or new R2 fitting.

## Scientific scope

1. Zero-fit fallback analysis on all120 existing cells (five graphs, seeds0/1/2, R0/R2, clean/U.4/U.6/Pair.4). Select across CE, three global strengths(.25,.5,1), four D1 group candidates. Primary selector: validation accuracy, checkpoint NLL, earlier best epoch, stable order with global anchors first. Also freeze a secondary balanced selector: arithmetic mean of validation accuracy and macro-F1 at each candidate's already selected checkpoint, then original ranking. Apply identical selection objective to global-only and expanded candidates. Do not change checkpoint choices or infer condition/noise rate. Describe retrospective reuse as exploration. Diagnose per-class recall and clean/corrupted agreement/conflict groups using truth only after frozen choices.
2. Matched KM-D1 on60 R0 cells (five graphs, three seeds, four conditions). Historical legacy SUPPORT32 and same-count KM, same normalized H, cached partitions, initialization, optimizer and validation selection. For each partition compare identical global four-candidate set and expanded eight-candidate set. Reuse legacy fits and KM lambda1; new KM .25/.5 plus four group rules (max360 fits). Ties use that partition's best global lambda. Primary selector as above; fixed per-arm results also reported. Granular-ball and KM comparison must match candidate budgets.
3. Same-tree combination on same60 R0 cells. Fixed deep-tree S32 vs support chosen by frozen historical C1 PROBE; each compared under global-only and expanded group selection. Historical C1 grid supplies CE/global .25/.5/1. Fit four group rules for each distinct selected/fixed support (max480 fits). Deduplicate identical support within cell. If PROBE chose CE, keep CE in the adaptive branch per approved rule. Freeze original C1 selection, do not pick a new support using final group outcomes. Report original C1 selected configuration as an additional anchor. Label new global-only support-refinement control explicitly. D1 group ties use best global lambda at that same support. Baseline and combo must share deep-tree partitions; historical legacy32 is supplementary only.

## Execution constraints

At most840 new scientific200epoch fits, no other stages; exact count from preflight dedup/reuse inventory. Up to4 bounded2epoch engineering smoke fits if adapter changes require them. Existing environment D:/codex_temp/noisygl-rocm721/Scripts/python.exe. Serial GPU, other jobs untouched. Prefer minimal adaptation of completed d1_c1_confirmation_20261024; never import historical code that writes its old output root. Reuse only with input/target/init/training/source provenance checked. No repeated complete fits. Source/input/config and selection hashes, checkpoints, probabilities, rawlogs and interruption receipts retained.

Targets and group semantics unchanged: beta.5, LOO/minvotes2, native alpha, four(agree,conflict) pairs (0,1),(.25,1),(1,.25),(1,0). Float64 targets tofloat32. RawX MLP64/dropout.5, Adam .01/wd.0005,200epochs, first strictly best validation accuracy; same student seeds and BN rules. Truth not used for targets/selection.

Freeze all selectors/reuse references before scoring any new tests; zero-fit stage may score after its own freeze and cannot alter later specs. Match baseline/group selection budgets and report validation reused for checkpoint/config selection. No clean-specific oracle switching. Secondary balanced selector is descriptive, predetermined, and must also be applied to baseline pool.

No hard success gate added. Describe paired seed-level mean/sampleSD and seeds1/2 separately. Useful combo size .3-.5pp, broad direction and clean/F1 costs assessed jointly; all groups complete irrespective of intermediate scientific result. Three seeds fixed historicalSSL, not new independentSSL.

Primary questions: fallback mitigates forced-group clean loss? group effect on KM vsGB? selected support plus grouping adds beyond strongest matched individual module? Report actual fitting/search cost including C1 upstream selection, not only extra group fits. Storage preflight must preserve3GiB reserve; use content-addressed targets and read-only aliases, no deleting historical artifacts. Insufficient space => report before new fits, not silently change scope.

## Deliverables

PLAN/config, input/source/reuse manifests, fit inventory, tests/smoke, rawlogs/checkpoints/probabilities, raw results/paired stats, seed1/2 summary, class/stratum diagnostics, selector records, audit and concise Chinese report. Save complete marker only after verification. Parent independently recomputes primary comparisons and reloads a stratified sample. No automatic next experiment.
