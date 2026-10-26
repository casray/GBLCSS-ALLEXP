# R2 + SUPPORT + D1 + C1 integration

Approved: five main graphs, seeds0/1/2, clean/U.4/U.6/Pair.4. No KM/kNN this round. Parent owns research/r2_integrated_20261026; sole gpt-6.1-sol/high worker owns experiments/r2_integrated_20261026. Historical files and manuscript read-only; no upload or extra experiments.

## Hypothesis and method
R2 transfers frozen GRACE representations into an attribute student; SUPPORT controls the evidence region; LCSS supplies leave-one-out targets; D1 allocates strength by observed agreement/conflict; C1 selects configurations through short validation-guided fits. Test whether the complete combination improves each component and ordinary robust supervision. Existing R0 combo and R2 fixed-support D1 are motivating evidence, not additive gains. Historical KM superiority stays documented, not re-tested.

Read root literature/history and prior d1_c1_confirmation/d1_fallback_km_combo plans, reports and code. Reuse verified fixed GRACE, per-seed100epoch R2 encoders, splits/noise. Frozen encoder/BN,64hidden head Adam lr.01 wd.0005,200epochs, first strictly highest validation accuracy; verify exact bridge. beta.5,min_votes2,LOO, tied votes unsharpened. Only observed training labels and clean validation labels enter fitting/selection; clean training/test truth is post-selection diagnostics. Seeds/graphs already developed on: integration validation, no claim of untouched confirmation.

## Scope
60R2 cells: cora,citeseer,corafull,coauthorcs,amazonpho x seeds0,1,2 x four conditions. R0 is read-only reference, no new R0 fits. No new SSL, teachers, weighting or extra seeds.
Anchors: CE,GCEq.7,OLD64-LCSS,legacySUPPORT32-LCSS. Exact oldfit reuse only if input/encoder/init/optimizer/checkpoint contracts match; otherwise refit in new folder and record why.

## Joint selection and matched ablations
Common deep tree, supports4/8/16/32/64/128. Seven rules per support: uniform .25/.5/1 plus D1(agree,conflict)=(0,1),(.25,1),(1,.25),(1,0). Groups use unique raw vote winner. NEW explicit tie convention: D1 ties use lambda1, avoiding dependence on future best-global fit. Use this convention in every new fixed/adaptive D1 arm; old D1 differs and is only supplementary. Target t=e_z+lambda_n*(t_original-e_z).
Joint C1:42to25epochs,retain14to75,retain4to200. Force support32/uniform1 anchor within survivor counts. Resume own full optimizer/RNG state. Rank: best validation accuracy descending, selected-checkpoint NLL ascending, lower mean applied lambda, larger support, earlier best epoch, fixed id. Freeze precise fields before training. Finalists compared to CE200; exactties prefer CE/global via frozen order.
Required same-tree ablations: fixed32/global(CE+3uniform); fixed32/expanded(add4D1); adaptive/global(original18->6->2 race plus CE, exactreuse allowed); adaptive/expanded(new joint42->14->4 plus CE); validation-selected stronger single module from fixed-expanded/adaptive-global.
Freeze races before completing shared fixed32 controls. Historical global-prefix replay permitted with retrospective provenance and strictly prefix-only selection. New group candidates require real training/resume. No exhaustive42fullgrid. Budgets differ: report updates/walltime and avoid attributing search-budget effects solely to the mechanism.

## Execution
Persist resolved config/hashes first. Inspect locks/completion/GPU/processes/storage. Python D:/codex_temp/noisygl-rocm721/Scripts/python.exe, serial GPU, >=3GiB reserve; never kill other tasks. Unit tests then <=8twoepoch smoke. Authorized writes limited new directory. Engineering errors retained and repaired, no scientific tuning after test inspection.
Tests:LOO,lambda0/1,ties,groups,pairedinit,frozenR2BN,resume,causal race/anchors/ranks,labelwhitelist,readonlyaliases,pairedSD. Preserve prefix logs, configurations/source/input hashes, checks/probabilities, actualsupport/cost, failures. Exact reused artifacts have aliases/hashes.

## Analysis and decisions
Primary: full minus validation-selected stronger single module; secondary CE/GCE/OLD64/SUPPORT and all ablations. Each graph/noise and clean, paired three-seed mean and sampleSD, macroF1, perclassrecall, diagnostic agreement/conflict strata. Modest positive increments retained with uncertainty/cost; clean/F1 damage reported. Nearzero increment favors simpler method. No hard .5pp cutoff, no automatic expansion.
Deliver config/provenance/tests/rawlogs/checkpoints/probabilities/results/summary/paired/selections/costs/failures/audit/ChineseREPORT/HANDOFF/complete. Parent independently checks selection, aggregation and checkpoint sample before final report.
