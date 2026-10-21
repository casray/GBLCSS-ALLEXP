# Completed non-training review: 2026-10-21

Read 综合结论与接续.md. Scope: boundary diagnostic + existing coarsening review only, no new fits or manuscript edits.
New20,656train-node/condition rows,12protocols,99correlations,394binnedrows; seed0, std=null, posthoc.
Geometry H/GB fixed. EligibleGB LOOsupport>=2; CS661/665,othersall. Labels only diagnostic.

Radial feature positive rho+endbin gap clean2/3,noise3/9; monotonic1/12, broad support false.
Cross-ball nearest-voter fraction clean3/3,noise7/9; monotonic4/12, broad descriptive support true, causal false.
Clean crossrho .177/.124/.224. Within-ball flips CoraU.4 andCFullU.6; weak associations undernoise.
Do not say fixed partitions inevitably lose to kNN. Highestcrossquartile cleanGB/KNNstrictwrong:
Cora34/137vs11/137;CS33/179vs25/179;CFull75/992vs30/992. Target error, NOT studenttesterror.
CSU.4 GBtargetBrier .006612lower butstudenttest .2148pp lower: no automatictarget→accuracy implication.
PreviousGB-on-R2 fullcontinuation remainsfalse, do not rescue with boundary correlations.

Budgetcoarsening1080oldrawrecords arithmetic rechecked,232pairedrows/264availmainarms/24absentoptional arms.
Fiveprimarygraphs×3noisy:cachedMLP C−B+.346,C−LCSS−1.218(4/15positive);GCN+1.148,+.565(6/15positive).
SomeGCNhighnoise three-seedpositivecases worth retaining, but not broadsuccess. R3−R0−.033,O2−R0+.281.
16cleancostcellsoriginalgate0passes,coldC/A1.22–2.17; sourcecoldscope reconstructed,notindependentcoldruns.
Differentarchitecture hidden256,originalgraphGCNdeployment; do not rank againstR2X-only64.
Strongpartition/noisyGNNcontrols missing. StopR3/efficiencyclaim;retainconditionalball-leveltraining signal only.

237frozenfilesunchanged,11tests passed,auditpassed,newfits0. Initialdiagnose completedboundary thenfailedoptional
CS B/Bprime/LCSS absence. Frozenoriginal preserved; failure.json+ENGINEERING.md record recoveries.
finish_review.py correctly handlesabsentarms/PRIMARYscope/sharedAdefaultbudget and verifiesallsavednumbers.
Do not rerun diagnose.py; hashes/frozen marker refuse overwrite. No pending sessions.
GCEcombos/fine-tuning/deeperhead/5seeds/GBSVMnotstarted. Next action requires authorization.
