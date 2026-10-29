# GCN teacher with SUPPORT+D1+C1 -> DIST: five graphs, three seeds

Approved2026-10-27. Expand teacher_integrated_dist_pilot_20261026, which covered3graphs/seed0/clean-U.4-U.6. New table fivegraphs x seeds0..2 x clean/U.4/U.6 =45cells, reuse9oldcells,36newcells. Keep six teacher routes CE/OLD64/LEGACY_SUPPORT32/DEEP_S32/FIXED_D1/JOINT and identicalDISTstudent perroute.270logicalstudent endpoints(54old,216new); physically identicalteacherprobabilities may alias only with sameinit/config/split. No R2teacher or R2studentwarmstart here, no newPair/KM/kNN/newSSL/graph/dataset/seed/hyperparameter/manuscript/upload.

## Hypothesis / GB role

Test whether controlled regional supervision improves originalGCNteachers and transfers to attribute-onlyMLPs acrossgraphs/seeds. GB supplies sharedLOOclasscounts in historicalfrozenGRACEgeometry. SUPPORT changes supportcut, D1 groups by unique rawvoteagreement/conflict, C1 validationselects. Main contrasts JOINT vs SUPPORT32 and fixedD1, teacher and student separately; CE/OLD64/deepS32 andclean/F1/recall/cost fullyreported. Priorpilot noisyteacher gain+.060pp/student+.176pp,CFcleanrepair; do not assume largerstudywillimprove. No newhardgainthreshold: complete/auditablematrix and explicitpositive/negativecases.

## Exact frozen protocol

Historical20/10/70orderedpercent_split_indices(seed), onlytrainlabelnoiseU.4/U.6, cleanvalidation. Reuseall15deepgeometry/legacy32/split/noisepacksandGRACE, nofreshSSL. GCN originalgblcss.models.GCN2layers64/dropout.5, originalcoalescedCOO graph,Adam.01/.0005,200epochs. Initresetseed=s, allocate/move discardedreferenceMLPthenGCN. Student originalMLP2layers64/BN/dropout.5,randomseed=s,Adam.01/.0005,200epochs,fullnodeofficialDIST T1/beta1/gamma1, frozenprobabilitiesonly. Everyarm firststrictmaxcleanvalidationaccuracycheckpoint, selectedcheckpointNLL onlyinter-candidatetie-break. No studentCE/regionalKL/graphinputs/R2warmstart. Fixedhistoricalbeta.5/minvotes2/LOO/tiesunsharpened;D1tieslambda1.

Candidates sixsupports4,8,16,32,64,128 x (threeuniformstrengths.25/.5/1 plusfourgroups(0,1),(.25,1),(1,.25),(1,0))=42. JOINT42->14->4 at25/75/200 withS32_L1anchorwithinquota. Eacharm ownoptimizer/RNGcontinuation, final4competeCE200. Rank originalaccuracydesc/NLL/meanlambda/largersupport/earlierbestepoch/fixedID. Teacher-specificvalidationchoices, noR2selectionreuse. Freezejointdecision beforecompletingfixed32controls(CE+3uniform+4groups), thenOLD64/legacy32baselines. Teacher45trajectoriespernewcell(42+CE/OLD64/legacy)=1620newcandidate trajectories, manyearlyeliminated; no exhaustive42fullgrid. Atoriginalpipelinelayer preserveallrawlogs/checkpoints/probabilities/prefixes/hash/cost.

## Execution / storage / selection discipline

Oldfilesread-only,newsourcehere; heavyD:/codex_temp/gcn_support_d1_c1_3seed_20261027.3GiBreserveandfree>=8GiB GPU. Beforelaunchreadonlyprocessinventory; ifotherownerswait, no kill. Singleexclusive lock, atomiccompletedfitreuse+prefixresume, nooverwriteoldfreeze. Old9teacher/studentrefsverifiedagainstoldglobalfreeze/artifacthashes; newpayload/hashmanifestsseparate. CPUpreflightall45cells beforeGPUsmoke. Testswhitelist/target/LOO/tie/rank/anchor/seeds/output/selection/recovery.<=6twoepochsmoketrajectories+2epocholdGCN/oldDISTbridges, verifyinitialization/modeloptimizerRNGcontinuation/reload. GPUserialanddurablehiddenbackgrounddriver.

Freezeall45teacherchoicesbeforestudents;all270logicalstudentchoicesbeforeanynewtestscore. Old9testalreadyreadidentifiedashistorical. DeduplicateonlyexactteacherprobabilitySHAandidenticalstudentinit/configwithaliases. Teacher/studentsourcesinputhashes/selectedepoch/config immutable. Sciencechangesnotdrivenbytest. Onlypostfreezeanalysis/auditmayloadcleantraintesttruth.

## Tables / analysis / interpretation

Pergraph/condition n3 mean/sampleSDddof1,rawseedvalues,accuracy/macroF1/perclassrecall,pairedJOINT-referencegains,student-minus-teacher, actualsupport/strength/choiceidentities andtrainingupdates/seconds/storage. SeedfirstmacroacrossU.4/U.6thengraphs;cleanseparate. Historicalsinglepilot/newcellsseparatecontext, SSLfixeddisclosed. Modelcapacity/teacherinformationdiffersfromdirectR2line;matchedsameseedcomparisononly. No n3ERASEclaimunlessavailablematchingartifacts; oldseed0referencesremainexploratory.

Analysis/sourcehashesseparatefromfitfreeze, independentCPUmodelmetricsandcausalraceaudit, allnegativeoutcomespreserved. Reportcountslogicalvsphysicaldistinct. Completeflagonlyafterfitting,analysisandindependentaudit. Currentdecisionauthorizedexecute; noautomaticmoreseeds/methodtuning/manuscriptediting/upload. RuntimeHandoff/statuslogsmonitorneeded, notduplicateworker.
