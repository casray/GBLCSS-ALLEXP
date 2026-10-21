# Strong KD screen handoff, 2026-10-07

Completed 100 new GPU fits, 20 reused references, 2 bit-exact replay fits.
No training remains. Read 实验结论.md, summary.json and paired_results.csv first.

Protocol: five datasets, uniform .4/.6, all seeds0; frozen ERASE/GRACE/GB,
same raw-X MLP64, 200epochs, clean-val checkpoint/candidate selection.
KD/DIST temperatures1/4; DGKD-T .7/1. Native losses plus optional regionKL1.
DGKD-T uses teacher argmax, not labels. See SOURCE_AUDIT.md for upstream caveat.

Matched regional increments: KD +.375169pp (8W2L), DIST +.009201 (5W1T4L),
DGKD-T +.092660 (6W1T3L); overall validation selector +.211091 (7W3L).
Continue gate passed narrowly. 73.26% of overall net gain comes from two
CiteSeer KD cells; other eight selected cells average +.070565pp.
Do not claim strong-KD universal benefit. DIST itself exceeds selected KD
by .731789pp on average (8/10); CoraFull gap to teacher is already eliminated
at seed0 by DIST without regions. DIST+GB CoraFull .6 adds .671238pp.

All120 selected models reloaded bit-exact; 8tests pass; frozen inputs unchanged.
Single seed: no seed std/significance claim. No kmeans-on-DIST or second teacher
tested. Proposed next step: independent full-protocol seeds, fixed DIST and GB
across full grid; no new modules or test-guided lambda search yet.

PLAN.md and analysis/training sources are frozen. Do not edit in place to
append conclusions; add follow-up report or create a separately frozen run.
No manuscript changed. New experiment code is confined to this directory.
