# Analysis recovery record

No fitting, target construction change, source-result edits or scientific threshold changes occurred.

1. diagnose.py completed all12boundary protocols and saved20656rows, bins, correlations, decision.
   It then stopped while validating historical main_accuracy_and_f1.csv: Coauthor-CS has no B/Bprime/LCSS
   runs, so treating a blank table cell as an available arm raised KeyError. failure.json retains traceback.
   Original frozen diagnose.py and existing outputs are preserved unchanged.
2. Added finish_review.py and four RED/GREEN regression tests. Missing optional arms are accepted ONLY
   when all three corresponding raw records are actually absent; partial seed availability raises.
3. First recovery stopped at assertion that each comparison has15cells. Source paired table includes a sixth
   cost dataset Coauthor-CS, while the original aggregate explicitly uses five PRIMARY graphs.
   Reviewed original analyze_completed.py and raw counts: A comparator has18 cells before PRIMARY filtering.
   Recovery now explicitly matches that original five-graph scope, not a newly selected favorable subset.
4. Reviewed original cost aggregation too: compression ratios share A at default ratio0.1, not a separately
   refitted A at0.05/0.3. Recovery explicitly uses this same baseline key when checking those cost records.

11unit/regression tests pass. Final status is in audit.json. No original input/source was changed to make checks pass.
