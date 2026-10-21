# Phase 40 Early Stop Note

This run stopped after `cora` and `citeseer` completed with seeds `0,1,2,3,4`.

Reason: both datasets failed all preregistered primary G1 gates for `s_xview` and `s_xview_soft`. Since the rule requires success on at least 2/3 datasets, the experiment could not become `GO` even if `coauthorcs` passed. `coauthorcs` was therefore not run, because raw-X granular-ball construction is expensive and no longer decision-relevant.

Conclusion: Phase 40 returns `STOP`. The cross-view granular-ball conflict signal did not provide a non-homologous noise signal beyond single-view kNN baselines on the decisive small-graph gate.
