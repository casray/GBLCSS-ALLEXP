# MLP-49 UTV-GB-LCSS implementation plan

1. Add unit tests for entropy/JSD reliability, non-training-only aggregation, tie-safe LOO equivalence, conservative fallback, and shuffled size preservation.
2. Implement the tested math in `utils/teacher_verified_gb.py` without modifying formal GB-LCSS.
3. Add a standalone Phase A runner that trains/caches one NRGNN teacher per protocol cell, obtains MC-dropout posteriors on the frozen best graph, constructs all control targets, and writes auditable CSV/JSON artifacts.
4. Run a Cora uniform-0.4 smoke and inspect probability, split, noise, and teacher cache integrity.
5. If smoke is healthy, run the frozen 3-dataset by 2-rate grid and evaluate the preregistered gate.
6. Write the detailed conclusion and update project evidence only after the gate result is fixed.

