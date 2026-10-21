# Session handoff: 2026-10-09

User authorized new adapted TINED experiment following the SimMLP screen.
Completed five main datasets, uniform0.4, seed0, GPU, 200 teacher+200 student
epochs each. All training and audit processes finished; no agents running.
No manuscript, old code or old result edits; no extra seeds or parameter searches.

Read PLAN.md and REPORT.md first; results.csv and artifact_audit.json contain
verified scores. Five fast unit tests, Cora2+2 GPU smoke and all five teacher/
student checkpoint reloads passed, including exact main split/noise hashes.

Teacher is exact common PyG GCN, hidden64/dropout.5/Adam .01/wd.0005;
native sparse adjacency and initialization consumption match Table8 CE routes.
Re-trained teachers here; original Table8 checkpoint identity has not been asserted.
Teacher uses last tied validation maximum; student uses first maximum.
Student remains official four-FC FT/GP imitation with teacher injection and
response KL+DE stages. Do not claim same two-FC student as original Table8 rows.
Actual GCNTINED FT/GP energy extraction uses GCN normalized propagation, not SAGE.

Author algorithm best-record presets are SAGE-transductive only. Reused these
predeclared knobs for common GCN; no full GCN-specific best preset exists in JSON.
Restored author student dropout: Cora0/CiteSeer0/Photo.5; CoraFull/CoCS use Cora
fallback0 as before. Retained main base lr.01/wd.0005, algorithm lr ratios,
512 batches, separate CE/KL/DE Adam steps including zero-weight CE updates.

Student test percentages: Cora74.3536, CiteSeer66.1512, CoraFull51.7503,
CoCS92.3245, Photo74.8086. Teacher tests:77.0449,64.4759,61.6673,91.8024,89.3557.
Compared to old adaptation, four improve and Photo drops8.74pp. CoraFull rises
32.56pp, but bundled changes cannot isolate causes; single seed no mean/std.
Cora bestval80.0 at6 falls to28.15 at200; Photo bestval75.69 at10 falls to27.32.
Both stored results use best validation checkpoints. Finite/reload-consistent
training does not imply late-stage stability. Photo teacher remains strong.

Potential optimizer/DE mismatch is unverified. Do not automatically tune, change
lr or dropout after seeing tests. Full official DGL training parity unverified.
Next user decision: keep strict common optimizer or allow predeclared method-
specific optimization and validation-only selection before formal multiseed runs.

Runtime D:/codex_temp/noisygl-rocm721/Scripts/python.exe; workdir this folder:

```powershell
python -X utf8 -m unittest test_gcn_adapter -v
python -X utf8 -u run_gcn.py --dataset cora --smoke
python -X utf8 -u run_gcn.py --grid
python -X utf8 -u report_results.py
```

Completed runs skip training; report_results.py only verifies saved artifacts.
