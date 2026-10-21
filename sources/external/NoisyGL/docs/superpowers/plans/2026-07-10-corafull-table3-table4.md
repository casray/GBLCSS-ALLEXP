# CoraFull Table 3/4 Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fill the CoraFull gaps for paper Table 3 (teacher distillation five-arm decomposition) and Table 4 (graph-dependent reference methods) under the existing MLP-21 protocol.

**Architecture:** Reuse the existing `run_standard_downstream.py` predictors and CoraFull GRACE embedding/configs. Run only missing CoraFull experiments, summarize them into a dedicated run directory, then update the v4 manuscript table text without changing the method.

**Tech Stack:** Python/ROCm, NoisyGL predictors, CSV summaries, Markdown manuscript.

---

### Task 1: Confirm Available Configs and Prior Results

**Files:**
- Read: `scripts/run_standard_downstream.py`
- Read: `config/*/*corafull*.yaml`
- Read: `runs/mlp21_dataset_replacement_20260710/mlp21_report.md`

- [ ] **Step 1: List CoraFull configs for table methods**

Run:
```powershell
rg --files config | rg "corafull|CoraFull"
```

Expected: configs exist for `mlp_distill_gcn`, `mlp_distill_gcn_xh`, `mlp_distill_softdist_xh`, `mlp_distill_knn_xh`, `mlp_distill_softdist_xh_shuffled`, and as many Table 4 methods as possible.

- [ ] **Step 2: Check existing CoraFull Table B rows**

Run:
```powershell
rg -n "CoraFull Graph-Free Table B|mlp_distill_softdist_xh|GLNN" runs/mlp21_dataset_replacement_20260710/mlp21_report.md
```

Expected: CoraFull uniform 0.4 graph-free rows exist and can be reused for `mlp_distill_gcn` and `mlp_distill_softdist_xh` where exact protocol matches.

### Task 2: Run CoraFull Table 3 Five-Arm Teacher Decomposition

**Files:**
- Output: `runs/mlp22_corafull_table3_table4_20260710/table3_corafull_smoke_seed0.csv`
- Output: `runs/mlp22_corafull_table3_table4_20260710/table3_corafull_uniform04_06_seeds_0_4.csv`

- [ ] **Step 1: Smoke run**

Run:
```powershell
& "C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe" scripts/run_standard_downstream.py `
  --datasets corafull `
  --methods mlp_distill_gcn,mlp_distill_gcn_xh,mlp_distill_softdist_xh,mlp_distill_knn_xh,mlp_distill_softdist_xh_shuffled `
  --seeds 0 `
  --noise_type uniform `
  --noise_rate 0.4 `
  --noise_scope train `
  --device cuda:0 `
  --split_type percent `
  --train_percent 0.2 `
  --val_percent 0.1 `
  --test_percent 0.7 `
  --output runs/mlp22_corafull_table3_table4_20260710/table3_corafull_smoke_seed0.csv `
  --stop_on_error
```

Expected: all five rows finish with `status=ok`.

- [ ] **Step 2: Full 5-seed run for uniform 0.4 and 0.6**

Run the same command twice, changing `--noise_rate` to `0.4` and `0.6`, with `--seeds 0,1,2,3,4` and output filenames:
```text
runs/mlp22_corafull_table3_table4_20260710/table3_corafull_uniform0p4_seeds_0_4.csv
runs/mlp22_corafull_table3_table4_20260710/table3_corafull_uniform0p6_seeds_0_4.csv
```

Expected: 25 rows per file, no NaN/OOM, train-only noise audit passes.

### Task 3: Run CoraFull Table 4 Graph-Dependent References

**Files:**
- Output: `runs/mlp22_corafull_table3_table4_20260710/table4_corafull_graphdeps_smoke_seed0.csv`
- Output: `runs/mlp22_corafull_table3_table4_20260710/table4_corafull_graphdeps_uniform0p4_seeds_0_4.csv`

- [ ] **Step 1: Smoke run feasible graph-dependent methods**

Run:
```powershell
& "C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe" scripts/run_standard_downstream.py `
  --datasets corafull `
  --methods gcn,gcn_xh_ce,nrgnn,nrgnn_xh,tfr_gcn,procon,gd2 `
  --seeds 0 `
  --noise_type uniform `
  --noise_rate 0.4 `
  --noise_scope train `
  --device cuda:0 `
  --split_type percent `
  --train_percent 0.2 `
  --val_percent 0.1 `
  --test_percent 0.7 `
  --output runs/mlp22_corafull_table3_table4_20260710/table4_corafull_graphdeps_smoke_seed0.csv
```

Expected: methods that are configured for CoraFull run; unavailable methods are recorded explicitly rather than silently assumed.

- [ ] **Step 2: Full 5-seed run for runnable graph-dependent methods**

Run only methods that passed smoke:
```powershell
& "C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe" scripts/run_standard_downstream.py `
  --datasets corafull `
  --methods <passed_methods> `
  --seeds 0,1,2,3,4 `
  --noise_type uniform `
  --noise_rate 0.4 `
  --noise_scope train `
  --device cuda:0 `
  --split_type percent `
  --train_percent 0.2 `
  --val_percent 0.1 `
  --test_percent 0.7 `
  --output runs/mlp22_corafull_table3_table4_20260710/table4_corafull_graphdeps_uniform0p4_seeds_0_4.csv
```

Expected: no protocol leakage; slow/OOM methods are documented as unavailable under this protocol.

### Task 4: Summarize and Patch Manuscript

**Files:**
- Create or use ad hoc summary output: `runs/mlp22_corafull_table3_table4_20260710/mlp22_summary.md`
- Modify: `docs/paper_final_zh_v4_mlp21_corafull_20260710.md`

- [ ] **Step 1: Aggregate mean/std**

Run a short Pandas aggregation over all new CSVs, grouping by `dataset`, `method`, `noise_type`, `noise_rate`.

Expected: table-ready mean±std cells for CoraFull Table 3 and Table 4.

- [ ] **Step 2: Update manuscript**

Replace the CoraFull TODO notes in Table 3 and Table 4 with completed CoraFull columns/rows where data exists. Keep any unavailable graph methods footnoted.

Expected: no unsupported claim that CoraFull five-arm or graph-dependent rows exist unless the CSV contains them.

