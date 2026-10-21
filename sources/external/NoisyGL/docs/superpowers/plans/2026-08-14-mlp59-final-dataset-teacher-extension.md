# MLP-59 Final Dataset and Teacher Extension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the formal GB-LCSS teacher evaluation on Coauthor-CS and Amazon-Photo, then re-evaluate all six historical breadth datasets with the final leave-one-out, tie-safe GB-LCSS target without post-hoc dataset replacement.

**Architecture:** Reuse the existing predictors and runner. Add only missing dataset configs, a dedicated MLP-59 summarizer, regression tests, and immutable result files. The frozen v59 manuscript remains unchanged unless a pre-registered gate explicitly authorizes a new manuscript version.

**Tech Stack:** Python, PyTorch, PyTorch Geometric, pandas, scipy, pytest, YAML.

---

## Frozen protocol and decision rules

### Teacher extension

- Datasets: `coauthorcs`, `amazonpho`.
- Noise: train-only uniform `0.4` and `0.6`.
- Split: percent `0.2/0.1/0.7`.
- Seeds: `0--4`.
- Methods: `mlp_distill_formal_lcss_x`, matched shuffled control, and `mlp_distill_gcn`.
- Formal target: `target_beta=0.5`, `leave_one_out=true`, `tie_mode=distribution_only`, teacher input `X,A`, student input `X`.
- Pass: formal teacher exceeds base GB-LCSS in all four cells; exceeds GCN teacher in at least 3/4 cells; exceeds shuffled teacher by at least 1 pp with at least 4/5 positive paired seeds in at least 3/4 cells. Any missing row, split/noise hash mismatch, leakage, non-finite result, or protocol mismatch invalidates the experiment.
- Follow-up: only if the gate passes, complete FF-G2M/KRD/VQGraph for a five-dataset Table 4. Do not restore NRGNN.

### Boundary dataset extension

- Datasets: `pubmed`, `amazoncom`, `coraml`, `attribwiki`, `citeseerfull`, `coauthorph`.
- Entry noise: train-only uniform `0.4`.
- Methods: final GB-LCSS and size-matched shuffled control, both with `leave_one_out=true` and `tie_mode=distribution_only`.
- Seeds: `0--4`.
- Existing seven same-backbone baselines may be reused only when dataset, seed, split hash, noise hash, split percentages, noise scope, type, and rate match.
- Entry pass for a dataset: GB-LCSS exceeds the strongest eligible baseline by at least 0.5 pp with at least 4/5 positive paired seeds, and exceeds shuffled by at least 1 pp with at least 4/5 positive paired seeds.
- Full-spectrum follow-up: only entry-pass datasets receive uniform `0.2/0.6`, pair `0.4`, and instance `0.4` runs. A dataset may be added, never substituted for an existing main dataset, only if GB-LCSS is non-inferior to the strongest eligible baseline in at least 4/5 settings, has non-negative cross-setting mean delta, and has no setting below the strongest baseline by more than 1 pp.

## Task 1: Lock configuration and summary contracts

**Files:**
- Create: `tests/test_mlp59_final_extensions.py`
- Create: `scripts/summarize_mlp59_final_extensions.py`

- [ ] Add failing tests for the two teacher configs and five missing boundary configs per real/shuffled method.
- [ ] Add synthetic-data tests for protocol rejection, hash pairing, paired deltas, teacher gates, boundary entry gates, and conditional full-spectrum gates.
- [ ] Run the focused test file and confirm failure before implementation.

## Task 2: Add missing configs

**Files:**
- Create: `config/mlp_distill_formal_lcss_x/mlp_distill_formal_lcss_x_{coauthorcs,amazonpho}.yaml`
- Create: `config/mlp_distill_formal_lcss_x_shuffled/mlp_distill_formal_lcss_x_shuffled_{coauthorcs,amazonpho}.yaml`
- Create: five missing configs under each `frozengrace_gb_mlp_softdist_beta05_loo*` directory for `amazoncom`, `coraml`, `attribwiki`, `citeseerfull`, and `coauthorph`.

- [ ] Copy shared optimization settings from the audited matching configs.
- [ ] Change only method identity, embedding path, leave-one-out, tie mode, and random-control identity where required.
- [ ] Run focused tests and verify pass.

## Task 3: Teacher smoke and formal runs

**Files:**
- Create: `runs/mlp59_final_extensions_20260814/teacher_smoke_u0p4_seed0.csv`
- Create: `runs/mlp59_final_extensions_20260814/teacher_u0p4_seeds0_4.csv`
- Create: `runs/mlp59_final_extensions_20260814/teacher_u0p6_seeds0_4.csv`

- [ ] Run seed-0 smoke on both datasets and three methods.
- [ ] Audit finite metrics, X-only student identity, teacher input, target fields, leakage counts, and hashes.
- [ ] Run five seeds for both rates.
- [ ] Summarize paired teacher deltas and apply the frozen teacher gate.

## Task 4: Boundary uniform-0.4 audit

**Files:**
- Create: `runs/mlp59_final_extensions_20260814/boundary_u0p4_seeds0_4.csv`
- Create: `runs/mlp59_final_extensions_20260814/boundary_baseline_controls.csv`

- [ ] Run final real/shuffled GB-LCSS on all six datasets.
- [ ] Locate eligible baseline rows and pair by split/noise hash.
- [ ] Re-run only missing or non-pairable baseline rows.
- [ ] Apply the frozen entry gate to every dataset and record pass/fail without changing thresholds.

## Task 5: Conditional full spectrum

**Files:**
- Create only if triggered: `runs/mlp59_final_extensions_20260814/boundary_full_spectrum.csv`

- [ ] Run the four missing settings only for entry-pass datasets.
- [ ] Pair against eligible baselines and shuffled controls.
- [ ] Apply the frozen main-table addition gate. Never replace an existing main dataset.

## Task 6: Reporting and manuscript lineage

**Files:**
- Create: `runs/mlp59_final_extensions_20260814/mlp59_aggregate.csv`
- Create: `runs/mlp59_final_extensions_20260814/mlp59_paired_deltas.csv`
- Create: `runs/mlp59_final_extensions_20260814/mlp59_protocol_audit.csv`
- Create: `runs/mlp59_final_extensions_20260814/mlp59_gate_decisions.csv`
- Create: `runs/mlp59_final_extensions_20260814/mlp59_report.md`
- Modify: `paper_pipeline/00_evidence/claim_evidence_ledger.md`
- Create only if evidence changes the paper: `paper_final_zh_v60_mlp59_extension_20260814.md`

- [ ] Record method identity, config snapshots, commands, hashes, completeness, mean/std, and every paired seed delta.
- [ ] Update the evidence ledger with supported, unsupported, and forbidden interpretations.
- [ ] Keep v59 frozen if no gate authorizes a manuscript change.
- [ ] Run focused tests, manuscript lint, reference continuity, table integrity, and evidence hash checks.

## Self-review

- The two experimental questions have separate gates and outputs.
- No NRGNN route is reintroduced.
- No historical non-LOO boundary result can enter the new headline.
- No dataset can replace a current main dataset based on observed accuracy.
- Every formal comparison requires exact protocol and hash pairing.
