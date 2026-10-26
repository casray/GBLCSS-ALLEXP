# Scheduled execution handoff

COMPLETED 2026-10-23: the one-shot E1/E2/E4 batch has finished. All expansion gates failed; E3/E5 were not launched. Read REPORT_zh.md and HANDOFF.md. Commands below are the original execution recipe, not an instruction to restart. The analysis-only repair is documented in ANALYSIS_AMENDMENT.md and final_audit.json; do not repeat the original fit pipeline on the already completed outputs.

User authorized designing the experiment now and starting about two hours later. No new permission is needed for the bounded first batch in PLAN.md. Do not start before config.json not_before. This is a single scheduled start, not a repeating training job.

Workspace: C:/Users/Admin/Desktop/实验8去噪5.19-/gb_c1_ratio_success_20260609/results_ratio_c1_20260609

1. Read PLAN.md/config.json and any current HANDOFF.md. Check whether diagnostics or training are already running/completed before starting. Inspect GPU occupancy; do not kill another task.
2. Start the ready E1a/E2-log diagnostic entry in the verified environment:

```powershell
& 'D:\codex_temp\noisygl-rocm721\Scripts\python.exe' -u 'experiments/sc_support_channel_pilot_20261023/run_diagnostics.py' --stage all
```

This CPU diagnostic builds label-value-free candidate trees, freezes SC choices, then analyzes clean training labels; it fits no neural network. It includes a wall-clock guard and an exclusive lock. All new output is inside this experiment directory.

3. Inspect actual-support/duplicate-cut/coverage and protocol checks. Do not change s, tree limits or beta using oracle scores. If candidates collapse, report the limitation; do not call a nominal four-vote setting four actual votes.
4. Implement narrowly scoped GPU fitting adapters under this directory using the inspected historical R0/R2/teacher helpers. Do not edit historical scripts. Add synthetic tests for target/selection logic and a short GPU bridge to verify exact initialization, frozen BN behavior, checkpoint selection, and deployment forward.
5. Run bounded E1b pilot, E2 length diagnostic and E4 six-arm seed0 pilot as specified in PLAN.md; use one GPU job at a time. Existing noisy teachers may be reused; clean SUPPORT teachers require explicit new fits already authorized in the plan. Freeze and audit selections before final test scoring. Seed1/2 expansions require the prewritten validation gates, not new ad hoc thresholds.
6. Save progress after every stage. If a fit fails, preserve its raw failure, diagnose, record any engineering amendment, and rerun only affected cells after a meaningful check. Do not silently alter scientific settings. Do not exit claiming all work complete while required processes are still running.
7. Produce a detailed Chinese report with paired mean/std, accuracy and macro-F1, all failures/negative results, selection leakage limitations, and next decision. E3/E5 are designed later phases, not automatically launched by this wake. Keep the paper and remote repositories unchanged.

Do not send repeated unchanged progress notifications. Report a meaningful start, stage completion, failure or required user decision. The scheduled action is one-shot; do not create a recurring training loop or duplicate this start.
