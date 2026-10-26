# Commands and purpose

Runtime: `D:/codex_temp/noisygl-rocm721/Scripts/python.exe`; PyTorch 2.9.1+rocm7.2.1, RX7700XT; all paths below relative to the experiment workspace.

1. `python -m unittest discover -s experiments/gce_lcss_fusion_20261022 -p test_kernels.py`: nine loss/gradient/tie tests; initial RED and later GREEN recorded in ENGINEERING.
2. `python experiments/gce_lcss_fusion_20261022/run.py prepare`: build and independently verify 36 protocols, save truth-only diagnostics, freeze sources. First pre-freeze attempt failed on CPU/GPU rounding; recovery documented.
3. `python experiments/gce_lcss_fusion_20261022/run.py smoke`: 14 three-epoch GPU engineering fits, excluded from formal results.
4. `python -u experiments/gce_lcss_fusion_20261022/run.py run`: 504 full GPU classifier fits; per-epoch logs, configs, X-only checkpoints, probabilities, status; stdout captured in raw_execution.log.
5. `python -u experiments/gce_lcss_fusion_20261022/analyze.py`: only after complete grid; checkpoint reloads, independent accuracy counts, 48 historical anchors, complete paired metrics and frozen decision gates.
6. `python experiments/gce_lcss_fusion_20261022/audit_posthoc.py`: independent saved trajectory, frozen encoder, hash and CSV arithmetic audit; checkpoint weight quantiles. Does not rerun training or choose hyperparameters.

Resume rule: an existing successful run is read, not overwritten. Failed/running artifacts require engineering review. Old experiment/code/manuscript paths remain read-only.

Numerical recovery: original run stopped after339 successes on soft-GCE underflow. Archived98 v1 GCE directories (97 successes plus1 failed), within the resolved experiment root, preserving all artifacts. `python -u experiments/gce_lcss_fusion_20261022/resume_v2.py` resumes with exp(q*log_p), reruns all GCE/SOFT_GCE arms and reuses unaffected successes. stdout: raw_execution_v2.log. Twelve unit tests pass, including finite gradients at extreme logits. Both frozen original and numerical-recovery source manifests must pass. 504 is the unique final grid, not the number of physical attempts including superseded runs.
