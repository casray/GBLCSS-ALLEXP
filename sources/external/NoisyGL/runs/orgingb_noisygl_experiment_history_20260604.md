# OrginGB NoisyGL Experiment History And Lessons

本文件只沉淀 `external/NoisyGL` 下的 OrginGB / ordinary granular ball noisy-label GCN 实验。它不整理其他 GB-GCL、GAGB、GRACE frozen embedding 或跨方向实验。

# 1. Current Overall Judgment

- Confirmed: NoisyGL 下已经完成 OrginGB-GCN v1 soft target、v2 reweight、offline gate scan、protocol-matched gate scan 和 Cora smoke training。
- Confirmed: OrginGB-GCN 当前不能作为主方法扩展。Cora uniform 0.4 seed=3000 中，`gcn`、`orgingb_gcn_reweight`、`orgingb_gcn_reweight_random`、`orgingb_gcn_reweight_noball` test acc 都是 0.705。
- Confirmed: v1 soft target mix 干预面过宽，但没有真正改变 target argmax；v2 reweight 更保守，但仍没有带来 validation/test 提升。
- Confirmed: protocol-matched scan 之后，ordinary GB 有一个局部 suspicious-node 诊断信号：`top_ratio=0.2, min_ball_conf=0.0` 时 real GB gated noisy precision 为 0.6071，高于 no-ball/random 的 0.4643。
- Risk: gate precision 不能替代训练结果。即使 real GB gate 更准，best checkpoint 仍来自 warmup，说明当前 loss/timing 没有利用这个信号。
- Risk: offline scan 必须和正式训练协议完全一致。旧 scan 因未复用 Cora `add_self_loop: false`，把旧阈值 precision 高估到 0.75。
- Decision: OrginGB ordinary GB reweight 在 NoisyGL 中暂定为 `diagnostic-only`，不进入 Cora/CiteSeer/PubMed 5-seed 正式实验。
- Next priority: 若继续 NoisyGL 标签噪声方向，应做 graph-aware suspicion smoke，而不是继续调 ordinary GB 的 `top_ratio/min_ball_conf/beta`。

# 2. Direction Index

| Direction ID | Direction Name | GB Role | Versions Included | Dataset | Best Trend | Main Failure | Decision | Paper Value |
|---|---|---|---|---|---|---|---|---|
| NGL-ORG-01 | OrginGB-GCN ordinary GB noisy-label reweight | ordinary embedding-space GB 生成 soft target 或 high-suspicion sample weights | v1 mix, v2 reweight, random control, no-ball control, offline scan, protocol-matched scan | Cora smoke；CiteSeer/PubMed planned but not run | real GB gate precision 0.6071 > no-ball/random 0.4643 | training acc 不超过 GCN，best_phase=warmup | diagnostic-only | diagnostic |

# 3. Detailed Record

## NGL-ORG-01: OrginGB-GCN ordinary GB noisy-label reweight

### 3.1 Direction Summary

- Target problem: 在 NoisyGL 图标签噪声节点分类任务中，验证 ordinary granular ball 是否能帮助 GCN 降低 noisy train labels 的影响。
- Core hypothesis: GCN warmup 后 hidden embedding 中可能形成局部一致区域；ordinary GB 可以提供 ball-level soft target 或 high-suspicion node gate。
- Main baseline: `gcn`。
- Controls: `orgingb_gcn_random`, `orgingb_gcn_reweight_random`, `orgingb_gcn_reweight_noball`。
- Dataset / noise / seed used: Cora uniform 0.4 seed=3000。
- Device: `cuda:0`。
- Current decision: diagnostic-only。

### 3.2 Main Related Files

- Design docs:
  - `OrginGB-GCN.html`
  - `OrginGB-GCN_deep_thinking_pack.html`
- Code:
  - `external/NoisyGL/predictor/OrginGB_GCN_Predictor.py`
  - `external/NoisyGL/utils/ordinary_gb.py`
  - `external/NoisyGL/utils/soft_targets.py`
  - `external/NoisyGL/scripts/diagnose_orgingb_gate_scan.py`
  - `external/NoisyGL/tests/test_orgingb_tools.py`
- Configs:
  - `external/NoisyGL/config/orgingb_gcn/orgingb_gcn_cora.yaml`
  - `external/NoisyGL/config/orgingb_gcn_reweight/orgingb_gcn_reweight_cora.yaml`
  - `external/NoisyGL/config/orgingb_gcn_reweight_random/orgingb_gcn_reweight_random_cora.yaml`
  - `external/NoisyGL/config/orgingb_gcn_reweight_noball/orgingb_gcn_reweight_noball_cora.yaml`
- Result and log files:
  - `external/NoisyGL/runs/orgingb_gcn_smoke_report_20260603.md`
  - `external/NoisyGL/runs/orgingb_gcn_v2_reweight_smoke_report_20260603.md`
  - `external/NoisyGL/runs/orgingb_gate_scan_20260603/orgingb_gate_scan_report_20260603.md`
  - `external/NoisyGL/runs/orgingb_gate_scan_protocol_matched_20260603/orgingb_gate_scan_protocol_matched_report_20260603.md`
  - `external/NoisyGL/runs/orgingb_gate_scan_protocol_matched_20260603/gate_scan.csv`
  - `external/NoisyGL/runs/orgingb_gate_scan_protocol_matched_20260603/training_smoke_protocol_matched.log`
  - `external/NoisyGL/runs/orgingb_gcn_diagnostics_v3.csv`

### 3.3 Versions / Attempts Included

| Version | Main Change | Result Trend | What It Tested | What It Showed |
|---|---|---|---|---|
| Design | 静态普通粒球 + soft target + sample reweight；不把 same-ball 当硬标签 | 未运行 | ordinary GB 是否值得先做低复杂度版本 | 设计边界正确：必须有 random/no-ball controls |
| v1 mix | warmup 30 epoch 后构 ordinary GB，用 `q_B` 与 noisy one-hot 做 soft target mix | `orgingb_gcn` test 0.705，和 GCN 持平；best_phase=warmup | ball-level soft target 是否能纠错 | 干预广但纠错弱；target_argmax_changed_ratio=0 |
| v2 reweight default | target 保持 noisy one-hot，只对 high-suspicion nodes 降权 | test 仍 0.705；best_phase=warmup | reweight 是否比 mix 更安全 | 安全但没有训练收益 |
| offline gate scan | 扫 `suspicion_top_ratio × min_ball_conf`，加 no-ball/random strength-matched controls | 旧 scan 选 `0.1/0.1`，real precision 0.75 | ordinary GB 是否能识别 noisy train nodes | 后续证明旧结果因协议不一致被高估 |
| selected old threshold smoke | 用旧 `0.1/0.1` 跑 training smoke | real GB precision 0.50，no-ball 0.6429；test 0.705 | scan 结果是否能复现到训练链路 | 不能，暴露 scan/training mismatch |
| protocol-matched scan | 修复 dataset construction，完整复用 Cora config，尤其 `add_self_loop=false` | 旧 `0.1/0.1` 回落到 real precision 0.50；新 best `0.2/0.0` real precision 0.6071 | 协议对齐后 ordinary GB 是否仍有信号 | 有弱诊断信号，但不再是旧的 0.75 |
| protocol-matched smoke | Cora reweight configs 切到 `top_ratio=0.2, min_ball_conf=0.0` | GCN、real、random、no-ball 全部 test 0.705；best_phase=warmup | 更准 gate 是否能提升训练 | 不能，停止正式扩展 |

### 3.4 Algorithm Role

- GB generated from: GCN hidden embedding after warmup。
- GB type: ordinary feature/embedding-space granular ball，不使用图拓扑构球。
- v1 target mode: `mix`，用 ball-level `q_B` soft target 混合 noisy one-hot。
- v2 target mode: `reweight`，不改 target，只改 train sample weight。
- Suspicion score:
  - GB mode: `1 - sqrt(warmup_prob_i[noisy_y_i] * q_B[noisy_y_i])`
  - no-ball mode: `1 - warmup_prob_i[noisy_y_i]`
- Clean label usage: 只用于 diagnostics 和 test evaluation，不进入 target、weight、gate 或构球。

### 3.5 Result Summary

| Setting | Method | Test Acc | Best Phase | Gated Ratio | Gated Precision | Note |
|---|---|---:|---|---:|---:|---|
| v1 mix | `orgingb_gcn` | 0.705 | warmup | 0.9500 | 0.3308 | target 改动广，但 argmax 不变 |
| v2 default | `orgingb_gcn_reweight` | 0.705 | warmup | 0.1500 | 0.4762 | 无收益 |
| old selected `0.1/0.1` | `orgingb_gcn_reweight` | 0.705 | warmup | 0.0571 | 0.5000 | no-ball 更高，为 0.6429 |
| protocol best `0.2/0.0` | `orgingb_gcn_reweight` | 0.705 | warmup | 0.2000 | 0.6071 | gate 优于 controls，但训练无收益 |
| protocol best `0.2/0.0` | `orgingb_gcn_reweight_random` | 0.705 | warmup | 0.2000 | 0.4643 | same-size random control |
| protocol best `0.2/0.0` | `orgingb_gcn_reweight_noball` | 0.705 | warmup | 0.2000 | 0.4643 | no-ball control |

Protocol-matched gate scan best row:

| top_ratio | min_ball_conf | real precision | no-ball precision | random precision | real recall | real clean ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 0.2 | 0.0 | 0.6071 | 0.4643 | 0.4643 | 0.3542 | 0.1196 |

### 3.6 What Is Directly Supported

- ordinary GB has a weak but real suspicious-node diagnostic signal on Cora uniform 0.4 seed=3000 after protocol matching。
- v1 broad soft-target correction is not suitable as a main mechanism。
- v2 reweight is safer than v1 but still does not improve the selected model。
- no-ball and random controls are mandatory; without them, gate precision can be misleading。
- protocol-matched data loading is mandatory; otherwise offline scan can produce false confidence。

### 3.7 What Cannot Be Concluded

- Cannot claim OrginGB-GCN improves noisy-label node classification。
- Cannot claim ordinary GB is better than no-ball generally; current evidence is single-seed diagnostic only。
- Cannot claim Cora/CiteSeer/PubMed formal performance; CiteSeer and PubMed were intentionally not run because smoke failed。
- Cannot compare single-seed OrginGB smoke to 5-seed baseline mean/std as a formal result。

### 3.8 Failure Analysis

- Main failure: gate precision did not translate into downstream validation/test gain。
- Mechanism reason: GB is built after 30 warmup epochs, but best checkpoint is epoch 6; the post-GB phase never surpasses warmup。
- v1-specific issue: soft target changed too many nodes weakly, causing broad clean exposure without target argmax correction。
- v2-specific issue: reweight is conservative, but weight changes are too weak/late to alter selected checkpoint。
- Diagnostic issue fixed: offline scan originally missed Cora `add_self_loop=false`, causing protocol mismatch。

### 3.9 Lessons Learned

1. Protocol-matched diagnostics before any claim。
   Evidence: old offline precision 0.75 disappeared after matching NoisyGL dataset construction。
   Future constraint: scan scripts must reuse dataset config, split, self-loop, seed, warmup and GB config。

2. Gate precision is only a screening metric。
   Evidence: real precision 0.6071 beat controls, but test acc stayed 0.705 and best_phase stayed warmup。
   Future constraint: no formal expansion unless training smoke also passes。

3. Ordinary GB should not be a broad soft-label corrector。
   Evidence: v1 effective coverage around 0.95, target_argmax_changed_ratio 0。
   Future constraint: avoid broad target mixing without a much stronger confidence gate。

4. Random and no-ball controls are non-negotiable。
   Evidence: old selected threshold had no-ball precision 0.6429 > real 0.50。
   Future constraint: every future GB gate must beat both controls at the same gate size。

### 3.10 Decision

- Decision: diagnostic-only。
- Reason: single-seed gate signal exists, but training objective has no gain and formal multi-dataset evidence is absent。
- Stop condition already met: Cora smoke failed; do not expand ordinary OrginGB to CiteSeer/PubMed 5-seed。
- What should not be tried again: only tuning ordinary GB thresholds (`top_ratio`, `min_ball_conf`, `beta`) as if it were a new main method。
- What remains useful: ordinary GB diagnostics, random-size matched controls, no-ball controls, protocol-matched gate scan。

# 4. Anti-Repetition Memory For NoisyGL OrginGB

| Pattern | Why It Failed | Evidence | Required New Evidence Before Retry |
|---|---|---|---|
| Broad ordinary GB soft target mix | Changed many targets weakly, but did not correct argmax or improve test | `orgingb_gcn_diagnostics_v3.csv` v1 row | Narrow high-confidence gate with GB-phase checkpoint improvement |
| Ordinary GB reweight threshold tuning | Better gate precision did not improve downstream | protocol best training smoke | `best_phase=gb` and test acc above GCN/control |
| Offline scan without protocol matching | Dataset construction mismatch inflated precision | old vs protocol-matched scan reports | Exact reuse of NoisyGL dataset config and training path |
| Claim from gate precision only | Gate selected noisier nodes but training tied GCN | `training_smoke_protocol_matched.log` | Both gate metric and training metric improve |

# 5. Useful Signals To Preserve

- Protocol-matched gate scan is useful as a screening tool。
- `gate_mask`, `node_ball_confidence`, `suspicion_auroc`, `gated_noisy_precision/recall`, and clean ratio should remain diagnostics。
- random-size matched assignment is a necessary control。
- no-ball confidence ranking is a strong baseline and must remain。
- ordinary GB can be used as a diagnostic baseline for future graph-aware suspicion。

# 6. Recommended Next NoisyGL Experiment

## NEXT-NGL-1: Graph-aware suspicion smoke

- Hypothesis: combining ordinary GB confidence with graph-neighborhood label conflict or warmup prediction inconsistency will detect noisy train nodes better than ordinary GB/no-ball/random。
- Change only: add graph-aware suspicion score; keep GCN, noise protocol, split, warmup epoch and evaluation unchanged。
- Minimal dataset: Cora uniform 0.4 seed=3000。
- Controls: GCN, ordinary GB reweight, no-ball reweight, random-size matched reweight。
- Success criterion:
  - graph-aware gate precision/recall exceeds ordinary GB and no-ball controls；
  - selected training run has best checkpoint from post-GB phase；
  - test acc exceeds GCN and controls。
- Failure criterion:
  - best checkpoint remains warmup；
  - test acc ties GCN；
  - no-ball or ordinary GB matches graph-aware gate。
- Expand only if smoke passes: then run Cora uniform 0.4, CiteSeer uniform 0.2, PubMed uniform 0.4 with seeds 3000-3004。

# 7. Paper Claim Boundary

- Supported: ordinary GB can provide a diagnostic suspicious-node signal in one protocol-matched Cora smoke。
- Not supported: ordinary GB reweight improves noisy-label GCN training。
- Not supported: ordinary GB works across Cora/CiteSeer/PubMed。
- Not supported: gate precision alone proves denoising effectiveness。
- Safe wording: In NoisyGL, ordinary GB reweight did not improve GCN in the smoke test, but its protocol-matched gate diagnostics can serve as a baseline for stronger graph-aware suspicion mechanisms。

# 8. Verification

- Unit tests: `python -m unittest discover -s tests` passed with 7 tests。
- Static check: `py_compile` passed for `diagnose_orgingb_gate_scan.py`, `soft_targets.py`, `OrginGB_GCN_Predictor.py`, `single_exp.py`, `total_exp.py`, and `test_orgingb_tools.py`。
- Protocol-matched scan output: 126 rows。
- Evidence files exist under `external/NoisyGL/runs/orgingb_gate_scan_protocol_matched_20260603/`。

# 9. NGL-ORG-02: ReliableGB-GCN reciprocal-kNN reweight (2026-06-07)

## 9.1 Direction Summary

- Target problem: 在 NoisyGL/GCN 宿主上，把 ordinary GB 换成 reciprocal mutual-kNN **reliable** 粒球 + 保守 reweight，并修复 OrginGB 的 `best_phase=warmup` 失败（周期重建球 + EMA hidden），验证可靠粒球去噪能否进入优化轨迹并超过 GCN/对照。
- GB role: weak local reliability signal driving conservative sample reweight。Clean labels diagnostics/eval only。
- Decision: **no-go on GCN host → 转 S2（label-free 对比表示）**。

## 9.2 Main Related Files

- Code: `utils/reliable_gb.py`（reciprocal mutual-kNN 构球，与 `ordinary_gb` 同 payload）、`predictor/ReliableGB_GCN_Predictor.py`（periodic rebuild + EMA）。
- Configs: `config/reliablegb_gcn_reweight{,_random,_noball}/*_{cora,citeseer}.yaml`。
- Tests: `tests/test_reliable_gb.py`（coverage/purity/determinism/random-match，4/4 pass；连 orgingb 共 11/11）。
- Report: `runs/reliablegb_s1_20260607/s1_report.md`。
- Archived buggy CSV: `runs/reliablegb_s1_20260607/reliablegb_gcn_diagnostics_BUGGY_noball.csv`。

## 9.3 Result Summary (Cora uniform 0.4 seed=3000, corrected controls)

| Method | warmup | Test | best_phase | gated precision | suspicion_mode |
|---|---:|---:|---|---:|---|
| gcn | – | 0.7050 | – | – | – |
| reliablegb_gcn_reweight | 30 | 0.7050 | warmup | 0.5000 | gb |
| reliablegb_gcn_reweight_noball | 30 | 0.7050 | warmup | 0.5000 | no_ball |
| reliablegb_gcn_reweight_random | 30 | 0.7050 | warmup | 0.4286 | gb(random_control) |
| reliablegb_gcn_reweight | 3/5 | 0.6970 | gb | – | gb |

## 9.4 Control-Bug Correction (peer-review finding, important)

- 第一版 no-ball 对照失效：`config/reliablegb_gcn_reweight_noball/*` 写了 `suspicion_mode: gb`，被 `_conf_value` 用来覆盖 predictor 类属性 `suspicion_mode="no_ball"`，导致 no-ball 实际又跑了一遍 real GB（旧 CSV `suspicion_mode=gb`、precision 与 real 相同）。
- 修复：config 改 `suspicion_mode: no_ball` 后重跑。random 对照不受影响（走 `random_control=True` 类属性，无 config 键覆盖）。
- 修正后结论从被污染的 “reliable ≡ no-ball” 变为有效对照下的 “real-GB gated precision 0.50 ties valid no-ball 0.50, only marginally > random 0.43”。证据更硬。

## 9.5 Root Cause Of No-Go (observed risk, not universal theorem)

- Cora uniform 0.4 seed3000：GCN val-acc 约在 epoch 6 见顶（0.442），随后随记忆 40% 噪声标签下降（epoch 30 时 train≈0.99、val≈0.37）。warmup=30 时构球建在已 noise-memorized 表示上 → `best_phase=warmup`，与 OrginGB 同因。
- 把 warmup 提前到 3/5：构球进入轨迹（`best_phase=gb`），但 test 不更好（0.697），且 reliable 仍不超过 no-ball。
- 结论：瓶颈不是球质量/去噪公式/时机，而是**监督 GCN 表示本身在标签噪声下不可靠**。

## 9.6 Anti-Repetition (append)

| Pattern | Why It Failed | Evidence | Required New Evidence Before Retry |
|---|---|---|---|
| reliable mutual-kNN GB reweight on supervised GCN hidden | test ties GCN, best_phase warmup or no gain; gating ties valid no-ball | `runs/reliablegb_s1_20260607/s1_report.md` | Build the same reliable balls on a label-free contrastive representation (S2a) |
| no-ball / random control configured via `gb:` block keys | `_conf_value` overrides class attrs → control silently re-runs real-GB | buggy vs corrected CSV | Verify `suspicion_mode` / control field actually differs in diagnostics CSV |

## 9.7 Paper Claim Boundary (append)

- Supported: reliable mutual-kNN ball construction is well-formed (full coverage, pure balls); but on the supervised GCN representation its denoising ties GCN and ties a valid no-ball control (single-seed smoke)。
- Not supported: reliable-GB reweight improves noisy-label GCN training。
- Safe wording: In this Cora uniform 0.4 seed-3000 smoke, reliable-GB reweight on supervised GCN hidden gave no training gain; S1 is no-go; the next step builds the same reliable balls on a label-free contrastive representation。

## 9.8 Turning Point Recorded

从 “图感知粒球” 正式转向 “**对比表示上的 reliable mutual-kNN 粒球去噪**”。理由见根目录 `02_experiment_history_and_lessons.md` 的 DCGB v2.8 / S1 turning-point 章节：图感知在 Cora/CiteSeer 拓扑增量近似为 0；真正稳定的正信号是 reciprocal mutual-kNN 构球；监督表示不可靠促使转向 label-free 对比表示（S2a）。
