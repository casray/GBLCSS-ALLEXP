# GAGB v2.1 Fair-Control Repair Minival Report

## 0. Validity Note

本报告基于修复后的结果目录：

- `reports/gagb_v2_1_fair_repair_fixed_20260602/`

旧目录 `reports/gagb_v2_1_fair_repair_20260602/` 不作为结论依据。原因是发现并修复了一个控制组路由 bug：`_method_grouping()` 先匹配 `V2_TARGET_METHODS`，导致 `gagb_shuffled_alpha09_matched`、`gagb_shuffled_replace_matched`、`khop_only_alpha09`、`khop_only_replace` 被提前路由到 GAGB grouping，错误地与 GAGB 修复方法完全相同。

修复内容：

- 新增单元测试 `test_repair_control_methods_route_to_requested_grouping`。
- 调整 `scripts/run_noisy_classifier.py::_method_grouping()` 判断顺序，让 shuffled / khop 等具体 repair controls 先于泛化的 V2 target 分支匹配。
- 重跑 20 个 frozen classifier runs，不重训 GRACE，不重提 embedding。

## 1. Research Hypothesis

本轮 v2.1 只检验一个问题：

> v2 target repair 中 `alpha09` / `replace_qmix` 的有效性，是否依赖真正的 GAGB grouping，而不是任意弱扰动、随机 grouping、kmeans grouping、topk neighborhood 或 khop-only topology grouping。

如果 GAGB 修复方法只比普通 `gb` 或 v1 `gagb` 好，但不能超过 matched controls，则不能主张 GAGB-specific value。

## 2. Role of Granular Ball

本轮仍把粒球视为保守的局部 soft-label generator：

- 不改 GRACE。
- 不改 GAGB 构球。
- 不做 view generation。
- 不做 prototype pulling。
- 不把 same-ball membership 当 hard positive label。

GAGB 只参与 target construction：先由 graph-aware grouping 得到局部节点组，再融合 noisy label distribution 与 noisy-train teacher prediction，最后通过 conflict gate 只修改高冲突训练节点。

## 3. Methods and Controls

主方法：

- `gagb_qb_conflict_gate_alpha09`
- `gagb_qb_conflict_gate_replace_qmix`

核心对照：

- `gb`
- `gagb`
- `khop_only`
- `gagb_qb_conflict_gate_alpha03`
- `gagb_shuffled_alpha09_matched`
- `gagb_shuffled_replace_matched`
- `kmeans_gagb_matched_alpha09`
- `kmeans_gagb_matched_replace`
- `topk_gagb_train_matched_alpha09`
- `topk_gagb_train_matched_replace`
- `khop_only_alpha09`
- `khop_only_replace`

注意：`topk_gagb_train_matched_*` 是 train-node effective group size matched 的 neighborhood control，不是严格 partition。它用于检验“局部邻域 target repair”是否足以解释结果。

## 4. Configuration

固定配置：

- Datasets: `Cora`, `CiteSeer`
- Split: `ratio_train0p2_val0p1_split0`
- Noise: symmetric label noise
- Noise rates: `0.2`, `0.4`
- Seeds: `0,1,2,3,4`
- Mode: frozen GRACE embedding
- Classifier epochs: `200`
- Patience: `30`
- Base alpha: `0.3`
- Teacher: linear teacher on frozen embedding, trained only with noisy train labels
- Teacher epochs: `100`
- Teacher lr: `0.01`
- Teacher weight decay: `5e-4`
- Teacher temperature: `1.0`

Commands:

```powershell
python tests\run_unit_tests.py

python scripts\run_noisy_classifier.py ... --methods gb gagb khop_only gagb_qb_conflict_gate_alpha03 gagb_qb_conflict_gate_alpha09 gagb_qb_conflict_gate_replace_qmix gagb_shuffled_alpha09_matched gagb_shuffled_replace_matched kmeans_gagb_matched_alpha09 kmeans_gagb_matched_replace topk_gagb_train_matched_alpha09 topk_gagb_train_matched_replace khop_only_alpha09 khop_only_replace

python scripts\summarize_gate_results.py --results-root results --output-dir reports\gagb_v2_1_fair_repair_fixed_20260602
```

## 5. Success and Failure Criteria

Hard criteria:

- `gagb_qb_conflict_gate_alpha09` or `gagb_qb_conflict_gate_replace_qmix` has `NCR > 0` and `CDR < 0.03`.
- At least 3/4 settings beat `gagb_qb_conflict_gate_alpha03`.
- At least 2/4 settings beat shuffled matched control.
- At least 2/4 settings beat kmeans and topk matched controls.

Strong criteria:

- At least 3/4 settings beat shuffled matched controls.
- At least 3/4 settings beat kmeans and topk matched controls.
- Accuracy gain reaches at least about `0.3 std` in at least 2 settings.
- No serious correction-by-class bias.

The last strong criterion was not evaluated because correction-by-class diagnostics are not yet implemented.

## 6. Main Results

Mean test accuracy / NCR / CDR:

| Setting | Method | Acc | NCR | CDR |
|---|---:|---:|---:|---:|
| Cora 0.2 | alpha03 | 0.8039 | 0.0000 | 0.0000 |
| Cora 0.2 | alpha09 | 0.8083 | 0.2519 | 0.0143 |
| Cora 0.2 | replace | 0.8090 | 0.2519 | 0.0143 |
| Cora 0.2 | shuffled alpha09 | 0.8038 | 0.0000 | 0.0005 |
| Cora 0.2 | kmeans alpha09 | 0.8095 | 0.1167 | 0.0023 |
| Cora 0.2 | topk alpha09 | 0.7990 | 0.6148 | 0.0806 |
| Cora 0.4 | alpha03 | 0.7949 | 0.0000 | 0.0000 |
| Cora 0.4 | alpha09 | 0.7907 | 0.0811 | 0.0025 |
| Cora 0.4 | replace | 0.7886 | 0.0811 | 0.0025 |
| Cora 0.4 | shuffled alpha09 | 0.7931 | 0.0000 | 0.0000 |
| CiteSeer 0.2 | alpha03 | 0.7058 | 0.0000 | 0.0000 |
| CiteSeer 0.2 | alpha09 | 0.7087 | 0.1444 | 0.0255 |
| CiteSeer 0.2 | replace | 0.7064 | 0.1444 | 0.0255 |
| CiteSeer 0.2 | shuffled alpha09 | 0.7064 | 0.0000 | 0.0004 |
| CiteSeer 0.4 | alpha03 | 0.6982 | 0.0000 | 0.0000 |
| CiteSeer 0.4 | alpha09 | 0.6981 | 0.0617 | 0.0075 |
| CiteSeer 0.4 | replace | 0.6997 | 0.0617 | 0.0075 |
| CiteSeer 0.4 | shuffled alpha09 | 0.6979 | 0.0000 | 0.0000 |

Paired setting wins:

| Comparison | alpha09 wins | replace wins |
|---|---:|---:|
| vs alpha03 | 2/4 | 3/4 |
| vs gb | 3/4 | 3/4 |
| vs gagb | 3/4 | 3/4 |
| vs shuffled matched | 3/4 | 3/4 |
| vs kmeans matched | 3/4 | 2/4 |
| vs topk matched | 4/4 | 3/4 |
| vs khop repair | 2/4 | 3/4 |

## 7. Interpretation

Supported findings:

- Target repair is no longer purely inert: `alpha09` and `replace_qmix` both produce positive NCR in all 4 settings.
- Clean damage remains bounded: CDR is below `0.03` in all 4 settings for both main repair methods.
- Compared with `alpha03`, `replace_qmix` passes the 3/4 hard gate; `alpha09` only wins 2/4.
- Compared with shuffled matched controls, both main methods win 3/4 settings after the routing bug is fixed.
- Compared with topk matched controls, both main methods are generally better on accuracy, but topk has much higher NCR and much higher CDR. This means topk can detect/modify more noisy nodes but damages many clean nodes.

Weakened or unsupported findings:

- Cora 0.4 is a clear weak setting: both `alpha09` and `replace_qmix` lose to `alpha03`, `gb`, and shuffled control.
- GAGB-specific value is only partially supported. `alpha09` beats kmeans 3/4, but `replace_qmix` beats kmeans only 2/4. Against khop repair, `alpha09` only wins 2/4 and `replace_qmix` wins 3/4.
- Gains are mostly small relative to standard deviation. This is not yet strong enough for a main paper claim.
- Shuffled controls often have near-zero target changes, so they are useful sanity controls but not fully intervention-strength matched controls.
- Correction-by-class bias was not evaluated, so strong success is incomplete.

## 8. Decision

Decision: continue, but narrow the claim.

The current result is mechanism-positive and partially method-positive:

- `replace_qmix` passes the hard gate.
- `alpha09` improves NCR/CDR but does not pass the alpha03 comparison gate.
- The evidence supports "conservative conflict-gated target repair can produce useful noise correction with bounded clean damage."
- The evidence does not yet support a strong claim that GAGB grouping itself is consistently superior to all fair matched grouping alternatives.

Next experiment should not add view generation or new GCL losses yet. The next step should diagnose and improve the gate/group quality:

- Add correction-by-class diagnostics.
- Add intervention-strength-matched shuffled controls, because current shuffled controls often change almost no nodes.
- Analyze why Cora 0.4 fails.
- Compare GAGB vs kmeans/khop on gated-node purity and target correctness, not only final accuracy.
- Consider calibrating conflict gate thresholds per dataset/noise rate only after documenting fixed thresholds clearly.

