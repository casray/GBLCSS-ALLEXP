# 执行计划：图领域专用去噪基线补跑，定稿 Table C（Phase MLP-18）

日期：2026-07-08
执行者：Codex
定位：**为论文新增的 Table C（图领域专用标签噪声鲁棒方法定位表）补齐当前协议下的基线数字，不是主表变更、不是新机制。** Table C 与主表 A 严格分离（MLP-7 铁律不变：图领域专用方法从不进同骨干主表 A），目的是用资源匹配的方式回答"与真正的图领域去噪 SOTA 相比，LCSS 处于什么位置"。

## 0. 背景与现状

- **★2026-07-08 新发现，本计划范围因此扩大：现有 NRGNN/TFR 数字本身已被污染，不能只补新方法。** 核查代码历史发现 `utils/labelnoise.py` 的 `label_process()` 函数里的 `eligible_mask` 参数是本仓库**尚未提交的修改**（最后一次提交 `7a6fc80`，2026-06-18，当时的 `label_process` 没有这个参数，噪声默认加在全图所有节点上，包括 val/test）。而 `runs/phase24_nrgnn_xh_20260616/nrgnn_xh_5seed.csv` 与 `runs/tfr_comparison_20260616/tfr_5seed.csv`——现有 `appendix_C_graph_dependent_references.md` 里 NRGNN/TFR 数字的唯一来源——生成于 2026-06-16，**早于** `eligible_mask` 出现。用数字验证：Cora 全图 2708 节点，这批 CSV 里 `modified_count`（全图被改标签数）= 1030 ≈ 2708×0.4，而 `modified_train_count`（train 内被改数）= 189；若噪声真只加在 20% train 节点，这两个数字理应相等（`modified_mask` 必然是 `eligible_idx` 的子集）。1030 远大于 189 且精确匹配"全图 40%"，**证实这批数据的 val/test 标签也被污染**，且 `Base_Predictor.train()` 内部早停用的是 `self.evaluate(self.noisy_label, self.val_mask)`，被污染的验证信号会影响模型选择，即便最终 `test()` 用 `self.clean_label`，整条链路也不干净。**这批数字与本项目其余数字（全部经 `run_ssl_swap_downstream.py --noise_scope train` 产出，`eligible_mask` 传递已核实无误）不在同一协议下，不可比、不可引用**——`paper_final_zh_v2_20260707.md` 第 324 行"Cora 80.60 对 81.07"这句具体数字对比需要先撤下或改为不带精确数字的定性表述，等本计划产出干净数字后再回填。
- **因此本计划范围扩大为：NRGNN、TFR 也要用干净协议重新跑**（不只是新增 RNCGLN/RTGNN/CLNode/CRGNN），全部统一走下面红线 0 规定的安全入口。
- 原有 `paper_assets_20260706/tables/appendix_C_graph_dependent_references.md` 只覆盖 **Cora / CiteSeer / 部分 CoauthorCS，且只有 uniform 0.4 一个噪声设置**——这个覆盖范围本身没问题，但底层数字需要全部重新产出。
- **RNCGLN 的旧数字不可用**：2026-06-02/03 的早期 baseline 筛查（`baseline_selection_diagnostic_review_20260603.md`）跑过 RNCGLN/RTGNN/CLNode/CRGNN/PIGNN，但用的是 NoisyGL 默认 split（如 PubMed train 60/val 500/test 1000），种子为 3000-3004，与当前主线协议（`split_type=percent, train_percent=0.2, val_percent=0.1, test_percent=0.7`，种子 0-4）不兼容。`related_work_cut_notes_20260702.md` 已明确标注这批数字"不可作对比数字引用"。本计划的核心任务就是在当前协议下重新产出这几个方法的可引用数字。
- 已确认的工程约束：RNCGLN 默认实现构造稠密 `x @ x.T` 节点对矩阵，PubMed（N=19717）在旧协议下已经 OOM（12GB GPU）。CoauthorCS（N≈18333）与 PubMed 同量级，**大概率同样 OOM，需在 5-seed 投入前用单 seed 验证**，不要假设它能跑通。Cora（2708）/CiteSeer（3327）/Amazon-Photo（7650）在旧协议下曾跑通，规模上应无问题，但仍需在新 split 下重新做健全性检查。

## 0.1 红线（预注册）

1. **进 Table C（含扩展版 Appendix C），绝不进主表 A**——这些方法在训练与推理端都使用更多资源（全图注意力/邻接依赖），与主表 A"同骨干、同资源"的比较前提冲突，MLP-7 铁律照旧适用。
2. **忠实适配优先**：核心机制照 NoisyGL 仓库自带实现跑，不新增、不删减原方法的组件；唯一允许的协议性替换是数据集 split（改为 percent 0.2/0.1/0.7）、噪声注入与种子（改为我们的噪声协议与 seeds 0-4）。若某方法的默认超参在新 split 下明显不收敛，允许一次预声明的超参网格重跑并记录两版结果，不允许静默调参到赢。
3. **协议显式声明**：`--split_type percent --train_percent 0.2 --val_percent 0.1 --test_percent 0.7`；噪声只加在 train（`noise_scope=train`）；ROCm `cuda:0`；同 seed 同 split，与主表 A、表 B 的协议完全一致。
4. **单 seed 健全门先行，全部方法全部数据集都要过**：status=ok、无 NaN/Inf、精度不低于同协议 GCN(X) 参照太多（若一个方法在某数据集上明显崩溃或报错，记录原因、跳过该格，不要为了凑数字勉强跑通）。**OOM 是预期中的合法结果，不是失败**——如实记录为"该方法在此规模下不可行"，这本身是论文可以引用的一条复杂度证据（呼应第 3.3 节 kNN 复杂度对比的论述方向）。
5. **判读纪律**：任何格照实进表，包括我们的方法在某格落后的情况；差距 <1×std 记为 comparable，不写 significant。
6. 本计划不改主表 A、不改主线 claim、不新增机制；只填 Table C 的行。

## 0.2 方法与优先级

已在 `scripts/run_standard_downstream.py` 的 `PREDICTOR_MODULES` 注册、可直接调用（无需新增代码）：

- **必跑（核心六个，含重跑）**：`nrgnn`、`nrgnn_xh`、`tfr`、`rncgln`、`rtgnn`、`clnode`、`crgnn`。前三个是重新产出干净协议数字（替换掉被污染的 2026-06-16 旧数据），后四个是新补的图领域基线（自纠正/双网络互校/课程学习/对比正则四种不同机制），且旧协议下都已验证过可运行（RNCGLN 的 PubMed/CoauthorCS 除外）。
- **可选、时间允许再做**：`pignn`。2026-06-03 的记录显示它单次 run 耗时长，5-seed 全量在当时被搁置；若前四个方法的时间预算允许，补一个 uniform 0.4 的 5-seed 作为补充行，不强制。
- **不新增**：DND-NET 未在本仓库实现（无对应 predictor），不在本计划范围内，仍按现有正文处理（引用文献数字，标注协议不可直比，不产出我们自己的复现数字）。

## 1. 阶段 A —— 单 seed 健全性筛查（seed 0）

对每个方法，在 **5 个数据集 × uniform 0.4**（代表性噪声设置）先跑单 seed，确认：
- 能否在当前 split 下跑通（尤其关注 RNCGLN 在 CoauthorCS/PubMed 上是否 OOM）；
- 精度是否显著低于同协议 GCN(X) 参照（如低于 GCN 5pp 以上且非预期设计，视为实现或超参问题，先排查再决定是否进入 5-seed）。

```
A1: {nrgnn, nrgnn_xh, tfr, rncgln, rtgnn, clnode, crgnn} × {cora, citeseer, pubmed, coauthorcs, amazonpho} × uniform0.4 × seed0 = 35 run
```

**★红线 0（本次修订新增，最高优先级）：命令入口必须是 `scripts/run_standard_downstream.py`，绝不使用 `total_exp.py` / `single_exp.py`。**

原因（2026-07-08 代码核查发现，务必让 Codex 知悉）：`total_exp.py`/`single_exp.py` 内部调用 `label_process(...)` 时**没有传 `eligible_mask` 参数**，默认值 `eligible_mask=None` 会让噪声污染全图所有节点，包括 val 和 test；而 `Base_Predictor.train()` 内部的验证/早停用的是 `self.noisy_label`（`evaluate(self.noisy_label, self.val_mask)`），意味着模型选择信号本身会被污染的验证标签污染，`predictor.test()` 虽然最终用 `self.clean_label`，但已经在被污染的验证集上做过早停/超参选择，整条链路都不干净。这与本项目主表/表 B/教师蒸馏使用的协议（噪声仅限 train，val/test 保持干净）完全不同，产出的数字不可比、不可引用。

`scripts/run_standard_downstream.py` 是本项目一贯用于产出可引用数字的安全入口：`--noise_scope` 为**必填**参数（`required=True, choices=["train","all"]`），内部显式 `eligible_mask=data.train_masks if args.noise_scope=="train" else None`，且 `PREDICTOR_MODULES` 字典已经注册了 `rncgln`/`rtgnn`/`clnode`/`crgnn`/`pignn`（与我们自己方法用的是同一个入口、同一套 fingerprint 校验），不需要新增代码，直接调用即可。

命令模板（唯一允许使用的形式）：

```
python scripts\run_standard_downstream.py \
  --datasets cora,citeseer,pubmed,coauthorcs,amazonpho \
  --methods nrgnn,nrgnn_xh,tfr,rncgln,rtgnn,clnode,crgnn \
  --seeds 0 \
  --noise_type uniform --noise_rate 0.4 \
  --noise_scope train \
  --split_type percent --train_percent 0.2 --val_percent 0.1 --test_percent 0.7 \
  --device cuda:0 \
  --output runs\mlp18_graphdomain_tablec_20260708\smoke_seed0.csv
```

执行前的强制检查（Codex 必须在跑批前确认，任一不满足则先修正代码/命令再跑）：
1. 命令行确实包含 `--noise_scope train`（该参数必填，若省略框架会直接报错拒绝执行，属于正常保护，不要绕过）。
2. 抽查输出 CSV 中的 `modified_val_count`/`modified_test_count` 字段（`run_standard_downstream.py` 的行结构含有这两列）——必须为 0；若非 0，说明协议被破坏，立即停止，不得继续扩量或汇总。
3. 不得改用 `total_exp.py`/`single_exp.py` 作为"更省事"的替代，即使 PREDICTOR_MODULES 里方法名相同——两者的噪声污染范围不同，产出数字不可互换。

预期结果记录表（对每个 method × dataset 组合填写）：status（ok / oom / nan / 明显欠力）、test_acc、`modified_val_count`/`modified_test_count`（须为 0）、备注。**RNCGLN 在 coauthorcs/pubmed 上如果 OOM，直接标注不可行并停止对该组合的后续尝试，不重跑、不缩批次强行凑通**（缩批次/降精度等同于改变原方法，违反红线 2）。

## 2. 阶段 B —— 5-seed 主跑（健全门通过的组合）

对阶段 A 通过健全门的每个 method × dataset 组合，扩展到全部 5 个噪声设置 × 5 seeds：

```
B1: 通过健全门的组合 × 5 设置(u0.2/0.4/0.6, pair0.4, instance0.4) × seeds 0-4
```

命令模板同上（仍是 `run_standard_downstream.py`），`--seeds 0,1,2,3,4`，`--noise_type`/`--noise_rate` 按 5 个设置展开为 5 次独立调用（该脚本的 `--noise_type`/`--noise_rate` 均为单值参数，不支持 nargs 多值，需要每个噪声设置单独起一次命令，输出到不同 CSV 后再合并）。

## 3. 汇总与交付物

- 汇总脚本产出格式需与现有 `appendix_C_graph_dependent_references.md` 兼容（列：dataset, method, test, noise_type, noise_rate, seed, source_file，另附聚合行 mean/n/std），追加写入同一份表，不新建平行格式。
- 额外产出一份简短的**可行性记录**（哪些 method×dataset 组合因 OOM/不收敛未纳入，原因各一句话），供正文 Table C 脚注与附录 C 说明引用，不要静默丢弃组合。
- 我们自己一侧的数字不需要重新跑：CoauthorCS/Amazon-Photo 的 LCSS 蒸馏学生精度已有（Table 2 现有 93.75/90.49），Cora/CiteSeer/PubMed 的教师与蒸馏学生数字已有（Table 3 现有 6 格 + 现有 appendix C 的 teacher 行）；本计划只补图领域专用竞品那一侧。
- 完成后交回：更新版 `appendix_C_graph_dependent_references.md`（或新文件 `appendix_C2_...`，注明与旧文件的关系）+ 可行性记录 + 一句话结论（哪些数据集/设置下 LCSS 蒸馏学生对图领域专用 SOTA 是持平/领先/落后），供撰写 Table C 正文与讨论段落使用。
