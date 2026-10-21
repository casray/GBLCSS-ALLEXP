# 带噪监督下的预算约束粒球图粗化

研究快照：2026-10-19。工作分支：`codex/budget-gb-coarsening-20261019`。

本目录保存新论文方向的设计、文献依据、理论检查和首轮成本冒烟。完整新算法尚未实现；当前没有确认效率优势或抗噪性能提升。

## 研究问题与方法草案

在有限计算预算下，如何选择粒球粗化的粒度，使训练少拟合随机错标，同时保留分类需要的属性与关系差异？粒球作为实际训练单元，用共享表示和预测压缩计算；标签评分用于分配有限的细分预算。

候选流程：无标签属性/关系坐标 → 有规模上限的候选树 → 表示保真项与带证据继承的留一不纯度选择切面 → 构建球级特征、`P_c=RPH`及训练标签软目标 → 粗图训练 → 按预定部署方式评估。

留一不纯度不包含自配对；少于两个训练标签的叶球继承最近一个有足够标签的祖先评分。固定树、训练位置与分区下，对称独立噪声的期望关系成立。该关系不消除用标签挑切面的选择偏差，也不保证实际训练准确率提高。

## 推荐阅读顺序

1. [文献交叉复核后的设计](research/literature_reassessment_20261019/05_DESIGN_REASSESSMENT.md)。
2. [留一不纯度的数学复核和反例](research/literature_reassessment_20261019/07_LOO_IMPURITY_REVIEW.md)。
3. [实验计划草案](experiments/budget_gb_coarsening_20261019/PLAN.md)。
4. [E0冒烟记录](experiments/budget_gb_coarsening_20261019/e0_cost/SMOKE_NOTES.md)。
5. [对E0、证据继承和实验口径的最新审查](research/literature_reassessment_20261019/08_CLAUDE_E0_SMOKE_REVIEW.md)。

PLAN中的历史状态“未实现、未训练”已被E0冒烟推进；其测量协议及oracle解释仍待按最新审查修订。这里保留原始文档，不将建议伪装成已完成的修复。涉及不同版本说法时，应结合日期和最新审查理解。

## 内容与状态

| 路径 | 内容 |
|---|---|
| `research/literature_reassessment_20261019/` | 五个领域36篇核心文献的卡片、检索与阅读范围记录、设计、理论和实验审查 |
| `experiments/budget_gb_coarsening_20261019/theory/` | 普通Gini、留一不纯度与证据继承的有限枚举 |
| `experiments/budget_gb_coarsening_20261019/e0_cost/` | 真实运行的冒烟代码、48条seed0原始结果、日志及结论 |
| `GB-LCSS_GitHub/src/gblcss/` | 冒烟导入所需的旧方法辅助模块快照；不等同完整旧项目 |
| `SNAPSHOT_MANIFEST.json` | 本次新增文件的SHA-256清单及来源范围 |

E0覆盖CS、CoraFull、Physics，clean/uniform0.4与三个目标压缩预算。u0.4下CS/Physics的缓存特征模型六格增益约0.46～0.80pp，但只有一个种子，不能推断稳定性。CoraFull存在明显压缩损失。显存与Physics计时暂不能用于正式效率结论。

最新审查指出：粗GCN选模方式与主测试方式不一致；当前标作SIGN的模型实际上是缓存多跳特征拼接MLP；0.3目标预算未全部达到；正式新方法的完整候选树成本尚未核算。代码与原始结果原样保存以便追溯。

## 运行理论检查

先进入本README所在目录。以下脚本不读取图数据，不训练模型；其中两个验证脚本会更新其同目录的数值JSON。

```powershell
python research/literature_reassessment_20261019/validate_math_notes.py
python research/literature_reassessment_20261019/validate_loo_impurity.py
python experiments/budget_gb_coarsening_20261019/theory/check_inherited_loo.py
```

文献清单检查（重新生成汇总文档和登记JSON）：

```powershell
python research/literature_reassessment_20261019/assemble_review.py
```

## E0环境与数据

依赖见 [requirements.txt](experiments/budget_gb_coarsening_20261019/requirements.txt)。原机器使用RX 7700 XT、PyTorch 2.9.1+ROCm 7.2.1、PyG 2.7.0。PyTorch需按本机GPU和操作系统安装对应版本；依赖文件不自动安装原机器的特殊GPU运行时。

冒烟脚本保留了原项目的数据路径约定：Git仓库根目录下的 `external/NoisyGL/data/`，以及 `gb_graph_aware_minival_6_1/data/CoraFull/`。本次快照不包含图数据、模型权重或PDF全文；运行者需准备相应PyG处理文件，或明确修改脚本中的 `DATA_FILES`。脚本要求可用的PyTorch CUDA接口（原ROCm环境也使用该接口）。

仅在满足数据和GPU条件后，才可执行原始冒烟命令：

```powershell
python experiments/budget_gb_coarsening_20261019/e0_cost/run_e0.py --datasets coauthorcs corafull physics --seeds 0 --noises uniform:0.0 uniform:0.4 --ratios 0.05 0.1 0.3 --out experiments/budget_gb_coarsening_20261019/e0_cost/local_rerun.jsonl
```

此命令复跑的是保留问题的历史冒烟实现，不是修复后的正式协议。下一步优先修复测量/选模，重新跑统一协议seed0/1/2，再进行少量clean-label切面诊断；不把O2称为分类上界。

## 文献与结论边界

文献卡片记录来源、阅读范围、机制和未核事项；登记检查不能替代逐项学术判断。已区分实验事实、合理解释、待验证假设和负结果。统计经典估计量不是新发明；潜在贡献在带噪图粗化问题中的粒度分配、计算组织及可验证收益。
