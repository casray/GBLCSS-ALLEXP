# 99 Session Handoff（2026-07-01）

## 最新追加：2026-10-21 边界诊断与粗化复盘完成

入口`gb_c1_ratio_success_20260609/results_ratio_c1_20260609/experiments/r2_boundary_diagnostic_20261021/综合结论与接续.md`。新训练0，无运行中任务，旧稿/模型/源码不动。三图四条件seed0，20656训练节点—条件，237源/11测试/99相关394分箱复核通过。

径向距离正rho且端组差clean2/3、带噪3/9，不支持通用球心边界解释；跨球近邻比例clean3/3、带噪7/9，有有限描述性支持但弱/部分球内反向，非因果。GB-on-R2原完整推进仍false，不据此加门控救援。

预算粗化1080旧记录数值复算成立；GCN球训练相对匹配平滑B+1.148pp、相对同骨干LCSS+.565(6/15正)，缓存MLP+.346/−1.218。保留条件性高噪声训练组织线索；R3−.033、O2+.281、16clean成本联合门0通过，冷C/A1.22–2.17。不同架构/原图部署，不与R2混排。旧表缺省臂恢复保留failure/ENGINEERING，最终auditpassed。没有启动GCE组合/c组/多seed或传统GBSVM；下一步需另行授权。

## 最新追加：2026-10-21 R2机制对照完成

入口`gb_c1_ratio_success_20260609/results_ratio_c1_20260609/experiments/r2_mechanism_controls_20261021/综合结论与接续.md`和目录交接。156新分类/三份打乱H预训练/三随机encoder完成，无待续训练，旧投稿未改。三图四条件seed0，同旧R2/100轮冻结分类。

带噪均值CE56.602、GB57.921、GCE59.710、MASS58.049、KNN59.134、KM59.277、随机55.720%。GB对GCE−1.789pp(1/9正)、MASS−.128(4/9)、KNN−1.213(0/9)、KM−1.355(1/9)；只对随机通过+2.202(8/9)。clean对CE仍−1.478/−1.246/−2.360pp，GB完整推进门false，停止当前固定GB-on-R2分支，不自动扩seed/解冻/加深或拼模块。

真实CE比随机/打乱分别+34.942/+39.298pp均9/9正，保留R2骨干；不能唯一归因拓扑。MASS匹配逐节点实际修正幅度，仍用GB强度；KM只匹配区域数，KNN精确票数匹配。BOOTdetach当前预测梯度等于0.8CE。172源、48目标、156重载、24旧R2精确锚、独立156指标/144差值/96近邻均核验，10测试通过，正式网格56.474s不含旧SSL/R2。单seed无std/显著性，GCE/KNN也有CoraFull clean损害。

## 最新追加：2026-10-21 表示迁移与鲁棒监督已完成

入口`gb_c1_ratio_success_20260609/results_ratio_c1_20260609/experiments/representation_supervision_20261021/综合结论与接续.md`及目录交接；无剩余训练。三主图、四条件、完整seed0，144主矩阵+24旧目标+24H参考。R2对齐100+冻结encoder/BN分类100，R0/R1监督200；原H与主实验分区固定。

带噪CE/LCSS/small-loss/组合：R0 35.515/43.420/39.451/47.356，R1 38.022/47.699/44.375/51.321，R2 56.602/57.921/50.658/53.896%。R2 LCSS对自身CE+1.320pp（8/9正），但clean三图-1.478/-1.246/-2.360pp，完整推进门未过；R2组合对逐格最好单项-4.062pp。

14测试、12协议、52冻结源、192检查点与独立指标核查通过。U.4 R2/LCSS75.673/71.005/47.333，对同seed SimMLP75.620/67.139/40.166；Cora仅差一个节点，不称SOTA。先读完整结论；表示/冻结因果、粒球独立作用、多seed仍未验证。旧投稿未改，不自动扩队列。

## 最新追加：2026-10-07 DIST 独立协议种子确认完成

优先读 `gb_c1_ratio_success_20260609/results_ratio_c1_20260609/experiments/gb_dist_protocol_seeds_20261007/实验结论.md`及同目录交接；方向历史41。
seeds1/2改变split/noise/teacher/student及分区，GRACE固定；20新ERASE教师、40学生、2回放全部完成。
固定DIST T1+区域lambda1平均增量+.120901pp（12/1/7），seed1+.210956、seed2+.030846，7/10格正。
预设>=.20pp门槛失败；固定epoch200+.134205也未达。58.62%净增量来自CiteSeer .6 seed1。
DIST自身已17/20达到或超过教师，加区域后仍17/20；不能把逼近教师归于粒球。
80学生重载差0；20教师最大概率差2.98e-7、全节点类别一致，初次严格断言与数值修正完整保留。
决策暂缓固定区域KL+DIST，不加模块、不自动继续实验；当前无待完成训练。投稿包未改。
下一窗口先读报告和原始表，再讨论具体失败机制；不要把两个种子的小增量写成稳定提升。

## 上一阶段：2026-10-07 强KD筛查已完成

优先读`gb_c1_ratio_success_20260609/results_ratio_c1_20260609/experiments/gb_strong_kd_screen_20261007/99_session_handoff.md`及同目录`实验结论.md`，方向历史40。
100新fit+20参考+2回放；全部seed0，固定ERASE/GRACE/GB及MLP。
区域增量KD+.375169、DIST+.009201、DGKD-T+.092660pp；验证选基线整体+.211091pp、7胜3负，gate刚过。
73.26%整体净增量来自CiteSeer普通KD两格，不能声称普遍改善强KD；DIST自身比KD+.731789pp。
8测试通过，120选中模型重载差0。当时建议的独立协议seed现已完成，结果见上；暂不加新模块。
下方为前一阶段记录，投稿包未改。

## 最新追加：2026-10-07 匹配区域数 k-means 控制已完成

后续区域蒸馏研究优先读：
`gb_c1_ratio_success_20260609/results_ratio_c1_20260609/experiments/gb_regional_kmeans_20261007/99_session_handoff.md`及同目录`实验结论.md`。
30次新KM训练、1次GB工程回放已完成，150新检查点重载概率差0，无剩余训练进程。
固定ERASE/split/noise/GRACEseed0，五数据集uniform0.4/0.6，学生seeds0/1/2；R0/R1参考复用上一轮统一密集矩阵数值实现。
平均GB−R0 +0.296753pp，KM−R0 +0.281181pp，GB−KM +0.015572pp；后者14胜1平15负、仅4/10格三seed均值为正。
通用区域gate通过，GB特有精度gate未通过。当时决策：修改定位，暂缓粒球独有优势claim；更强KD现已筛查（见上方），不同教师/独立协议seeds仍未验证。
方向历史见`experiment_history_and_lessons/02_experiment_history_and_lessons 39.md`。下面是旧历史，投稿包未更改。

## 最新补充：Phase MLP-5 β=0.5 标签噪声主表（2026-07-01）

用户要求按 `external/NoisyGL/docs/execution_plan_mlp_maintable_20260701.md` 开始实验。已完成 FINE / DivideMix-lite 的同 backbone 适配、配置、runner、summary、单测，并用 ROCm GPU 跑完整主表。

新增或更新文件：
- `external/NoisyGL/predictor/MLPNoiseBaselines_Predictor.py`
- `external/NoisyGL/config/mlp_fine/`
- `external/NoisyGL/config/mlp_dividemix_lite/`
- `external/NoisyGL/config/mlp_coteaching/mlp_coteaching_pubmed.yaml`
- `external/NoisyGL/config/mlp_elr/mlp_elr_pubmed.yaml`
- `external/NoisyGL/config/mlp_sce/mlp_sce_pubmed.yaml`
- `external/NoisyGL/scripts/run_mlp_maintable_20260701.ps1`
- `external/NoisyGL/scripts/summarize_mlp_maintable.py`
- `external/NoisyGL/tests/test_mlp_maintable.py`
- `external/NoisyGL/tests/test_mlp_noise_baselines.py`
- `external/NoisyGL/docs/execution_plan_mlp_maintable_20260701.md`
- `gb-direction-turning-point-2026-06.md`

实验输出：
- `external/NoisyGL/runs/mlp_maintable_20260701/mlp_maintable_report.md`
- `external/NoisyGL/runs/mlp_maintable_20260701/mlp_maintable_aggregate.csv`
- `external/NoisyGL/runs/mlp_maintable_20260701/mlp_maintable_paired_deltas.csv`
- 完整性：850/850 rows，missing=0，stderr 长度 0。

验证：
- `tests.test_mlp_noise_baselines tests.test_mlp_maintable tests.test_mlp_softdist tests.test_mlp_labelrate -v`：15 tests OK。
- `py_compile predictor/MLPNoiseBaselines_Predictor.py single_exp.py scripts/run_standard_downstream.py scripts/run_ssl_swap_downstream.py scripts/summarize_mlp_maintable.py`：通过。

主结论：
- Cora / CiteSeer 强成立。β=0.5 在两个数据集、五类噪声设置下相对经典 MLP 抗噪 baseline、FINE-adapted、DivideMix-lite-adapted 和 shuffled 全面占优。Cora：vs classic 25/25 胜，vs latest 10/10 胜，vs shuffled 5/5 胜；CiteSeer 同样 25/25、10/10、5/5 胜。
- PubMed 是明确边界。β=0.5 vs shuffled 5/5 胜，说明真实球结构仍有效；但对 small-loss / co-teaching / JoCoR / DivideMix-lite 多数不占优，不能写成全数据集标签噪声 SOTA。
- 全主表最稳证据：β=0.5 vs shuffled 15/15 main settings 胜，平均 delta +9.34pp。
- 低标签端点 5-seed：Cora/CiteSeer train=0.05 下 β=0.5 对 small-loss / JoCoR / FINE / shuffled 均 5/5 胜；train=0.65 优势缩小，Cora vs JoCoR 只有 +0.18pp 且 2/5 seeds 正。

论文定位：
- 可写：同一 X-only weak MLP backbone 下，SoftDist-MLP β=0.5 在 Cora/CiteSeer 型 citation 数据的多噪声设置中稳定优于经典和 MLP-adapted 最新抗噪策略，同时真实球结构对 shuffled 有稳定增量。
- 必须写：PubMed 边界；FINE/DivideMix-lite 是 MLP-adapted baseline，不是原图像域方法复现；不声称模型无关 SOTA。

## 最新补充：Phase 41 免K粒球图节点聚类诊断（2026-06-30）

用户要求按 `external/NoisyGL/docs/execution_plan_gb_clustering_feasibility_20260630.md` 执行 Phase 41，验证是否可放弃 SoftDist 分类线、转向“免K粒球图节点聚类”新主线。已完成脚本、测试、Cora seed0 smoke、三数据集 5-seed 诊断和 go/no-go 报告。

新增或更新文件：
- `external/NoisyGL/scripts/diagnose_gb_clustering.py`
- `external/NoisyGL/tests/test_gb_clustering.py`
- `external/NoisyGL/runs/gb_clustering_feasibility_20260630/`
- `external/NoisyGL/docs/execution_plan_gb_clustering_feasibility_20260630.md`
- `02_experiment_history_and_lessons 19.md`
- `gb-direction-turning-point-2026-06.md`
- `项目核心总结_新窗口上下文.md`

验证：`py -3 -m pytest tests/test_gb_clustering.py -q` -> `4 passed`。脚本编译检查通过。

执行范围：`cora,citeseer,pubmed` × seeds `0,1,2,3,4`，split sensitivity `0.02,0.05,0.1`，主判定 split `0.05`。聚类使用全图节点，无噪声注入；真实标签只用于 ACC/NMI/ARI 评测，不进入构球、球图或 modularity 凝聚决策。

主结果：
| Dataset | true K | K_auto | GB ACC | kmeans trueK ACC | ACC gap | GB NMI | kmeans trueK NMI | wrong-K ACC adv |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Cora | 7 | 3.00 | 0.4823 | 0.6317 | -0.1493 | 0.3404 | 0.5193 | -0.1208 |
| CiteSeer | 6 | 4.40 | 0.5342 | 0.6395 | -0.1053 | 0.2942 | 0.3782 | +0.0240 |
| PubMed | 3 | 3.40 | 0.5248 | 0.6766 | -0.1518 | 0.1918 | 0.3107 | +0.0096 |

结论：Phase 41 判定 **NO-GO**。G1/G2 三数据集全失败；G4 也全失败，说明当前球图凝聚没有独立优于 `kmeans_H_autoK`。因此“免K + 误设K鲁棒”的图聚类新主线不成立。后续不建议继续投入 `GRACE H + ordinary balls + 无参凝聚` 的图聚类变体；更稳的路线是回到 SoftDist 分类论文，按受控增量与适用性边界诚实收口。

## 最新补充：Phase 40 跨视角粒球冲突诊断（2026-06-29）

用户要求按 `external/NoisyGL/docs/execution_plan_cross_view_conflict_20260629.md` 开始 Phase 40。已完成脚本、测试、smoke 与 go/no-go 诊断。

新增或更新文件：
- `external/NoisyGL/scripts/diagnose_cross_view_conflict.py`
- `external/NoisyGL/tests/test_cross_view_conflict.py`
- `external/NoisyGL/runs/cross_view_conflict_20260629/`
- `external/NoisyGL/docs/execution_plan_cross_view_conflict_20260629.md`
- `02_experiment_history_and_lessons 18.md`

验证：`py -3 -m pytest tests/test_cross_view_conflict.py -q` -> `5 passed`。

执行范围：`cora,citeseer` × seeds `0,1,2,3,4`。未继续跑 `coauthorcs`，因为 `cora` 和 `citeseer` 两个数据集已经全部未通过 primary G1，而预注册标准要求至少 `2/3` 数据集通过；即使 `coauthorcs` 通过也无法改变全局 `STOP`。

关键结果：
- Cora `s_xview/s_xview_soft` 条件 AUC 均未过线，双干净交集相对 `s_combo_2knn` 分别为 `-0.2868/-0.2950`。
- CiteSeer `s_xview/s_xview_soft` 条件 AUC 均未过线，双干净交集相对 `s_combo_2knn` 分别为 `-0.3036/-0.2788`。
- 冲突节点并不更脏：Cora conflict noise `0.3970` < non-conflict `0.4287`；CiteSeer conflict noise `0.4036` < non-conflict `0.4351`。

结论：Phase 40 判定 `STOP`。跨视角 X/H 粒球冲突没有提供单空间 kNN 拿不到的有效噪声信号。后续不建议继续开新的粒球形状、投票、椭球、GMM 或 X/H 硬冲突变体；更合理的论文定位是“图神经表示下粒球去噪的适用性边界”。

## 本轮目标

使用 `$gb-gcl-project-context` 模式A，同步刚完成的 GEAF-style 粒椭球噪声过滤桥接实验，使新窗口不依赖聊天记忆即可恢复实验配置、代码改动、结果和下一步决策。

项目根目录：`C:\Users\Admin\Desktop\实验8去噪5.19-`

## 本轮读取的关键文件

- `AGENTS.md`
- `C:\Users\Admin\.codex\skills\gb-gcl-project-context\SKILL.md`
- `C:\Users\Admin\.codex\skills\gb-gcl-project-context\references\project-sources.md`
- `项目核心总结_新窗口上下文.md`
- `99_session_handoff.md`
- `02_experiment_history_and_lessons 16.md`
- `external/NoisyGL/runs/geaf_bridge_20260629_gpu/geaf_bridge_report.md`
- `external/NoisyGL/runs/geaf_bridge_20260629_gpu/geaf_bridge_gate.csv`
- `external/NoisyGL/runs/geaf_bridge_downstream_seed0_20260629_gpu/geaf_bridge_downstream_seed0_report.md`
- `external/NoisyGL/runs/geaf_bridge_downstream_seed0_20260629_gpu/geaf_bridge_downstream_seed0_summary.csv`
- `external/NoisyGL/runs/geaf_bridge_softdist_success_20260629_gpu/geaf_bridge_report.md`
- `external/NoisyGL/runs/geaf_bridge_softdist_success_20260629_gpu/geaf_bridge_gate.csv`
- `external/NoisyGL/runs/geaf_bridge_softdist_success_downstream_seed0_20260629_gpu/geaf_bridge_downstream_seed0_report.md`
- `external/NoisyGL/runs/geaf_bridge_softdist_success_downstream_seed0_20260629_gpu/geaf_bridge_downstream_seed0_summary.csv`

本轮也运行了发现脚本：

```powershell
py C:\Users\Admin\.codex\skills\gb-gcl-project-context\scripts\discover_context_sources.py --root C:\Users\Admin\Desktop\实验8去噪5.19- --limit 120
```

## 本轮改动或新增文件

- 更新：`项目核心总结_新窗口上下文.md`
- 更新：`99_session_handoff.md`
- 新增：`02_experiment_history_and_lessons 17.md`

本轮是上下文同步，没有重新运行训练实验。

## 本轮前已完成的代码与实验

GEAF bridge 实现位于 `external/NoisyGL`：

- `utils/ordinary_gb.py`：新增 label-free size-adaptive ellipsoid 构球。
- `utils/frozen_grace_targets.py`：新增 GEAF hard correction 与分区诊断。
- `predictor/FrozenGraceGB_GCN_Predictor.py`：新增 `frozengrace_geaf_gcn_xh` 与 shuffled 方法。
- `scripts/diagnose_geaf_bridge.py`：新增离线诊断脚本。
- `scripts/run_geaf_bridge_downstream.py`：新增按 gate 跑 downstream seed0 的脚本。
- `tests/test_geaf_bridge.py`：新增单元测试。

验证结果：

- `tests/test_frozen_grace_targets.py tests/test_geaf_bridge.py tests/test_ellipsoid_param_grid.py tests/test_xh_features.py -q`：`38 passed, 1 warning`
- `py_compile scripts\diagnose_geaf_bridge.py scripts\run_geaf_bridge_downstream.py`：通过

用户要求不用 CPU 后，实验命令已改为 `--device cuda:0`。注意：sklearn KMeans 构球仍是 CPU 实现，GPU 主要用于 NoisyGL tensor/downstream 训练路径。

## 主四数据集 GEAF bridge 结果

离线目录：`external/NoisyGL/runs/geaf_bridge_20260629_gpu/`

downstream 目录：`external/NoisyGL/runs/geaf_bridge_downstream_seed0_20260629_gpu/`

协议：DBLP、PubMed、AmazonPhoto、AmazonCom；offline seeds `0..4`；confidence `0.0/0.6/0.7`；uniform `0.4`；train-only noise；percent split `0.2/0.1/0.7`；device `cuda:0`。

离线规模：

- raw rows: `120`
- gate passed settings: `6`

离线 gate：

| Dataset | Confidence | Gate | hard_net | NCR | CDR | positive seeds | 结论 |
|---|---:|---|---:|---:|---:|---:|---|
| AmazonCom | 0.0 | FAIL | 0.3682 | 0.5763 | 0.2081 | 5 | CDR 高 |
| AmazonCom | 0.6 | FAIL | 0.1604 | 0.1906 | 0.0302 | 5 | CDR 略超 |
| AmazonCom | 0.7 | PASS | 0.0574 | 0.0646 | 0.0072 | 5 | 通过 |
| AmazonPhoto | 0.0 | FAIL | 0.4264 | 0.5952 | 0.1687 | 5 | CDR 高 |
| AmazonPhoto | 0.6 | PASS | 0.1963 | 0.2220 | 0.0257 | 5 | 通过 |
| AmazonPhoto | 0.7 | PASS | 0.0739 | 0.0809 | 0.0070 | 5 | 通过 |
| DBLP | 0.0 | FAIL | 0.5054 | 0.6836 | 0.1782 | 5 | CDR 高 |
| DBLP | 0.6 | PASS | 0.1955 | 0.2209 | 0.0254 | 5 | 通过 |
| DBLP | 0.7 | PASS | 0.0685 | 0.0761 | 0.0076 | 5 | 通过 |
| PubMed | 0.0 | FAIL | 0.4429 | 0.6489 | 0.2060 | 5 | CDR 高 |
| PubMed | 0.6 | FAIL | 0.1886 | 0.2246 | 0.0360 | 5 | CDR 高 |
| PubMed | 0.7 | PASS | 0.0642 | 0.0736 | 0.0094 | 5 | 通过 |

downstream seed0：

| Dataset | Confidence | GCN[X‖H] | SoftDist | SoftDist shuffled | GEAF | GEAF shuffled | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| AmazonCom | 0.7 | 0.8564 | 0.8558 | 0.8578 | **0.8642** | 0.8564 | 唯一明确正例 |
| AmazonPhoto | 0.6 | 0.9109 | 0.8964 | 0.8995 | **0.9115** | 0.9109 | 只比 GCN 高约 0.06pp |
| AmazonPhoto | 0.7 | 0.9109 | 0.8964 | 0.8995 | 0.9046 | 0.9109 | 不过门 |
| DBLP | 0.6 | 0.8429 | 0.8426 | 0.8426 | 0.8349 | 0.8429 | 负 |
| DBLP | 0.7 | 0.8429 | 0.8426 | 0.8426 | 0.8429 | 0.8429 | 持平 |
| PubMed | 0.7 | 0.8249 | **0.8252** | 0.8241 | 0.8243 | 0.8249 | 负于 GCN/SoftDist |

结论：只建议 AmazonCom `confidence=0.7` 进入 5-seed sanity。DBLP、PubMed、AmazonPhoto 不建议继续。

## Cora/CiteSeer/Coauthor-CS 补充结果

离线目录：`external/NoisyGL/runs/geaf_bridge_softdist_success_20260629_gpu/`

downstream 目录：`external/NoisyGL/runs/geaf_bridge_softdist_success_downstream_seed0_20260629_gpu/`

离线 raw rows: `90`；gate passed settings: `2`。

只有 Coauthor-CS `confidence=0.6/0.7` 通过 gate：

- Coauthor-CS 0.6：hard_net `0.2263`，NCR `0.2405`，CDR `0.0142`
- Coauthor-CS 0.7：hard_net `0.0890`，NCR `0.0935`，CDR `0.0045`

Coauthor-CS downstream seed0：

| Confidence | GCN[X‖H] | SoftDist | SoftDist shuffled | GEAF | GEAF shuffled |
|---:|---:|---:|---:|---:|---:|
| 0.6 | 0.9189 | **0.9208** | 0.9199 | 0.9185 | 0.9189 |
| 0.7 | 0.9189 | **0.9208** | 0.9199 | 0.9189 | 0.9189 |

结论：Coauthor-CS 上 GEAF hard correction 不能替代当前 SoftDist；即使离线 gate 通过，下游仍低于 SoftDist。

## 新增结论

### 已被实验支持

- GEAF-style hard correction 有离线噪声修正信号，`confidence=0.6/0.7` 能显著降低 CDR。
- `confidence=0.0` 不可用，四个主诊断数据集均因 CDR 过高失败。
- downstream seed0 中只有 AmazonCom `confidence=0.7` 明确超过 GCN[X‖H]、GEAF shuffled 与当前 SoftDist。
- DBLP/PubMed/AmazonPhoto 的离线 gate 通过没有稳定转化为 downstream。
- Coauthor-CS 补充实验说明 GEAF hard correction 不能替代当前 SoftDist 主方法。

### 合理猜测

- AmazonCom 的局部 H-space 粒椭球多数类可能更接近真实可纠正噪声，GEAF 修正没有被 GCN[X‖H] 完全吸收。
- DBLP/PubMed 的失败可能来自 corrected labels 与 `[X‖H]` baseline 信息冗余，或边界样本误改抵消收益。
- AmazonPhoto `confidence=0.6` 的 `+0.06pp` 更可能是极弱边际或 seed0 波动。

### 尚未验证

- AmazonCom `confidence=0.7` 的 5-seed 稳定性。
- deployable confidence selection policy。
- soft/hybrid GEAF 是否能比 hard correction 更稳定。
- `gradient_weighted_target_gain` 是否能预测 offline-to-downstream 转化。

### 已被负结果削弱

- 离线 hard-correction gate 通过即可推进 downstream。
- GEAF-style hard correction 能普适修复 SoftDist offline-to-downstream 断裂。
- DBLP/PubMed 只要显式标签修正就能恢复下游收益。
- Coauthor-CS 可用 GEAF 替代 SoftDist。

## 下一窗口第一步

建议只做一个可选确认实验：

当前 `scripts\run_geaf_bridge_downstream.py` 只支持单个 `--seed`，并且会消费 gate 里全部 passed rows；它还没有 `--datasets` 或 `--confidence_thresholds` 过滤参数。因此 AmazonCom-only 5-seed 不能直接用一个命令完成。下一窗口应先二选一：

- 小改脚本：给 `run_geaf_bridge_downstream.py` 增加 `--seeds`、`--datasets`、`--confidence_thresholds` 过滤，再跑 AmazonCom `confidence=0.7` 的 5 seeds。
- 不改脚本：先生成一个只含 `dataset=amazoncom` 且 `majority_confidence=0.7` 的临时 gate CSV，再对 `--seed 0/1/2/3/4` 手动循环运行。

决策规则：

- AmazonCom 5-seed 若稳定超过 `gcn_xh_ce`、`frozengrace_geaf_gcn_xh_shuffled` 和当前 SoftDist，写成 AmazonCom-specific bridge case。
- 若不稳定，GEAF bridge 写成又一个 offline-positive 但 downstream 不稳的边界实验。
- 不建议 DBLP/PubMed/AmazonPhoto/Coauthor-CS 继续 seed5 或强 baseline。

## 2026-07-10 MLP-23 数据集广度补充

- 目标：按固定 LCSS/MLP/噪声协议寻找 1-3 个新数据集；不为新数据集调球参数。
- 完成 Cora_ML、Attributed-Wiki、CiteSeerFull、Coauthor-Physics 的 uniform 0.4 全 13 方法×5 seeds。四者的真实分组均 5/5 seeds 优于 shuffled，但没有一个稳定领先最强 baseline ≥1pp。
- 正文新增两个高价值广度数据集：CiteSeerFull（LCSS 59.40±1.73，JoCoR 59.48±1.60，shuffled 46.25±3.18）与 Coauthor-Physics（LCSS 93.11±0.25，JoCoR 93.09±0.28，shuffled 88.90±2.05）。表述限定为“相当 + 真实结构贡献”。
- Cora_ML（+0.22pp vs best，2/5）与 Attributed-Wiki（−2.41pp vs best，0/5）只进附录筛选记录。
- 高/低噪声边界：CiteSeerFull u0.2 +2.28pp、u0.6 −4.71pp；Coauthor-Physics u0.2 −0.69pp、u0.6 −2.29pp。核心 5 数据集×5 噪声设置 claim 不变。
- Coauthor-Physics 采用与原 InfoNCE 数学等价的 anchor chunking（1024）解决 ROCm OOM，跑满 1000 epochs；H=34493×256、全有限、行归一化、节点顺序对齐，球纯度约 0.9235。
- 证据：`external/NoisyGL/runs/mlp23_dataset_breadth_20260710/`、`external/NoisyGL/docs/mlp23_dataset_breadth_table_20260710.md`、`external/NoisyGL/docs/paper_final_zh_v5_mlp23_breadth_20260710.md`、`02_experiment_history_and_lessons 20.md`。

## 2026-07-12 MLP-24 尺度自适应球数

- 已实现并执行 `B_eff=max(64,ceil(sqrt(N)))`，只改变球数上限；11 数据集几何 33/33、seed0 下游 44/44，协议/泄漏/完整性审计均通过。
- 几何门通过：Coauthor-Physics 64→186 球、平均规模下降 65.6%，构球时间可接受。
- seed0 门失败：Coauthor-Physics adaptive-fixed 仅 +0.10pp，低于 +0.50pp；七受影响数据集平均约 +0.33pp。
- 按预注册停止 Phase C/D，固定 64 保持主方法默认；不要因 CoraFull +0.93pp 或 Amazon-Computers +0.71pp 单点复活该方向。
- 下一窗口第一步：继续论文收口，不再调球数公式。证据：`external/NoisyGL/runs/mlp24_scaleadaptive_balls_20260712/`、`02_experiment_history_and_lessons 21.md`。

## 2026-07-12 MLP-25 中等票数粗球 Holdout

- 最终五臂 smoke 完成 15/15：fixed、vote-target real/shuffled、冻结 MLP-13 candidate real/shuffled；holdout 为 CiteSeerFull/Cora_ML/Attributed-Wiki。
- vote-target 相对 fixed：+0.27/+1.24/−5.29pp；candidate：+0.88/+0.76/−1.96pp。两个候选均未过“2/3 数据集 +1pp、无 >0.5pp 回退”门。
- 不跑 5-seed，fixed 64 保持默认，粒度轴关闭。不要再以“20 votes 普适最优”为由重启同类参数实验。
- 证据：`external/NoisyGL/runs/mlp25_coarse_holdout_smoke_20260712/`、`02_experiment_history_and_lessons 22.md`。

## 最新补充：D2 Stable-GB Reliability A0/A1（2026-07-14）

已在隔离分支 `codex/d2-stable-gb-a0a1` 与 worktree `C:\Users\Admin\.config\superpowers\worktrees\实验8去噪5.19-\d2-stable-gb-a0a1` 完成实现、测试和正式运行。历史 NoisyGL 代码与输入只读。

实现：`next_paper/stable_gb_reliability/`；正式产物：`next_paper/runs/stable_gb_reliability_a0a1_20260714/`；详细报告：`next_paper/docs/stable_gb_reliability_a0a1_report.md`；编号历史：`02_experiment_history_and_lessons 26.md`。

协议：Cora/CiteSeer、uniform0.6 seed0、20/10/70 split、frozen H、原始+8 个 10% shared-dimension mask views、GB 64/32/0.05、参考球分组 5-fold OOF、5,000 cluster bootstrap、100 permutation。MLP34 selection trace 未使用。

A0 两数据集全部 PASS。A1 两数据集五类 Gate 全部 FAIL：Cora 两项主 ΔAUPRC 为 -0.0142/-0.0242；CiteSeer为 +0.0170/+0.0157 但 CI 跨 0 且未超过 permutation p95。固定 `R_GB` uplift 两数据集为正但 CI 跨 0，真实 GB 未稳定超过 size-preserving shuffle。

最终决策：**D2 NO-GO，不进入 A2。** 下一窗口不要调整 masking rate、公式、数据集或门槛复活本轮方向，也不要进入真实 GRACE views 或 Stable-GB 下游训练。可转向 direction ranking 中其他候选；本实验不能声称节点分类性能改善。

## 最新补充：MLP-37 GRACE 预训练种子分层方差（2026-07-16）

- 第一篇 LCSS/SoftDist-MLP 的唯一新增实验已经完成：Cora/CiteSeer/CoraFull × GRACE seeds 0/1/2 × downstream seeds 0/1/2，uniform0.4、train-only、20/10/70 split；72/72 下游和 27/27 几何单元完整。
- 三数据集全部通过跑前预注册门。任一 H-seed 下 LCSS−shuffled 最小均值增量为 `+11.43/+11.61/+4.73pp`，LCSS−CE 为 `+11.70/+10.81/+4.82pp`；LCSS−JoCoR 在全部 9 个 dataset×H-seed 单元均为正。
- H-seed SD 为 `0.53/0.07/0.44pp`，downstream RMS SD 为 `0.87/1.11/0.39pp`。可声称三数据集、三个预训练种子下核心增量方向稳健，不能声称全面解决预训练不确定性。
- 跨 H-seed 球划分 ARI 均值 `0.300`（`0.247--0.365`），说明具体划分会变，但聚合共识收益稳定；这是与结果一致的机制解释，不是因果定律。
- 九份 GRACE checkpoint/H/config/loss/成本/hash 已归档。CoraFull 平均预训练 `74.48 min`、峰值 allocated `12.73 GiB`。
- v15 候选稿已加入 3×3 结果、双口径、噪声公式、LOO 声明和成本表：`external/NoisyGL/docs/paper_final_zh_v15_submission_candidate_20260716.md`。
- 投稿收口已完成：匿名工件 v3、v15 内嵌 HTML、数字/措辞/路径泄漏终审均通过；最终相关测试 `63 passed`。下一窗口应做作者本人全文通读和投稿材料填写，不要再改 β、粒度、算法或增加数据集。

## 最新补充：MLP-39 LOO-LCSS + Soft-Small-Loss（2026-07-17）

- 完成 Cora/CiteSeer/CoraFull、uniform0.4、seed0 的 6-run 最小 smoke，协议与 MLP-30/36 锚点 exact-hash 配对，val/test 零污染。
- LOO combo−默认 combo 为 `−0.79/+0.39/−0.82pp`，平均 `−0.41pp`，只有 1/3 正向；预注册方向门、均值门和最大回退门失败。
- real−shuffled 仍为 `+13.46/+16.24/+6.49pp`。结论是 LOO 没有叠加到 small-loss，而不是局部共识结构失效。
- 决策：NO-GO，不扩 5-seed、不调参数。第一篇继续冻结；下一篇 `LCSS+soft-small-loss` 保留默认非 LOO target。历史见 `02_experiment_history_and_lessons 29.md`。

## 最新补充：MLP-40 缺失噪声谱闸门（2026-07-17）

- 完成 Cora_ML 的 uniform0.6/pair0.4/instance0.4 与 Coauthor-Physics 的 pair0.4/instance0.4，共 45/45 个九臂 seed0 GPU run；hash 配对和 train-only 泄漏审计全部通过。
- Cora_ML 相对最强 baseline 为 `−4.20/+1.57/−2.62pp`，平均 `−1.75pp`，未过门。
- Coauthor-Physics 相对最强 baseline 为 `+0.09/+0.69pp`，平均 `+0.39pp`，低于冻结 +0.5pp 门，也未过门。
- 两者 real−shuffled 均明显为正，但不能转化为稳定强 baseline 优势。严格停止，不补 seeds 1–4；第一篇核心主表不变。证据：`external/NoisyGL/runs/mlp40_coraml_coauthorph_fullspectrum_20260717/`；历史：`02_experiment_history_and_lessons 30.md`。

## 最新补充：MLP-41 固定 k=5 的 kNN 共识教师（2026-07-17）

- 历史 MLP-11 的 k=5 是“kNN target 直接训练 MLP”，与表 4 的“kNN-GCN[X||H] teacher 蒸馏到 X-only MLP”角色不同，因此重新实现固定 k=5 teacher 并完成 30/30 个五种子 GPU run。
- 固定 k=5 相对球式的差值：u0.4 的 Cora/CiteSeer/CoraFull 为 `+0.63/−0.58/+1.21pp`；u0.6 为 `−0.14/+0.13/+1.42pp`。k=5 在 4/6 个均值更高，球式在 CiteSeer u0.4 和 Cora u0.6 更高。
- 最终论文决定：正文表 4 恢复原票数匹配 kNN teacher，以控制有效共识票数；固定 `k=5` 作为附录 I 的尺度敏感性结果完整保留。该实验不能支持“粒球普遍优于 kNN”，只能支持两种局部共识实现随数据集/噪声率互有胜负。
- 证据：`external/NoisyGL/runs/mlp41_knn_k5_teacher_20260717/`；历史：`02_experiment_history_and_lessons 31.md`。

## 最新补充：MLP-48 CF-BLAGB 固定预算标签辅助粒球（2026-08-05）

- 只修改第二阶段粒球分裂顺序：五折含噪标签通过有界 JSD 乘子重排标签无关几何分裂，不能否决分裂；球数固定为正式 `K_geo`。
- 候选树先完全复现正式 assignment，再扩展更深候选。Cora/CiteSeer/CoraFull 的正式预算为 52/64/64，候选容量为 52/65/128，真实数据前缀哈希全部匹配。
- Phase A 的 21/21 七臂目标单元和 60/60 fold 分区完整，协议 hash 可追溯，val/test 零污染。
- 判定为 NO-GO：固定预算 3/3 通过，但 CF-BLAGB 同时改善 geometry-only 与 shuffled Brier 为 0/3，clean oracle headroom 只有 1/3；平均 target-accuracy 增量仅 `+0.00118`。
- 不进入下游 Phase B，不调 JSD、先验、折数或候选树阈值。结合 MLP-46，直接使用当前含噪训练标签改进粒球生成的方向关闭。
- 证据：`external/NoisyGL/runs/mlp48_cf_blagb_20260805/`；执行备案：`external/NoisyGL/docs/execution_plan_mlp48_cf_blagb_20260805.md`。
