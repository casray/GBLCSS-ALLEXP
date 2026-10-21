# v2 runner 独立实现审查

日期：2026-10-19。状态：**核心目标/评估路径审查通过；完整实验启动前须修复下列 P1 问题并复核**。本报告检查的是审查时的 staging 文件；之后的修改不自动继承通过状态。

范围：`PLAN_V2.md`、`config_v2.json`、`prepare.py`、`run_arm.py`、`runtime.py`、`mechanisms.py`、`granularity.py`、`run_suite.py`、`summarize.py` 和四组 CPU 测试；另核对 `GB-LCSS_GitHub/src/gblcss/data.py` 与 `targets.py`。未训练、未调用 GPU、未操作 Git。只创建本审查报告。

## 1. 优先修复的问题

### P1：冻结 manifest 没有覆盖实际划分/噪声源码

位置：`run_suite.py:31–34`，`run_arm.py:26–28`。

`source_fingerprint` 只哈希 v2 目录内非测试 `.py`，但 `run_arm.py` 实际从工作区 `GB-LCSS_GitHub/src` 导入 `gblcss.data` 的划分、噪声和 RNG。该文件若在队列期间变化，现有 source hash 不变，冻结检测仍通过，旧结果和新结果可能使用不同噪声/划分实现而被混合。应冻结实际外部依赖文件（至少 data.py，若有本地传递依赖也包括），并在每次启动子进程前校验。`WORKSPACE=Path(__file__).resolve().parents[3]` 对当前目录层级本身正确，不需要改成其他层。

### P1：称为完整冷启动的时间缺少准备进程开销

位置：`run_suite.py:207–216` 的准备子进程；`run_arm.py:266–273` 的 shared/cold 计算；`prepare.py:91–92` 的 provenance。

训练 worker 的启动/导入开销由 parent wall 与 run body 差补回，这一步正确。但准备子进程的 `_execute_child` 返回 wall 没有保存，shared 只加显式 load/adjacency/features/coordinates/tree 阶段；准备端 Python/torch/PyG 导入、启动/退出、来源哈希及未覆盖间隙没有归入 cold。因而当前 `cold_total_s` 是已归属准备阶段加训练进程墙钟，不能直接称完整冷启动或据此评定 G0。

建议记录准备进程 wall 与各阶段和，给出公共启动/来源校验开销，并用同一规则归属给各臂；或同时给出真实执行总墙钟和准确命名的归属估算。不要将准备全部六图的总耗时随意加到每个单任务；单任务与共享批任务边界须明确。

### P1：复用缓存未校验内容与冻结源

位置：`prepare.py:38–43`；`run_arm.py:88–93` 和 `142–147`。

已有 metadata 时 preparation 只比 config hash 就返回，不验证 source 文件当前 hash、准备源码或 x/y/P/features/tree 文件完整性。切面缓存也直接读取 JSON/NPY，未校验 tree/split/score-label/M/assignment hash。新空 run-root 首次运行风险较小，但续跑和“相同新球严格配对”的保证不足，且无法证明缓存未变。

至少保存并验证 source data、实际准备源码、tree 与每个已生成缓存文件的内容哈希；切面 metadata 保存依赖与 assignment 哈希，使用前比对。可避免每个训练臂重复哈希巨大 features：由父进程在队列启动/恢复时验证共同 cache，再校验轻量签名，但须保证运行期间的来源约束。不能用结果记录里的 assignment hash 代替加载前验证。

## 2. 应补充或准确命名的读数

### P2：摊销计划尚无实现输出

`config_v2.json` 固定了 `amortization_K=[1,3,10,30]`，但审查时 runner/summary 未消费该字段。已有 shared 与 cut_credit 只能部分支持后处理；必须区分可复用 R0 切面与随 noisy labels/split 改变的 R3/O2 切面，给 A 与粗臂相同缓存权利。不能只将 cold_total 除以 K。可在训练启动前锁定后处理公式，不必为此新增训练。

### P2：inference 显存为训练对象仍驻留时的峰值

位置：`run_arm.py:250–264`。

重置 peak 只重置统计，不释放 optimizer、参数梯度、最后一轮 logits/loss、fit_x/fit_y/weights 等训练对象。GCN 粗臂又同时持有 original x/P 与 coarse x/P，这是当前原图验证所需，完整任务峰值必须包括，不能隐藏。

现有 `gpu_peak_allocated_gib=max(train_peak,infer_peak)` 作为实际 runner 的完整 GPU allocator 峰值基本合理；`gpu_primary_inference_peak_gib` 应明确叫“训练对象驻留时主推理峰值”，或清理优化器/训练临时张量后另测部署峰值。不要从这一读数推断纯部署最低显存。`gpu_training_reserved_gib` 不是完整 reserved 峰值；若总 reserved 是报告承诺，应补记录推理/全程 reserved。

### P2：主机 RSS 不含 CPU preparation 峰值

`run_arm.py` 的采样线程只覆盖训练子进程，并从 `run()` 启动后开始采样；`prepare.py` 没有 RSS 采样。因此 `host_rss_peak_gib` 不能命名为完整流程主机峰值。准备阶段含原图 dense X、缓存构造、SVD、树以及 float64 geometry 拷贝，可能才是最大值。应分开记录 preparation/worker 峰值；串行阶段完整峰值取二者最大值，不能相加。

### P2：汇总缺少 A 配对与部分已计算指标

`summarize.py:41` 的 R0 C 只与 B/Bprime/LCSS/D 配对，遗漏 C−A 与 D−A，CS 又只有 A/C/D，因此无法直接看对应原图完整成本收益。A 只训练 .1 合理，但 CoraFull 敏感性 .05/.3 应引用相同 dataset/noise/seed/backbone 的 .1 A，不应当成缺失。

raw JSON 已有 macro-F1、最差类召回、secondary lift accuracy 和阶段时间，但 summary 只含 accuracy、cold/run wall、GPU allocated 与 parent wall。补充汇总能防止只看 accuracy 忽视 CoraFull 少数类损失。主/次指标必须仍分开，不取最大。

### P2：次要预测与恢复工件完整性不足

当前 NPZ 只保存 primary test indices/prediction；secondary 只有汇总 accuracy。建议保存 secondary prediction（若存在），以及 checkpoint 身份/目标哈希/训练支持摘要，便于回算。续跑 `validate_completed_result` 只检查 status/job/config/source，未检查 predictions 工件或关键数值完整性；缺预测/损坏结果会被当 completed。当前单元测试也将缺大部分字段的 fake completed 视为可接受。应把“可识别的子进程成功”与“正式完整结果”验证区分。

## 3. 核心正确性：已经核对的部分

1. **目标不读 held-out 类别。** `protocol_labels` 保留 clean y 供验证/最终测试和 O2，但传入目标构造的是非训练位置固定为 0 的 permitted；build_targets 内先索引训练位置。O2 只把 clean train 写入 cut scoring，后续 build_targets 仍使用 permitted noisy。未看到 test 指标控制 checkpoint/预算/臂。
2. **噪声方向一致。** 原 helper 是 `transition[c,c-1]=rate`，即 c→(c−1) mod C；PLAN 写法一致。uniform 是翻到其他类，每类 η/(C−1)，不是含自身的均匀重采样。
3. **B/C 配对。** 非 A 臂使用同一 R0_Mbudget 缓存；B 是训练节点 inclusive q；C 用同一 q 并按 support/L 加权。GCN C 的 fit_idx 只取有监督球，空训练球仍参与消息传递；cached C 只训练 M+ 行。这符合协议。
4. **Bprime/LCSS 区分。** Bprime n_B≥2 做纯 LOO、单票回退自身；LCSS n_B≥3 才改目标，alpha、tie-safe empirical target、beta_target=.5 与原 helper 数值一致。beta_cut 和 beta_target 没有混用。
5. **cached C 定义正确。** 先在原图缓存 [X,PX,P²X]，随后 r@features，未改成粗图重新传播。GCN 使用 r@p@h，custom forward 无二次加环/归一化。
6. **D 匹配正确。** 对完整 assignment 随机置换，逐球 bincount 与 C 完全相同；未根据标签拒绝随机分组。M+ 可变且有记录。
7. **树与 exact M。** build_tree 无标签/mask 参数；geometry 先按通道根 SSE 缩放。select_cut 使用精确固定预算树 DP，预算可达时实际 M 精确；小树穷举测试通过。继承父链按建树的父先于子顺序求值；n<2 的根有预定义 prior fallback。O2 只是 clean-label cut diagnostic。
8. **checkpoint。** 原图/原节点 validation 每 5 epoch，patience=10 checks；primary/secondary 独立 snapshot，CPU clone/deepcopy 避免末轮覆盖；停止只依赖 primary。最终 primary test 加载 primary_state，secondary 加载 secondary_state。无两种测试择优。
9. **进程隔离。** suite 串行启动每个独立 arm，失败/超时停止队列，未同时驻留旧模型进程。GCN 不加载 cached features，MLP 不加载 GPU graph；同一初始模型 seed 在 warmup 后重建，warmup 不改变正式初始化。
10. **主/敏感性范围。** 清单 1080 个唯一作业，其中 E2 48 个额外 C R3/O2，R0 复用；A 不因 ratio 重复训练；CS 只有 A/C/D。WikiCS 未出现在执行列表。旧 E0 seed0 未混入新结果。

CE 身份仍须按条件解释：同球共同预测 p_B 时，sum_i CE(p_B,y_i)=n_B CE(p_B,q_B)。B 实际节点预测 p_i 不同，不能由此推出 B 与 C 训练等价。现有测试已加入“不同节点预测打破恒等式”的反例；这支持限定说法。

## 4. 实际验证

执行命令（CPU，未调用 run_arm 训练入口）：

```powershell
& 'D:/codex_temp/noisygl-rocm721/Scripts/python.exe' -B -m unittest discover -s 'experiments/budget_gb_coarsening_20261019/v2' -p 'test_*.py' -v
```

输出：**34 tests，2.734s，OK**。覆盖目标、held-out 隔离、LCSS 原公式、共享 CE 条件、随机 exact sizes、退化树 fallback、DP 穷举对照、固定预算、继承、checkpoint、冻结检测、队列超时/恢复及负配对保留。出现 joblib 物理核检测警告后使用逻辑核，不是测试失败。

该测试集未实际执行真实数据 preparation、GPU sparse算子、真实 run_arm 全流程或显存计量，故不能证明 GPU 环境、真实大树运行时、真实缓存和数据路径已通过。GPU 连通检查应在上述启动前修复完成且新 hash 冻结后进行；连通结果不进入正式统计。

## 5. 最终判断

算法比较和主选模路径可以接受；P1 是可追溯性与效率主张的真实缺口，而非要求扩展算法。完成外部源码冻结、缓存验证和完整准备成本归属后，复核变更、重跑受影响 CPU 测试，再启动正式清单。不要用本报告的核心正确性认可跳过这些修复，也不要为修复新增算法变体。

## 6. 修复后的第二轮定向审查

状态更新：**此前 P1 启动阻断项已处理；未发现剩余关键阻断项。GPU 连通检查通过并冻结正式配置/源码后，可开始正式描述性筛查。** 本节取代第 1 节与第 5 节中“仍须修复才能启动”的旧状态；第 1–5 节保留为审查历史。

本轮只读复核修改后的 preparation、worker、suite、summary、runtime、配置及测试。再次执行同一 CPU unittest 命令，实际输出 **37 tests，3.226s，OK**。未运行训练、未调用 GPU、未读取连通实验成绩作选型依据。joblib 仍有物理核检测警告，未造成失败。

另用 SHA256 比较 staging 与正式 Newidea worktree 下八个文件：`run_arm.py`、`prepare.py`、`runtime.py`、`run_suite.py`、`summarize.py`、`mechanisms.py`、`granularity.py`、`config_v2.json`，**八项全部一致**。正式 worktree 路径为 `C:/Users/Admin/.codex/worktrees/newidea-budget-gb-coarsening/Newidea/gb_c1_ratio_success_20260609/results_ratio_c1_20260609/experiments/budget_gb_coarsening_20261019/v2`。

确认的修复：

1. **实际外部源码已冻结。** source_fingerprint 除 v2 实现，还纳入 gblcss 的 `__init__.py`、`data.py`、`granular_ball.py`、`targets.py`，覆盖 import 时的本地依赖；suite 启动前与每作业前后检查漂移。
2. **准备缓存已可校验。** metadata 记录准备源码、x/y/P/features/tree 的大小与 SHA256；正常 suite 的 prepare/恢复路径全量验证文件，worker 检查大小。原始 source_data_sha256 保留生成这些固定缓存时的数据来源，后续比较应以冻结缓存为准。
3. **切面缓存有内容与标签源检查。** 读取 assignment 校验其哈希；R3/O2 校验对应训练标签源哈希。固定 run-root 的图/树/config/代码由外层冻结，seed 和 noise 写入切面键，因此当前已锁定矩阵可保证配对。不得在正式运行期间手工替换缓存；`--skip-prepare` 不应被用来绕过首次/恢复完整性验证。
4. **冷成本口径已修正。** parent 保存准备进程完整 wall，计算公共启动/未覆盖间隙与 provenance 开销；worker 的必要准备阶段成本与 parent 测量的 worker wall 合成，并保留原始阶段/进程墙钟。PLAN 和进度说明明确：cold_total 是重建值，非一次独立无缓存端到端冷跑实测。该透明口径足以进入筛查，不能将其包装成更强的实测证据。
5. **固定 K 已输出。** K=1/3/10/30，按 `cold−shared+shared/K` 计算每任务摊销读数；标签相关切面不放入 shared，逐任务计费。R0 切面仍按任务计费，属于保守口径，不是最大可能复用收益。任何实际多任务总成本应再乘 K，而不是把此每任务读数直接称 K 任务总时间。
6. **内存与推理过渡更清楚。** 推理前删除 optimizer、最后 logits/loss，清梯度并释放 allocator 空缓存；该清理也计时。准备阶段记录 Windows peak working set，worker RSS 与 preparation 峰值分开保留。原图验证所需 full/coarse 输入仍可能同时驻留，因此 inference 数值仍是当前完整 runner 驻留输入下的峰值，不是独立最简部署进程的最低显存。完整 GPU 峰值仍覆盖原图验证和推理。
7. **工件与汇总补齐。** primary 与 secondary prediction 均写入 NPZ，保存 SHA256；正式 runner 生成的工件在恢复和汇总时校验存在、路径边界与内容哈希。C/D 对 A 的配对已增加，CoraFull 敏感性明确引用相同单元 `.1` A，并输出 reference_ratio；新测试覆盖该行为。
8. **干净诊断在测试后。** clean training Brier/球混合度只在训练/选模及最终测试之后计算，不返回方法决策；R3/O2 的 downstream noisy 目标路径未改变。无新增训练变体。

已知而不阻断筛查的报告边界：三个 seed 各一次是准确率与成本的描述性筛查，不是每配置三次 timing repeat；prepare 的 host 峰值是统一 preparation 流程（包含两骨干缓存和候选树）的峰值，不能据此单独声称某臂必需的最小 host 内存。推理 reserved 总峰值、macro-F1/最差类召回/secondary 指标的汇总列仍可由已保存 raw JSON 补充，正式结论应读这些原始指标，不能只引用当前 accuracy CSV。

恢复验证器为 CPU fake worker 测试保留了“结果没有声明 prediction_file/SHA 时跳过工件检查”的兼容路径；生产 runner 始终写这两个字段，因此不影响当前首次冻结运行。若以后引入其他 worker 或外部结果导入，应要求正式 schema 必填，不能借兼容路径接收缺工件的真实结果。

本轮审查认可的是**实现与筛查启动条件**，不是训练结果、效率门槛或论文贡献已成立。仍需由主代理确认独立连通清单完成、正式 hash 冻结且运行期不编辑代码，随后按全部预定条件解释正负结果。
