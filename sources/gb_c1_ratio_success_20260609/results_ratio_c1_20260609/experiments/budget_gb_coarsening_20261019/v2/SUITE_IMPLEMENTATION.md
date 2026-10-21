# v2 调度与汇总实现

日期：2026-10-19。此文档记录实现与 CPU 验证，未声称正式实验已完成。

## 文件与接口

- `run_suite.py`：`build_jobs(config)`、`freeze_protocol(config_path, run_root, source_dir)`、`validate_completed_result(result, job, config_sha256, source_sha256, run_root=None)`、`validate_prediction_artifact(result, run_root)`、`run_suite(config_path, run_root, limit=None, skip_prepare=False, source_dir=SOURCE_DIR)`。
- `summarize.py`：`summarize(run_root)` 与纯数据函数 `paired_rows(results)`；只依赖 Python 标准库。
- `test_suite.py`：标准库 unittest，CPU 临时目录与无 GPU 子进程测试。

固定配置生成 **1080 项**：1032 项 E0 和 48 项追加 E2。A 对每个 dataset/noise/seed/backbone 只训练一次，ratio 标注为 .1。CoraFull .05/.3 敏感性覆盖其余五臂；CS 只含 A/C/D。E2 只增加 C 的 R3/O2，R0 指向 E0 C。顺序为 seed、配置内 dataset 次序、u.4/pair.4/clean/u.6、主 ratio 后敏感性；每个对应主 ratio E0 块后追加 E2；cached_mlp 先于 gcn。

Job 是包含 `id,dataset,noise,seed,backbone,arm,ratio,rule,experiment` 的完整字典。主实验 experiment=E0/rule=R0；追加实验 experiment=E2。ID 为 ASCII，浮点小数点替换为 p。

## 命令

在 v2 目录中，CPU 测试：

```powershell
& 'D:\codex_temp\noisygl-rocm721\Scripts\python.exe' test_suite.py
```

正式运行接口（由主任务执行；本子任务没有启动训练）：

```powershell
& 'D:\codex_temp\noisygl-rocm721\Scripts\python.exe' run_suite.py --config config_v2.json --run-root <独立正式运行目录>
```

调试使用独立运行目录，且保持其自己的配置文件：

```powershell
& 'D:\codex_temp\noisygl-rocm721\Scripts\python.exe' run_suite.py --config <调试配置.json> --run-root <独立调试目录> --limit 1
```

`--skip-prepare` 仅用于已具有 worker 所需缓存的目录。它不绕过配置、源码、清单或结果匹配校验。仅重建汇总：

```powershell
& 'D:\codex_temp\noisygl-rocm721\Scripts\python.exe' summarize.py --run-root <运行目录>
```

## 冻结与续跑

在任何结果产生之前保存输入配置原始字节为 `config.json`，保存 `protocol.json`（config_sha256、source_sha256、source_hashes、时间、解释器、超时处置），冻结总清单 `jobs.json` 和 `jobs/<id>.json`。所有 v2 顶层生产 `.py` 纳入 SHA256，`test_*` 排除；实际导入的 `GB-LCSS_GitHub/src/gblcss/{__init__.py,data.py,granular_ball.py,targets.py}` 也纳入，键名为 `support/gblcss/<文件名>`。source_sha256 为排序文件哈希字典的规范 JSON SHA256。外部数据和缓存的额外哈希由 prepare/worker 负责。

续跑必须匹配配置原始字节哈希、当前生产源码全部哈希、总清单及每个 job。存在旧结果却无 protocol 时拒绝接管。每个 completed 结果必须与 job 完全相同、config/source 哈希匹配，并具有父进程测得的完整墙钟；父进程尚未补计成本即中断的结果不能直接当作正式完成。任何已存在但非 completed 的结果也不会被默默跳过或覆盖，需要先核查现场。

结果包含 prediction_file/prediction_sha256 时，必须同时提供两项；解析后的真实路径必须位于该运行的 results 目录内，文件须存在且流式 SHA256 匹配。新结果接收、续跑和单独汇总均校验。损坏、缺失或路径越界的预测文件不能被汇总为完成。CPU stub 允许两项均不提供。

所有 CPU 准备先顺序完成，之后训练 worker 串行启动。调用均为当前 `sys.executable`，CLI 参数保持 prepare/worker 的约定。原始输出保存为 `logs/prepare_<dataset>.txt` 或 `logs/<id>.txt`，追加运行时间与命令。`status.json` 原子写入状态、completed/total/current、parent_pid/child pid、ISO 时间与当前子进程墙钟。

每个子进程按配置超时（正式配置 1800s）；超时或父任务异常会终止当前直接子进程并等待退出，保留文件，立即停队列。worker 必须不再启动独立训练子进程。非零退出、无结果、来源不符或结果非 completed 同样停队列。源文件在每次启动前及每次训练结果接收后重新核对。

父进程在子进程退出后添加 `parent_process_wall_s`（别名 `process_wall_s`）。该计时覆盖解释器启动、导入、加载和 worker 执行。`startup_overhead_s=max(0,parent_process_wall_s-run_total_s)`；原 worker 冷成本保存在 `body_cold_total_s`，正式 `cold_total_s` 加入该 overhead。等待使用子进程带超时等待，避免把固定轮询睡眠人为计入短作业成本。

准备进程在 `cache/<dataset>/metadata.json` 追加 `preparation_process_wall_s`，以及 `preparation_common_overhead_s=max(0,prepare_wall-sum(times))+provenance_s`，由 worker 分配到必要共享阶段。已有完整准备测量不被后续缓存读取的短墙钟覆盖。冷总成本是必要阶段成本重建总和，不能称独立无缓存端到端冷运行的直接测量；原始准备墙钟和 worker 墙钟分别保留。

父进程按配置固定 K=1/3/10/30 写入 `amortized_total_s[str(K)]=cold_total_s-shared_cold_s+shared_cold_s/K`。K=1 等于完整冷成本。假设为相同图、表示、候选树的 K 次任务，任务可以有不同 mask、噪声与模型 seed；标签相关切面评分不摊销，每次完整计入。原节点对照按其自己的共享依赖同样允许复用。汇总和配对表包含全部 K 的成本列。

## 汇总与结论边界

每完成一个作业即重建汇总；失败时也尽力重建。`runs.csv` 逐个保留计划作业（包含缺失、失败、无效），`summary.csv` 按 dataset/noise/backbone/arm/ratio/rule 汇总均值、样本标准差、有效观测数和 seed；精度仍为 [0,1] 比例。缺失、非数值或非有限字段输出 NA，不补零。显存、冷成本、执行成本及父进程墙钟分别保留。

`paired_deltas.csv` 比较 C_R0−A_R0/B_R0/Bprime_R0/LCSS_R0/D_R0、D_R0−A_R0，以及 C_R3/O2−C_R0。按 dataset/noise/seed/backbone 精确匹配；除 A 外还要求 ratio 相同。A 与预算无关且只训练 .1，因此敏感性预算明确用 reference_ratio=.1 指向同条件 A。输出配对 seed、逐 seed 差、百分点差 mean/std 和成本差；缺失参考不强配。所有负差保留。

`PROGRESS.md` 明示部分完成、失败原因与未知组合；少于三个 seed 不声称稳定。三 seed 只是描述性筛查，不构成显著性或等价证据，也不是同一条件的三次 timing repeat。±.5pp 是描述带。O2 只称 clean-label 切面诊断。

## CPU 验证记录

先添加测试并运行，观察到缺失 `run_suite` 模块；实现后七项测试通过。随后增加真实 CPU stub 子进程测试，观察到 `body_cold_total_s` 缺失的失败，再实现成本补计。审查后新增支持模块哈希与摊销断言，观察到相应失败后补齐。最后执行 `test_suite.py`，九项全部通过（0.289s），包括 1080 项清单、唯一 ID、A 去重、CS 限臂、E2 无基线重训、配置/源码/作业不匹配拒绝、四个实际支持模块哈希、严格 seed 配对、负差保留、缺字段汇总、真实子进程/续跑、固定 K 成本与超时终止。

这些测试没有导入 GPU/ML 运行时，没有训练模型，不能替代主任务的 worker 连通检查或研究实验验证。

最终冻结前追加 CS 的 C−A/D−A 与敏感性 A 参考测试、预测文件篡改/缺失/目录越界测试，先观察相应缺接口和缺配对失败。补齐后最新 CPU 验证 **11/11 通过（0.309s）**。
