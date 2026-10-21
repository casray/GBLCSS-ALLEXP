# GBLCSS-ALLEXP

GB-LCSS and related GB–GCL research archive, assembled on 2026-10-21 from the owner's historical workspace.
本仓库保留历史代码、配置、日志、结果、实验复盘和稿件版本，包括失败实验；不是只展示最终成功结果的精简代码仓库。

## Start here / 阅读入口

- [实验目录索引](EXPERIMENT_INDEX.md)：当前主实验目录下的 52 个历史分支。
- [项目核心上下文](sources/项目核心总结_新窗口上下文.md)、[项目交接](sources/99_session_handoff.md)。
- [历史实验记录](sources/experiment_history_and_lessons)、[文献研究地图](sources/01_literature_and_research_map.md)。
- [GB-LCSS implementation](sources/gb_c1_ratio_success_20260609/results_ratio_c1_20260609/GB-LCSS_code)、[全部主实验目录](sources/gb_c1_ratio_success_20260609/results_ratio_c1_20260609/experiments)。
- [后续研究与复盘](sources/gb_c1_ratio_success_20260609/results_ratio_c1_20260609/research)、[更早的GRACE实验](sources/gb_graph_aware_minival_6_1)。
- [近期R2机制对照](sources/gb_c1_ratio_success_20260609/results_ratio_c1_20260609/experiments/r2_mechanism_controls_20261021/综合结论与接续.md)。
- [边界诊断与粗化复盘](sources/gb_c1_ratio_success_20260609/results_ratio_c1_20260609/experiments/r2_boundary_diagnostic_20261021/综合结论与接续.md)。

## What is actually included / 归档范围

复制了 32,158 个源文件（2.363 GB，十进制）。
`sources/` 下保留相对原项目根目录的路径，区分原方法、后续尝试、基线实现与历史版本。
模型权重、数组缓存和原始数据按所有者选择保留本地，**没有声称它们已上传**。
完整盘点 51,241 个文件，逐文件路径、大小、SHA-256及上传/留存原因见 [files.csv](manifests/files.csv)。
[retained_locally.jsonl](manifests/retained_locally.jsonl) 专门列出本地留存项；
[archive_summary.json](manifests/archive_summary.json) 给出数量与排除范围。

环境、依赖缓存、Git内部对象、编译临时文件、重复分发压缩包不作为实验源文件上传。
潜在凭据签名匹配的文件需人工审核，未公开其内容；清单只给路径与检测类型。
参考论文库、NIDS和开放意图的独立完整项目未并入本仓库。
`next_paper/` 中保留用于GB-LCSS选题复盘的路线比较与小规模转向记录，不将它们冒充GB-LCSS主实验。
相关图对比前身与第三方对照源码保留供追溯。

## Reproduction and interpretation / 复现边界

这是忠实的历史快照，不是承诺一条命令即可重跑全部实验的统一软件包。旧脚本中的绝对Windows路径没有擅自改写。
原根目录为 `C:/Users/Admin/Desktop/实验8去噪5.19-`；迁移到其他机器需按各分支说明调整路径与环境，
并另行取得清单中的本地数据/权重。不要仅因目录里存在结果就认定该方向通过了成功门槛。
单种子、开发集反复使用、不同骨干/部署条件和负结果均须以原分支报告为准。

## Third-party material / 第三方代码

`sources/external/`、`upstream/`、`vendor/` 等保留对应源代码中已有的LICENSE及来源说明。
本仓库不为所有第三方文件统一重新许可，也不把基线源码宣称为本项目原创。

## Integrity / 完整性

源文件复制前后SHA-256一致，原实验文件不改动。上传完成后核对Git树中的文件字节与清单。
归档脚本及范围规则在 [tools/](tools/)，完整上传状态与远端核验见 [UPLOAD_STATUS.md](UPLOAD_STATUS.md)。

另有9份下载目录中的GB-LCSS稿件原件，见 [补充材料](supplemental/README.md) 和独立校验清单。
