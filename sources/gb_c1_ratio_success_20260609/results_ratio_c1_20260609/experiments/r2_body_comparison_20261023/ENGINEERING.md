# 工程记录

原始冻结入口完成480个R2分类头后，复用Cora uniform.4的GSSC时读取旧config.json失败。
检查确认原结果把配置存储在result.json的config字段；旧目录没有独立config.json。
新增external_v2.py只修改这一处配置读取；external.py和原sources.json保留。
recover.py核对原源文件、跳过已完成单元，从外部基线阶段恢复。
全部R2结果、外部已完成的clean GSSC和两个SimMLP探针均保留；失败发生于复用读取，未启动新拟合。
failure.json保留首次失败；recovery_sources.json记录修复代码摘要。
本次恢复不改变科学参数、选模规则、训练预算或结果取舍。

## 后台接续
外部大图GSSC单元约12分钟，完整剩余队列较长，切换为独立后台串行任务。
只停止本实验run.lock记录并核实路径的前台Python进程113432；未操作其他任务。
当前CoraFull pair.4未完成单元移至interrupted/corafull_pair04_GSSC_foreground，
保留全部已运行日志；该单元从相同seed0及相同配置重新完整拟合，旧部分轨迹不进入结果表。
全部已完成R2及外部单元跳过。background.py按recover.py → verify_and_interpret.py
自动执行，输出background_console.log与background_status.json。
原run.lock因强制结束留下，核实进程退出后移除；新队列使用独立background.lock防重复。

## 全部训练后的汇总恢复
480个R2和40个外部结果全部完成，recover.py末尾from analyze导入了旧实验的同名模块，
原因是借用的历史运行器修改了sys.path。未产生汇总结果，不涉及训练失败。
新增finalize_results.py用绝对路径载入本目录analyze.py，再给独立复核器绑定本模块，
保留原始冻结源与后台失败日志，只恢复分析和480模型重载审计。
