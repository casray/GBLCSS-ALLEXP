# R2 正文对照实验交接

## 完成更新
480个R2头与40个外部单元全部完成。后台末尾因同名analyze模块导入冲突未汇总；
finalize_results.py通过绝对路径恢复分析与480模型重载，未重训。
REPORT_zh.md、summary.csv、paired.csv、matched_seed0.csv、decision.json、delivery_audit.json均已生成。
审计通过：480 checkpoint概率误差0，104额外源哈希，五份GSSC划分/噪声一致。
三seed五图带噪均值 CE67.630/GCE71.151/OLD64 69.940/SUPPORT70.615/KM70.985。
SUPPORT比CE+2.985pp、比OLD64+.675；比GCE−.536、比KM−.370，逐数据集KM五图均更高。
clean对CE：Cora−1.953pp、CoraFull−1.030pp；验证选固定竞争者后SUPPORT五图均负，门失败。
外部相同seed0的SimMLP在4/5图高于SUPPORT，CS例外；GSSC本协议下较弱。
最终保留R2骨干扩展和LCSS的带噪增量，停止扩当前粒球监督变体；无待续训练。
以下为历史启动记录。

用户已授权设计并运行对照实验。PLAN/config冻结。全部480个R2分类头已完成；
SimMLP/GSSC五图四条件seed0外部对照正在执行，恢复入口recover.py。
前台exec session 51627已退出，当前由background.py独立串行接续；不要重复启动。
读取background_status.json、status.json和external各epochs.jsonl查看进度。
原始session63382在复用GSSC独立config缺失处退出；failure.json保留。
修复external_v2只改为读取old result.json中的config，旧external.py和source manifest保持。

外部对照全部结束后，recover.py自动运行analyze.py产生results/summary/paired/REPORT/audit/complete。
后台会自动接着执行：
D:/codex_temp/noisygl-rocm721/Scripts/python.exe experiments/r2_body_comparison_20261023/verify_and_interpret.py
它会重载480模型、复核编码器与初始化、历史GSSC划分噪声、补相同seed0对照表和门槛。
additional_sources.json、recovery_sources.json另存新增审计源。

注意：等待全部拟合完成再评估test，不提前用test决定任何扩展。
GSSC完整预算为edge_pretrain + classifier_pretrain +200 main，CoraFull有约12.7万边，每轮248batch，耗时较大。
保留此预算与作者结构，外部方法和R2的端到端成本单独说明。
四个单元测试通过；Cora uniform.4的R2 SUPPORT与历史桥接最大概率差2.384e-7，target差5.96e-8。
正文/旧实验/远端未改。结束后更新本文件、99_session_handoff.md和实验历史入口，完整汇报正负结果。
