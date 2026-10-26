# 工程记录

1. 测试先行：实现 router.py 前，5 项测试入口因模块未创建而失败；实现后 5 项全部通过。
2. 20 个单元的 620 个逻辑回归拟合槽位、选模和评分完成后，最终哈希审计报中文路径不存在。
   最小复现实验：sources.json 默认编码读取，0/9 路径存在；显式 UTF-8 读取，9/9 存在。
   原因是 UTF-8 写入配合 Windows 默认编码读取。
3. run.py 只在 sources.json 的 read_text 调用增加 encoding='utf-8'。
   科学代码、数据、参数、模型、预测与首次选模文件保持原样。
   原 sources.json 与 failure.txt 保留，finalize.py 将修复行还原后计算摘要核对原版本，
   同时记录修复后代码摘要。后处理恢复仅执行 finalize.py，未重跑实验拟合。
4. raw.jsonl 的 complete 事件表示分析完成；交付以 complete.json 和 final_audit.json 为准。
