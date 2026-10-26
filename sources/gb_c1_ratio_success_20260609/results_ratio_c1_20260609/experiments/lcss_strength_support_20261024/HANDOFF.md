# 完成交接

状态：COMPLETE。本地audit及补充审计通过，科学训练05:39:34结束，05:42:05后处理完成；没有待续训练。
完成量：1500个完整分类拟合、20个CPU预测器、5个两轮smoke。另有122更新的受控中断轨迹，原样保存在repairs。
报告入口：REPORT_zh.md；主表results.csv含1500行；summary.csv、selected_results.csv及paired_differences.csv保留全部条件。
优先结论：D1 R0分组对逐格最佳全局+1.123pp，对精确均量/打乱+2.793/+2.215；R2增量较弱。C1接近FULL_VAL；C2及D2暂缓。详细小类代价和负结果见报告。
独立复核入口：audit.json、supplement_manifest.json、run_passports.json、postprocessing_sources.json、repair_sources.json、sources.json/inputs.json。
34个checkpoint重载及72组连续/续训桥的全节点概率误差均0；11项最终测试通过；seed0 std留空。
CS R0历史SUPPORT32四格未列入新增预算，保持缺失；CiteSeer四格因分区/目标完全一致而有精确别名。没有额外图、种子或融合训练。
原冻结根目录源码保持不变；D2第9静态特征在首个D2拟合前经postprocess/run_repaired.py纠正。Windows路径及alias展示通过postprocess/finalize.py独立后处理修复，未重训。
future只读审计可从检查点与固定输入重载；当前无需再次启动run.py。若重生成报表，使用指定Python -B postprocess/finalize.py并先保留人工结论。
所有变更文件见CHANGED_FILES.txt；完整交付哈希见DELIVERY_SHA256.csv。所有写入仅在本实验目录；research/旧实验/稿件/远端均未改。

## 已解决的原分析入口故障（原始轨迹保留）
Traceback (most recent call last):
  File "C:\Users\Admin\Desktop\实验8去噪5.19-\gb_c1_ratio_success_20260609\results_ratio_c1_20260609\experiments\lcss_strength_support_20261024\run.py", line 452, in main
    analyze()
  File "C:\Users\Admin\Desktop\实验8去噪5.19-\gb_c1_ratio_success_20260609\results_ratio_c1_20260609\experiments\lcss_strength_support_20261024\analyze.py", line 115, in analyze
    add('B1', name, b, cond, family+'_'+policy, fitpath('B1', name, b, cond, arm))
  File "C:\Users\Admin\Desktop\实验8去噪5.19-\gb_c1_ratio_success_20260609\results_ratio_c1_20260609\experiments\lcss_strength_support_20261024\analyze.py", line 101, in add
    selector=selector, folder=folder, seed=0, seed_std=None, **by_folder[folder]))
                                                               ~~~~~~~~~^^^^^^^^
KeyError: 'fits/B1/cora/R0/clean/SUPPORT_L1.0'
