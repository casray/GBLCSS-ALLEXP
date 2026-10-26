# 主会话复核与结论

684次新正式拟合，4次两轮smoke，11项测试；GPU工作结束。旧结果只读复用，无稿件修改/上传。主侧独立核验2040个选择、2040条配对差、3672行摘要、30个CPU重载和sklearn指标；最大概率误差7.7486e-7，测试类别一致。详见PARENT_AUDIT.json。

## 主要结果

所有主数字用accuracy选择器；三噪声、五图等权，seed内先平均，再样本SD。balanced为事先冻结的次分析。新训练只跑R0，R2回退完全复用。

- 回退：legacy粒球扩展八候选对全局四候选，带噪R0 +.777±.151pp、R2 +.535±.120pp；与强制分组相比带噪几乎持平。R0 clean相对全局：Cora/CF/CS/Photo均为0，CiteSeer accuracy-.172、macroF1-1.396。回退缓解了大部分先前代价，保留CiteSeer负结果。
- 匹配KM：相同区域数、候选和预算下，GB扩展−KM扩展为-.713±.270pp，新增seed1/2 -.696。KM分组对KM全局+.968±.378；GB分组对GB全局+.777。两种分区均受益，局部监督分配具有跨分区用途。相对KM的粒球优势未出现。
- 同树组合：adaptive-expanded−fixed-global +1.229±.221pp；对fixed-expanded +.326±.290；对adaptive-global +.753±.186；对验证集选出的single-module对照+.249±.082。新增seed1/2对single-module+.246。逐图该增量Cora+.668、CiteSeer+.248、CF+.088、CS+.357、Photo-.118。
- 主侧额外复算combo−KM扩展：seed0/1/2为-.5887686/-.4169406/-.0791946pp，均值-.3616346、SD.2592498。保持比较口径，KM为legacy同数量区域、combo支持量自适应。
- balanced下组合对single-module+.168pp，对KM的legacy差-.615pp；总体方向相近，未依据测试重选主selector。

## 代价与决定

分组方法可作为GB-LCSS的监督侧增强；回退列入完整候选集。组合有小幅叠加效果，幅度应连同额外搜索成本报告。八候选global+group需1600更新，组合含上游PROBE及补齐三λ后的分支预算1200–2350；旧全网格缓存复用仍计其部署补训需求。本轮实际684*200=136800新更新。

当前优先保留简单的分组+回退机制，组合为可选增强。普通KM同样受益并更强，论文机制可围绕局部证据冲突与监督分配展开。已有五图/种子已用于开发，这轮属于固定规则的回顾性探索；后续确认需预先冻结方法，另行授权。无自动更多实验。

完整结果：experiments/d1_fallback_km_combo_20261024/REPORT_zh.md、paired_summary.csv、seeds12_paired_summary.csv、selected_results.csv、per_class_recall.csv、strata.csv、deployment_costs.csv及全部日志/概率/权重/复用哈希。
