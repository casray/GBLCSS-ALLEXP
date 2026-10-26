# D1回退、匹配KM与同树C1+D1实验报告

完成：2026-10-24T09:11:12.587107+00:00。本轮新科学fit 684，每个200epoch；4个两轮smoke。全部旧结果只读引用，目标内容寻址去重。
五图三seed、clean/U.4/U.6/Pair.4。零fit回退覆盖120格；新KM与同树组合仅60格R0。固定历史H/SSL与对应seed的split、noise、初始化。
本轮沿用已经开发和查看过的历史结果，归为预先固定规则的回顾性探索。规则、源码和复用清单冻结，全部选择冻结后才评分测试集；验证集同时用于checkpoint与配置选择。

## 假设与控制

1. CE及全局力度回退可缓解强制GROUP的clean/F1代价。accuracy主selector；balanced次selector=(validation accuracy+macro-F1)/2，仅使用每臂既定最佳准确率checkpoint，平票回原accuracy/NLL/epoch/稳定臂序。两个selector同时应用于基线和扩展池。
2. KM与粒球均采用四全局候选及八扩展候选；同一归一化H、相同区域数、相同学生协议。比较预算匹配的选择结果，并保存固定臂结果。
3. 深树固定S32/历史C1 PROBE选支持量 × global-only/expanded四格。PROBE若为CE，自适应分支保持CE。原C1所选配置单列，same-support global refinement单列；支持量不会依据新GROUP结果重选。
粒球用于局部留一软标签证据。beta=.5/min_votes2、四种分组力度及平票规则保持历史定义。全部条件完成，无新增成功硬门。

## 带噪宏均值

单位pp。先在seed内对五图三噪声等权平均，再计算seed均值和样本SD(ddof=1)。新增seed1/2单列。
|selector|骨干|方法−对照|三seed accuracy±SD|新seed accuracy±SD|三seed macro-F1差|
|---|---|---|---:|---:|---:|
|accuracy|R0|LEGACY_EXPANDED−LEGACY_GLOBAL|+0.777 ± 0.151|+0.714 ± 0.146|+1.056|
|accuracy|R0|LEGACY_EXPANDED−LEGACY_FORCED|-0.006 ± 0.023|-0.016 ± 0.023|+0.052|
|accuracy|R0|LEGACY_EXPANDED−CE|+10.340 ± 0.376|+10.411 ± 0.502|+10.920|
|accuracy|R0|LEGACY_GLOBAL−KM_GLOBAL|-0.523 ± 0.024|-0.527 ± 0.032|-0.745|
|accuracy|R0|LEGACY_EXPANDED−KM_EXPANDED|-0.713 ± 0.270|-0.696 ± 0.380|-0.702|
|accuracy|R0|KM_EXPANDED−KM_GLOBAL|+0.968 ± 0.378|+0.883 ± 0.493|+1.013|
|accuracy|R0|ADAPTIVE_EXPANDED−FIXED_GLOBAL|+1.229 ± 0.221|+1.291 ± 0.272|+1.496|
|accuracy|R0|ADAPTIVE_EXPANDED−FIXED_EXPANDED|+0.326 ± 0.290|+0.440 ± 0.300|+0.699|
|accuracy|R0|ADAPTIVE_EXPANDED−ADAPTIVE_GLOBAL|+0.753 ± 0.186|+0.859 ± 0.031|+0.887|
|accuracy|R0|ADAPTIVE_EXPANDED−ORIGINAL_C1|+0.753 ± 0.186|+0.859 ± 0.031|+0.887|
|accuracy|R0|ADAPTIVE_GLOBAL−ORIGINAL_C1|+0.000 ± 0.000|+0.000 ± 0.000|+0.000|
|accuracy|R0|ADAPTIVE_EXPANDED−SINGLE_MODULE_VAL|+0.249 ± 0.082|+0.246 ± 0.115|+0.530|
|accuracy|R0|FIXED_EXPANDED−FIXED_GLOBAL|+0.903 ± 0.091|+0.851 ± 0.029|+0.797|
|accuracy|R0|ADAPTIVE_GLOBAL−FIXED_GLOBAL|+0.476 ± 0.186|+0.432 ± 0.240|+0.608|
|balanced|R0|LEGACY_EXPANDED−LEGACY_GLOBAL|+0.961 ± 0.068|+1.000 ± 0.015|+1.212|
|balanced|R0|LEGACY_EXPANDED−LEGACY_FORCED|-0.004 ± 0.123|-0.019 ± 0.171|+0.057|
|balanced|R0|LEGACY_EXPANDED−CE|+10.390 ± 0.325|+10.496 ± 0.379|+11.008|
|balanced|R0|LEGACY_GLOBAL−KM_GLOBAL|-0.657 ± 0.216|-0.728 ± 0.252|-0.813|
|balanced|R0|LEGACY_EXPANDED−KM_EXPANDED|-0.615 ± 0.240|-0.560 ± 0.312|-0.713|
|balanced|R0|KM_EXPANDED−KM_GLOBAL|+0.919 ± 0.416|+0.832 ± 0.549|+1.112|
|balanced|R0|ADAPTIVE_EXPANDED−FIXED_GLOBAL|+1.243 ± 0.213|+1.362 ± 0.073|+1.566|
|balanced|R0|ADAPTIVE_EXPANDED−FIXED_EXPANDED|+0.318 ± 0.230|+0.401 ± 0.255|+0.577|
|balanced|R0|ADAPTIVE_EXPANDED−ADAPTIVE_GLOBAL|+0.654 ± 0.192|+0.761 ± 0.071|+0.901|
|balanced|R0|ADAPTIVE_EXPANDED−ORIGINAL_C1|+0.654 ± 0.192|+0.761 ± 0.071|+0.901|
|balanced|R0|ADAPTIVE_GLOBAL−ORIGINAL_C1|+0.000 ± 0.000|+0.000 ± 0.000|+0.000|
|balanced|R0|ADAPTIVE_EXPANDED−SINGLE_MODULE_VAL|+0.168 ± 0.097|+0.161 ± 0.137|+0.334|
|balanced|R0|FIXED_EXPANDED−FIXED_GLOBAL|+0.925 ± 0.144|+0.961 ± 0.182|+0.990|
|balanced|R0|ADAPTIVE_GLOBAL−FIXED_GLOBAL|+0.589 ± 0.022|+0.601 ± 0.001|+0.666|
|accuracy|R2|LEGACY_EXPANDED−LEGACY_GLOBAL|+0.535 ± 0.120|+0.604 ± 0.016|+0.713|
|accuracy|R2|LEGACY_EXPANDED−LEGACY_FORCED|-0.015 ± 0.053|-0.041 ± 0.039|+0.052|
|accuracy|R2|LEGACY_EXPANDED−CE|+3.467 ± 0.116|+3.443 ± 0.153|+3.198|
|balanced|R2|LEGACY_EXPANDED−LEGACY_GLOBAL|+0.565 ± 0.057|+0.532 ± 0.014|+0.789|
|balanced|R2|LEGACY_EXPANDED−LEGACY_FORCED|-0.070 ± 0.053|-0.052 ± 0.060|+0.095|
|balanced|R2|LEGACY_EXPANDED−CE|+3.398 ± 0.215|+3.291 ± 0.152|+3.367|

## 主selector逐图结果

|骨干|图|方法−对照|三seed带噪accuracy±SD|新seedaccuracy差|
|---|---|---|---:|---:|
|R0|cora|LEGACY_EXPANDED−LEGACY_GLOBAL|+1.736 ± 0.286|+1.574|
|R0|cora|LEGACY_EXPANDED−KM_EXPANDED|-0.850 ± 0.828|-1.020|
|R0|cora|ADAPTIVE_EXPANDED−FIXED_EXPANDED|+0.563 ± 1.021|+0.871|
|R0|cora|ADAPTIVE_EXPANDED−ADAPTIVE_GLOBAL|+1.489 ± 0.541|+1.671|
|R0|cora|ADAPTIVE_GLOBAL−ORIGINAL_C1|+0.000 ± 0.000|+0.000|
|R0|cora|ADAPTIVE_EXPANDED−SINGLE_MODULE_VAL|+0.668 ± 0.628|+1.029|
|R2|cora|LEGACY_EXPANDED−LEGACY_GLOBAL|+0.428 ± 0.481|+0.695|
|R0|citeseer|LEGACY_EXPANDED−LEGACY_GLOBAL|+1.136 ± 0.529|+1.296|
|R0|citeseer|LEGACY_EXPANDED−KM_EXPANDED|-0.692 ± 0.690|-0.365|
|R0|citeseer|ADAPTIVE_EXPANDED−FIXED_EXPANDED|+0.248 ± 0.869|-0.064|
|R0|citeseer|ADAPTIVE_EXPANDED−ADAPTIVE_GLOBAL|+1.145 ± 0.300|+1.310|
|R0|citeseer|ADAPTIVE_GLOBAL−ORIGINAL_C1|+0.000 ± 0.000|+0.000|
|R0|citeseer|ADAPTIVE_EXPANDED−SINGLE_MODULE_VAL|+0.248 ± 0.869|-0.064|
|R2|citeseer|LEGACY_EXPANDED−LEGACY_GLOBAL|+0.778 ± 0.041|+0.802|
|R0|corafull|LEGACY_EXPANDED−LEGACY_GLOBAL|+0.498 ± 0.488|+0.372|
|R0|corafull|LEGACY_EXPANDED−KM_EXPANDED|-0.934 ± 0.157|-1.022|
|R0|corafull|ADAPTIVE_EXPANDED−FIXED_EXPANDED|+0.079 ± 0.577|+0.380|
|R0|corafull|ADAPTIVE_EXPANDED−ADAPTIVE_GLOBAL|+0.361 ± 0.313|+0.279|
|R0|corafull|ADAPTIVE_GLOBAL−ORIGINAL_C1|+0.000 ± 0.000|+0.000|
|R0|corafull|ADAPTIVE_EXPANDED−SINGLE_MODULE_VAL|+0.088 ± 0.288|+0.070|
|R2|corafull|LEGACY_EXPANDED−LEGACY_GLOBAL|+0.202 ± 0.028|+0.214|
|R0|coauthorcs|LEGACY_EXPANDED−LEGACY_GLOBAL|+0.378 ± 0.075|+0.400|
|R0|coauthorcs|LEGACY_EXPANDED−KM_EXPANDED|+0.078 ± 0.317|+0.210|
|R0|coauthorcs|ADAPTIVE_EXPANDED−FIXED_EXPANDED|+0.416 ± 0.341|+0.574|
|R0|coauthorcs|ADAPTIVE_EXPANDED−ADAPTIVE_GLOBAL|+0.371 ± 0.132|+0.443|
|R0|coauthorcs|ADAPTIVE_GLOBAL−ORIGINAL_C1|+0.000 ± 0.000|+0.000|
|R0|coauthorcs|ADAPTIVE_EXPANDED−SINGLE_MODULE_VAL|+0.357 ± 0.116|+0.422|
|R2|coauthorcs|LEGACY_EXPANDED−LEGACY_GLOBAL|+0.571 ± 0.115|+0.579|
|R0|amazonpho|LEGACY_EXPANDED−LEGACY_GLOBAL|+0.139 ± 0.440|-0.072|
|R0|amazonpho|LEGACY_EXPANDED−KM_EXPANDED|-1.166 ± 0.738|-1.282|
|R0|amazonpho|ADAPTIVE_EXPANDED−FIXED_EXPANDED|+0.324 ± 0.933|+0.439|
|R0|amazonpho|ADAPTIVE_EXPANDED−ADAPTIVE_GLOBAL|+0.396 ± 0.352|+0.591|
|R0|amazonpho|ADAPTIVE_GLOBAL−ORIGINAL_C1|+0.000 ± 0.000|+0.000|
|R0|amazonpho|ADAPTIVE_EXPANDED−SINGLE_MODULE_VAL|-0.118 ± 0.507|-0.224|
|R2|amazonpho|LEGACY_EXPANDED−LEGACY_GLOBAL|+0.695 ± 0.110|+0.728|

## Clean与F1代价

所有clean配对结果均在paired_summary；下表突出clean accuracy下降>2pp或macro-F1下降>1pp。

|selector|seed范围|骨干|图/条件|方法−对照|accuracy差|F1差|
|---|---|---|---|---|---:|---:|
|accuracy|0,1,2|R0|cora/uniform0.4|LEGACY_GLOBAL−KM_GLOBAL|-1.372|-1.437|
|accuracy|0,1,2|R0|cora/uniform0.4|LEGACY_EXPANDED−KM_EXPANDED|-1.249|-1.332|
|balanced|0,1,2|R0|cora/uniform0.4|LEGACY_GLOBAL−KM_GLOBAL|-1.372|-1.437|
|balanced|0,1,2|R0|cora/uniform0.4|LEGACY_EXPANDED−KM_EXPANDED|-1.214|-1.010|
|accuracy|0,1,2|R0|cora/uniform0.6|LEGACY_GLOBAL−KM_GLOBAL|-1.055|-1.061|
|accuracy|0,1,2|R0|cora/uniform0.6|LEGACY_EXPANDED−KM_EXPANDED|-1.249|-1.279|
|balanced|0,1,2|R0|cora/uniform0.6|LEGACY_GLOBAL−KM_GLOBAL|-1.055|-1.061|
|balanced|0,1,2|R0|cora/uniform0.6|LEGACY_EXPANDED−KM_EXPANDED|-1.847|-1.918|
|balanced|0,1,2|R0|cora/pair0.4|LEGACY_GLOBAL−KM_GLOBAL|-1.214|-1.008|
|accuracy|0,1,2|R0|citeseer/clean|LEGACY_EXPANDED−LEGACY_GLOBAL|-0.172|-1.396|
|accuracy|0,1,2|R0|citeseer/clean|FIXED_EXPANDED−FIXED_GLOBAL|-0.172|-1.396|
|accuracy|0,1,2|R2|citeseer/uniform0.4|LEGACY_EXPANDED−CE|+0.100|-1.666|
|balanced|0,1,2|R0|citeseer/uniform0.6|ADAPTIVE_EXPANDED−SINGLE_MODULE_VAL|-1.260|-1.247|
|accuracy|0,1,2|R0|citeseer/pair0.4|LEGACY_GLOBAL−KM_GLOBAL|-1.188|-1.132|
|balanced|0,1,2|R0|citeseer/pair0.4|LEGACY_GLOBAL−KM_GLOBAL|-1.188|-1.132|
|accuracy|0,1,2|R0|corafull/uniform0.6|LEGACY_GLOBAL−KM_GLOBAL|-1.133|-1.282|
|accuracy|0,1,2|R0|corafull/uniform0.6|LEGACY_EXPANDED−KM_EXPANDED|-0.926|-1.215|
|balanced|0,1,2|R0|corafull/uniform0.6|LEGACY_GLOBAL−KM_GLOBAL|-1.133|-1.282|
|accuracy|0,1,2|R0|amazonpho/uniform0.4|LEGACY_GLOBAL−KM_GLOBAL|-0.081|-1.910|
|accuracy|0,1,2|R0|amazonpho/uniform0.4|LEGACY_EXPANDED−KM_EXPANDED|-0.554|-2.059|
|balanced|0,1,2|R0|amazonpho/uniform0.4|LEGACY_GLOBAL−KM_GLOBAL|-0.081|-1.910|
|accuracy|0,1,2|R0|amazonpho/uniform0.6|LEGACY_GLOBAL−KM_GLOBAL|-0.504|-1.776|
|accuracy|0,1,2|R0|amazonpho/uniform0.6|LEGACY_EXPANDED−KM_EXPANDED|-1.326|-2.592|
|balanced|0,1,2|R0|amazonpho/uniform0.6|LEGACY_GLOBAL−KM_GLOBAL|-0.504|-1.776|
|balanced|0,1,2|R0|amazonpho/uniform0.6|LEGACY_EXPANDED−KM_EXPANDED|-1.008|-1.859|
|accuracy|0,1,2|R0|amazonpho/pair0.4|KM_EXPANDED−KM_GLOBAL|+1.139|-1.772|
|accuracy|0,1,2|R0|amazonpho/pair0.4|FIXED_EXPANDED−FIXED_GLOBAL|+0.965|-1.552|
|balanced|0,1,2|R0|amazonpho/pair0.4|LEGACY_EXPANDED−KM_EXPANDED|-1.743|-1.711|
|accuracy|1,2|R0|cora/uniform0.4|LEGACY_GLOBAL−KM_GLOBAL|-1.108|-1.397|
|accuracy|1,2|R0|cora/uniform0.4|LEGACY_EXPANDED−KM_EXPANDED|-1.187|-1.498|
|balanced|1,2|R0|cora/uniform0.4|LEGACY_GLOBAL−KM_GLOBAL|-1.108|-1.397|
|balanced|1,2|R0|cora/uniform0.4|LEGACY_EXPANDED−KM_EXPANDED|-1.135|-1.014|
|accuracy|1,2|R0|cora/uniform0.6|LEGACY_GLOBAL−KM_GLOBAL|-1.504|-1.293|
|accuracy|1,2|R0|cora/uniform0.6|LEGACY_EXPANDED−KM_EXPANDED|-1.794|-1.598|
|balanced|1,2|R0|cora/uniform0.6|LEGACY_GLOBAL−KM_GLOBAL|-1.504|-1.293|
|balanced|1,2|R0|cora/uniform0.6|LEGACY_EXPANDED−KM_EXPANDED|-2.691|-2.556|
|balanced|1,2|R0|cora/pair0.4|LEGACY_GLOBAL−KM_GLOBAL|-2.190|-1.544|
|accuracy|1,2|R0|citeseer/clean|LEGACY_EXPANDED−LEGACY_GLOBAL|-0.408|-1.493|
|accuracy|1,2|R0|citeseer/clean|ADAPTIVE_EXPANDED−FIXED_GLOBAL|-0.408|-1.493|
|accuracy|1,2|R0|citeseer/clean|ADAPTIVE_EXPANDED−ADAPTIVE_GLOBAL|-0.408|-1.493|
|accuracy|1,2|R0|citeseer/clean|FIXED_EXPANDED−FIXED_GLOBAL|-0.408|-1.493|
|balanced|1,2|R0|citeseer/clean|LEGACY_EXPANDED−LEGACY_GLOBAL|+0.043|-1.079|
|balanced|1,2|R0|citeseer/clean|LEGACY_EXPANDED−KM_EXPANDED|-0.752|-1.512|
|balanced|1,2|R0|citeseer/clean|ADAPTIVE_EXPANDED−FIXED_GLOBAL|+0.043|-1.079|
|balanced|1,2|R0|citeseer/clean|ADAPTIVE_EXPANDED−ADAPTIVE_GLOBAL|+0.043|-1.079|
|balanced|1,2|R0|citeseer/clean|FIXED_EXPANDED−FIXED_GLOBAL|+0.043|-1.079|
|accuracy|1,2|R2|citeseer/uniform0.4|LEGACY_EXPANDED−CE|+0.558|-2.388|
|balanced|1,2|R0|citeseer/uniform0.6|ADAPTIVE_EXPANDED−FIXED_EXPANDED|-1.847|-1.724|
|balanced|1,2|R0|citeseer/uniform0.6|ADAPTIVE_EXPANDED−SINGLE_MODULE_VAL|-1.847|-1.724|
|accuracy|1,2|R0|corafull/uniform0.4|LEGACY_GLOBAL−KM_GLOBAL|-1.476|-1.171|
|accuracy|1,2|R0|corafull/uniform0.4|LEGACY_EXPANDED−KM_EXPANDED|-1.909|-1.572|
|balanced|1,2|R0|corafull/uniform0.4|LEGACY_GLOBAL−KM_GLOBAL|-1.476|-1.171|
|accuracy|1,2|R0|corafull/uniform0.6|LEGACY_GLOBAL−KM_GLOBAL|-1.241|-1.375|
|accuracy|1,2|R0|corafull/uniform0.6|LEGACY_EXPANDED−KM_EXPANDED|-0.826|-1.075|
|balanced|1,2|R0|corafull/uniform0.6|LEGACY_GLOBAL−KM_GLOBAL|-1.241|-1.375|
|accuracy|1,2|R0|amazonpho/uniform0.4|LEGACY_GLOBAL−KM_GLOBAL|-0.149|-1.593|
|accuracy|1,2|R0|amazonpho/uniform0.4|LEGACY_EXPANDED−KM_EXPANDED|-0.682|-1.162|
|accuracy|1,2|R0|amazonpho/uniform0.4|FIXED_EXPANDED−FIXED_GLOBAL|+0.149|-1.337|
|balanced|1,2|R0|amazonpho/uniform0.4|LEGACY_GLOBAL−KM_GLOBAL|-0.149|-1.593|
|balanced|1,2|R0|amazonpho/uniform0.4|FIXED_EXPANDED−FIXED_GLOBAL|+0.261|-1.016|
|accuracy|1,2|R0|amazonpho/uniform0.6|LEGACY_GLOBAL−KM_GLOBAL|-0.775|-1.451|
|accuracy|1,2|R0|amazonpho/uniform0.6|LEGACY_EXPANDED−KM_EXPANDED|-1.671|-3.708|
|balanced|1,2|R0|amazonpho/uniform0.6|LEGACY_GLOBAL−KM_GLOBAL|-0.775|-1.451|
|balanced|1,2|R0|amazonpho/uniform0.6|LEGACY_EXPANDED−KM_EXPANDED|-1.195|-2.610|
|accuracy|1,2|R0|amazonpho/pair0.4|LEGACY_GLOBAL−KM_GLOBAL|-1.186|-1.914|
|accuracy|1,2|R0|amazonpho/pair0.4|KM_EXPANDED−KM_GLOBAL|+0.317|-4.117|
|accuracy|1,2|R0|amazonpho/pair0.4|FIXED_EXPANDED−FIXED_GLOBAL|+1.251|-1.459|
|balanced|1,2|R0|amazonpho/pair0.4|LEGACY_GLOBAL−KM_GLOBAL|-1.186|-1.914|
|balanced|1,2|R0|amazonpho/pair0.4|LEGACY_EXPANDED−KM_EXPANDED|-1.681|-1.625|

## 回退与搜索成本

- R0/accuracy/clean 新seed回退分布：{'L0.25': 6, 'CE': 2, 'GROUP_A0.25_C1.0': 1, 'L0.5': 1}。
- R0/accuracy/uniform0.4 新seed回退分布：{'GROUP_A0.25_C1.0': 6, 'GROUP_A0.0_C1.0': 4}。
- R0/accuracy/uniform0.6 新seed回退分布：{'GROUP_A0.0_C1.0': 6, 'GROUP_A0.25_C1.0': 3, 'L1.0': 1}。
- R0/accuracy/pair0.4 新seed回退分布：{'GROUP_A0.0_C1.0': 6, 'L1.0': 1, 'GROUP_A0.25_C1.0': 3}。
- R0/balanced/clean 新seed回退分布：{'L0.25': 4, 'CE': 2, 'GROUP_A0.25_C1.0': 1, 'GROUP_A1.0_C0.25': 2, 'L0.5': 1}。
- R0/balanced/uniform0.4 新seed回退分布：{'GROUP_A0.0_C1.0': 5, 'GROUP_A0.25_C1.0': 4, 'L1.0': 1}。
- R0/balanced/uniform0.6 新seed回退分布：{'L1.0': 2, 'GROUP_A0.0_C1.0': 6, 'GROUP_A0.25_C1.0': 2}。
- R0/balanced/pair0.4 新seed回退分布：{'L1.0': 2, 'GROUP_A0.0_C1.0': 5, 'GROUP_A0.25_C1.0': 3}。
- R2/accuracy/clean 新seed回退分布：{'L0.5': 1, 'CE': 2, 'L1.0': 1, 'GROUP_A1.0_C0.25': 2, 'GROUP_A1.0_C0.0': 3, 'L0.25': 1}。
- R2/accuracy/uniform0.4 新seed回退分布：{'GROUP_A0.0_C1.0': 7, 'GROUP_A0.25_C1.0': 1, 'L1.0': 1, 'L0.5': 1}。
- R2/accuracy/uniform0.6 新seed回退分布：{'GROUP_A0.0_C1.0': 8, 'L0.25': 1, 'GROUP_A0.25_C1.0': 1}。
- R2/accuracy/pair0.4 新seed回退分布：{'GROUP_A0.0_C1.0': 9, 'GROUP_A0.25_C1.0': 1}。
- R2/balanced/clean 新seed回退分布：{'L0.5': 1, 'CE': 1, 'GROUP_A1.0_C0.25': 2, 'L1.0': 1, 'GROUP_A1.0_C0.0': 4, 'L0.25': 1}。
- R2/balanced/uniform0.4 新seed回退分布：{'GROUP_A0.0_C1.0': 4, 'GROUP_A0.25_C1.0': 3, 'CE': 1, 'L1.0': 1, 'L0.5': 1}。
- R2/balanced/uniform0.6 新seed回退分布：{'GROUP_A0.0_C1.0': 3, 'L0.5': 1, 'L0.25': 1, 'L1.0': 2, 'GROUP_A0.25_C1.0': 2, 'GROUP_A1.0_C0.25': 1}。
- R2/balanced/pair0.4 新seed回退分布：{'GROUP_A0.0_C1.0': 7, 'GROUP_A0.25_C1.0': 3}。

本轮实际新学生更新136800次，训练循环计时2457.448s，输入、哈希、序列化与分析时间另计。历史完整C1网格已支付432000候选更新，作为上游既有研究成本披露。
预算匹配KM/GB的global-only各800更新，expanded各1600；本轮增量复用CE/KMλ1与全部legacy。深树固定支持global-only800、expanded1600。
自适应分支部署成本逐cell含历史PROBE1000+CE200，再补齐同支持三λ未完成轮数，最后四group800：global总更新范围1200–1550，expanded范围1200–2350。
缓存中的完整λ候选在本次GPU上通过别名复用；部署估计仍补计原PROBE尚未支付的更新。CE分支保留上游1200成本。共同执行四格时共享支持、CE和相同目标；单分支成本与实际新拟合数量分别报告。
per_class_recall.csv与strata.csv提供逐类召回、干净/错标及同意/冲突/平票交叉诊断；真值仅用于冻结后的分析。

## 审计与决策说明

审计passed：684新fit，2256逻辑引用，120冻结cell，27重载样本最大概率误差0.0。
研究判断依赖新seed的方向、跨图覆盖、与匹配强对照的差及clean/F1成本。+.3–.5pp作组合量级参照，无自动成功硬门；所有负结果保留。
已支持事实为表中配对结果及审计；分组改变纠错/保留权衡是机制解释；独立SSL随机性、新图迁移与额外规则仍待验证。后续结论补充于报告末尾，当前脚本不会发起下一轮实验。
