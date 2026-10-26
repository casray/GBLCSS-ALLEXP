# R2 + SUPPORT + D1 + C1 集成实验报告

## 研究问题与协议

在冻结历史 GRACE 与每seed R2编码器上，检验支持量、分组力度与短训联合选择的增量。粒球提供 LOO 局部软标签证据，R2冻结编码器/BN，只训练属性学生头。五图、clean/U.4/U.6/Pair.4、seed0/1/2，共60格；所有图与seed已有开发历史，定位为回顾性集成验证。

beta=.5、min_votes=2、平票不锐化；D1使用原始投票唯一赢家定义同意/冲突。本轮明确修改平票力度为λ1，固定S32与联合候选完全一致。训练目标仅由observed标签构建，选择仅用clean validation；真值诊断与test评分在全部选择冻结后进行。

42候选=6支持×(3全局力度+4分组规则)，42→14→4，25/75/200轮。强制S32λ1包含在存活名额内。排名依次为验证accuracy、最佳checkpoint NLL、全训练节点平均施加λ、更大支持、更早best epoch、冻结ID（CE/global优先）。18→6→2全局对照沿用原排名，逐格验证旧PROBE和两轮survivors。

## 实现与复用

fit.py与上轮验证内核逐字节一致；checkpoint格式head_with_encoder_reference。历史全局候选通过完整轨迹的因果前缀回放复用，当前前缀之外的成绩不进入排名。所有新D1真实训练，逐arm保存及恢复optimizer/RNG，25/75前缀快照保留。联合决策冻结后补齐固定S32控制。OLD64旧格式缺完整input/RNG契约，60格按共同内核重训并在末尾做概率桥。未训练R0、SSL、KM/kNN、教师或额外seed。

## 带噪总体结果

先对每seed内五图×三噪声条件等权平均，再报三seed均值和样本SD(ddof1)，单位pp。seed1/2单列用于观察方向；本轮三seed均有先前开发历史。

|比较|accuracy均值±SD|macroF1均值±SD|seed1/2 accuracy均值±SD|
|---|---:|---:|---:|
|ADAPTIVE_EXPANDED−SINGLE_MODULE_VAL|+0.307±0.165|+0.474±0.242|+0.385±0.137|
|ADAPTIVE_EXPANDED−CE|+3.901±0.254|+3.871±0.486|+3.899±0.359|
|ADAPTIVE_EXPANDED−GCE|+0.379±0.298|+1.029±0.316|+0.548±0.078|
|ADAPTIVE_EXPANDED−OLD64|+1.591±0.075|+1.524±0.156|+1.553±0.054|
|ADAPTIVE_EXPANDED−LEGACY_SUPPORT32|+0.916±0.214|+1.343±0.252|+1.017±0.172|
|ADAPTIVE_EXPANDED−FIXED_S32|+0.940±0.208|+1.373±0.208|+1.032±0.189|
|ADAPTIVE_EXPANDED−FIXED_GLOBAL|+0.995±0.230|+1.428±0.153|+1.078±0.255|
|ADAPTIVE_EXPANDED−FIXED_EXPANDED|+0.488±0.171|+0.753±0.189|+0.509±0.236|
|ADAPTIVE_EXPANDED−ADAPTIVE_GLOBAL|+0.579±0.077|+0.792±0.079|+0.594±0.102|
|FIXED_EXPANDED−FIXED_GLOBAL|+0.507±0.108|+0.675±0.083|+0.569±0.018|
|ADAPTIVE_GLOBAL−FIXED_GLOBAL|+0.416±0.278|+0.635±0.204|+0.484±0.357|

## 逐图主比较及clean/F1代价

|图/条件|参照|accuracy差±SD|macroF1差±SD|
|---|---|---:|---:|
|cora/clean|SINGLE_MODULE_VAL|+0.229±0.309|+0.249±0.389|
|cora/clean|CE|-0.035±0.351|-0.045±0.298|
|cora/clean|FIXED_GLOBAL|+0.053±0.346|+0.059±0.361|
|cora/NOISY_MACRO|SINGLE_MODULE_VAL|+0.610±0.583|+0.768±0.794|
|citeseer/clean|SINGLE_MODULE_VAL|-0.100±0.292|-0.037±0.952|
|citeseer/clean|CE|+0.315±0.252|+0.590±0.649|
|citeseer/clean|FIXED_GLOBAL|+0.200±0.286|+0.445±0.983|
|citeseer/NOISY_MACRO|SINGLE_MODULE_VAL|+0.220±0.147|+0.230±0.250|
|corafull/clean|SINGLE_MODULE_VAL|+0.128±0.318|+0.340±0.505|
|corafull/clean|CE|-0.046±0.531|-0.286±1.044|
|corafull/clean|FIXED_GLOBAL|+0.111±0.347|+0.462±0.188|
|corafull/NOISY_MACRO|SINGLE_MODULE_VAL|+0.048±0.200|+0.314±0.402|
|coauthorcs/clean|SINGLE_MODULE_VAL|-0.023±0.099|-0.058±0.137|
|coauthorcs/clean|CE|+0.013±0.044|+0.011±0.019|
|coauthorcs/clean|FIXED_GLOBAL|-0.026±0.104|-0.081±0.177|
|coauthorcs/NOISY_MACRO|SINGLE_MODULE_VAL|+0.213±0.039|+0.594±0.387|
|amazonpho/clean|SINGLE_MODULE_VAL|+0.380±0.330|+0.842±0.751|
|amazonpho/clean|CE|+0.430±0.843|+0.263±1.284|
|amazonpho/clean|FIXED_GLOBAL|+0.143±0.350|+0.070±0.977|
|amazonpho/NOISY_MACRO|SINGLE_MODULE_VAL|+0.446±0.375|+0.467±0.209|

## 机制诊断与选择分布

主比较三seed宏差：[0.15284735608034375, 0.48141066462346654, 0.2876753486901862] pp。决策：保留探索性增量，停止本轮训练并结合成本评估。没有新增硬门槛，不自动扩实验。

逐类召回12720行；干净/错标及同意/冲突/平票分层4132行。原表和三seed、seed1/2汇总均已保存；稀缺分层记录实际可用seed数。GCE的one-hot target仅表示其observed输入，GCE损失仍按q=.7计算。

- clean 联合选择：{'S4_G_A1.0_C0.0': 3, 'S32_L0.25': 1, 'S16_G_A0.0_C1.0': 1, 'S4_G_A1.0_C0.25': 1, 'S16_L1.0': 1, 'S4_L0.5': 1, 'S8_G_A1.0_C0.25': 1, 'S128_G_A1.0_C0.0': 1, 'S128_G_A1.0_C0.25': 1, 'S64_G_A1.0_C0.0': 1, 'CE': 1, 'S4_L0.25': 1, 'S16_L0.5': 1}。
- uniform0.4 联合选择：{'S16_L1.0': 1, 'S16_G_A0.25_C1.0': 1, 'S32_G_A0.0_C1.0': 5, 'S64_G_A0.0_C1.0': 1, 'S32_L1.0': 4, 'S16_L0.5': 1, 'S4_G_A0.0_C1.0': 1, 'S128_G_A0.0_C1.0': 1}。
- uniform0.6 联合选择：{'S32_L1.0': 2, 'S16_G_A0.0_C1.0': 2, 'S32_G_A0.0_C1.0': 2, 'S128_G_A0.25_C1.0': 1, 'S8_L1.0': 1, 'S64_G_A0.0_C1.0': 3, 'S4_G_A0.0_C1.0': 2, 'S8_G_A0.0_C1.0': 2}。
- pair0.4 联合选择：{'S64_G_A0.0_C1.0': 6, 'S32_G_A0.0_C1.0': 2, 'S128_G_A0.0_C1.0': 4, 'S16_G_A0.0_C1.0': 2, 'S32_G_A0.25_C1.0': 1}。

## 成本与审计

实际新物理候选1500，其中完整200轮407；科学更新123475，训练循环计时279.966s（I/O、哈希、审计另计）。4个2epoch smoke单列。

联合race候选2250更新+CE200=2450；原全局1000+CE200=1200。固定global800、固定expanded1600。较强单模块验证选择还需支付两个分支的搜索成本，其并集上界2600更新（已扣重复CE）。逐格实际新训练、历史别名与部署预算在costs.csv分列。历史全局完整网格每格已付3600候选更新，当前回放新增更新为0。联合/全局搜索预算有差异，机制与预算贡献需共同解释。

本地审计passed：2760目标重建、120race重算、1939前缀状态、30checkpoint重载，最大误差0.0。源码及输入哈希未变。工程失败如有保留failures.jsonl；正式状态以complete.json为准。

已支持结论为上述配对结果及审计。分组力度改变保留/纠错权衡属于机制解释；独立SSL随机性和新图迁移尚待验证。此前R0匹配KM较强的历史结论继续保留，本轮按授权省略KM/kNN；R0只读参考另存，搜索/tie协议差异已注明。父侧独立审计由research目录维护，交付时另核其完成状态。

## 文件索引

config.json / sources.json / inputs.json / reuse_manifest.json / physical_fits.json / races / selections / selections_frozen.json / fits / targets / results.csv / selected_results.csv / summary.csv / paired_summary.csv / seeds12_* / per_class_recall* / strata* / costs.csv / old64_bridges.csv / audit.json。
