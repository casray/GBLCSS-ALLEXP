# GCN SUPPORT/D1/C1 三 seed 扩展

45 cells，五图 x seed0/1/2 x clean/U.4/U.6；六路教师与同 seed DIST 学生。
固定历史 GRACE 几何；旧三图 seed0 九格只读复用，其余 36 格新拟合。
全部选择冻结后测试；宏平均先 seed 内平均条件和图，再对三个 seed 求均值与样本 SD(ddof=1)。
所有 seed 原值、负差、clean、逐类召回、配置身份、成本保留于 CSV。

## 跨图带噪结果

|阶段|方法|accuracy % mean +/- SD|macro-F1 % mean +/- SD|
|---|---|---:|---:|
|teacher|CE|73.906 +/- 1.059|67.657 +/- 1.274|
|teacher|OLD64|75.959 +/- 0.987|69.280 +/- 1.331|
|teacher|LEGACY_SUPPORT32|76.099 +/- 0.239|69.213 +/- 0.795|
|teacher|DEEP_S32|76.107 +/- 0.251|69.401 +/- 0.975|
|teacher|FIXED_D1|76.572 +/- 0.516|70.277 +/- 1.454|
|teacher|JOINT|76.676 +/- 0.697|70.164 +/- 1.496|
|student|CE|74.905 +/- 0.655|67.987 +/- 1.001|
|student|OLD64|76.633 +/- 0.571|69.737 +/- 0.911|
|student|LEGACY_SUPPORT32|76.913 +/- 0.167|69.682 +/- 0.822|
|student|DEEP_S32|76.897 +/- 0.280|69.813 +/- 1.055|
|student|FIXED_D1|77.290 +/- 0.308|70.755 +/- 1.256|
|student|JOINT|77.464 +/- 0.449|70.862 +/- 1.401|

## 配对增量及负结果

|图|条件|阶段|参照|accuracy pp mean +/- SD|seed0/1/2 pp|
|---|---|---|---|---:|---|
|amazonpho|clean|teacher|CE|+0.037 +/- 0.135|-0.075/+0.187/+0.000|
|ALL|clean|teacher|CE|-0.125 +/- 0.324|-0.064/+0.165/-0.475|
|amazonpho|clean|teacher|OLD64|+1.693 +/- 0.075|+1.625/+1.681/+1.774|
|ALL|clean|teacher|OLD64|+2.054 +/- 0.312|+2.258/+2.208/+1.694|
|amazonpho|clean|teacher|LEGACY_SUPPORT32|+1.917 +/- 0.434|+1.587/+1.755/+2.409|
|ALL|clean|teacher|LEGACY_SUPPORT32|+1.530 +/- 0.497|+0.960/+1.871/+1.761|
|amazonpho|clean|teacher|DEEP_S32|+1.948 +/- 0.403|+1.662/+1.774/+2.409|
|ALL|clean|teacher|DEEP_S32|+1.467 +/- 0.517|+0.872/+1.727/+1.802|
|amazonpho|clean|teacher|FIXED_D1|+0.280 +/- 0.314|-0.075/+0.523/+0.392|
|ALL|clean|teacher|FIXED_D1|-0.104 +/- 0.134|-0.098/+0.027/-0.241|
|amazonpho|NOISY_MACRO|teacher|CE|+2.213 +/- 0.503|+1.643/+2.400/+2.596|
|ALL|NOISY_MACRO|teacher|CE|+2.770 +/- 0.370|+3.169/+2.704/+2.437|
|amazonpho|NOISY_MACRO|teacher|OLD64|+0.831 +/- 0.989|-0.103/+1.867/+0.728|
|ALL|NOISY_MACRO|teacher|OLD64|+0.717 +/- 0.327|+1.010/+0.776/+0.364|
|amazonpho|NOISY_MACRO|teacher|LEGACY_SUPPORT32|+1.127 +/- 1.621|-0.037/+0.439/+2.979|
|ALL|NOISY_MACRO|teacher|LEGACY_SUPPORT32|+0.577 +/- 0.461|+0.052/+0.763/+0.916|
|amazonpho|NOISY_MACRO|teacher|DEEP_S32|+1.071 +/- 1.708|-0.065/+0.243/+3.035|
|ALL|NOISY_MACRO|teacher|DEEP_S32|+0.569 +/- 0.468|+0.049/+0.701/+0.957|
|amazonpho|NOISY_MACRO|teacher|FIXED_D1|-0.140 +/- 0.413|-0.598/+0.205/-0.028|
|ALL|NOISY_MACRO|teacher|FIXED_D1|+0.104 +/- 0.218|-0.128/+0.303/+0.137|
|citeseer|clean|teacher|CE|-0.215 +/- 0.352|+0.172/-0.515/-0.301|
|citeseer|clean|teacher|OLD64|+2.234 +/- 1.903|+4.424/+1.289/+0.988|
|citeseer|clean|teacher|LEGACY_SUPPORT32|+0.186 +/- 0.305|-0.086/+0.129/+0.515|
|citeseer|clean|teacher|DEEP_S32|+0.186 +/- 0.305|-0.086/+0.129/+0.515|
|citeseer|clean|teacher|FIXED_D1|-0.029 +/- 0.345|+0.301/-0.387/+0.000|
|citeseer|NOISY_MACRO|teacher|CE|+5.548 +/- 0.875|+6.551/+4.940/+5.155|
|citeseer|NOISY_MACRO|teacher|OLD64|+0.988 +/- 1.283|+2.448/+0.473/+0.043|
|citeseer|NOISY_MACRO|teacher|LEGACY_SUPPORT32|+0.473 +/- 0.478|-0.043/+0.902/+0.558|
|citeseer|NOISY_MACRO|teacher|DEEP_S32|+0.473 +/- 0.478|-0.043/+0.902/+0.558|
|citeseer|NOISY_MACRO|teacher|FIXED_D1|+0.243 +/- 0.110|+0.150/+0.365/+0.215|
|coauthorcs|clean|teacher|CE|+0.122 +/- 0.108|+0.164/+0.000/+0.203|
|coauthorcs|clean|teacher|OLD64|+1.034 +/- 0.179|+0.951/+0.912/+1.239|
|coauthorcs|clean|teacher|LEGACY_SUPPORT32|+0.719 +/- 0.160|+0.584/+0.678/+0.896|
|coauthorcs|clean|teacher|DEEP_S32|+0.655 +/- 0.163|+0.545/+0.577/+0.842|
|coauthorcs|clean|teacher|FIXED_D1|+0.094 +/- 0.142|+0.023/+0.000/+0.257|
|coauthorcs|NOISY_MACRO|teacher|CE|+0.735 +/- 0.309|+0.530/+1.091/+0.584|
|coauthorcs|NOISY_MACRO|teacher|OLD64|-0.051 +/- 0.097|-0.082/+0.058/-0.129|
|coauthorcs|NOISY_MACRO|teacher|LEGACY_SUPPORT32|-0.056 +/- 0.154|+0.113/-0.094/-0.187|
|coauthorcs|NOISY_MACRO|teacher|DEEP_S32|+0.009 +/- 0.066|+0.055/+0.039/-0.066|
|coauthorcs|NOISY_MACRO|teacher|FIXED_D1|-0.065 +/- 0.074|-0.094/+0.019/-0.121|
|cora|clean|teacher|CE|-0.510 +/- 1.455|-0.633/+1.003/-1.900|
|cora|clean|teacher|OLD64|+1.865 +/- 1.500|+0.633/+3.536/+1.425|
|cora|clean|teacher|LEGACY_SUPPORT32|+1.829 +/- 2.072|-0.369/+3.747/+2.111|
|cora|clean|teacher|DEEP_S32|+1.478 +/- 1.626|-0.369/+2.691/+2.111|
|cora|clean|teacher|FIXED_D1|-0.739 +/- 0.739|-0.739/+0.000/-1.478|
|cora|NOISY_MACRO|teacher|CE|+5.128 +/- 1.596|+6.966/+4.327/+4.090|
|cora|NOISY_MACRO|teacher|OLD64|+0.572 +/- 0.877|+1.583/+0.106/+0.026|
|cora|NOISY_MACRO|teacher|LEGACY_SUPPORT32|+0.844 +/- 0.735|+0.026/+1.451/+1.055|
|cora|NOISY_MACRO|teacher|DEEP_S32|+0.774 +/- 0.654|+0.026/+1.240/+1.055|
|cora|NOISY_MACRO|teacher|FIXED_D1|+0.431 +/- 0.374|+0.026/+0.501/+0.765|
|corafull|clean|teacher|CE|-0.058 +/- 0.280|+0.051/+0.152/-0.375|
|corafull|clean|teacher|OLD64|+3.443 +/- 0.344|+3.659/+3.623/+3.046|
|corafull|clean|teacher|LEGACY_SUPPORT32|+3.000 +/- 0.112|+3.082/+3.046/+2.873|
|corafull|clean|teacher|DEEP_S32|+3.067 +/- 0.433|+2.606/+3.464/+3.132|
|corafull|clean|teacher|FIXED_D1|-0.125 +/- 0.217|+0.000/+0.000/-0.375|
|corafull|NOISY_MACRO|teacher|CE|+0.226 +/- 0.504|+0.155/+0.761/-0.238|
|corafull|NOISY_MACRO|teacher|OLD64|+1.245 +/- 0.119|+1.205/+1.379/+1.151|
|corafull|NOISY_MACRO|teacher|LEGACY_SUPPORT32|+0.496 +/- 0.537|+0.198/+1.115/+0.173|
|corafull|NOISY_MACRO|teacher|DEEP_S32|+0.518 +/- 0.487|+0.274/+1.079/+0.202|
|corafull|NOISY_MACRO|teacher|FIXED_D1|+0.052 +/- 0.324|-0.126/+0.426/-0.144|
|amazonpho|clean|student|CE|+0.062 +/- 0.526|-0.430/+0.616/+0.000|
|ALL|clean|student|CE|-0.047 +/- 0.304|+0.118/+0.138/-0.398|
|amazonpho|clean|student|OLD64|+1.581 +/- 0.374|+1.158/+1.867/+1.718|
|ALL|clean|student|OLD64|+2.016 +/- 0.260|+2.277/+2.014/+1.757|
|amazonpho|clean|student|LEGACY_SUPPORT32|+1.693 +/- 0.518|+1.120/+1.830/+2.129|
|ALL|clean|student|LEGACY_SUPPORT32|+1.382 +/- 0.476|+0.835/+1.605/+1.706|
|amazonpho|clean|student|DEEP_S32|+1.780 +/- 0.588|+1.102/+2.110/+2.129|
|ALL|clean|student|DEEP_S32|+1.296 +/- 0.561|+0.657/+1.526/+1.705|
|amazonpho|clean|student|FIXED_D1|+0.162 +/- 0.699|-0.430/+0.934/-0.019|
|ALL|clean|student|FIXED_D1|-0.123 +/- 0.258|-0.008/+0.057/-0.418|
|amazonpho|NOISY_MACRO|student|CE|+2.004 +/- 1.114|+0.971/+1.858/+3.184|
|ALL|NOISY_MACRO|student|CE|+2.559 +/- 0.315|+2.847/+2.222/+2.608|
|amazonpho|NOISY_MACRO|student|OLD64|+1.061 +/- 0.588|+0.663/+1.737/+0.784|
|ALL|NOISY_MACRO|student|OLD64|+0.832 +/- 0.222|+1.021/+0.588/+0.886|
|amazonpho|NOISY_MACRO|student|LEGACY_SUPPORT32|+1.208 +/- 1.903|+0.252/-0.028/+3.399|
|ALL|NOISY_MACRO|student|LEGACY_SUPPORT32|+0.551 +/- 0.458|+0.187/+0.401/+1.065|
|amazonpho|NOISY_MACRO|student|DEEP_S32|+1.547 +/- 1.446|+0.906/+0.532/+3.203|
|ALL|NOISY_MACRO|student|DEEP_S32|+0.567 +/- 0.417|+0.346/+0.306/+1.048|
|amazonpho|NOISY_MACRO|student|FIXED_D1|+0.044 +/- 0.293|-0.187/+0.373/-0.056|
|ALL|NOISY_MACRO|student|FIXED_D1|+0.174 +/- 0.168|+0.054/+0.101/+0.366|
|citeseer|clean|student|CE|+0.129 +/- 0.888|+1.117/-0.601/-0.129|
|citeseer|clean|student|OLD64|+2.005 +/- 2.133|+4.467/+0.730/+0.816|
|citeseer|clean|student|LEGACY_SUPPORT32|-0.086 +/- 0.324|-0.129/-0.387/+0.258|
|citeseer|clean|student|DEEP_S32|-0.086 +/- 0.324|-0.129/-0.387/+0.258|
|citeseer|clean|student|FIXED_D1|+0.014 +/- 0.848|+0.988/-0.387/-0.558|
|citeseer|NOISY_MACRO|student|CE|+4.646 +/- 0.496|+5.219/+4.360/+4.360|
|citeseer|NOISY_MACRO|student|OLD64|+0.709 +/- 0.777|+1.525/-0.021/+0.623|
|citeseer|NOISY_MACRO|student|LEGACY_SUPPORT32|+0.379 +/- 0.180|+0.494/+0.172/+0.473|
|citeseer|NOISY_MACRO|student|DEEP_S32|+0.379 +/- 0.180|+0.494/+0.172/+0.473|
|citeseer|NOISY_MACRO|student|FIXED_D1|+0.379 +/- 0.569|+0.838/-0.258/+0.558|
|coauthorcs|clean|student|CE|+0.065 +/- 0.177|-0.070/+0.000/+0.265|
|coauthorcs|clean|student|OLD64|+0.753 +/- 0.427|+0.670/+0.374/+1.216|
|coauthorcs|clean|student|LEGACY_SUPPORT32|+0.545 +/- 0.366|+0.366/+0.304/+0.966|
|coauthorcs|clean|student|DEEP_S32|+0.338 +/- 0.164|+0.281/+0.210/+0.522|
|coauthorcs|clean|student|FIXED_D1|+0.057 +/- 0.074|+0.140/+0.000/+0.031|
|coauthorcs|NOISY_MACRO|student|CE|+0.634 +/- 0.130|+0.545/+0.573/+0.783|
|coauthorcs|NOISY_MACRO|student|OLD64|+0.166 +/- 0.194|+0.094/+0.019/+0.386|
|coauthorcs|NOISY_MACRO|student|LEGACY_SUPPORT32|+0.086 +/- 0.086|+0.152/-0.012/+0.117|
|coauthorcs|NOISY_MACRO|student|DEEP_S32|+0.165 +/- 0.128|+0.105/+0.078/+0.312|
|coauthorcs|NOISY_MACRO|student|FIXED_D1|+0.151 +/- 0.098|+0.140/+0.058/+0.253|
|cora|clean|student|CE|-0.334 +/- 1.204|-0.106/+0.739/-1.636|
|cora|clean|student|OLD64|+2.111 +/- 1.059|+1.372/+3.325/+1.636|
|cora|clean|student|LEGACY_SUPPORT32|+1.706 +/- 1.961|-0.475/+3.325/+2.269|
|cora|clean|student|DEEP_S32|+1.390 +/- 1.616|-0.475/+2.375/+2.269|
|cora|clean|student|FIXED_D1|-0.686 +/- 0.398|-0.739/-0.264/-1.055|
|cora|NOISY_MACRO|student|CE|+5.215 +/- 2.000|+7.493/+3.747/+4.406|
|cora|NOISY_MACRO|student|OLD64|+0.721 +/- 0.974|+1.504/-0.369/+1.029|
|cora|NOISY_MACRO|student|LEGACY_SUPPORT32|+0.598 +/- 0.637|-0.106/+1.135/+0.765|
|cora|NOISY_MACRO|student|DEEP_S32|+0.457 +/- 0.725|-0.132/+0.237/+1.266|
|cora|NOISY_MACRO|student|FIXED_D1|+0.405 +/- 0.678|+0.000/+0.026/+1.187|
|corafull|clean|student|CE|-0.159 +/- 0.296|+0.079/-0.065/-0.491|
|corafull|clean|student|OLD64|+3.630 +/- 0.202|+3.717/+3.775/+3.399|
|corafull|clean|student|LEGACY_SUPPORT32|+3.051 +/- 0.209|+3.291/+2.952/+2.909|
|corafull|clean|student|DEEP_S32|+3.058 +/- 0.479|+2.505/+3.320/+3.349|
|corafull|clean|student|FIXED_D1|-0.164 +/- 0.283|+0.000/+0.000/-0.491|
|corafull|NOISY_MACRO|student|CE|+0.294 +/- 0.284|+0.004/+0.570/+0.307|
|corafull|NOISY_MACRO|student|OLD64|+1.500 +/- 0.159|+1.317/+1.573/+1.610|
|corafull|NOISY_MACRO|student|LEGACY_SUPPORT32|+0.484 +/- 0.308|+0.141/+0.736/+0.574|
|corafull|NOISY_MACRO|student|DEEP_S32|+0.285 +/- 0.271|+0.357/+0.512/-0.014|
|corafull|NOISY_MACRO|student|FIXED_D1|-0.108 +/- 0.413|-0.520/+0.307/-0.112|

## 成本与判读

新教师轨迹 1620；新物理学生 197；逻辑学生 270(旧54/新216)。
新拟合累计更新 173925。epochs 为各物理轨迹最终累计轮数，前缀与续训合计一次。
成本表将历史复用与新拟合分列；计时沿用 fit.seconds，存储另列选中产物与全运行目录总量。
已支持结论：以上表格及逐 seed 配对差；教师增量与学生传递分别判读，负差完整呈现。
机制解释：区域监督可能改变教师概率结构与蒸馏效果，该解释仍需机制证据。
尚待验证：跨未开发划分、更多随机种子及其他 SSL 几何的稳定性。
负结果削弱方向：逐图/条件/seed 负差直接记录于 paired_differences.csv，保留其参照身份。
最终决策：保留本次全部结果并停止自动扩展；结合教师、学生、clean、F1、逐类召回及成本表做下一步研究决策。
CPU 审计单独执行，最终状态见 complete.json。
