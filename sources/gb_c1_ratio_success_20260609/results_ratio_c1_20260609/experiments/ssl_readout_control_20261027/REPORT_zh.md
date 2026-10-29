# B批：GraphECL统一读出实验、结论与后续决策

完成时间：2026-10-27 22:47（Asia/Shanghai）。64/64新分类头、4/4原生工程复现、64个历史引用；128项预测完成评分，失败0。独立复核通过。

核心判断：CS上原R2的差距会随归一化与读出设置大幅缩小；Photo上同样处理效果有限且clean退化，PCA64配合原始粒球软目标则表现较好。后续应分别处理特征读出和迁移路径，保留原生GraphECL属性分支作为算法底座。

## 研究问题与固定协议

- 假设：此前原生H与R2的差距同时包含64维压缩、属性迁移和读出设置因素；在同一目标下比较原生H、PCA64和归一化R2可定位问题。
- 粒球角色：固定原始GraphECL H产生的SUPPORT32目标；全部新表示沿用同一球成员与LOO软标签，beta=.5、min_votes=2、tie-safe。
- 数据：Coauthor-CS、Amazon-Photo；协议seed0/1，各自关联已有SSL/R2 seed；clean、Uniform .4、Uniform .6、Pair .4。
- 新增64头：2图×2seed×2新特征×4条件×2监督；CE/SUPPORT32。原生32头、旧raw R2 32头只读引用。
- 原生H：与旧probe一致的行L2归一化输入，CS1024维、Photo2048维。PCA64：全节点无标签中心化、float64协方差特征分解、64维、符号规范化、不白化，输出float32再行L2归一化。
- R2_unit64：从冻结encoder.pt按原encode(X)提取eval特征，BN状态固定，再行L2归一化；未重训SSL或R2。
- 同一线性读出：含bias，Adam lr=.01、wd=.0005、200轮，第一次严格最高干净验证计数选模。PCA64与R2_unit64同维、同seed初始化哈希全部相同。
- 旧raw R2参考还使用旧分类头初始化；与R2_unit64的差值包含归一化与初始化变化。PCA包含中心化、截断和再归一化三个操作。
- 完成全部预定拟合并锁定后才评分测试。每个seed先平均三噪声，再报告两个seed均值与样本SD。
- 代码改动限于本独立目录的变换、调用、分析、测试、启动器；复用旧probe.fit，通过固定成员回调维持同一目标。旧训练代码与缓存未改。

## 完整主结果

单位：%；带噪列为三噪声平均。SD反映两个协议seed的差异。

|数据集|特征|监督|带噪Accuracy ± SD|验证均值|clean Accuracy|带噪Macro-F1|
|---|---|---|---:|---:|---:|---:|
|coauthorcs|native|CE|88.206 ± 0.020|88.407|91.171|81.391|
|coauthorcs|pca64|CE|86.532 ± 0.220|86.570|91.471|81.293|
|coauthorcs|r2_unit64|CE|88.166 ± 0.485|87.807|92.075|82.548|
|coauthorcs|r2_raw64|CE|84.805 ± 0.444|84.970|92.936|79.088|
|coauthorcs|native|SUPPORT32|89.198 ± 0.501|89.462|90.871|84.612|
|coauthorcs|pca64|SUPPORT32|88.613 ± 0.511|88.552|90.953|83.910|
|coauthorcs|r2_unit64|SUPPORT32|89.700 ± 0.527|89.698|91.627|85.201|
|coauthorcs|r2_raw64|SUPPORT32|87.676 ± 0.134|87.834|92.060|82.669|
|amazonpho|native|CE|86.813 ± 0.453|86.906|93.371|82.446|
|amazonpho|pca64|CE|85.711 ± 0.295|85.773|93.380|82.239|
|amazonpho|r2_unit64|CE|82.219 ± 2.020|82.789|88.627|77.784|
|amazonpho|r2_raw64|CE|81.198 ± 2.144|82.593|91.905|77.114|
|amazonpho|native|SUPPORT32|89.153 ± 0.084|89.172|92.549|85.526|
|amazonpho|pca64|SUPPORT32|89.872 ± 0.158|90.022|92.764|86.912|
|amazonpho|r2_unit64|SUPPORT32|86.078 ± 1.184|86.688|88.049|81.344|
|amazonpho|r2_raw64|SUPPORT32|85.801 ± 1.048|87.124|91.485|82.023|

## SUPPORT逐条件结果

|图|特征|条件|Accuracy ± SD|验证均值|Macro-F1|
|---|---|---|---:|---:|---:|
|coauthorcs|native|clean|90.871 ± 0.182|91.380|88.244|
|coauthorcs|native|uniform0.4|90.785 ± 0.017|91.435|88.118|
|coauthorcs|native|uniform0.6|90.072 ± 0.022|89.907|84.412|
|coauthorcs|native|pair0.4|86.737 ± 1.499|87.043|81.307|
|coauthorcs|pca64|clean|90.953 ± 0.231|90.780|88.506|
|coauthorcs|pca64|uniform0.4|91.000 ± 0.143|91.189|88.388|
|coauthorcs|pca64|uniform0.6|90.166 ± 0.176|89.989|84.734|
|coauthorcs|pca64|pair0.4|84.672 ± 1.565|84.479|78.609|
|coauthorcs|r2_unit64|clean|91.627 ± 0.028|91.517|88.306|
|coauthorcs|r2_unit64|uniform0.4|91.853 ± 0.050|91.844|89.101|
|coauthorcs|r2_unit64|uniform0.6|91.218 ± 0.011|91.135|87.106|
|coauthorcs|r2_unit64|pair0.4|86.028 ± 1.620|86.116|79.396|
|coauthorcs|r2_raw64|clean|92.060 ± 0.364|92.390|89.405|
|coauthorcs|r2_raw64|uniform0.4|91.078 ± 0.253|91.135|87.927|
|coauthorcs|r2_raw64|uniform0.6|89.200 ± 0.860|89.280|84.868|
|coauthorcs|r2_raw64|pair0.4|82.752 ± 1.008|83.088|75.213|
|amazonpho|native|clean|92.549 ± 0.026|91.895|90.968|
|amazonpho|native|uniform0.4|92.316 ± 0.198|91.961|90.091|
|amazonpho|native|uniform0.6|91.373 ± 0.343|91.699|87.717|
|amazonpho|native|pair0.4|83.772 ± 0.106|83.856|78.770|
|amazonpho|pca64|clean|92.764 ± 0.277|91.895|91.631|
|amazonpho|pca64|uniform0.4|92.652 ± 0.251|92.288|91.281|
|amazonpho|pca64|uniform0.6|91.653 ± 0.370|91.503|89.256|
|amazonpho|pca64|pair0.4|85.313 ± 0.145|86.275|80.199|
|amazonpho|r2_unit64|clean|88.049 ± 2.166|87.974|83.201|
|amazonpho|r2_unit64|uniform0.4|89.477 ± 1.730|89.412|86.023|
|amazonpho|r2_unit64|uniform0.6|88.543 ± 1.307|89.085|84.766|
|amazonpho|r2_unit64|pair0.4|80.215 ± 0.515|81.569|73.244|
|amazonpho|r2_raw64|clean|91.485 ± 0.634|91.634|89.838|
|amazonpho|r2_raw64|uniform0.4|90.373 ± 0.198|91.438|88.199|
|amazonpho|r2_raw64|uniform0.6|87.880 ± 0.713|89.085|84.383|
|amazonpho|r2_raw64|pair0.4|79.150 ± 2.232|80.850|73.488|

## 成对变化

差值单位pp；两个seed按0/1排列。

|图|监督|A−B|带噪测试差|seed0 / seed1|带噪验证差|clean测试差|clean验证差|
|---|---|---|---:|---:|---:|---:|---:|
|coauthorcs|CE|pca64 − r2_unit64|-1.634|-2.133 / -1.135|-1.237|-0.604|-0.846|
|coauthorcs|CE|pca64 − native|-1.674|-1.816 / -1.533|-1.837|+0.300|-0.355|
|coauthorcs|CE|r2_unit64 − native|-0.040|+0.317 / -0.397|-0.600|+0.904|+0.491|
|coauthorcs|CE|r2_unit64 − r2_raw64|+3.361|+3.390 / +3.333|+2.837|-0.861|-0.791|
|coauthorcs|SUPPORT32|pca64 − r2_unit64|-1.087|-1.075 / -1.099|-1.146|-0.674|-0.736|
|coauthorcs|SUPPORT32|pca64 − native|-0.586|-0.592 / -0.579|-0.909|+0.082|-0.600|
|coauthorcs|SUPPORT32|r2_unit64 − native|+0.501|+0.483 / +0.519|+0.236|+0.756|+0.136|
|coauthorcs|SUPPORT32|r2_unit64 − r2_raw64|+2.023|+1.745 / +2.301|+1.864|-0.432|-0.873|
|amazonpho|CE|pca64 − r2_unit64|+3.492|+4.712 / +2.272|+2.985|+4.753|+4.052|
|amazonpho|CE|pca64 − native|-1.102|-1.631 / -0.573|-1.133|+0.009|-0.523|
|amazonpho|CE|r2_unit64 − native|-4.594|-6.343 / -2.845|-4.118|-4.743|-4.575|
|amazonpho|CE|r2_unit64 − r2_raw64|+1.021|+1.108 / +0.934|+0.196|-3.277|-3.660|
|amazonpho|SUPPORT32|pca64 − r2_unit64|+3.794|+4.743 / +2.845|+3.333|+4.715|+3.922|
|amazonpho|SUPPORT32|pca64 − native|+0.719|+0.890 / +0.548|+0.850|+0.215|+0.000|
|amazonpho|SUPPORT32|r2_unit64 − native|-3.075|-3.853 / -2.297|-2.484|-4.500|-3.922|
|amazonpho|SUPPORT32|r2_unit64 − r2_raw64|+0.277|+0.181 / +0.373|-0.436|-3.436|-3.660|

## CS：读出设置解释了原R2差距中的较大部分

R2_unit64的CE从旧R2的84.805%升至88.166%，+3.361pp；SUPPORT从87.676%升至89.700%，+2.023pp，两个seed分别+1.745/+2.301pp，验证平均+1.864pp。固定编码器和粒球目标后即可恢复较多表现，故此前“R2普遍损失GraphECL语义”的解释需要收窄。
与原生SUPPORT的89.198%相比，R2_unit64测试均值高.501pp，两seed+.483/+.519；对应验证仅+.236pp，两个seed−.200/+.673。该比较表现为测试正向、验证方向不一致，继续保留两条路径。
PCA64+SUPPORT为88.613%，较原生低.586pp、较R2_unit64低1.087pp。PCA保留91.3%–91.6%的中心化方差，仍出现性能损失；后续应检查哪些局部方向影响含噪监督，而非只看总保留方差。
逐噪声表显示PCA相对原生的主要损失集中于Pair .4；两个Uniform条件基本保持。R2_unit64对旧R2的改善也在较强噪声和Pair条件下更明显。
代价：R2_unit64相对旧R2的clean CE下降.861pp，clean SUPPORT下降.432pp；SUPPORT clean验证下降.873pp。保留带噪收益和clean代价，下一轮需在相同初始化下拆开读出因素。

## Photo：PCA64保留有效空间，R2迁移路径仍有明显剩余差距

PCA64+SUPPORT为89.872 ± .158%，原生为89.153 ± .084%，测试+.719pp，两seed+.890/+.548；验证+.850pp，两seed均正，clean验证均值持平，clean测试+.215pp。已通过预登记PCA接近原生门，并呈现超过原生的双seed信号。
同为64维、单位范数、相同初始化及目标，PCA64比R2_unit64高3.794pp；两seed+4.743/+2.845，验证+3.333pp、clean测试+4.715pp。CE下PCA64也比R2_unit64高3.492pp。两条监督均通过PCA优于R2的预登记门。
这使Photo后续优先级转向保留原生属性表示或改善迁移后的几何关系。64维PCA仍能得到较好结果，为“64维容量普遍不足”的解释增加了反例。PCA的中心化及归一化效应仍需在后续机制消融中分开。
R2_unit64相对旧R2：带噪SUPPORT仅+.277pp，验证−.436pp，Macro-F1−.679pp；clean测试−3.436pp、clean验证−3.660pp。因此全数据集统一套用该读出改动的方向受到负结果削弱。

## 新发现：表示变换与粒球软监督发生交互

Photo上PCA64相对原生：CE下降1.102pp，SUPPORT提升.719pp。同一表示变换，在两种监督下得到相反方向。
|图|特征|SUPPORT相对CE的带噪Accuracy增量pp|
|---|---|---:|
|coauthorcs|native|+0.992|
|coauthorcs|pca64|+2.081|
|coauthorcs|r2_unit64|+1.534|
|coauthorcs|r2_raw64|+2.872|
|amazonpho|native|+2.340|
|amazonpho|pca64|+4.161|
|amazonpho|r2_unit64|+3.859|
|amazonpho|r2_raw64|+4.603|

Photo的PCA路径将SUPPORT相对CE的收益从2.340pp扩大到4.161pp。所有表示使用的target.pt逐元素相同，目标Brier也完全相同；此次变化来自特征与读出怎样利用同一份监督。这为“面向粒球软监督的表示学习”提供了具体研究切入点。
合理解释：中心化和截断可能削弱容易拟合噪声的方向，软目标又降低错误标签的梯度干扰，二者共同改变分类边界。该机制尚未被单因素消融确认。
读出尺度的机制解释：线性头某类权重的梯度为(p_ic−t_ic)u_i；原始特征范数会影响每个样本的梯度及logit大小，并改变固定权重衰减下的有效约束。R2的投影余弦对齐目标主要约束投影方向，后续直接使用隐藏层特征时，其尺度值得检查。Photo上单位化clean退化也说明范数或其对应读出条件可能携带有用信息。该段为待验证解释。

## 变换诊断

|图|seed|原生维度|PCA64保留中心化方差%|原R2范数 min / mean / max|R2提取预测误差|
|---|---:|---:|---:|---:|---:|
|coauthorcs|0|1024|91.571|2.062 / 3.957 / 20.801|0.0e+00|
|coauthorcs|1|1024|91.328|2.131 / 4.016 / 22.127|0.0e+00|
|amazonpho|0|2048|78.350|2.272 / 3.739 / 11.855|0.0e+00|
|amazonpho|1|2048|76.172|2.117 / 3.528 / 17.490|0.0e+00|

## 验证标准、失败标准与结果

预登记主门：PCA64−R2_unit64的三噪声验证均值≥+.5pp，两个seed正，clean验证均值≥−.5pp。CS两种监督均未通过；Photo两种监督均通过。
预登记PCA接近原生门：带噪与clean验证均值≥−.5pp，各seed带噪≥−.5pp。仅Photo SUPPORT通过。工程失败门为来源/目标/模型/选模核验异常，本批无工程失败。
两个seed用于初筛和配对方向判断；下一轮复现需继续报告逐seed结果。

## 下一步：限定一次读出确认，然后进入粒球机制

1. 建议先做最小2×2读出确认：固定同一R2特征和目标，交叉原始/单位化特征与旧/统一分类头初始化。已有两条对角路径；补raw+统一初始化、unit+旧初始化两个缺格，在两图两seed四条件CE/SUPPORT下共64新头，0次SSL。用相同初始化对比范数效应，避免把初始化作用归给归一化。预登记带噪验证均值、两seed方向及clean−.5pp约束；按图汇报。此项仅建议，未运行。
2. 之后的自研对比算法优先基于GraphECL的原生属性分支。导出代码graphecl.Model.embedding(x)只调用encoder_target(x)，推理输入为属性；新方法可直接接分类头，减少额外R2迁移干扰。部署时PCA需保存训练阶段无标签拟合的均值与基。
3. 将Photo上“PCA与SUPPORT配合更好”作为工作假设：学习能保留可靠局部共识、抑制噪声敏感方向的属性表示。第一轮仅比较原GraphECL与一个弱粒球关系约束，保持原始监督目标和统一读出固定；粒球关系用无标签稳定性加权，低置信关系拒绝。
4. 若粒球约束直接改变H，先用旧目标测试表示效应，再用新H构球单独测试目标效应；沿用A批交叉思路。同时记录CE、SUPPORT、clean、Macro-F1及Pair条件，成功条件覆盖CS和Photo。
5. 继续保留GRACE/GCA参照。新读出只在GraphECL验证过；跨SSL排名需给其他方法匹配读出后再更新。CiteSeer的谱集中单独处理，现有两图结果继续保留各自机制差异。
最终决策：继续GraphECL原生属性分支与粒球软监督适配方向；修改“CS主要因不可逆R2语义损失而弱”的解释；暂缓全图统一单位化、盲目增大R2维度与复杂模块堆叠。本批结束，后续实验未自动启动。

## 证据状态

- 已被实验支持：CS读出改动后的噪声收益；Photo PCA64/SUPPORT的双seed正向配对；Photo同64维下PCA与R2的剩余差距；固定目标下表示与监督的交互。
- 合理解释：特征尺度改变噪声梯度与有效正则；PCA截断和软监督可能互补。
- 尚未验证：范数与初始化的各自贡献；PCA各步骤的单独作用；新的粒球对比约束是否改善两图；所有SSL经过统一读出后的排名。
- 负结果削弱：统一R2单位化适合全部数据；原R2差距主要由64维容量决定；只凭总保留方差预测含噪分类效果。

## 工程验证与文件索引

- 7项数学/任务测试：先失败后实现通过；语法和模块导入通过。
- 497项来源哈希、冻结特征清单和全部目标核验通过；4个原生复现预测误差0，模型/初始化/最佳轮次一致。
- 64个新头CPU重载概率最大差4.7683716e-7；32组PCA/R2初始化一致；全部200轮与首次最佳验证选模通过。
- 独立NumPy逐节点LOO重算16组目标，最大差1.0353431e-7；sklearn/NumPy重算128项分数，最大差4.7522259e-8；均值/样本SD差2.0428104e-14。
- 配置/流程：config.json、PLAN.md、run.py、operations.py、analyze.py、launch.py、test_operations.py。
- 可复核结果：results/summary.json、scores.csv、per_condition.json、comparisons.json、gates.json、audit.json；review/check_readout.py及JSON。
- 原始权重/概率/逐轮日志/来源绑定：D:/codex_temp/ssl_readout_control_20261027/formal/linear/；变换权重与统计：同根features/。
- 启动/接续入口：START_HERE.md。只读最终状态：同运行根queue_status.json与formal/final.json。
