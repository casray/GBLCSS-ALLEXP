# 异配机制诊断与首轮分类实验 Implementation Plan

> 执行：subagent-driven-development；用户2026-10-14已批准“设计实验，先做机制诊断，然后开始实验”。本计划先于读取新诊断结果冻结。旧论文/实验代码只读，新实现置于本目录。

## 研究假设、角色和动机

H1：更细粒球能改善CoraFull共识，但收益可能只是容量；H2：低同配图的几何邻域对根类别可能存在负边际，随机噪声浓缩不能修复；H3：ESS/尺度JS收缩只有胜过同幅度常数收缩及打乱因子才支持节点特异可靠性；H4：目标改善不必转成X-only分类增益。粒球仅组织冻结表示上的训练票源，不作为硬对比标签。

承接研究05审查及历史43：旧多尺度JS筛查未过门。本次新增CoraFull/CoauthorCS大图及Roman-empire真实异配图、完整软目标Brier及普通收缩对照，属于有界新诊断，不是恢复已关闭实现。

## 数据和固定协议

- CoraFull与CoauthorCS：只读已有processed/data.pt与正式seed0 GRACE H，20/10/70旧percent_split_indices，协议seed0–4；split/noise均用该seed。原节点顺序不变。
- Roman-empire：作者 yandex-research/heterophilous-graphs 的 roman_empire.npz，官方split0–4；仅训练标签污染。新图不与旧20/10/70跨论文数字直接比较。
- 条件clean(rate0)、uniform0.4、pair0.4；噪声函数复用GB-LCSS_code。验证/测试干净，不进入候选、几何、投票权重。clean真值仅在目标数组保存后用于诊断。
- 既有GRACE每图一个预训练seed。新GraphACL/新图GRACE先seed0作探索；不宣称五个SSL种子。GraphACL准备须核官方源码/公式及依赖，不能用未经验证的近似方法冒名。
- 旧图属性不改预处理。新图保留官方属性，表示方法的输入预处理逐一记录，不将模型差异全部归因目标函数。

## 阶段A：目标质量诊断

固定H行L2归一化；粒球复用正式构造，Bmax64/128、min_child32、split_improvement.05、partition_seed=protocol_seed（核查正式run_gblcss后确认，不能固定0冒充原协议）。k-means n_init10/相同protocol_seed、簇数匹配实际球数。每训练节点的kNN仅检索其他训练节点，唯一票数与GB逐节点精确匹配；这是匹配票源对照，不冒充论文原固定K表值。

每个预算保存GB、KM、kNN候选关系；等权投票沿用min_voters2、beta.5、并列只用频率的原LCSS规则。额外对GB128加入exp(cosine/.2)几何权重；不改候选，不新增跨球票。

固定GB128的r/q，比较 original、half(0.5alpha)、ess、js、full、mean_matched、shuffled。ESS取(sumw)^2/sumw²；ESS因子n_eff/(n_eff+10)。JS在GB64与GB128的留一分布间计算，标准双分布JS、自然对数，因子exp(-JS/.1)；只有两个尺度均有至少2票时计算，否则JS因子设1并标注缺失。full为两因子乘积。mean_matched用同一数组使平均alpha严格匹配full，仅诊断控制；shuffled置换完整收缩因子(seed=10000+protocolseed)。二者跨节点统计会混入自身标签影响，不能用于声称逐点Brier独立性。weighted128作单独候选质量消融，不与固定q收缩贡献混同。

报告每节点/每组：支持数、有效票数、无票率、根类别干净margin和正margin比例、q正确概率/准确率、target Brier、初始干净/错标两组Brier与argmax纠错/误修正率、平均alpha。保存q/alpha/targets后才读取clean真值评价；诊断不用于宣称独立测试效果。

预注册推进筛选（探索门槛，非显著性结论）：
- 容量增量：GB128对GB64，两个噪声条件宏平均Brier降低>=.005、至少一个数据集4/5seed同向；其他图退化单列。
- 粒球独立信号：同表示GB128对匹配kNN128和KM128均宏平均Brier降低>=.005，且至少2图同向；若只有1张异配图，不能声称跨异配泛化。
- 可靠性：full对mean_matched Brier降低>=.005且4/5seed同向，并胜shuffled、没有平均干净节点误修正率上升；不通过则不训练full作为新方法。
- 均值/std及配对差值全报，不以门槛替代统计推断。失败即暂缓对应主张，不扩展参数搜索。

## 阶段B：首轮下游

基线首轮用于检查诊断到分类的转化，预先固定noisy CE、GB64、GB128、matched-kNN128；各已准备图先协议seed0、clean/uniform.4/pair.4。所有方法同MLP(2层含BN、hidden64、dropout.5)、Adam lr.01/wd.0005、200epoch、干净验证准确率选checkpoint、同初始化/训练seed。复用正式train_x_only_mlp；每run记录结果/耗时/seed/输入hash。首轮seed0明确是pilot，不作为稳定收益；完整5seed为后续确认阶段。本轮不因单次测试分数改参数。

## 表示分支

独立准备 prepare_representations.py 与 representation_sources.md：下载公开Roman数据和必要官方GraphACL源码，只写本目录，记录URL/hash；固定训练预算和配置后才训练，禁止用标签选表示checkpoint。GRACE采用已有实现的精确分块损失避免全N²显存，数值/梯度与原dense小图比对。GraphACL若需移植，核官方公式、负样本策略、EMA、编码器，并提供小图测试。该分支有明确差异/限制时单独报告，不阻塞已有H诊断。

## 文件、任务和验收

- [ ] PLAN.md与config.json先冻结；环境/来源hash清单。
- [ ] test_diagnostic.py：先测试留一与原targets一致、自身标签不影响q、并列/不足票回退、唯一票数匹配、固定q与平均alpha匹配、指标手算。
- [ ] diagnostic.py：数据/协议、分球/近邻、target生成、冻结文件、事后指标与paired摘要；原代码不改。
- [ ] prepare_representations.py、test_representations.py、representation_sources.md：新图数据/异配表示，独立审查后GPU运行。
- [ ] run_students.py：复用正式训练器、配置/目标/原始日志/seed结果与均值std，失败显式记录。
- [ ] 运行诊断、独立检查协议hash/目标归一化/无自投票/来源未变；之后运行baseline pilot。
- [ ] 实验结论.md、decision.json、99_session_handoff.md：写结果、解释边界、继续/暂缓决策；同步workspace交接，保留历史。

命令前缀：`& 'D:\codex_temp\noisygl-rocm721\Scripts\python.exe'`。先 `-m pytest experiments/heterophily_mechanism_20261014/test_diagnostic.py -q`；再 `experiments/heterophily_mechanism_20261014/diagnostic.py`；再 `experiments/heterophily_mechanism_20261014/run_students.py`。表示脚本CLI由该子任务核实后记录具体命令。单GPU任务串行，CPU诊断可与只读文献/代码准备并行。不得同时运行两个GPU训练占满显存。

## 实验结果与最终决策

启动前状态：尚未运行；结果写入独立实验结论文档，不覆盖这份预注册标准。若依赖或数据受阻，完成无依赖部分并记缺口；不伪造新图/方法结果。
