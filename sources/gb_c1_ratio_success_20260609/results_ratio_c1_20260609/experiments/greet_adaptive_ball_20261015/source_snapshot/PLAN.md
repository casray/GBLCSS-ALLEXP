# GREET / GRACE 自适应粒球重新实验

2026-10-15，用户授权更改粒球构造规则并再次实验。本计划在新目标与分类结果产生前固定；原实验与默认生成器不改。

**研究假设**：GREET旧规则仅1球是几何门槛不适配；放开相对半径门槛、维持合理投票支持后，局部证据可能恢复，最终是否优于GRACE仍须验证。

**粒球角色与动机**：冻结表示中的局部软监督区域；不是硬正样本，不用标签纯度划分。复用已有无标签自适应生成器，避免另搜大量阈值。sqrt仅规模预算，不是最优理论。

**规则**：两空间均L2归一化H；最大总SSE球优先、2-means n_init10；子球最少ceil(N/(4B))节点和16训练节点；不满足时用中心差方向可行投影切分；正SSE收益才接受，退化不强凑。B=64和ceil(sqrt(N))。支持16沿用已登记经验，不从本轮test选取。训练集合位置可用于支持约束，类别不能输入生成器。内部float64/分裂次序/fallback也有改变，不把整体差异只归因于取消5%门槛。

**配置与对照**：Roman / CoraFull / CS，协议seed0/1/2；clean/U0.4/Pair0.4，原划分和污染逐项哈希核对；同一已冻结GRACE/GREET H，每空间OLD64/ADAPT64/SQRT16/KM_MATCH/KNN_MATCH，另共同CE，共11臂×27条件=297目标和297学生。KM匹配SQRT16实际球数；kNN为每个训练节点匹配SQRT16其他训练投票数，并排除自身。相同LCSS、X-only MLP、200轮与初始化；clean验证选择epoch，测试不参与选择。

**代码改动**：本目录run.py、test_protocol.py、audit_report.py；使用GB-LCSS_code/src/gblcss/granular_ball_adaptive.py已有研究版，不改原版granular_ball.py或正式默认入口。新增config/PLAN/source_manifest及分区、目标、学生、raw log、统计CSV和审计。

**预期机制**：细分可能改善干净类别边际，也可能切碎真实语义。球数达标、SSE降低不等于去噪成功。新规则对两种表示都生效，用相同规则配对比较隔离表示作用；OLD64是历史锚。

**成功标准**：工程层面分区覆盖且无重复、支持≥16（全图训练不足16时单球例外）、预算/几何停止正确、目标与logits审计通过。研究推进标准为主规则带噪配对：每图3seed×2噪声平均Brier至少降低.005、6配对中至少5个改善；学生准确率平均至少+0.5pp、6配对至少5个改善，且不以显著clean损害换取。分别相对旧GRACE与同规则GRACE判断；粒球特有增量还须超过同空间KM与kNN。此为探索筛查，不是显著性检验；不通过也完整训练和报告。

**失败标准**：达到球数但目标/学生退化；仅一个噪声改善；不超过匹配近邻；clean平均退化>1pp；都据实报告，不以换阈值或只选成功图绕开。

**范围限制**：每空间只一个SSL seed，三协议seed不能代表三独立预训练。GREET是400epoch portable MC版本，旧GRACE缓存预算不同；不做公平计算预算算法排名，不声称HALF复现。本轮不叠HF或新损失，不扩更多数据集。

## 执行步骤

- [ ] 新增协议测试并确认缺实现时失败；实现无标签适配与固定队列；新增测试及既有生成器测试通过。
- [ ] 冻结执行源码、数据/H哈希；先全量生成分区和目标，再诊断，不根据中间分数修改计划。
- [ ] 完成297个200轮学生；每次保存logits与完整配置。独立审计原划分/噪声/旧目标/票数/softtarget/logits。
- [ ] 汇总mean/std与配对差、记录成功及负结果，写详细实验结论和继续/修改/暂缓/放弃决策。

运行：`& 'D:\codex_temp\noisygl-rocm721\Scripts\python.exe' 'experiments/greet_adaptive_ball_20261015/run.py' --device cuda`

核验：`& 'D:\codex_temp\noisygl-rocm721\Scripts\python.exe' 'experiments/greet_adaptive_ball_20261015/audit_report.py'`

实验结果与最终决策在实验结论.md及decision.json中填写；完成前不声称收益。
