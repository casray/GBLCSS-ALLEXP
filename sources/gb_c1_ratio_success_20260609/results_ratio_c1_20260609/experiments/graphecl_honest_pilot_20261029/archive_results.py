"""Archive audited small tables and an explicit protocol/results narrative."""
from common import *


def ms(v,scale=1):
    v=np.asarray(v)*scale
    return f'{v.mean():.3f} ± {v.std(ddof=1):.3f}'


def main():
    audit=read(ROOT/'review/audit.json');assert audit['status']=='passed'
    gate=read(ROOT/'crossfit/decision.json');screen=read(ROOT/'stage1/decision.json')
    scores=read(ROOT/'crossfit/scores.json');pairs=read(ROOT/'crossfit/paired.json')
    geometry=read(ROOT/'review/split_geometry.json');aux=read(ROOT/'review/auxiliary_quality.json')
    losses=read(ROOT/'review/loss_trends.json');drift=read(ROOT/'review/embedding_drift.json')
    lookup={(r['dataset'],r['seed'],r['arm']):r for r in scores}
    names=dict(a0='原空间＋全票',b0='原空间＋折内票',a20='无标签续训＋全票',b20='无标签续训＋折内票',real='交叉拟合真实目标',shuffle='交叉拟合打乱目标',CE='固定H0＋CE')
    lines=['# 交叉拟合粒球对比：实验解释与完整结果','',
        '日期：2026-10-29。全部数字来自冻结矩阵和独立复核。所有 mean±SD 的统计单位是两个协议 seed；数据集先在各 seed 内等权聚合。',
        '', '## 研究问题与方法', '',
        'H1：图/属性分支的粒球集合兼容分布，能否补充现有几何与投票集中度对目标质量的预测信息。H2：另一折标签驱动的粒球对比空间，能否改善本折的含噪局部共识，并抵消拆折代价。',
        '', 'H1在CS/Photo、seed0/1、U.4/U.6/Pair.4上使用缓存诊断。H2限定CS/Photo、seed0/1、Uniform 0.4，20轮短续训、原学习率×0.1，原始GraphECL图结构和损失保持。每图每seed包含一条无标签续训、两折真实监督、两折打乱监督，合计20端点。',
        '', '随机分A/B；E_A只接收A折节点的原LCSS完整软目标，E_B对称。所有节点的X/图仍用于原无标签目标。原空间按该折训练掩码裁剪SUPPORT32。辅助正对权重为同球候选上的 max(t_i·t_j−1/C,0)，去除自身并逐行归一化；对比项使用属性分支的投影输出，候选key是本折所有训练节点。每轮至多512个查询，温度0.5。',
        '', 'E_A训练后，在其原生属性H中按B折掩码构球，只用B折观测标签做LOO投票，E_B对称。两份目标拼回原训练节点顺序；所有分类头固定使用原H0和同初值，200轮、首次最大验证正确数选模。该阶段直接检验上游产生目标的价值；新表示完整读出留待净收益信号成立后测试。',
        '', '辅助权重按初始参数梯度校准为 min(1,0.1||g_ECL||/||g_aux||)，每折固定，打乱分支共用该权重。正式过程无验证集调节上游权重、轮数或折划分。原先无标签起点、折外标签隔离、单向构造目标是理论条件的实现基础。',
        '', '## 缓存筛查及反事实', '',
        f"路线1的GB增量AUROC={screen['gb_delta_auc']:+.6f}，KMeans={screen['km_delta_auc']:+.6f}，Brier回归MSE下降量={screen['gb_brier_mse_reduction']:+.6f}；推进门通过={screen['passed']}。缓存筛查未进入新增集合KL编码器训练。诊断拟合按原粒球分组验证，人工真值只作为事后响应。",
        '', 'ContextECL的完整2×2反事实、票源变动和类别票差见REPORT_zh.md及results/stage1；本轮新增12个反事实头，并复现12个原头。',
        '', '## 交叉拟合准确率', '', '|方法|CS测试%|Photo测试%|CS验证%|Photo验证%|','|---|---:|---:|---:|---:|']
    for arm in ['CE','a0','b0','a20','b20','real','shuffle']:
        values=[ms([lookup[d,s,arm][m] for s in CFG['seeds']],100) for m in ['accuracy','validation'] for d in CFG['datasets']]
        lines.append('|'+names[arm]+'|'+'|'.join(values)+'|')
    lines+=['','配对差值先逐seed相减，以下SD是差值的样本SD。','',
            '|图|比较|测试差pp|验证差pp|目标Brier差|','|---|---|---:|---:|---:|']
    for r in pairs:
        values=[f"{r[k]['mean']:+.3f} ± {r[k]['sd']:.3f}" for k in ['accuracy','validation']]
        lines.append(f"|{r['dataset']}|{r['arm']}−{r['reference']}|"+'|'.join(values)+f"|{r['brier']['mean']:+.6f}|")
    lines+=['','## 预登记门与决策','',
        '主门：真实监督相对同预算折内基线验证平均≥1pp，两个seed宏增量为正；优于打乱监督；不低于原全票验证；Brier宏改善且两图均不恶化。clean未参与否决。',
        '',f"实际验证 real−b20={gate['validation_real_minus_b20_pp']:+.3f}pp，seed宏差={gate['validation_seed_deltas_pp']}；real−shuffle={gate['validation_real_minus_shuffle_pp']:+.3f}pp；real−a0={gate['validation_real_minus_a0_pp']:+.3f}pp。两图Brier差={gate['brier_by_dataset']}。门通过={gate['passed']}。",
        '', '最终决策：'+('继续保留信号，补普通软监督对比以检查粒球额外作用。' if gate['passed'] else '暂缓本规格，停止自动扩噪声、轮数与额外消融。'),
        '', '## 拆折与正对机制', '', '|图|区域|平均其他训练票数|节点所属区域总节点数|真实类别票差|同类票比例|','|---|---|---:|---:|---:|---:|']
    for d in CFG['datasets']:
        for arm in ['a0','b0']:
            rows=[r for r in geometry if r['dataset']==d and r['arm']==arm]
            lines.append(f'|{d}|{arm}|'+ '|'.join(f"{np.mean([r[k] for r in rows]):.4f}" for k in ['support','region_nodes','true_margin','true_class_fraction'])+'|')
    lines+=['','下表正对同类率使用真实类别做事后诊断，按查询逐行正对权重加权后平均；有效正对数为1/Σw²。','',
        '|图|监督|加权正对同类率|同球均匀正对同类率|有效正对数|有效查询比例|','|---|---|---:|---:|---:|---:|']
    for d in CFG['datasets']:
        for arm in ['real','shuffle']:
            rows=[r for r in aux if r['dataset']==d and r['arm']==arm]
            lines.append(f'|{d}|{arm}|'+ '|'.join(f"{np.mean([r[k] for r in rows]):.4f}" for k in ['positive_class_precision','same_ball_uniform_precision','effective_positive_count','active_fraction'])+'|')
    lines+=['','## 干预与实际成本','', '|图|监督|辅助损失首轮|末轮|H相对等预算ECL平均余弦|','|---|---|---:|---:|---:|']
    for d in CFG['datasets']:
        for arm in ['real','shuffle']:
            ls=[r for r in losses if r['dataset']==d and r['arm']==arm];ds=[r for r in drift if r['dataset']==d and r['arm']==arm]
            lines.append(f"|{d}|{arm}|{np.mean([r['first_aux'] for r in ls]):.4f}|{np.mean([r['last_aux'] for r in ls]):.4f}|{np.mean([r['cosine_to_equal_budget_ecl_mean'] for r in ds]):.6f}|")
    lines+=['',f"20个正式分支合计400轮更新，分支训练与保存计时之和{sum(r['seconds'] for r in losses)/60:.2f}分钟，构球、输入核验、标定与诊断时间另计；另有工程冒烟更新。共44新头（24反事实/复现、20交叉拟合）。",
        '', '第一次前台进程在7个SSL、5个头完成后退出，未捕获到退出原因；恢复时已完成产物经哈希复用。隐藏启动器保存resume_stdout/stderr及退出码，中断现场保存在interruption_recovery。没有更改训练规格。',
        '', '## 验证与复用', '',
        f"9项机制测试通过。独立核查：{audit['formal_ssl']}个SSL、{audit['new_heads']}个新头、{audit['ssl_updates']}轮SSL日志、{audit['head_updates']}轮分类头日志、{audit['manifest_file_hashes']}个产物哈希；独立指标最大误差{audit['max_independent_metric_error']:.3g}。12个原头概率复现逐位一致；所有正式SSL保存时完成GPU表示逐位重载，交叉拟合最终28头额外GPU概率重载通过。",
        '', '人工干净训练标签仅用于相关阶段拟合锁定后的离线诊断；验证标签用于按固定规则选择分类头。两seed小规模筛查保留逐格结果；普通软监督对比、其他噪声条件与独立开发集确认尚待信号成立后进行。',
        '', '代码/配置：本目录config.json、PLAN.md、mechanisms.py、common.py、stage1.py、crossfit.py、analyze.py、review_audit.py及两份test文件。来源收据和小表在results/。权重、预测、逐节点明细和原始日志：D:/codex_temp/graphecl_honest_pilot_20261029。', '']
    (CODE/'RESULT_INTERPRETATION_zh.md').write_text('\n'.join(lines),encoding='utf-8')
    shutil.copy2(ROOT/'REPORT_zh.md',CODE/'REPORT_zh.md')
    for stage in ['stage1','crossfit','review']:
        dest=CODE/'results'/stage;dest.mkdir(parents=True,exist_ok=True)
        for pattern in ['*.json','*.csv']:
            for p in (ROOT/stage).glob(pattern):shutil.copy2(p,dest/p.name)
    for name in ['receipt.json','status.json','launcher_status.json','resume_launch.json']:
        p=ROOT/name
        if p.exists():shutil.copy2(p,CODE/'results'/name)
    for name in ['config.json','PLAN.md']:
        shutil.copy2(CODE/name,ROOT/name)
    evidence={str(p.relative_to(CODE)):sha(p) for p in (CODE/'results').rglob('*') if p.is_file()}
    (CODE/'ARCHIVE_MANIFEST.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(archived=len(evidence),report=str(CODE/'RESULT_INTERPRETATION_zh.md'),gate=gate['passed']),ensure_ascii=False))


if __name__=='__main__':main()
