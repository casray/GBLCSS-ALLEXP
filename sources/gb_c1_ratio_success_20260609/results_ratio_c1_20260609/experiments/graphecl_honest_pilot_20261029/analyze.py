"""Independent post-lock scores, paired effects, gates and Chinese report."""
from common import *
from mechanisms import *
import statistics


def mean_sd(values):
    return dict(mean=float(np.mean(values)),sd=float(np.std(values,ddof=1)) if len(values)>1 else 0.)


def stage1_summary():
    rows=read(ROOT/'stage1/scores.json');changes=read(ROOT/'stage1/support_changes.json');summary=[]
    for d in CFG['datasets']:
        lookup={(r['seed'],r['condition'],r['arm']):r for r in rows if r['dataset']==d}
        out=dict(dataset=d)
        for arm in ['H0_q1','H1_q0','H1_q1']:
            ds=[100*np.mean([lookup[s,c,arm]['accuracy']-lookup[s,c,'H0_q0']['accuracy'] for c in CFG['diagnostic_conditions']]) for s in CFG['seeds']]
            out[arm]=dict(**mean_sd(ds),seeds=ds)
        dr=[r for r in changes if r['dataset']==d]
        for key in ['changed_fraction','jaccard','old_support','new_support','delta_brier','lost_same_class_support_fraction','old_true_margin','new_true_margin','churn_brier_spearman']:
            out[key]=float(np.mean([r[key] for r in dr if r[key] is not None]))
        summary.append(out)
    save(ROOT/'stage1/summary.json',summary);return summary


def target_geometry(d,s,arm,data,truth):
    train=data['train'].numpy();parts=split_folds(train,s);out=[]
    for f,positions in enumerate(parts):
        if arm=='b0':folder=ROOT/'crossfit/packages'/d/f'seed{s}'/f'fold{f}'/'partition'
        else:
            name=f'b20_fold{f}' if arm=='b20' else f'{arm}_vote_fold{f}'
            folder=ROOT/'crossfit/targets'/d/f'seed{s}'/name/'partition'
        assignment=load(folder/'support32.pt').numpy()[train[positions]];y=truth[positions]
        detail=vote_details(y,assignment,data['classes']);margin,purity=true_margins(y,assignment,data['classes'])
        out.append(dict(dataset=d,seed=s,arm=arm,fold=f,nodes=len(y),support=float(detail['support'].mean()),true_margin=float(margin.mean()),
            true_class_fraction=float(purity.mean()),nonpositive_margin_fraction=float((margin<=0).mean()),
            s_margin2_positive=float(np.mean(detail['support']*np.maximum(margin,0)**2))))
    return out


def crossfit_summary():
    lock=read(ROOT/'crossfit/fits_locked.json')
    if len(lock['ssls'])!=20 or len(lock['heads'])!=20:raise RuntimeError('Incomplete frozen matrix')
    rows=[];audits=[];geometry=[]
    for d in CFG['datasets']:
        for s in CFG['seeds']:
            lab=old.gvec.analysis_labels(d,s);data=old.supervised(d,s,CFG['crossfit_condition']);h0=load(source_ssl(d,s)/'H.pt')
            for arm in ['a0','CE','b0','a20','b20','real','shuffle']:
                if arm in ['a0','CE']:folder=source_head(d,s,CFG['crossfit_condition'],head='fixed' if arm=='a0' else 'CE')
                else:folder=ROOT/'crossfit/formal/heads'/d/f'seed{s}'/arm
                audits.append(verify_head(folder,h0));rows.append(dict(dataset=d,seed=s,arm=arm,**score(folder,lab,data)))
                if arm in ['b0','b20','real','shuffle']:geometry.extend(target_geometry(d,s,arm,data,lab['y'][data['train']].numpy()))
    save(ROOT/'crossfit/scores.json',rows);csv_write(ROOT/'crossfit/scores.csv',rows)
    save(ROOT/'crossfit/geometry.json',geometry);csv_write(ROOT/'crossfit/geometry.csv',geometry)
    save(ROOT/'crossfit/independent_audit.json',dict(heads=28,gpu_bitwise=all(r['gpu_bitwise'] for r in audits),rows=audits))
    lookup={(r['dataset'],r['seed'],r['arm']):r for r in rows};pairs=[]
    for d in CFG['datasets']:
        for arm,ref in [('b0','a0'),('b20','a0'),('a20','a0'),('real','b20'),('real','a0'),('real','shuffle'),('shuffle','b20')]:
            row=dict(dataset=d,arm=arm,reference=ref)
            for metric in ['accuracy','macro_f1','validation','brier']:
                delta=[(100 if metric!='brier' else 1)*(lookup[d,s,arm][metric]-lookup[d,s,ref][metric]) for s in CFG['seeds']]
                row[metric]=dict(**mean_sd(delta),seeds=delta)
            pairs.append(row)
    save(ROOT/'crossfit/paired.json',pairs)
    vals=lambda ref,metric:[np.mean([lookup[d,s,'real'][metric]-lookup[d,s,ref][metric] for d in CFG['datasets']]) for s in CFG['seeds']]
    vb=vals('b20','validation');vd=vals('shuffle','validation');va=vals('a0','validation')
    br=[np.mean([lookup[d,s,'real']['brier']-lookup[d,s,'b20']['brier'] for s in CFG['seeds']]) for d in CFG['datasets']]
    gate=dict(validation_real_minus_b20_pp=100*float(np.mean(vb)),validation_seed_deltas_pp=[100*float(v) for v in vb],
        validation_real_minus_shuffle_pp=100*float(np.mean(vd)),validation_real_minus_a0_pp=100*float(np.mean(va)),brier_by_dataset=[float(v) for v in br],
        passed=bool(np.mean(vb)>=.01 and min(vb)>0 and np.mean(vd)>0 and np.mean(va)>=0 and np.mean(br)<0 and max(br)<=0),
        test_real_minus_b20_pp=100*float(np.mean(vals('b20','accuracy'))),test_real_minus_a0_pp=100*float(np.mean(vals('a0','accuracy'))),
        test_real_minus_shuffle_pp=100*float(np.mean(vals('shuffle','accuracy'))))
    save(ROOT/'crossfit/decision.json',gate);return rows,pairs,gate


def make_report(first,second=None):
    screen=read(ROOT/'stage1/decision.json')
    lines=['# GraphECL 两阶段小规模实验结果','', '日期：2026-10-29。CS、Photo，seed0/1；旧源只读，新结果独立保存。','',
           '## 1. ContextECL 反事实补全','', '表中为相对 H0+q0 的测试准确率变化，先每 seed 聚合三种噪声，再报告 mean±样本SD，单位pp。','',
           '|图|H0+新目标|新H+旧目标|新H+新目标|','|---|---:|---:|---:|']
    for row in first:
        vals=[f"{row[a]['mean']:+.3f} ± {row[a]['sd']:.3f}" for a in ['H0_q1','H1_q0','H1_q1']]
        lines.append('|'+row['dataset']+'|'+'|'.join(vals)+'|')
    for row in first:lines+=['',f"{row['dataset']}：训练节点票源集合改变 {100*row['changed_fraction']:.2f}%，平均Jaccard {row['jaccard']:.4f}；真实类别票差 {row['old_true_margin']:.4f}→{row['new_true_margin']:.4f}；Brier变化 {row['delta_brier']:+.6f}。票源变动与Brier变化的Spearman均值 {row['churn_brier_spearman']:.4f}，仅作描述性关联。"]
    lines+=['','## 2. 路线1缓存筛查','',f"GB增量AUROC {screen['gb_delta_auc']:+.6f}；KMeans {screen['km_delta_auc']:+.6f}；GB的Brier回归MSE下降量 {screen['gb_brier_mse_reduction']:+.6f}。两个seed AUROC增量 {screen['seed_delta_auc']}。推进门：{'通过' if screen['passed'] else '未通过，停止当前KL定义的编码器训练'}。",'',
            '这些诊断使用人工真值训练响应，按原GB区域分组交叉验证；属于探索性开发证据。只有缓存级KMeans，无新KMeans编码器训练。']
    if second:
        rows,pairs,gate=second
        lines+=['','## 3. 交叉拟合 Uniform .4','', '固定原H0读出；b20为同轮数无标签续训后的折内票源，real/shuffle均20轮。','',
            '|图|real−b20 测试pp|real−a0 测试pp|real−shuffle 测试pp|real−b20 Brier|','|---|---:|---:|---:|---:|']
        for d in CFG['datasets']:
            get=lambda ref:next(r for r in pairs if r['dataset']==d and r['arm']=='real' and r['reference']==ref)
            vals=[f"{get(ref)['accuracy']['mean']:+.3f} ± {get(ref)['accuracy']['sd']:.3f}" for ref in ['b20','a0','shuffle']]
            lines.append('|'+d+'|'+'|'.join(vals)+f"|{get('b20')['brier']['mean']:+.6f}|")
        lines+=['',f"预登记验证门：{'通过' if gate['passed'] else '未通过'}。real−b20验证 {gate['validation_real_minus_b20_pp']:+.3f}pp，两个seed {gate['validation_seed_deltas_pp']}；real−shuffle {gate['validation_real_minus_shuffle_pp']:+.3f}pp；real−原全票 {gate['validation_real_minus_a0_pp']:+.3f}pp。",'',
            '本轮共有20个短续训端点、20个新分类头；最终比较另引用8个原头。全部新头与原参照按同一H0、同初值、200轮和首次最大验证正确数选模。固定H0使CE保持相同，交互量退化为SUPPORT增量。', '',
            '## 4. 决策与解释', '',
            '已支持的事实以上表和逐seed原始记录为准。机制解释结合geometry.csv中的支持量、真实类别票差和目标Brier；方法归因需要后续普通软监督对比。',
            '当前决策：'+('保留交叉拟合信号，先做普通监督对比再讨论扩噪声。' if gate['passed'] else '停止本规格扩噪声和额外消融；保留正负结果，先分析折拆分代价与标签模块增量。')]
    lines+=['','## 可追溯记录','', 'config.json与PLAN.md固定科学规格；运行根receipt.json和crossfit/receipt.json保存来源哈希；stage1/和crossfit/保存逐节点CSV、预测、权重、原始epoch日志、分区和目标；独立GPU重载与双精度评分结果归档。','']
    old.art.atomic(ROOT/'REPORT_zh.md',lambda p:Path(p).write_text('\n'.join(lines),encoding='utf-8'))


def main():
    prepare();first=stage1_summary();second=crossfit_summary() if (ROOT/'crossfit/fits_locked.json').exists() else None
    make_report(first,second)
    if second:status('PILOT_COMPLETE',stage1_passed=read(ROOT/'stage1/decision.json')['passed'],crossfit_passed=second[2]['passed'])
    else:print(json.dumps(first,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
