"""Post-freeze new two-graph scoring and immutable five-graph table merge."""
import csv
from pathlib import Path
import shutil
import statistics
import importlib.util

OLD=Path(__file__).resolve().parents[1]/'r2_teacher_dist_five_seed_20261026/analysis.py'
spec=importlib.util.spec_from_file_location('csphoto_summary_reference',OLD)
reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
summarize=reference.summarize


def csvread(path):
    with Path(path).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))


def analyze(run):
    p,ROOT,CONFIG=run.p,run.ROOT,run.CONFIG
    refs=p.read(ROOT/'teacher_refs.json');students=p.read(ROOT/'student_refs.json')
    rows,pairs,recalls,transfers,costs=[],[],[],[],[]
    for n in CONFIG['datasets']:
        raw=run.torch.load(p.track(p.common.DATA_FILES[n]),map_location='cpu',weights_only=False)[0]
        y=(raw['y'] if isinstance(raw,dict) else raw.y).long()
        for s in CONFIG['seeds']:
            for c in CONFIG['conditions']:
                v,geo=run.data(n,s,c);te=geo['test'];indexed={}
                for route in CONFIG['teacher_routes']:
                    teacher=refs[f'{run.cid(n,s,c)}/{route}']
                    student=next(r for r in students if (r['dataset'],r['seed'],r['condition'],r['route'])==(n,s,c,route))
                    pred={}
                    for kind,ref in [('teacher',teacher),('student',student)]:
                        folder=Path(ref['folder']);r=p.read(p.track(folder/'result.json'))
                        prob=p.load(folder/'probabilities.pt');pr=prob[te].argmax(1);pred[kind]=pr
                        row=dict(dataset=n,seed=s,condition=c,route=route,kind=kind,folder=str(folder),
                            probabilities_sha256=p.sha(folder/'probabilities.pt'),best_epoch=r['best_epoch'],validation=100*r['validation'],
                            accuracy=100*float((pr==y[te]).double().mean()),macro_f1=100*run.fitter.f1_score(pr,y[te],v['classes']))
                        rows.append(row);indexed[route,kind]=row
                        for cl in range(v['classes']):
                            mask=y[te]==cl
                            recalls.append(dict(dataset=n,seed=s,condition=c,route=route,kind=kind,class_id=cl,
                                test_count=int(mask.sum()),recall=100*float((pr[mask]==cl).double().mean()) if mask.any() else None))
                    tc,sc=pred['teacher']==y[te],pred['student']==y[te]
                    transfers.append(dict(dataset=n,seed=s,condition=c,route=route,test_nodes=len(te),
                        teacher_wrong_student_correct=int((~tc&sc).sum()),teacher_correct_student_wrong=int((tc&~sc).sum()),
                        net_gain_pp=100*float((sc.double()-tc.double()).mean())))
                    costs.append(dict(dataset=n,seed=s,condition=c,route=route,new_updates=200,
                        seconds=p.read(Path(student['folder'])/'result.json')['seconds'],teacher_new_updates=0,
                        historical_alignment_updates=100,teacher_search_updates=2450 if route=='R2_FULL' else 200))
                for a,ak,b,bk in [('R2_FULL','student','R2_CE','student'),('R2_FULL','teacher','R2_CE','teacher'),
                                   ('R2_FULL','student','R2_FULL','teacher'),('R2_CE','student','R2_CE','teacher')]:
                    pairs.append(dict(dataset=n,seed=s,condition=c,method=a,kind=ak,reference=b,reference_kind=bk,
                        **{m+'_diff_pp':indexed[a,ak][m]-indexed[b,bk][m] for m in ('accuracy','macro_f1')}))
    assert len(rows)==120 and len(students)==60
    for name,value in [('results',rows),('paired_differences',pairs),('per_class_recall',recalls),('transfers',transfers),('costs',costs)]:
        p.csv_save(ROOT/(name+'.csv'),value)
    for name,value,identity,metrics in [('summary',rows,['dataset','condition','route','kind'],['accuracy','macro_f1','validation']),
        ('paired_summary',pairs,['dataset','condition','method','kind','reference','reference_kind'],['accuracy_diff_pp','macro_f1_diff_pp'])]:
        p.csv_save(ROOT/(name+'.csv'),summarize(value,identity,metrics,CONFIG['seeds']))
    old=Path(CONFIG['previous_dist_runtime'])
    oldfreeze=p.read(p.track(old/'selections_frozen.json'))
    assert p.sha(p.track(old/'student_refs.json'))==oldfreeze['student_refs_sha256']
    previous=csvread(p.track(old/'results.csv'));previous_pairs=csvread(p.track(old/'paired_differences.csv'))
    for r in previous:
        r['seed']=int(r['seed'])
        for m in ('accuracy','macro_f1','validation'):r[m]=float(r[m])
    for r in previous_pairs:
        r['seed']=int(r['seed'])
        for m in ('accuracy_diff_pp','macro_f1_diff_pp'):r[m]=float(r[m])
    assert {r['dataset'] for r in previous}=={'cora','citeseer','corafull'} and len(previous)==180
    merged=previous+rows;mergedpairs=previous_pairs+pairs
    assert len(merged)==300 and len(mergedpairs)==300
    summary=summarize(merged,['dataset','condition','route','kind'],['accuracy','macro_f1','validation'],CONFIG['seeds'])
    ps=summarize(mergedpairs,['dataset','condition','method','kind','reference','reference_kind'],['accuracy_diff_pp','macro_f1_diff_pp'],CONFIG['seeds'])
    dest=ROOT/'tables_five_graphs'
    for name,value in [('results',merged),('paired_differences',mergedpairs),('summary',summary),('paired_summary',ps)]:
        p.csv_save(dest/(name+'.csv'),value)
    for metric in ('accuracy','macro_f1'):
        lines=['% Five protocol/student seeds0..4, sample SD, fixed historical GRACE.',r'\begin{tabular}{lccccc}',
            r'\hline',r'Method & Cora & CiteSeer & CoraFull & Coauthor-CS & Amazon-Photo \\',r'\hline']
        for condition in CONFIG['conditions']+['NOISY_MACRO']:
            lines.append(r'\multicolumn{6}{l}{'+condition.replace('_',r'\_')+r'} \\')
            for route,kind,label in [('R2_CE','teacher','R2+CE teacher'),('R2_CE','student','R2+CE to DIST'),
                                     ('R2_FULL','teacher','Complete R2 teacher'),('R2_FULL','student','Complete R2 to DIST')]:
                values=[]
                for n in ['cora','citeseer','corafull','coauthorcs','amazonpho']:
                    r=next(x for x in summary if (x['dataset'],x['condition'],x['route'],x['kind'])==(n,condition,route,kind))
                    values.append(f"${r[metric+'_mean']:.2f} \\pm {r[metric+'_std_ddof1']:.2f}$")
                lines.append(label+' & '+' & '.join(values)+r' \\')
            lines.append(r'\hline')
        lines.append(r'\end{tabular}')
        p.output(dest/(metric+'.tex')).write_text('\n'.join(lines)+'\n',encoding='utf-8')
    report=['# Coauthor-CS / Amazon-Photo 蒸馏线补齐','',
        '两图seed0..4，clean/U0.4/U0.6，完整R2与R2+CE冻结教师各接原MLP64/DIST200轮，共60新学生、12000更新。所有选模先冻结再测试。旧三图原结果保持，合并五图表重新等权聚合。下表单位%，均值±样本标准差ddof1。',
        '', '|图|CE教师→DIST|完整教师|完整教师→DIST|完整学生−CE学生 pp|完整学生−教师 pp|',
        '|---|---:|---:|---:|---:|---:|']
    for n in ['cora','citeseer','corafull','coauthorcs','amazonpho','ALL']:
        pick=lambda data,**kw:next(r for r in data if all(r[k]==v for k,v in kw.items()))
        common=dict(dataset=n,condition='NOISY_MACRO')
        values=[]
        for route,kind in [('R2_CE','student'),('R2_FULL','teacher'),('R2_FULL','student')]:
            r=pick(summary,**common,route=route,kind=kind);values.append(f"{r['accuracy_mean']:.3f} +/- {r['accuracy_std_ddof1']:.3f}")
        for route,kind in [('R2_CE','student'),('R2_FULL','teacher')]:
            r=pick(ps,**common,method='R2_FULL',kind='student',reference=route,reference_kind=kind)
            values.append(f"{r['accuracy_diff_pp_mean']:+.3f} +/- {r['accuracy_diff_pp_std_ddof1']:.3f}")
        report.append('|'+n+'|'+'|'.join(values)+'|')
    report+=['','Clean/macro-F1、全部逐seed和负格详CSV。原论文主线三种噪声，当前DIST表只覆盖U0.4/U0.6和clean，噪声范围注明。最终结论待独立CPU审计。',
        '', 'GPU在另一条SSL任务结束后串行运行，未停止或修改该任务。完整模型和学生均X-only，本轮为同结构再蒸馏；历史SSL固定，没有新增预训练或教师选择。',
        '代码/测试/源输入hash/队列状态保留在新目录，旧结果未覆盖，稿件及GitHub未改。']
    p.output(ROOT/'REPORT_zh.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    p.save(ROOT/'analysis_complete.json',dict(time=p.now(),new_rows=120,merged_rows=300,new_students=60,new_updates=12000))
    run.CODE.mkdir(exist_ok=True)
    for f in list(ROOT.glob('*.csv'))+[ROOT/'REPORT_zh.md']:
        shutil.copy2(f,run.CODE/f.name)
    local=run.CODE/'tables_five_graphs';local.mkdir(exist_ok=True)
    for f in dest.glob('*'):shutil.copy2(f,local/f.name)
