"""Post-fit scoring, independent NumPy target/statistic checks, and paired reports."""
import csv,json,itertools
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score,f1_score
from runtime import *

def independent_target(y,assignment,classes):
    y=np.asarray(y);assignment=np.asarray(assignment);out=np.zeros((len(y),classes),np.float64)
    for group in np.unique(assignment):
        ids=np.flatnonzero(assignment==group);counts=np.bincount(y[ids],minlength=classes)
        for i in ids:
            c=counts.copy();c[y[i]]-=1;n=c.sum();own=np.eye(classes)[y[i]]
            if n<2:out[i]=own;continue
            freq=c/n;alpha=np.clip((freq.max()-1/classes)/(1-1/classes),0,1)
            beta=.5 if np.sum(c==c.max())==1 else 0
            q=(1-beta)*freq+beta*np.eye(classes)[c.argmax()];out[i]=(1-alpha)*own+alpha*q
    return out

def analyze():
    lock=read(ROOT/'formal/fits_locked.json');assert len(lock['completed'])==84 and not lock['failures']
    verify(read(ROOT/'receipt.json')['signature']);rows=[];max_error=0.;max_target_error=0.
    for d,s in itertools.product(CFG['datasets'],CFG['seeds']):
        label=labels_after_lock(d,s);test=label['test'].numpy();y=label['y'].numpy()
        for arm in CFG['arms']:
            sf=ssl_dir(d,s,arm);manifest=read(sf/'complete.json')
            for name,hsh in manifest['files'].items():assert sha(sf/name)==hsh
            h=F.normalize(load(sf/'H.pt').float(),dim=1)
            for c in CFG['conditions']:
                data=supervised(d,s,c);tr=data['train'].numpy()
                for head in (['CE','fixed'] if arm=='ecl' else ['CE','fixed','own']):
                    p=head_dir(d,s,arm,c,head);m=read(p/'complete.json');r=read(p/'result.json')
                    for name,hsh in m['files'].items():assert sha(p/name)==hsh
                    prob=load(p/'probabilities.pt');model=load(p/'student.pt');target=load(p/'target.pt')
                    calc=F.linear(h,model['weight'],model['bias']).softmax(1)
                    err=float((calc-prob).abs().max());max_error=max(max_error,err)
                    assert err<=2e-5 and torch.equal(calc.argmax(1),prob.argmax(1)),(d,s,arm,c,head,err)
                    if head=='CE':expected=np.eye(data['classes'])[data['observed'].numpy()]
                    else:
                        origin='ecl' if head=='fixed' else arm
                        ass=load(ROOT/'formal/targets'/d/f'seed{s}'/origin/'support32.pt')[data['train']].numpy()
                        expected=independent_target(data['observed'].numpy(),ass,data['classes'])
                    terr=float(np.abs(expected-target.numpy()).max());max_target_error=max(max_target_error,terr);assert terr<2e-6
                    logs=[json.loads(line) for line in (p/'epochs.jsonl').read_text().splitlines()]
                    first=max(logs,key=lambda a:a['validation_correct']);assert first['epoch']==r['best_epoch'] and len(logs)==CFG['head_epochs']
                    valid_correct=int((prob[data['valid']].argmax(1)==data['valid_y']).sum());assert valid_correct==r['valid_correct']
                    pred=prob.argmax(1).numpy();true=np.eye(data['classes'])[y[tr]]
                    rows.append(dict(dataset=d,seed=s,arm=arm,condition=c,head=head,test_accuracy=float(accuracy_score(y[test],pred[test])),test_macro_f1=float(f1_score(y[test],pred[test],labels=list(range(data['classes'])),average='macro',zero_division=0)),validation_accuracy=valid_correct/len(data['valid']),target_brier=float(np.mean(np.sum((target.numpy()-true)**2,axis=1))),best_epoch=r['best_epoch'],source=str(p)))
    assert len(rows)==720
    save(ROOT/'analysis/scores.json',rows)
    with (ROOT/'analysis/scores.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    metrics=['test_accuracy','test_macro_f1','validation_accuracy','target_brier'];summary=[];pairs=[]
    lookup={(r['dataset'],r['seed'],r['arm'],r['condition'],r['head']):r for r in rows}
    for d in CFG['datasets']:
        for arm in CFG['arms']:
            for head in (['CE','fixed'] if arm=='ecl' else ['CE','fixed','own']):
                vals={m:[np.mean([lookup[d,s,arm,c,head][m] for c in CFG['conditions']]) for s in CFG['seeds']] for m in metrics}
                summary.append(dict(dataset=d,arm=arm,head=head,**{m:dict(mean=float(np.mean(v)),sd=float(np.std(v,ddof=1)),seeds=list(map(float,v))) for m,v in vals.items()}))
                if arm=='ecl':continue
                basehead='CE' if head=='CE' else 'fixed'
                for comparator in (['ecl', 'cos'+arm[2:], 'shuffle'+arm[2:]] if arm.startswith('gb') else ['ecl']):
                    comphead=basehead if comparator=='ecl' else head
                    differences={m:[100*np.mean([lookup[d,s,arm,c,head][m]-lookup[d,s,comparator,c,comphead][m] for c in CFG['conditions']]) for s in CFG['seeds']] for m in metrics}
                    pairs.append(dict(dataset=d,arm=arm,head=head,comparator=comparator,**{m:dict(mean_pp=float(np.mean(v)),sd_pp=float(np.std(v,ddof=1)),seed_pp=list(map(float,v))) for m,v in differences.items()}))
    save(ROOT/'analysis/summary.json',summary);save(ROOT/'analysis/paired.json',pairs)
    gates=[]
    for arm in ['gb025','gb100']:
        selected=[r for r in pairs if r['arm']==arm and r['head']=='fixed' and r['dataset'] in CFG['primary_datasets']]
        base=[r for r in selected if r['comparator']=='ecl'];delta=np.mean([r['validation_accuracy']['seed_pp'] for r in base],axis=0)
        cos=np.mean([r['validation_accuracy']['mean_pp'] for r in selected if r['comparator']=='cos'+arm[2:]])
        sh=np.mean([r['validation_accuracy']['mean_pp'] for r in selected if r['comparator']=='shuffle'+arm[2:]])
        positive=sum(r['validation_accuracy']['mean_pp']>0 for r in base)
        gates.append(dict(arm=arm,mean_validation_pp=float(delta.mean()),seed_validation_pp=delta.tolist(),positive_primary_graphs=positive,vs_cos_validation_pp=float(cos),vs_shuffle_validation_pp=float(sh),advance=bool(delta.mean()>=.5 and (delta>0).all() and positive>=3 and cos>0 and sh>0)))
    save(ROOT/'analysis/gates.json',gates)
    # Labels are first used for relation semantics here, after all model/target fits.
    semantics=[]
    for d,s in itertools.product(CFG['datasets'],CFG['seeds']):
        labels=labels_after_lock(d,s)['y'].numpy();e=features(d,s)['edge_index'].numpy();e=e[:,e[0]!=e[1]]
        cache=load(ROOT/'formal/warmup'/d/f'seed{s}'/'context/context.pt');score=cache['scores'].numpy();same=labels[e[0]]==labels[e[1]];cuts=np.quantile(score,np.linspace(0,1,6))
        for q in range(5):
            sel=(score>=cuts[q])&((score<=cuts[q+1]) if q==4 else (score<cuts[q+1]))
            semantics.append(dict(dataset=d,seed=s,quantile=q,edges=int(sel.sum()),same_class_fraction=float(same[sel].mean()) if sel.any() else None,lower=float(cuts[q]),upper=float(cuts[q+1])))
    save(ROOT/'analysis/relation_semantics.json',semantics)
    audit=dict(status='passed',scored=len(rows),ssl_endpoints=84,max_cpu_probability_error=max_error,max_independent_target_error=max_target_error,sources=len(read(ROOT/'receipt.json')['signature']))
    save(ROOT/'analysis/audit.json',audit)
    lines=['# GB-ContextECL 六图两seed结果','', '全部84个SSL端点与720个分类头拟合完成后评分；五主图与Roman分列，两个剂量及负结果全部保留。先在每seed内平均三噪声，再计算两seed均值/样本SD。','', '|数据|方法|监督|测试Accuracy±SD %|Macro-F1 %|验证 %|','|---|---|---|---:|---:|---:|']
    for r in summary:
        lines.append(f"|{r['dataset']}|{r['arm']}|{r['head']}|{100*r['test_accuracy']['mean']:.3f} ± {100*r['test_accuracy']['sd']:.3f}|{100*r['test_macro_f1']['mean']:.3f}|{100*r['validation_accuracy']['mean']:.3f}|")
    lines+=['','## 五主图固定目标验证门',json.dumps(gates,ensure_ascii=False,indent=2),'','## 判读','GB对ECL衡量附加机制增量；同剂量Cos和Shuffle用于解释增量来源。fixed与own的同H比较定位监督目标变化。Roman为另列机制扩展，其适配配置与五主图不同。clean不进入推进门。详细逐seed/逐条件及目标Brier见scores；梯度和权重剂量见formal/warmup。','',json.dumps(audit,ensure_ascii=False,indent=2)]
    (ROOT/'analysis/REPORT_zh.md').write_text('\n'.join(lines),encoding='utf-8')
    save(ROOT/'formal/final.json',dict(status='complete_audited',scored=len(rows),report=str(ROOT/'analysis/REPORT_zh.md'),audit_sha256=sha(ROOT/'analysis/audit.json')))

if __name__=='__main__':analyze()
