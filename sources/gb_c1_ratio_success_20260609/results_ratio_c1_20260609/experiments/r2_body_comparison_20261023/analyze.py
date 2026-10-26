"""Post-grid scoring, paired reports and source/selection integrity checks."""
import csv
import json
from pathlib import Path
import numpy as np
import torch
import run as r

def csvfile(path,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',encoding='utf-8-sig',newline='') as z:
        w=csv.DictWriter(z,fields);w.writeheader();w.writerows(rows)
def analyze():
    rows=[];checks=[]
    for name in r.CFG['datasets']:
        raw=r.f.r2.raw_data(name);y=raw.y.cpu().numpy();classes=int(y.max())+1
        for seed in r.CFG['seeds']:
            _,geo=r.fixed_data(name,seed)
            for cond in r.CFG['conditions']:
                for arm in r.CFG['arms']:
                    folder=r.ROOT/'fits'/name/f'seed{seed}'/cond/arm
                    meta=r.f.read(folder/'result.json');assert meta['status']=='complete'
                    assert r.sha(folder/'probabilities.pt')==meta['probabilities_sha256']
                    history=[json.loads(s) for s in (folder/'epochs.jsonl').read_text(encoding='utf-8').splitlines()]
                    best=max(h['validation_correct'] for h in history)
                    assert next(h['epoch'] for h in history if h['validation_correct']==best)==meta['best_epoch']
                    probs=torch.load(folder/'probabilities.pt',map_location='cpu',weights_only=True).numpy()
                    pred=probs.argmax(1);vi=geo['valid'].numpy();ti=geo['test'].numpy()
                    assert int((pred[vi]==y[vi]).sum())==meta['valid_correct']
                    m=r.accuracy_f1(y[ti],pred[ti],classes)
                    rows.append(dict(dataset=name,seed=seed,condition=cond,arm=arm,group='R2',accuracy=m['accuracy'],macro_f1=m['macro_f1'],validation=best/len(vi),best_epoch=meta['best_epoch'],seconds=meta['seconds']))
                    checks.append(dict(folder=str(folder),selection=True,hash=True))
        _,geo=r.fixed_data(name,0);ti=geo['test'].numpy();vi=geo['valid'].numpy()
        for cond in r.CFG['conditions']:
            for arm in ['SIMMLP','GSSC']:
                folder=r.ROOT/'external'/name/cond/arm;meta=r.f.read(folder/'result.json')
                assert meta['status']=='complete' and r.sha(folder/'probabilities.pt')==meta['probabilities_sha256']
                probs=torch.load(folder/'probabilities.pt',map_location='cpu',weights_only=True).numpy();p=probs.argmax(1)
                m=r.accuracy_f1(y[ti],p[ti],classes)
                rows.append(dict(dataset=name,seed=0,condition=cond,arm=arm,group='external',accuracy=m['accuracy'],macro_f1=m['macro_f1'],validation=float(np.mean(p[vi]==y[vi])),seconds=meta['seconds']))
    csvfile(r.ROOT/'results.csv',rows)
    summary=[];pairs=[];idx={(z['dataset'],z['seed'],z['condition'],z['arm']):z for z in rows}
    for name in r.CFG['datasets']:
        for cond in r.CFG['conditions']:
            for arm in r.CFG['arms']+['SIMMLP','GSSC']:
                subset=[z for z in rows if z['dataset']==name and z['condition']==cond and z['arm']==arm]
                summary.append(dict(dataset=name,condition=cond,arm=arm,seeds=len(subset),
                    accuracy_pct=100*np.mean([z['accuracy'] for z in subset]),
                    accuracy_std_pp=100*np.std([z['accuracy'] for z in subset],ddof=1) if len(subset)>1 else None,
                    macro_f1_pct=100*np.mean([z['macro_f1'] for z in subset])))
        for seed in r.CFG['seeds']:
            for cond in r.CFG['conditions']:
                for baseline in [a for a in r.CFG['arms'] if a!='SUPPORT']:
                    a,b=idx[name,seed,cond,'SUPPORT'],idx[name,seed,cond,baseline]
                    pairs.append(dict(dataset=name,seed=seed,condition=cond,baseline=baseline,
                         delta_accuracy_pp=100*(a['accuracy']-b['accuracy']),delta_macro_f1_pp=100*(a['macro_f1']-b['macro_f1']),delta_validation_pp=100*(a['validation']-b['validation'])))
    csvfile(r.ROOT/'summary.csv',summary);csvfile(r.ROOT/'paired.csv',pairs)
    lines=['# R2 同骨干监督与外部方法比较','',
       'R2 三seed固定历史GRACE，外部方法seed0。所有预定拟合完成后统一读取测试标签。',
       'PosteL为no-IPL；ALS/SALS为原稿机制适配。SimMLP复用作者无标签encoder并统一重训探针；GSSC作者核心移植。',
       '外部结构与辅助预算各自保留，跨组均值的种子数不同。','',
       '## 带噪三条件均值准确率 (%)','',
       '| 数据集 | CE | GCE | OLD64 | SUPPORT | KM | PosteL | SALS | ALS | SimMLP(s0) | GSSC(s0) |',
       '|---|'+'---:|'*10]
    for name in r.CFG['datasets']:
        vals=[]
        for arm in r.CFG['arms']+['SIMMLP','GSSC']:
            vals.append(np.mean([z['accuracy'] for z in rows if z['dataset']==name and z['condition']!='clean' and z['arm']==arm])*100)
        lines.append('| '+name+' | '+' | '.join(f'{v:.3f}' for v in vals)+' |')
    lines+=['','## SUPPORT的配对增量','', '| 数据集 | 对CE pp | 对GCE pp | 对KM pp | 对PosteL pp | 对SALS pp | 对ALS pp |','|---|'+'---:|'*6]
    for name in r.CFG['datasets']:
        vals=[np.mean([p['delta_accuracy_pp'] for p in pairs if p['dataset']==name and p['condition']!='clean' and p['baseline']==a]) for a in ['CE','GCE','KM','POSTEL','SALS','ALS']]
        lines.append('| '+name+' | '+' | '.join(f'{v:+.3f}' for v in vals)+' |')
    lines+=['','完整分条件mean/std与macro-F1见summary.csv；逐seed差值见paired.csv。',
        '同骨干增量直接评价监督规则；外部对照评价完整方法的训练/部署效果。',
        '旧beta=.5为历史开发配置。本轮保留它以隔离骨干和监督变化。',
        '所有负结果、clean损害和失败记录随目录保存。']
    (r.ROOT/'REPORT_zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    frozen=r.f.read(r.ROOT/'sources.json')
    for path,digest in frozen.items():assert r.sha(path)==digest,path
    inputs=r.f.read(r.ROOT/'inputs.json')
    for path,digest in inputs.items():assert r.sha(path)==digest,path
    r.save(r.ROOT/'audit.json',dict(passed=True,r2_cells=len(checks),external_cells=40,rows=len(rows),sources=len(frozen),inputs=len(inputs)))
    r.save(r.ROOT/'complete.json',dict(status='complete',r2_fits=480,external_cells=40))

if __name__=='__main__':analyze()
