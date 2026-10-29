"""Post-lock replay, independent scores and paired 2x2 effects."""
import json,csv,statistics as st
from pathlib import Path
from functools import lru_cache
import torch
import torch.nn.functional as F
from run import ROOT,CFG,read,sha,save,load,ref,folder,init_dir,verify_signature,down
from contract import effects,eligible
from data_bridge import load_supervised,analysis_labels,state_hash

@lru_cache(maxsize=8)
def feature(path):return load(path)

def score(task,path,receipt,new):
    ds,s,scale,init,c,head=(task[k] for k in ['dataset','seed','scale','initialization','condition','head'])
    rr=ref(receipt,ds,s);r=read(path/'result.json');p=load(path/'probabilities.pt');target=load(path/'target.pt')
    if new or init=='unified':
        record=read(path/'complete.json')
        for name,digest in record['files'].items():assert sha(path/name)==digest
    else:
        for name,key in [('student.pt','student_sha256'),('probabilities.pt','probabilities_sha256'),('target.pt','target_file_sha256'),('config.json','config.json_sha256'),('epochs.jsonl','epochs.jsonl_sha256')]:assert sha(path/name)==r[key]
    initial=load(init_dir(ds,s)/(init+'.pt'));initial_hash=state_hash(initial)
    if new:
        assert r['receipt_sha256']==sha(ROOT/'receipt.json') and r['initial_hash']==initial_hash and state_hash(load(path/'initial.pt'))==initial_hash
        assert r['feature_sha256']==sha(rr['features'][scale]) and r['target_source_sha256']==sha(Path(rr['raw_r2'][c][head])/'target.pt')
    elif init=='unified':assert r['initial_hash']==initial_hash
    else:assert r['initial_hash']==read(init_dir(ds,s)/'stats.json')['legacy_deployment_initial_hash']
    data=load_supervised(ds,s,c);truth=analysis_labels(ds,s)
    expected=F.one_hot(data['observed'],data['classes']).float()
    if head=='SUPPORT32':expected=down.build_support32_targets(data['observed'],load(rr['assignment'])[data['train']],classes=data['classes'])
    assert torch.equal(target,expected) and torch.equal(target,load(Path(rr['raw_r2'][c][head])/'target.pt'))
    logs=[json.loads(x) for x in (path/'epochs.jsonl').read_text().splitlines()]
    best=max(logs,key=lambda x:x['validation_correct']);assert len(logs)==200 and best['epoch']==r['best_epoch']
    valid_correct=int((p[data['valid']].argmax(1)==data['valid_y']).sum());assert valid_correct==r['valid_correct']==best['validation_correct']
    assert torch.isfinite(p).all() and (p>=0).all() and (p.sum(1)-1).abs().max()<2e-6
    error=None
    if new:
        weights=load(path/'student.pt')
        with torch.no_grad():replay=F.linear(feature(rr['features'][scale]),weights['weight'],weights['bias']).softmax(1)
        error=float((replay-p).abs().max());assert error<=2e-5
    y=truth['y'][truth['test']];pred=p[truth['test']].argmax(1);classes=p.shape[1]
    conf=torch.bincount(y*classes+pred,minlength=classes*classes).reshape(classes,classes).double()
    f1=float((2*conf.diag()/(conf.sum(0)+conf.sum(1)).clamp_min(1)).mean())
    return dict(task,new=new,validation_accuracy=valid_correct/len(data['valid']),test_accuracy=int((pred==y).sum())/len(y),test_macro_f1=f1,target_brier=float((target-F.one_hot(truth['y'][truth['train']],classes)).square().sum(1).mean()),best_epoch=r['best_epoch'],initial_hash=initial_hash,cpu_replay_max_error=error,source=str(path))

def aggregate(rows,metric):
    vals=[st.mean(r[metric] for r in rows if r['seed']==s and r['condition']!='clean') for s in CFG['seeds']]
    clean=[next(r[metric] for r in rows if r['seed']==s and r['condition']=='clean') for s in CFG['seeds']]
    return dict(noisy=dict(seeds=vals,mean=st.mean(vals),sd=st.stdev(vals)),clean=dict(seeds=clean,mean=st.mean(clean),sd=st.stdev(clean)))

def analyze():
    receipt=read(ROOT/'receipt.json');lock=read(ROOT/'formal/fits_locked.json');assert lock['receipt_sha256']==sha(ROOT/'receipt.json')
    verify_signature(receipt);verify_signature(read(ROOT/'initializations_locked.json'))
    scores=[score(t,folder(t),receipt,True) for t in lock['completed']]
    for rr in receipt['references']:
        for scale,init,key in [('raw','legacy','raw_r2'),('unit','unified','unit_unified')]:
            for c in CFG['conditions']:
                for head in CFG['heads']:
                    t=dict(dataset=rr['dataset'],seed=rr['seed'],scale=scale,initialization=init,condition=c,head=head)
                    scores.append(score(t,Path(rr[key][c][head]),receipt,False))
    out=ROOT/'analysis';save(out/'scores.json',scores)
    with (out/'scores.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(scores[0]));w.writeheader();w.writerows(scores)
    lookup={(r['dataset'],r['seed'],r['scale'],r['initialization'],r['condition'],r['head']):r for r in scores}
    metrics=['validation_accuracy','test_accuracy','test_macro_f1','target_brier'];summary=[];percondition=[];effect_rows=[];effect_summary=[];gates=[];norm_pairs=0
    for ds in CFG['datasets']:
        for head in CFG['heads']:
            for scale in ['raw','unit']:
                for init in ['legacy','unified']:
                    sub=[dict(r) for r in scores if (r['dataset'],r['head'],r['scale'],r['initialization'])==(ds,head,scale,init)]
                    if len(sub)!=8:continue
                    row=dict(dataset=ds,head=head,scale=scale,initialization=init)
                    for metric in metrics:
                        for r in sub:r[metric]*=1 if metric=='target_brier' else 100
                        row[metric]=aggregate(sub,metric)
                    summary.append(row)
                    for c in CFG['conditions']:
                        row=dict(dataset=ds,head=head,scale=scale,initialization=init,condition=c)
                        for metric in metrics:
                            vals=[r[metric] for r in sub if r['condition']==c]
                            row[metric]=dict(mean=st.mean(vals),sd=st.stdev(vals),seeds=vals)
                        percondition.append(row)
            for s in CFG['seeds']:
                for c in CFG['conditions']:
                    keys=[(ds,s,scale,init,c,head) for scale,init in [('raw','legacy'),('raw','unified'),('unit','legacy'),('unit','unified')]]
                    if not all(k in lookup for k in keys):continue
                    a,b,cc,d=[lookup[k] for k in keys]
                    assert a['initial_hash']==cc['initial_hash'] and b['initial_hash']==d['initial_hash'];norm_pairs+=2
                    for metric in metrics:
                        factor=1 if metric=='target_brier' else 100
                        ef=effects(*(lookup[k][metric]*factor for k in keys))
                        assert abs(ef['normalization_main']+ef['initialization_main']-ef['diagonal'])<1e-10
                        for effect,value in ef.items():effect_rows.append(dict(dataset=ds,head=head,seed=s,condition=c,metric=metric,effect=effect,value=value))
            for effect in effects(0,0,0,0):
                row=dict(dataset=ds,head=head,effect=effect)
                for metric in metrics:
                    sub=[r for r in effect_rows if (r['dataset'],r['head'],r['metric'],r['effect'])==(ds,head,metric,effect)]
                    if len(sub)!=8:break
                    row[metric]=aggregate(sub,'value')
                else:
                    effect_summary.append(row)
                    if effect.startswith('normalization_') and effect!='normalization_main':
                        vals=row['validation_accuracy']['noisy']['seeds'];clean=row['validation_accuracy']['clean']['seeds']
                        gates.append(dict(dataset=ds,head=head,initialization=effect.split('_')[1],validation_noisy_seed_deltas_pp=vals,validation_noisy_mean_delta_pp=st.mean(vals),validation_clean_seed_deltas_pp=clean,validation_clean_mean_delta_pp=st.mean(clean),noise_gate=st.mean(vals)>=.5 and all(v>0 for v in vals),full_gate=eligible(vals,clean)))
    save(out/'summary.json',summary);save(out/'per_condition.json',percondition);save(out/'factorial_rows.json',effect_rows);save(out/'factorial_summary.json',effect_summary);save(out/'gates.json',gates)
    audit=dict(new_heads=len(lock['completed']),aliases=64,total_scored=len(scores),failures=lock['failures'],source_hashes=len(receipt['signature']),matching_scale_initialization_pairs=norm_pairs,targets_unchanged=True,first_validation_max_verified=True,max_new_cpu_replay_error=max((r['cpu_replay_max_error'] for r in scores if r['new']),default=None))
    save(out/'audit.json',audit)
    lines=['# C批：R2尺度×初始化2×2结果','',f"新增{len(lock['completed'])}/64；历史引用64；失败{len(lock['failures'])}。所有头同训练协议，分别报告两seed样本SD。",'',
           '|图|监督|尺度|初始化|噪声Accuracy%|SD|clean Accuracy%|噪声验证%|','|---|---|---|---|---:|---:|---:|---:|']
    for r in summary:lines.append(f"|{r['dataset']}|{r['head']}|{r['scale']}|{r['initialization']}|{r['test_accuracy']['noisy']['mean']:.3f}|{r['test_accuracy']['noisy']['sd']:.3f}|{r['test_accuracy']['clean']['mean']:.3f}|{r['validation_accuracy']['noisy']['mean']:.3f}|")
    lines+=['','## 成对效应','', '|图|监督|效应|带噪测试pp|clean测试pp|带噪验证pp|clean验证pp|','|---|---|---|---:|---:|---:|---:|']
    for r in effect_summary:lines.append(f"|{r['dataset']}|{r['head']}|{r['effect']}|{r['test_accuracy']['noisy']['mean']:+.3f}|{r['test_accuracy']['clean']['mean']:+.3f}|{r['validation_accuracy']['noisy']['mean']:+.3f}|{r['validation_accuracy']['clean']['mean']:+.3f}|")
    lines+=['','## 预登记门','',json.dumps(gates,ensure_ascii=False,indent=2),'','## 审计','',json.dumps(audit,ensure_ascii=False,indent=2)]
    (out/'REPORT_zh.md').write_text('\n'.join(lines),encoding='utf-8')
    save(ROOT/'formal/final.json',dict(status='complete' if len(lock['completed'])==64 and not lock['failures'] else 'complete_with_failures',scored=len(scores),audit_sha256=sha(out/'audit.json')))
    save(ROOT/'formal/status.json',dict(status='complete',completed=len(lock['completed']),total=64,failures=lock['failures']))
    print(json.dumps(dict(audit=audit,gates=gates)),flush=True)

if __name__=='__main__':analyze()
