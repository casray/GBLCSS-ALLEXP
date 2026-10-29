"""Post-lock scoring and frozen-encoder/target provenance audit."""
import csv,json,statistics as st
from pathlib import Path
import torch
import torch.nn.functional as F
from run import ROOT,CFG,PARENT,read,save,sha,load,ref,folder,down,verify_signature
from data_bridge import analysis_labels,load_features,load_supervised,state_hash
from contract import factorial_effects,validate_target

def score(task,path,receipt,head='SUPPORT32',new=False):
    ds,seed,f,t,c=(task[k] for k in ['dataset','seed','features','targets','condition'])
    r=read(path/'result.json');p=load(path/'probabilities.pt');target=load(path/'target.pt')
    for filename,key in [('student.pt','student_sha256'),('probabilities.pt','probabilities_sha256'),('target.pt','target_file_sha256'),('config.json','config.json_sha256'),('epochs.jsonl','epochs.jsonl_sha256')]:
        if sha(path/filename)!=r[key]:raise RuntimeError('Changed output '+str(path/filename))
    data=load_supervised(ds,seed,c);truth=analysis_labels(ds,seed);fr=ref(receipt,ds,seed,f);tr=ref(receipt,ds,seed,t)
    if head=='SUPPORT32':
        expected=down.build_support32_targets(data['observed'],load(tr['assignment'])[data['train']],classes=data['classes'])
        validate_target(target,expected,len(data['train']),data['classes'])
        assert torch.equal(target,load(Path(tr['conditions'][c]['SUPPORT32'])/'target.pt'))
    else:assert torch.equal(target,F.one_hot(data['observed'],data['classes']).float())
    rows=[json.loads(line) for line in (path/'epochs.jsonl').read_text().splitlines()]
    best=max(rows,key=lambda x:x['validation_correct'])
    assert len(rows)==200 and best['epoch']==r['best_epoch']
    correct=int((p[data['valid']].argmax(1)==data['valid_y']).sum())
    assert correct==r['valid_correct']
    cpu_error=None
    if new:
        bind=read(path/'source_binding.json');assert bind['task']==task and bind['result_sha256']==sha(path/'result.json')
        assert bind['feature_sha256']==sha(fr['encoder'])
        assert bind['target_source_file_sha256']==sha(Path(tr['conditions'][c]['SUPPORT32'])/'target.pt')
        x=load_features(ds,seed)['x'];m=down._CORE.MLP(x.shape[1],64,data['classes'],2,.5).eval();s=load(path/'student.pt');m.load_state_dict(s)
        enc=load(fr['encoder'])
        for k,v in s.items():
            if k.startswith(('layers.0.','batch_norms.')):assert torch.equal(v,enc['backbone.'+k])
        with torch.no_grad():replay=m(x).softmax(1)
        cpu_error=float((replay-p).abs().max())
        if cpu_error>2e-5:raise RuntimeError('CPU replay error '+str(cpu_error))
    y=truth['y'][truth['test']];pred=p[truth['test']].argmax(1);classes=p.shape[1]
    conf=torch.bincount(y*classes+pred,minlength=classes*classes).reshape(classes,classes).double()
    f1=float((2*conf.diag()/(conf.sum(0)+conf.sum(1)).clamp_min(1)).mean())
    brier=float((target-F.one_hot(truth['y'][truth['train']],classes)).square().sum(1).mean())
    return dict(**task,head=head,new=new,validation_accuracy=correct/len(data['valid']),test_accuracy=float((pred==y).float().mean()),test_macro_f1=f1,target_brier=brier,best_epoch=r['best_epoch'],cpu_replay_max_error=cpu_error,source=str(path))

def analyze():
    lock=read(ROOT/'formal/fits_locked.json');receipt=read(ROOT/'receipt.json')
    assert lock['receipt_sha256']==sha(ROOT/'receipt.json');verify_signature(receipt)
    scores=[]
    for task in lock['completed']:scores.append(score(task,folder(task),receipt,new=True))
    for ds,pair in CFG['pairs'].items():
        for seed in CFG['seeds']:
            for a in pair:
                rr=ref(receipt,ds,seed,a)
                for c in CFG['conditions']:
                    task=dict(dataset=ds,seed=seed,features=a,targets=a,condition=c)
                    for head in ['CE','SUPPORT32']:scores.append(score(task,Path(rr['conditions'][c][head]),receipt,head=head))
    out=ROOT/'analysis';save(out/'scores.json',scores)
    with (out/'scores.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(scores[0]));w.writeheader();w.writerows(scores)
    lookup={(x['dataset'],x['seed'],x['features'],x['targets'],x['condition'],x['head']):x for x in scores}
    summary=[];effects=[];gates=[];deltas=[]
    for ds,pair in CFG['pairs'].items():
        for f in pair:
            for t in pair:
                sub=[x for x in scores if x['dataset']==ds and x['features']==f and x['targets']==t and x['head']=='SUPPORT32']
                if len(sub)!=8:continue
                rr=dict(dataset=ds,features=f,targets=t)
                for metric in ['validation_accuracy','test_accuracy','test_macro_f1','target_brier']:
                    vals=[st.mean(x[metric] for x in sub if x['seed']==s and x['condition']!='clean') for s in CFG['seeds']]
                    scale=1 if metric=='target_brier' else 100
                    rr[metric]=dict(seeds=[scale*v for v in vals],mean=scale*st.mean(vals),sd=scale*st.stdev(vals))
                    rr['clean_'+metric]=scale*st.mean(x[metric] for x in sub if x['condition']=='clean')
                summary.append(rr)
                if f!=t:
                    sd=[];clean=[];td=[]
                    for seed in CFG['seeds']:
                        vd=[];testd=[]
                        for c in CFG['conditions']:
                            a=lookup[(ds,seed,f,t,c,'SUPPORT32')];b=lookup[(ds,seed,f,f,c,'SUPPORT32')]
                            delta=dict(dataset=ds,seed=seed,features=f,targets=t,condition=c)
                            for metric in ['validation_accuracy','test_accuracy','test_macro_f1','target_brier']:
                                delta[metric+'_difference']=(a[metric]-b[metric])*(1 if metric=='target_brier' else 100)
                            deltas.append(delta)
                            if c=='clean':clean.append(delta['validation_accuracy_difference'])
                            else:vd.append(delta['validation_accuracy_difference']);testd.append(delta['test_accuracy_difference'])
                        sd.append(st.mean(vd));td.append(st.mean(testd))
                    passed=st.mean(sd)>=.5 and all(v>0 for v in sd) and st.mean(clean)>=-.5
                    gates.append(dict(dataset=ds,features=f,targets=t,validation_noisy_seed_deltas_pp=sd,validation_noisy_mean_delta_pp=st.mean(sd),validation_clean_mean_delta_pp=st.mean(clean),test_noisy_seed_deltas_pp=td,test_noisy_mean_delta_pp=st.mean(td),passed=passed))
        a,b=pair
        for seed in CFG['seeds']:
            for c in CFG['conditions']:
                keys=[(ds,seed,f,t,c,'SUPPORT32') for f,t in [(a,a),(a,b),(b,a),(b,b)]]
                if all(k in lookup for k in keys):
                    for metric in ['validation_accuracy','test_accuracy','test_macro_f1']:
                        effects.append(dict(dataset=ds,seed=seed,condition=c,metric=metric,A=a,B=b,**factorial_effects(*(100*lookup[k][metric] for k in keys))))
    save(out/'summary.json',summary);save(out/'gates.json',gates);save(out/'paired_differences.json',deltas);save(out/'factorial_effects.json',effects)
    audit=dict(new_cells=len(lock['completed']),baseline_aliases=64,total_scored=len(scores),failures=lock['failures'],source_signature_verified=True,target_source_verified=True,encoder_unchanged_verified=True,selection_verified=True,max_new_cpu_replay_error=max((x['cpu_replay_max_error'] for x in scores if x['new']),default=None))
    save(out/'audit.json',audit)
    lines=['# 冻结R2 × 粒球目标交叉结果','',f"新拟合 {len(lock['completed'])}/32；已有对照64个；失败{len(lock['failures'])}。",'',
           '每个seed先平均三噪声，再报告双seed均值±样本SD。列：学生特征来源、粒球目标来源。','',
           '|图|学生|目标|带噪Accuracy%|SD|验证均值%|clean测试%|目标Brier|','|---|---|---|---:|---:|---:|---:|---:|']
    for x in summary:lines.append(f"|{x['dataset']}|{x['features']}|{x['targets']}|{x['test_accuracy']['mean']:.3f}|{x['test_accuracy']['sd']:.3f}|{x['validation_accuracy']['mean']:.3f}|{x['clean_test_accuracy']:.3f}|{x['target_brier']['mean']:.4f}|")
    lines+=['','## 预登记验证门','',json.dumps(gates,ensure_ascii=False,indent=2),'','## 审计','',json.dumps(audit,ensure_ascii=False,indent=2),
            '','仅此批交叉训练；未重训SSL/R2，未启动后续读出或新算法。逐噪声差值见paired_differences.json；2×2交互见factorial_effects.json。']
    (out/'REPORT_zh.md').write_text('\n'.join(lines),encoding='utf-8')
    save(ROOT/'formal/final.json',dict(status='complete' if len(lock['completed'])==32 and not lock['failures'] else 'complete_with_failures',scored=len(scores),audit_sha256=sha(out/'audit.json'),gates=gates))
    save(ROOT/'formal/status.json',dict(status='complete',completed=len(lock['completed']),total=32,failures=lock['failures']))
    print(json.dumps(dict(audit=audit,gates=gates),ensure_ascii=False),flush=True)

if __name__=='__main__':analyze()
