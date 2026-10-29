"""Post-lock scoring, independent head replay, paired readout comparison."""
import csv,json,statistics as st
from pathlib import Path
from functools import lru_cache
import torch
import torch.nn.functional as F
from run import ROOT,CFG,read,save,sha,load,ref,folder,feature_dir,down,verify_signature
from data_bridge import analysis_labels,load_supervised

@lru_cache(maxsize=8)
def feature(ds,seed,rep):return load(feature_dir(ds,seed)/(rep+'.pt'))

def score(task,path,receipt,new=False):
    ds,seed,rep,c,head=(task[k] for k in ['dataset','seed','representation','condition','head'])
    rr=ref(receipt,ds,seed);r=read(path/'result.json');p=load(path/'probabilities.pt');target=load(path/'target.pt')
    if rep!='r2_raw64':
        record=read(path/'complete.json')
        for name,digest in record['files'].items():assert sha(path/name)==digest
    else:
        for name,key in [('student.pt','student_sha256'),('probabilities.pt','probabilities_sha256'),('target.pt','target_file_sha256'),('config.json','config.json_sha256'),('epochs.jsonl','epochs.jsonl_sha256')]:assert sha(path/name)==r[key]
    assert torch.isfinite(p).all() and (p>=0).all() and (p.sum(1)-1).abs().max()<2e-6
    data=load_supervised(ds,seed,c);truth=analysis_labels(ds,seed)
    expected=F.one_hot(data['observed'],data['classes']).float()
    if head=='SUPPORT32':expected=down.build_support32_targets(data['observed'],load(rr['assignment'])[data['train']],classes=data['classes'])
    assert torch.equal(target,expected) and torch.equal(target,load(Path(rr['native'][c][head])/'target.pt'))
    epochs=[json.loads(x) for x in (path/'epochs.jsonl').read_text().splitlines()]
    best=max(epochs,key=lambda x:x['validation_correct'])
    assert len(epochs)==200 and best['epoch']==r['best_epoch']
    correct=int((p[data['valid']].argmax(1)==data['valid_y']).sum())
    assert correct==r['valid_correct']==best['validation_correct']
    error=None
    if new:
        bind=read(path/'source_binding.json')
        assert bind['task']==task and bind['receipt_sha256']==sha(ROOT/'receipt.json') and bind['result_sha256']==sha(path/'result.json')
        assert bind['feature_sha256']==sha(feature_dir(ds,seed)/(rep+'.pt')) and bind['target_source_sha256']==sha(Path(rr['native'][c][head])/'target.pt')
        weights=load(path/'student.pt')
        with torch.no_grad():replay=F.linear(feature(ds,seed,rep),weights['weight'],weights['bias']).softmax(1)
        error=float((replay-p).abs().max())
        if error>2e-5:raise RuntimeError('CPU head replay mismatch '+str(error))
    y=truth['y'][truth['test']];pred=p[truth['test']].argmax(1);classes=p.shape[1]
    conf=torch.bincount(y*classes+pred,minlength=classes*classes).reshape(classes,classes).double()
    f1=float((2*conf.diag()/(conf.sum(0)+conf.sum(1)).clamp_min(1)).mean())
    brier=float((target-F.one_hot(truth['y'][truth['train']],classes)).square().sum(1).mean())
    return dict(**task,new=new,validation_accuracy=correct/len(data['valid']),test_accuracy=int((pred==y).sum())/len(y),test_macro_f1=f1,target_brier=brier,best_epoch=r['best_epoch'],initial_hash=r['initial_hash'],cpu_replay_max_error=error,source=str(path))

def analyze():
    lock=read(ROOT/'formal/fits_locked.json');receipt=read(ROOT/'receipt.json')
    assert lock['receipt_sha256']==sha(ROOT/'receipt.json');verify_signature(receipt);verify_signature(read(ROOT/'features_locked.json'))
    scores=[score(task,folder(task),receipt,new=True) for task in lock['completed']]
    for rr in receipt['references']:
        for rep,key in [('native','native'),('r2_raw64','raw_r2')]:
            for c in CFG['conditions']:
                for head in CFG['heads']:
                    task=dict(dataset=rr['dataset'],seed=rr['seed'],representation=rep,condition=c,head=head)
                    scores.append(score(task,Path(rr[key][c][head]),receipt))
    out=ROOT/'analysis';save(out/'scores.json',scores)
    with (out/'scores.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(scores[0]));writer.writeheader();writer.writerows(scores)
    lookup={(r['dataset'],r['seed'],r['representation'],r['condition'],r['head']):r for r in scores}
    metrics=['validation_accuracy','test_accuracy','test_macro_f1','target_brier']
    summary=[];bycondition=[];gates=[];paired=[];comparisons=[];init_pairs=0
    for ds in CFG['datasets']:
        for head in CFG['heads']:
            for rep in ['native','pca64','r2_unit64','r2_raw64']:
                sub=[r for r in scores if r['dataset']==ds and r['representation']==rep and r['head']==head]
                if len(sub)!=8:continue
                row=dict(dataset=ds,representation=rep,head=head)
                for metric in metrics:
                    scale=1 if metric=='target_brier' else 100
                    vals=[scale*st.mean(r[metric] for r in sub if r['seed']==s and r['condition']!='clean') for s in CFG['seeds']]
                    clean=[scale*next(r[metric] for r in sub if r['seed']==s and r['condition']=='clean') for s in CFG['seeds']]
                    row[metric]=dict(seeds=vals,mean=st.mean(vals),sd=st.stdev(vals))
                    row['clean_'+metric]=dict(seeds=clean,mean=st.mean(clean),sd=st.stdev(clean))
                summary.append(row)
                for c in CFG['conditions']:
                    crow=dict(dataset=ds,representation=rep,head=head,condition=c)
                    for metric in metrics:
                        vals=[r[metric]*(1 if metric=='target_brier' else 100) for r in sub if r['condition']==c]
                        crow[metric]=dict(mean=st.mean(vals),sd=st.stdev(vals),seeds=vals)
                    bycondition.append(crow)
            for a,b in [('pca64','r2_unit64'),('pca64','native'),('r2_unit64','native'),('r2_unit64','r2_raw64')]:
                delta=[]
                for s in CFG['seeds']:
                    for c in CFG['conditions']:
                        ka,kb=(ds,s,a,c,head),(ds,s,b,c,head)
                        if ka not in lookup or kb not in lookup:continue
                        x,y=lookup[ka],lookup[kb]
                        row=dict(dataset=ds,head=head,a=a,b=b,seed=s,condition=c)
                        for metric in metrics:row[metric]=(x[metric]-y[metric])*(1 if metric=='target_brier' else 100)
                        delta.append(row);paired.append(row)
                        if a=='pca64' and b=='r2_unit64':
                            assert x['initial_hash']==y['initial_hash'];init_pairs+=1
                if len(delta)!=8:continue
                row=dict(dataset=ds,head=head,a=a,b=b)
                for metric in metrics:
                    vals=[st.mean(x[metric] for x in delta if x['seed']==s and x['condition']!='clean') for s in CFG['seeds']]
                    clean=[next(x[metric] for x in delta if x['seed']==s and x['condition']=='clean') for s in CFG['seeds']]
                    row[metric]=dict(seeds=vals,mean=st.mean(vals),sd=st.stdev(vals))
                    row['clean_'+metric]=dict(seeds=clean,mean=st.mean(clean),sd=st.stdev(clean))
                comparisons.append(row)
                val=row['validation_accuracy'];clean=row['clean_validation_accuracy']['mean']
                if a=='pca64' and b=='r2_unit64':
                    gates.append(dict(dataset=ds,head=head,gate='PCA_above_R2',passed=val['mean']>=.5 and all(v>0 for v in val['seeds']) and clean>=-.5,validation_noisy=val,validation_clean_delta_pp=clean))
                if a=='pca64' and b=='native':
                    gates.append(dict(dataset=ds,head=head,gate='PCA_near_native',passed=val['mean']>=-.5 and all(v>=-.5 for v in val['seeds']) and clean>=-.5,validation_noisy=val,validation_clean_delta_pp=clean))
    for name,rows in [('summary',summary),('per_condition',bycondition),('comparisons',comparisons),('paired_differences',paired),('gates',gates)]:save(out/(name+'.json'),rows)
    audit=dict(new_heads=len(lock['completed']),native_aliases=32,raw_r2_aliases=32,total_scored=len(scores),failures=lock['failures'],source_hashes_verified=len(receipt['signature']),feature_signature_verified=True,all_targets_identical_to_original=True,all_selections_first_validation_max=True,matching_64dim_initialization_pairs=init_pairs,max_new_cpu_replay_error=max((r['cpu_replay_max_error'] for r in scores if r['new']),default=None))
    save(out/'audit.json',audit)
    lines=['# B批：GraphECL统一读出结果','',f"新拟合{len(lock['completed'])}/64；原生引用32；旧R2引用32；失败{len(lock['failures'])}。",'',
           '每个seed先平均三个噪声条件，再报告两个seed均值和样本SD。所有路径使用相同原始GraphECL粒球目标。',
           '旧raw R2还具有不同分类头初始化；其差异记作特征归一化与统一读出合并效果。PCA同时含中心化、截断与行归一化。','',
           '|图|读出特征|监督|噪声Accuracy%|SD|验证%|clean Accuracy%|噪声MacroF1%|','|---|---|---|---:|---:|---:|---:|---:|']
    for r in summary:lines.append(f"|{r['dataset']}|{r['representation']}|{r['head']}|{r['test_accuracy']['mean']:.3f}|{r['test_accuracy']['sd']:.3f}|{r['validation_accuracy']['mean']:.3f}|{r['clean_test_accuracy']['mean']:.3f}|{r['test_macro_f1']['mean']:.3f}|")
    lines+=['','## 验证门','',json.dumps(gates,ensure_ascii=False,indent=2),'','## 审计','',json.dumps(audit,ensure_ascii=False,indent=2)]
    (out/'REPORT_zh.md').write_text('\n'.join(lines),encoding='utf-8')
    save(ROOT/'formal/final.json',dict(status='complete' if len(lock['completed'])==64 and not lock['failures'] else 'complete_with_failures',scored=len(scores),audit_sha256=sha(out/'audit.json'),gates=gates))
    save(ROOT/'formal/status.json',dict(status='complete',completed=len(lock['completed']),total=64,failures=lock['failures']))
    print(json.dumps(dict(audit=audit,gates=gates)),flush=True)

if __name__=='__main__':analyze()
