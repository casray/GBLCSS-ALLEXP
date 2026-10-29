"""Five-graph noisy-only paired analysis after fitting is locked."""
import json,csv,statistics as st
from pathlib import Path
import torch
import torch.nn.functional as F
from run import ROOT,CFG,read,load,save,sha,verify,ref,folder,features_dir,checked_head,down,load_supervised,analysis_labels,state_hash
from contract import noisy_gate

def score(t,path,receipt,new=False):
 checked_head(path,set());r=read(path/'result.json');p=load(path/'probabilities.pt');target=load(path/'target.pt');d,s,c,h=t['dataset'],t['seed'],t['condition'],t['head'];rr=ref(receipt,d,s)
 data=load_supervised(d,s,c);truth=analysis_labels(d,s)
 expected=F.one_hot(data['observed'],data['classes']).float()
 if h=='SUPPORT32':expected=down.build_support32_targets(data['observed'],load(rr['assignment'])[data['train']],classes=data['classes'])
 assert torch.equal(target,expected)
 epochs=[json.loads(x) for x in (path/'epochs.jsonl').read_text().splitlines()];best=max(epochs,key=lambda x:x['validation_correct'])
 assert len(epochs)==200 and best['epoch']==r['best_epoch']
 correct=int((p[data['valid']].argmax(1)==data['valid_y']).sum());assert correct==best['validation_correct']==r['valid_correct']
 assert torch.isfinite(p).all() and (p>=0).all() and (p.sum(1)-1).abs().max()<2e-6
 error=None
 if new:
  fp=features_dir(d,s)/(t['scale']+'.pt');assert r['feature_sha256']==sha(fp) and r['receipt_sha256']==sha(ROOT/'receipt.json')
  initial=load(features_dir(d,s)/'initial.pt');assert state_hash(load(path/'initial.pt'))==state_hash(initial)==r['initial_hash']
  w=load(path/'student.pt')
  with torch.no_grad():replay=F.linear(load(fp),w['weight'],w['bias']).softmax(1)
  error=float((replay-p).abs().max());assert error<=2e-5
 y=truth['y'][truth['test']];pred=p[truth['test']].argmax(1);k=p.shape[1];conf=torch.bincount(y*k+pred,minlength=k*k).reshape(k,k).double()
 return dict(t,new=new,validation_accuracy=correct/len(data['valid']),test_accuracy=int((pred==y).sum())/len(y),test_macro_f1=float((2*conf.diag()/(conf.sum(0)+conf.sum(1)).clamp_min(1)).mean()),target_brier=float((target-F.one_hot(truth['y'][truth['train']],k)).square().sum(1).mean()),best_epoch=r['best_epoch'],cpu_error=error,source=str(path))

def stats(v):return dict(seeds=v,mean=st.mean(v),sd=st.stdev(v))

def analyze():
 receipt=read(ROOT/'receipt.json');lock=read(ROOT/'formal/fits_locked.json');assert lock['receipt_sha256']==sha(ROOT/'receipt.json');verify(receipt['signature']);verify(read(ROOT/'features_locked.json')['signature'])
 scores=[score(t,folder(t),receipt,True) for t in lock['completed']]+[score(x['task'],Path(x['path']),receipt) for x in receipt['aliases']]
 out=ROOT/'analysis';save(out/'scores.json',scores)
 with (out/'scores.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(scores[0]));w.writeheader();w.writerows(scores)
 metrics=['validation_accuracy','test_accuracy','test_macro_f1','target_brier'];datasets=CFG['new_datasets']+CFG['reuse_datasets'];summary=[];paired=[];gates=[];percondition=[];aggregate=[]
 index={(r['dataset'],r['seed'],r['scale'],r['condition'],r['head']):r for r in scores}
 for d in datasets:
  for head in CFG['heads']:
   for scale in ['raw','unit']:
    sub=[r for r in scores if (r['dataset'],r['head'],r['scale'])==(d,head,scale)]
    if len(sub)!=6:continue
    row=dict(dataset=d,head=head,scale=scale)
    for m in metrics:row[m]=stats([st.mean(x[m] for x in sub if x['seed']==s)*(1 if m=='target_brier' else 100) for s in [0,1]])
    summary.append(row)
   diffs=[]
   for s in [0,1]:
    for c in CFG['conditions']:
     a,b=(d,s,'unit',c,head),(d,s,'raw',c,head)
     if a not in index or b not in index:continue
     r=dict(dataset=d,seed=s,condition=c,head=head)
     for m in metrics:r[m]=(index[a][m]-index[b][m])*(1 if m=='target_brier' else 100)
     diffs.append(r);paired.append(r)
   if len(diffs)!=6:continue
   for c in CFG['conditions']:
    row=dict(dataset=d,head=head,condition=c)
    for m in metrics:row[m]=stats([next(x[m] for x in diffs if x['seed']==s and x['condition']==c) for s in [0,1]])
    percondition.append(row)
   vals=[st.mean(x['validation_accuracy'] for x in diffs if x['seed']==s) for s in [0,1]]
   row=dict(dataset=d,head=head,passed=noisy_gate(vals),validation_difference=stats(vals))
   for m in ['test_accuracy','test_macro_f1']:row[m+'_difference']=stats([st.mean(x[m] for x in diffs if x['seed']==s) for s in [0,1]])
   gates.append(row)
 for cohort,ds in [('new_three',CFG['new_datasets']),('existing_two',CFG['reuse_datasets']),('all_five',datasets)]:
  for head in CFG['heads']:
   for scale in ['raw','unit']:
    sub=[x for x in summary if x['dataset'] in ds and x['head']==head and x['scale']==scale]
    if len(sub)!=len(ds):continue
    row=dict(cohort=cohort,head=head,scale=scale)
    for m in metrics:row[m]=stats([st.mean(x[m]['seeds'][s] for x in sub) for s in [0,1]])
    aggregate.append(row)
 for n,v in [('summary',summary),('paired',paired),('per_condition_differences',percondition),('gates',gates),('aggregate',aggregate)]:save(out/(n+'.json'),v)
 audit=dict(new_heads=len(lock['completed']),aliases=len(receipt['aliases']),scored=len(scores),source_hashes=len(receipt['signature']),failures=lock['failures'],targets_verified=True,selection_verified=True,max_cpu_error=max((x['cpu_error'] for x in scores if x['new']),default=None))
 save(out/'audit.json',audit)
 lines=['# 五图带噪单位化扩展','',f"新增{len(lock['completed'])}/36；引用84；总评分{len(scores)}；失败{len(lock['failures'])}。旧初始化、同一目标与200轮预算；clean无否决门。",'', '|图|监督|尺度|带噪Accuracy ± SD %|验证%|Macro-F1%|','|---|---|---|---:|---:|---:|']
 for x in summary:lines.append(f"|{x['dataset']}|{x['head']}|{x['scale']}|{x['test_accuracy']['mean']:.3f} ± {x['test_accuracy']['sd']:.3f}|{x['validation_accuracy']['mean']:.3f}|{x['test_macro_f1']['mean']:.3f}|")
 lines+=['','## 配对结果','', '|图|监督|测试差pp|两seed测试差|验证差pp|验证门|','|---|---|---:|---|---:|---|']
 for x in gates:lines.append(f"|{x['dataset']}|{x['head']}|{x['test_accuracy_difference']['mean']:+.3f}|{x['test_accuracy_difference']['seeds']}|{x['validation_difference']['mean']:+.3f}|{x['passed']}|")
 (out/'REPORT_zh.md').write_text('\n'.join(lines),encoding='utf-8')
 save(ROOT/'formal/final.json',dict(status='complete' if len(lock['completed'])==36 and not lock['failures'] else 'complete_with_failures',scored=len(scores),audit_sha256=sha(out/'audit.json')))
 save(ROOT/'formal/status.json',dict(status='complete',completed=len(lock['completed']),total=36,failures=lock['failures']));print(json.dumps(audit),flush=True)

if __name__=='__main__':analyze()
