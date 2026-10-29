"""Post-lock scoring, independent CPU checkpoint replay and paired statistics."""
import csv
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from artifacts import ROOT, PROTOCOL as P, read, save, sha, reusable
from data_bridge import analysis_labels, final_scores, load_features, load_supervised
from downstream import _CORE, build_support32_targets

def analyze():
    lock=ROOT/'formal/fits_locked.json'
    if not lock.exists():raise RuntimeError('All fits must be frozen before test scoring')
    if read(lock)['receipt_sha256']!=sha(ROOT/'receipt.json'):raise RuntimeError('Locked receipt changed')
    rows=[];audits=[];diagnostics=[]
    paths=list((ROOT/'downstream').glob('*/*/*/conditions/*/*/result.json'))
    paths+=list((ROOT/'formal/linear').glob('*/*/*/*/*/result.json'))
    for path in paths:
        r=read(path)
        if r['status']!='complete':continue
        folder=path.parent;name,seed,arm=r['dataset'],r['seed'],r['arm']
        condition,head=r['condition'],r['head'];native='linear' in path.parts
        if native:
            marker=read(folder/'complete.json');reusable(folder,marker['expected'])
            if arm=='graphecl' and condition=='clean' and head=='CE':
                alias=read(folder/'native_alias.json')
                if alias['config_sha256']!=sha(folder/'config.json') or alias['h_sha256']!=r['h_sha256']:
                    raise RuntimeError('Native alias signature mismatch')
        else:
            for filename,field in [('config.json','config.json_sha256'),('epochs.jsonl','epochs.jsonl_sha256'),
                                   ('student.pt','student_sha256'),('probabilities.pt','probabilities_sha256'),('target.pt','target_file_sha256')]:
                if sha(folder/filename)!=r[field]:raise RuntimeError('Downstream artifact signature changed')
        probs=torch.load(folder/'probabilities.pt',map_location='cpu',weights_only=True)
        target=torch.load(folder/'target.pt',map_location='cpu',weights_only=True)
        labels=analysis_labels(name,seed);data=load_supervised(name,seed,condition)
        state=torch.load(folder/'student.pt',map_location='cpu',weights_only=True)
        if native:
            sslarm='grace' if arm=='grace200' else arm
            epoch=200 if arm=='grace200' else (P['graphecl']['dataset_configs'][name]['epochs'] if arm=='graphecl' else 1000)
            hp=ROOT/'formal/ssl'/name/f'seed{seed}'/sslarm/f'H_epoch{epoch}.pt'
            h=F.normalize(torch.load(hp,map_location='cpu',weights_only=True)['H'].float(),dim=1)
            model=torch.nn.Linear(h.shape[1],data['classes']);model.load_state_dict(state);model.eval()
            with torch.no_grad():replay=model(h).softmax(1)
        else:
            x=load_features(name,seed)['x']
            model=_CORE.MLP(x.shape[1],64,data['classes'],2,.5);model.load_state_dict(state);model.eval()
            with torch.no_grad():replay=model(x).softmax(1)
        difference=float((replay-probs).abs().max())
        if difference>2e-5:raise RuntimeError(f'CPU replay mismatch {folder}: {difference}')
        logs=[json.loads(line) for line in (folder/'epochs.jsonl').read_text().splitlines()]
        best=max(logs,key=lambda v:v['validation_correct'])
        if best['epoch']!=r['best_epoch']:raise RuntimeError('Checkpoint selection mismatch')
        validation=int((probs[data['valid']].argmax(1)==data['valid_y']).sum())
        if validation!=r['valid_correct']:raise RuntimeError('Validation replay mismatch')
        if head=='CE':expected=F.one_hot(data['observed'],data['classes']).float()
        else:
            assignment_path=(folder.parents[1]/'support32.pt') if native else folder.parents[2]/'r2/support32.pt'
            assignment=torch.load(assignment_path,map_location='cpu',weights_only=True)
            expected=build_support32_targets(data['observed'],assignment[data['train']],classes=data['classes'])
            _, inv, sizes=torch.unique(assignment,return_inverse=True,return_counts=True)
            classes=data['classes'];train=data['train'];obs=data['observed']
            counts=torch.zeros(len(sizes),classes);counts.index_add_(0,inv[train],F.one_hot(obs,classes).float())
            local=counts[inv[train]]-F.one_hot(obs,classes);votes=local.sum(1)
            ties=(local==local.max(1).values[:,None]).sum(1)>1
            _,alpha=_CORE.make_targets(obs,assignment[train],classes)
            true_counts=torch.zeros(len(sizes),classes)
            true_counts.index_add_(0,inv,F.one_hot(labels['y'],classes).float())
            pairs=(sizes*(sizes-1)).sum()
            same_pairs=(true_counts*(true_counts-1)).sum()
            diagnostics.append(dict(dataset=name,seed=seed,arm=arm,path='linear' if native else 'R2',condition=condition,
                 balls=len(sizes),size_min=int(sizes.min()),size_mean=float(sizes.float().mean()),size_max=int(sizes.max()),
                 loo_vote_coverage=float((votes>0).float().mean()),low_vote_rate=float((votes<2).float().mean()),
                 tie_rate=float(ties.float().mean()),zero_alpha_fallback_rate=float((alpha==0).float().mean()),
                 true_same_class_pair_ratio=float(same_pairs/pairs) if pairs>0 else None))
        if not torch.equal(target,expected):raise RuntimeError('Target replay mismatch')
        truth=F.one_hot(labels['y'][labels['train']],data['classes']).float()
        brier=float((target-truth).square().sum(1).mean())
        score=final_scores(name,seed,probs)
        path_name='linear' if native else 'R2'
        rows.append(dict(dataset=name,seed=seed,arm=arm,path=path_name,condition=condition,head=head,
                         target_brier=brier,**score,source=str(folder)))
        audits.append(dict(source=str(folder),cpu_replay_max_error=difference,selection=True,target=True,
                           probabilities_sha256=sha(folder/'probabilities.pt')))
    out=ROOT/'analysis';out.mkdir(parents=True,exist_ok=True)
    save(out/'scores.json',rows);save(out/'audit.json',audits)
    save(out/'ball_diagnostics.json',diagnostics)
    expected_ids=set()
    for name in P['datasets']:
        for seed in P['seeds']:
            arms=['grace','gca_degree','progcl_weight_degree','graphecl']+(['grace200'] if name in ('cora','citeseer') else [])
            for arm in arms:
                for condition in P['conditions']:
                    for head in ['CE','SUPPORT32']:
                        expected_ids.add((name,seed,arm,'R2',condition,head))
                        if arm=='graphecl':expected_ids.add((name,seed,arm,'linear',condition,head))
                expected_ids.add((name,seed,arm,'linear','clean','CE'))
    found={(r['dataset'],r['seed'],r['arm'],r['path'],r['condition'],r['head']) for r in rows}
    missing=sorted(expected_ids-found);unexpected=sorted(found-expected_ids)
    if len(found)!=len(rows) or unexpected:raise RuntimeError('Duplicate or unexpected result identities')
    save(out/'coverage.json',dict(expected=len(expected_ids),found=len(found),missing=missing,task_failures=read(lock)['failures']))
    increments=[]
    lookup={(r['dataset'],r['seed'],r['arm'],r['path'],r['condition'],r['head']):r for r in rows}
    for key,r in lookup.items():
        if key[-1]=='SUPPORT32' and (*key[:-1],'CE') in lookup:
            ce=lookup[(*key[:-1],'CE')]
            increments.append(dict(dataset=r['dataset'],seed=r['seed'],arm=r['arm'],path=r['path'],condition=r['condition'],
                  accuracy_gain_pp=100*(r['test_accuracy']-ce['test_accuracy']),macro_f1_gain_pp=100*(r['test_macro_f1']-ce['test_macro_f1'])))
    save(out/'support_minus_ce.json',increments)
    with (out/'scores.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]) if rows else ['dataset']);writer.writeheader();writer.writerows(rows)
    summary=[];paired=[]
    def values(arm,path,head,dataset=None,clean=False):
        result=[]
        for seed in P['seeds']:
            chosen=[r for r in rows if r['arm']==arm and r['path']==path and r['head']==head and r['seed']==seed
                    and (dataset is None or r['dataset']==dataset) and (r['condition']=='clean')==clean]
            wanted=(1 if dataset else len(P['datasets']))*(1 if clean else 3)
            result.append(float(np.mean([r['test_accuracy']*100 for r in chosen])) if len(chosen)==wanted else None)
        return result
    for arm in ['grace','gca_degree','progcl_weight_degree','graphecl']:
        for path_name in ['R2','linear']:
            for head in ['CE','SUPPORT32']:
                vals=values(arm,path_name,head)
                if None not in vals:
                    summary.append(dict(arm=arm,path=path_name,head=head,seed_values=vals,mean=float(np.mean(vals)),sd=float(np.std(vals,ddof=1))))
    for arm in ['gca_degree','progcl_weight_degree','graphecl']:
        for comparator in ['grace1000']:
            graph_deltas=[];clean_deltas=[];seed_deltas=np.zeros(2);available=True
            for name in P['datasets']:
                ref='grace200' if comparator=='strong_reference' and name in ('cora','citeseer') else 'grace'
                a,b=values(arm,'R2','SUPPORT32',name),values(ref,'R2','SUPPORT32',name)
                ac,bc=values(arm,'R2','SUPPORT32',name,True),values(ref,'R2','SUPPORT32',name,True)
                if any(v is None for v in a+b+ac+bc):available=False;break
                delta=np.array(a)-b;seed_deltas+=delta/len(P['datasets'])
                graph_deltas.append(float(delta.mean()));clean_deltas.append(float((np.array(ac)-bc).mean()))
            if available:
                passed=seed_deltas.mean()>=.5 and sum(v>0 for v in graph_deltas)>=2 and all(seed_deltas>0) and all(v>=-.5 for v in clean_deltas)
                paired.append(dict(arm=arm,comparator=comparator,seed_deltas_pp=seed_deltas.tolist(),mean_pp=float(seed_deltas.mean()),
                    sd_pp=float(seed_deltas.std(ddof=1)),dataset_deltas_pp=graph_deltas,clean_deltas_pp=clean_deltas,passed=bool(passed)))
    save(out/'summary.json',summary);save(out/'paired.json',paired)
    decisions={arm:('advance' if len([r for r in paired if r['arm']==arm])==1 and all(r['passed'] for r in paired if r['arm']==arm)
                   else ('error' if any(cell[2]==arm for cell in missing) else 'defer')) for arm in ['gca_degree','progcl_weight_degree','graphecl']}
    save(out/'decision.json',decisions)
    lines=['# 冻结 SSL 空间对照结果','',f'完成评分与 CPU 复核：{len(rows)}/{len(expected_ids)} 个预测单元，缺格 {len(missing)}。完整配对项目报告两 seed 样本标准差（ddof=1）。',
           '','|方法|路径|头|带噪均值 %|seed 标准差|','|---|---|---|---:|---:|']
    for r in summary:lines.append(f"|{r['arm']}|{r['path']}|{r['head']}|{r['mean']:.3f}|{r['sd']:.3f}|")
    lines+=['','## 配对门槛',json.dumps(paired,ensure_ascii=False,indent=2),'','## 决策',json.dumps(decisions,ensure_ascii=False,indent=2),
            '','## 审计与追溯','scores.csv 保留各图、seed、噪声、clean、Macro-F1 与目标 Brier；audit.json 保存 CPU 重载误差、首次最佳选择与目标复核。',
            'formal/fits_locked.json 保留全部成功/失败任务。结果来自固定受控配置；GraphECL 原生宽度及日程另有登记。']
    (out/'REPORT_zh.md').write_text('\n'.join(lines),encoding='utf-8')
    save(ROOT/'formal/final.json',dict(status='complete' if not missing else 'complete_with_failures',scored=len(rows),missing=missing,decisions=decisions,audit_sha256=sha(out/'audit.json')))

if __name__=='__main__':analyze()
