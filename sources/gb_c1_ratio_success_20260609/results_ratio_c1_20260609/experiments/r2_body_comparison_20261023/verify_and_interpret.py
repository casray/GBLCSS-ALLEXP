"""Reconstruct deployed models and write matched-seed interpretation."""
import csv
from pathlib import Path
import numpy as np
import torch
import run as r
from analyze import csvfile

def main():
    assert (r.ROOT/'complete.json').exists()
    torch.set_num_threads(4)
    records=[];max_error=0.;bridges=[];protocol_checks=[]
    for name in r.CFG['datasets']:
        raw=r.f.r2.raw_data(name);x=raw.x.float().cuda();classes=int(raw.y.max())+1
        h,_,_=r.f.r2.load_h(name,raw)
        _,seed0geo=r.fixed_data(name,0)
        historical=r.f.read(r.EXP/'new_graph_mlp_seed0_20261009/runs'/name/'gssc/result.json')
        split_digest=r.f.r2.fingerprint(*[seed0geo[k] for k in ['train','valid','test']])
        assert historical['split_hash']==split_digest
        base,_=r.fixed_data(name,0);pack=r.load(base/'uniform0.4.pt')
        clean=raw.y.long().cpu();observed=clean.clone();observed[seed0geo['train']]=pack['observed']
        assert historical['noise_hash']==r.f.r2.fingerprint(clean,observed)
        protocol_checks.append(dict(dataset=name,reused_gssc_split_exact=True,reused_gssc_noise_exact=True))
        for seed in r.CFG['seeds']:
            enc=r.load(r.f.historical_encoder(name,seed));m=r.model(x,classes,h.shape[1],seed,enc)
            encoder_hash=r.state_hash(m.encoder_state())
            _,geo=r.fixed_data(name,seed)
            with torch.no_grad():hidden=m.encode(x).detach()
            initial=r.state_hash(m.deployment_state())
            for cond in r.CFG['conditions']:
                for arm in r.CFG['arms']:
                    folder=r.ROOT/'fits'/name/f'seed{seed}'/cond/arm;meta=r.f.read(folder/'result.json')
                    assert meta['initial_hash']==initial and meta['encoder_hash']==encoder_hash
                    assert r.sha(folder/'student.pt')==meta['checkpoint_sha256']
                    assert r.state_hash({'target':r.load(folder/'target.pt')})==meta['target_hash']
                    state=r.load(folder/'student.pt');m.backbone.load_state_dict(state)
                    assert r.state_hash(m.encoder_state())==encoder_hash
                    with torch.no_grad():prob=m.backbone.layers[-1](hidden).softmax(1).cpu()
                    saved=r.load(folder/'probabilities.pt');err=float((saved-prob).abs().max());max_error=max(max_error,err)
                    assert err<1e-6
                    assert torch.equal(saved.argmax(1),prob.argmax(1))
                    assert torch.isfinite(saved).all() and torch.allclose(saved.sum(1),torch.ones(len(saved)),atol=1e-5)
                    records.append(dict(dataset=name,seed=seed,condition=cond,arm=arm,max_error=err))
            if seed==0:
                old=r.EXP/'sc_support_channel_pilot_20261023/fits/E4'/name/'seed0/uniform0.4/C_head/probabilities.pt'
                current=r.ROOT/'fits'/name/'seed0/uniform0.4/SUPPORT/probabilities.pt'
                if old.exists():bridges.append(dict(dataset=name,max_probability_error=float((r.load(old)-r.load(current)).abs().max())))
        del x,hidden,m;torch.cuda.empty_cache()
    csvfile(r.ROOT/'checkpoint_replay.csv',records)
    extras=r.f.read(r.ROOT/'additional_sources.json')['sha256']
    for path,digest in extras.items():assert r.sha(path)==digest,path
    with (r.ROOT/'results.csv').open(encoding='utf-8-sig') as s:rows=list(csv.DictReader(s))
    for row in rows:
        for k in ['seed']:row[k]=int(row[k])
        for k in ['accuracy','macro_f1','validation']:row[k]=float(row[k])
    all_arms=r.CFG['arms']+['SIMMLP','GSSC']
    matched=[]
    for name in r.CFG['datasets']:
        for arm in all_arms:
            z=[p for p in rows if p['dataset']==name and p['seed']==0 and p['condition']!='clean' and p['arm']==arm]
            matched.append(dict(dataset=name,arm=arm,seed=0,accuracy_pct=100*np.mean([p['accuracy'] for p in z]),macro_f1_pct=100*np.mean([p['macro_f1'] for p in z])))
    csvfile(r.ROOT/'matched_seed0.csv',matched)
    costs=[]
    for name in r.CFG['datasets']:
        for cond in r.CFG['conditions']:
            for method in ['SIMMLP','GSSC']:
                meta=r.f.read(r.ROOT/'external'/name/cond/method/'result.json')
                reused=method=='GSSC' and cond=='uniform0.4'
                oldroot=r.EXP/('new_graph_mlp_seed0_20261009' if method=='GSSC' else 'simmlp_tined_audit_seed0_20261009')
                previous=r.f.read(oldroot/'runs'/name/('gssc' if method=='GSSC' else 'simmlp')/'result.json')
                costs.append(dict(dataset=name,condition=cond,method=method,result_reused=reused,
                    fresh_fit_seconds=None if reused else meta['seconds'],
                    old_complete_pipeline_seconds=previous.get('wall_seconds'),
                    encoder_pretraining_reused=method=='SIMMLP',
                    scope='probe only; reused encoder pretraining cost separate' if method=='SIMMLP' else 'all author-core phases'))
    csvfile(r.ROOT/'cost_scope.csv',costs)
    # Aggregate each seed across all noisy cells before computing seed SD.
    macro=[]
    for arm in r.CFG['arms']:
        av=[100*np.mean([p['accuracy'] for p in rows if p['arm']==arm and p['condition']!='clean' and p['seed']==seed]) for seed in r.CFG['seeds']]
        macro.append(dict(arm=arm,mean_pct=float(np.mean(av)),seed_sd_pp=float(np.std(av,ddof=1)),seed_means=av))
    r.save(r.ROOT/'macro_seed_summary.json',macro)
    idx={(p['dataset'],p['seed'],p['condition'],p['arm']):p for p in rows}
    gaps={};clean={};seed_gaps=[]
    controls=['GCE','POSTEL','SALS','ALS','KM']
    # Validation selects one fixed competitor per dataset; test only scores that choice.
    selected={n:max(controls,key=lambda a:np.mean([p['validation'] for p in rows if p['dataset']==n and p['arm']==a and p['condition']!='clean'])) for n in r.CFG['datasets']}
    for name in r.CFG['datasets']:
        gaps[name]=100*np.mean([idx[name,s,c,'SUPPORT']['accuracy']-idx[name,s,c,selected[name]]['accuracy'] for s in r.CFG['seeds'] for c in r.CFG['conditions'] if c!='clean'])
        clean[name]=100*np.mean([idx[name,s,'clean','SUPPORT']['accuracy']-idx[name,s,'clean','CE']['accuracy'] for s in r.CFG['seeds']])
    for s in r.CFG['seeds']:seed_gaps.append(100*np.mean([idx[n,s,c,'SUPPORT']['accuracy']-idx[n,s,c,selected[n]]['accuracy'] for n in r.CFG['datasets'] for c in r.CFG['conditions'] if c!='clean']))
    passed=np.mean(list(gaps.values()))>=.5 and sum(v>0 for v in gaps.values())>=3 and min(seed_gaps)>0 and min(clean.values())>=-1
    decision=dict(passed=bool(passed),comparison='SUPPORT versus validation-selected fixed competitor per dataset',selected=selected,
                  test_gain_pp=gaps,clean_vs_ce_pp=clean,seed_macro_gain_pp=seed_gaps,decision='continue confirmation' if passed else 'retain scope evidence; stop expanding this supervision variant')
    r.save(r.ROOT/'decision.json',decision)
    lines=['','## 相同seed0的整体方法比较','', '| 数据集 | R2 CE | R2 GCE | R2 SUPPORT | R2 KM | SimMLP | GSSC |','|---|'+'---:|'*6]
    mi={(p['dataset'],p['arm']):p for p in matched}
    for name in r.CFG['datasets']:lines.append('| '+name+' | '+' | '.join(f"{mi[name,a]['accuracy_pct']:.3f}" for a in ['CE','GCE','SUPPORT','KM','SIMMLP','GSSC'])+' |')
    lines+=['','## R2 三seed的五图带噪宏均值','', '| 方法 | 准确率均值 % | seed SD pp |','|---|---:|---:|']
    for p in macro:lines.append(f"| {p['arm']} | {p['mean_pct']:.3f} | {p['seed_sd_pp']:.3f} |")
    lines+=['','## 预注册实用门槛', '',json_format(decision),'',
        '验证集为每图选择一个固定竞争方法，测试集评分保持全部预定结果；该选择规则只用于汇总对照，不改变任何训练或超参数。',
        '门槛作为本轮实用筛查，3seed与固定SSL缓存对应的独立性范围保留在解释中。',
        '', '## 审计', '',f'480个部署checkpoint重载，最大概率误差 {max_error:.3g}；编码器/BN及各臂初始分类头核对通过。',
        f'新增借用源与历史分区哈希 {len(extras)} 份通过。历史SUPPORT桥接误差详见delivery_audit.json。']
    with (r.ROOT/'REPORT_zh.md').open('a',encoding='utf-8') as s:s.write('\n'.join(lines)+'\n')
    recovery=r.f.read(r.ROOT/'recovery_sources.json') if (r.ROOT/'recovery_sources.json').exists() else {}
    for path,digest in recovery.items():assert r.sha(path)==digest
    r.save(r.ROOT/'delivery_audit.json',dict(passed=True,checkpoints=480,max_replay_probability_error=max_error,additional_hashes=len(extras),bridges=bridges,protocol_checks=protocol_checks,recovery_sources_verified=len(recovery)))
    print(json_format(decision))
def json_format(x):
    import json
    return json.dumps(x,ensure_ascii=False,indent=2)
if __name__=='__main__':main()
