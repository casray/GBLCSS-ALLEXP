"""Counterfactual readouts and pre-registered, grouped cache diagnostics."""
from common import *
from mechanisms import *
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits


def grouped_probe(base,extra,y,brier,groups):
    k=min(CFG['screen_group_folds'],len(np.unique(groups)))
    if k<2 or len(np.unique(y))<2:raise RuntimeError('Insufficient diagnostic groups/classes')
    score0=np.zeros(len(y));score1=np.zeros(len(y));reg0=np.zeros(len(y));reg1=np.zeros(len(y))
    for train,test in GroupKFold(k).split(base,y,groups):
        assert not set(groups[train])&set(groups[test])
        for x,prob,reg in [(base,score0,reg0),(np.c_[base,extra],score1,reg1)]:
            if len(np.unique(y[train]))==1:prob[test]=y[train].mean()
            else:
                model=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=1000))
                model.fit(x[train],y[train]);prob[test]=model.predict_proba(x[test])[:,1]
            model=make_pipeline(StandardScaler(),Ridge(alpha=1));model.fit(x[train],brier[train]);reg[test]=model.predict(x[test])
    auc0=float(roc_auc_score(y,score0));auc1=float(roc_auc_score(y,score1))
    mse0=float(np.mean((reg0-brier)**2));mse1=float(np.mean((reg1-brier)**2))
    return dict(base_auc=auc0,augmented_auc=auc1,delta_auc=auc1-auc0,base_brier_mse=mse0,augmented_brier_mse=mse1,mse_reduction=mse0-mse1,
                nodes=len(y),groups=len(np.unique(groups)),folds=k,correct_fraction=float(y.mean()))


def cache_features(d,s):
    dest=ROOT/'stage1/cache'/d/f'seed{s}'
    if (dest/'features.pt').exists():return load(dest/'features.pt')
    status('CACHE_RELATIONS',dataset=d,seed=s)
    data=old.supervised(d,s,'uniform0.4');ids=data['train'].numpy();a=load(source_assignment(d,s)).numpy()
    h=load(source_ssl(d,s)/'H.pt');cfg,x,edge,adj=model_data(d,s);model=original_model(d,s,cfg,x);model.eval()
    with torch.no_grad():g,_=model(x,adj);g=F.normalize(g,dim=1).cpu().numpy()
    del model,x,edge,adj;gc.collect();torch.cuda.empty_cache()
    hn=h.numpy();k=len(np.unique(a));started=time.perf_counter()
    with threadpool_limits(limits=1):
        ka=KMeans(k,n_init=CFG['kmeans_n_init'],random_state=s+29002,max_iter=300,algorithm='lloyd').fit_predict(hn)
        gb=relation_features(g,hn,a,ids,CFG['temperature']);km=relation_features(g,hn,ka,ids,CFG['temperature'])
    trh=normalized(hn[ids]);groups=a[ids];sums=np.zeros((a.max()+1,trh.shape[1]));np.add.at(sums,groups,trh)
    count=np.bincount(groups,minlength=a.max()+1)[groups]-1
    mean_cos=(trh*(sums[groups]-trh)).sum(1)/np.maximum(count,1)
    out=dict(gb=torch.from_numpy(gb),km=torch.from_numpy(km),assignment=torch.from_numpy(a),km_assignment=torch.from_numpy(ka),
             base_geometry=torch.from_numpy(np.c_[mean_cos,np.log1p(count)]))
    save_tensor(dest/'features.pt',out);save(dest/'meta.json',dict(balls=k,kmeans_seconds_including_relations=time.perf_counter()-started,
        source_model_sha=sha(source_ssl(d,s)/'model.pt'),h_sha=sha(source_ssl(d,s)/'H.pt'),gb_sizes=np.bincount(a).tolist(),km_sizes=np.bincount(ka).tolist()))
    return out


def fit_counterfactuals():
    audits=[]
    for d in CFG['datasets']:
        for s in CFG['seeds']:
            h=load(source_ssl(d,s)/'H.pt')
            for c in CFG['diagnostic_conditions']:
                data=old.supervised(d,s,c)
                for arm,source in [('replay',source_head(d,s,c)),('counterfactual',source_head(d,s,c,'gb100','own'))]:
                    dest=ROOT/'stage1/heads'/d/f'seed{s}'/c/arm
                    status('COUNTERFACTUAL_HEAD',dataset=d,seed=s,condition=c,arm=arm)
                    fit_head(dest,h,load(source/'target.pt'),data,d,s,c,arm)
                    audit=verify_head(dest,h)
                    if arm=='replay':
                        same=torch.equal(load(dest/'probabilities.pt'),load(source/'probabilities.pt'))
                        if not same:raise RuntimeError('Original head replay mismatch')
                        audit['original_probability_bitwise']=True
                    audits.append(audit)
    save(ROOT/'stage1/fits_locked.json',dict(count=24,audits=audits))


def diagnose():
    if not (ROOT/'stage1/fits_locked.json').exists():raise RuntimeError('Counterfactual fitting lock missing')
    changes=[];probes=[];scores=[];classes_rows=[]
    for d in CFG['datasets']:
        for s in CFG['seeds']:
            feats=cache_features(d,s);lab=old.gvec.analysis_labels(d,s)
            for c in CFG['diagnostic_conditions']:
                status('CACHE_DIAGNOSTIC',dataset=d,seed=s,condition=c)
                data=old.supervised(d,s,c);ids=data['train'].numpy();y=lab['y'][ids].numpy();z=data['observed'].numpy();classes=data['classes']
                a0=load(source_assignment(d,s)).numpy()[ids];a1=load(source_assignment(d,s,'gb100')).numpy()[ids]
                v0=vote_details(z,a0,classes);v1=vote_details(z,a1,classes)
                t0=load(source_head(d,s,c)/'target.pt').numpy();t1=load(source_head(d,s,c,'gb100','own')/'target.pt').numpy()
                err=max(float(np.abs(v0['target']-t0).max()),float(np.abs(v1['target']-t1).max()))
                if err>2e-7:raise RuntimeError('Independent target mismatch')
                change=support_changes(a0,a1);b0=quality(t0,y);b1=quality(t1,y);delta=b1-b0
                margin0,purity0=true_margins(y,a0,classes);margin1,purity1=true_margins(y,a1,classes)
                r0=vote_details(y,a0,classes)['counts'];r1=vote_details(y,a1,classes)['counts']
                same0=r0[np.arange(len(y)),y];same1=r1[np.arange(len(y)),y]
                corr=spearmanr(1-change['jaccard'],delta).statistic
                changes.append(dict(dataset=d,seed=s,condition=c,changed_fraction=float(change['changed'].mean()),jaccard=float(change['jaccard'].mean()),
                    old_support=float(change['old_support'].mean()),new_support=float(change['new_support'].mean()),old_brier=float(b0.mean()),new_brier=float(b1.mean()),
                    delta_brier=float(delta.mean()),brier_worse_fraction=float((delta>1e-8).mean()),lost_same_class_support_fraction=float((same1<same0).mean()),
                    old_true_margin=float(margin0.mean()),new_true_margin=float(margin1.mean()),churn_brier_spearman=float(corr) if np.isfinite(corr) else None,target_rebuild_max_error=err))
                pred0=load(source_head(d,s,c)/'probabilities.pt').argmax(1).numpy()[ids]
                predcf=load(ROOT/'stage1/heads'/d/f'seed{s}'/c/'counterfactual/probabilities.pt').argmax(1).numpy()[ids]
                rows=[dict(node=int(ids[i]),true_class=int(y[i]),old_group=int(a0[i]),new_group=int(a1[i]),old_support=int(change['old_support'][i]),new_support=int(change['new_support'][i]),
                    jaccard=float(change['jaccard'][i]),same_class_votes_old=float(same0[i]),same_class_votes_new=float(same1[i]),margin_old=float(margin0[i]),margin_new=float(margin1[i]),
                    brier_old=float(b0[i]),brier_new=float(b1[i]),delta_brier=float(delta[i]),prediction_changed=bool(pred0[i]!=predcf[i])) for i in range(len(ids))]
                csv_write(ROOT/'stage1/nodes'/d/f'seed{s}'/(c+'.csv'),rows)
                for cls in range(classes):
                    mask=y==cls
                    if mask.any():classes_rows.append(dict(dataset=d,seed=s,condition=c,cls=cls,nodes=int(mask.sum()),old_same_support=float(same0[mask].mean()),new_same_support=float(same1[mask].mean()),delta_brier=float(delta[mask].mean())))
                top=np.sort(v0['vote'],axis=1);base=np.c_[feats['base_geometry'].numpy(),top[:,-1],top[:,-1]-top[:,-2]]
                correct=(v0['q'].argmax(1)==y).astype(int)
                for method in ['gb','km']:
                    with threadpool_limits(limits=1):probe=grouped_probe(base,feats[method].numpy(),correct,b0,a0)
                    probes.append(dict(dataset=d,seed=s,condition=c,method=method,**probe))
                for arm,folder in [('H0_q0',source_head(d,s,c)),('H1_q0',source_head(d,s,c,'gb100','fixed')),('H1_q1',source_head(d,s,c,'gb100','own')),('H0_q1',ROOT/'stage1/heads'/d/f'seed{s}'/c/'counterfactual')]:
                    scores.append(dict(dataset=d,seed=s,condition=c,arm=arm,**score(folder,lab,data)))
    for name,rows in [('support_changes',changes),('probes',probes),('scores',scores),('class_changes',classes_rows)]:
        save(ROOT/'stage1'/f'{name}.json',rows);csv_write(ROOT/'stage1'/f'{name}.csv',rows)
    gb=[r for r in probes if r['method']=='gb'];km=[r for r in probes if r['method']=='km']
    dg=float(np.mean([r['delta_auc'] for r in gb]));dk=float(np.mean([r['delta_auc'] for r in km]));mse=float(np.mean([r['mse_reduction'] for r in gb]))
    seeds=[float(np.mean([r['delta_auc'] for r in gb if r['seed']==s])) for s in CFG['seeds']]
    gate=dict(gb_delta_auc=dg,km_delta_auc=dk,gb_brier_mse_reduction=mse,seed_delta_auc=seeds,passed=bool(dg>=CFG['screen_auc_gate'] and dg>dk and min(seeds)>0 and mse>0))
    save(ROOT/'stage1/decision.json',gate);status('STAGE1_COMPLETE',**gate)


def main():
    prepare();fit_counterfactuals();diagnose()


if __name__=='__main__':
    try:main()
    except Exception as exc:
        import traceback
        status('STAGE1_FAILED',error=repr(exc));traceback.print_exc();raise
