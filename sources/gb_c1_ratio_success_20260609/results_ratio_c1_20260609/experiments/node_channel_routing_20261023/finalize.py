"""Independent prediction, fold isolation, selection and hash audit after UTF-8 fix."""
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import run as runner
from router import CONFIG, student_features, predict_model, apply_route
from routing_kernels import choose_threshold, choose_weight
from metrics import accuracy_f1

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent/CONFIG['source']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def check_model(model, features, a, c, y):
    use = (a==y) ^ (c==y)
    x, target = features[use], (c[use]==y[use]).astype(int)
    assert model['n']==len(target) and model['positives']==int(target.sum())
    if len(np.unique(target)) < 2:
        assert model['kind']=='constant' and model['value']==(int(target[0]) if len(target) else 0)
        return 1
    scaler = StandardScaler().fit(x)
    fit = LogisticRegression(C=1,solver='lbfgs',max_iter=2000).fit(scaler.transform(x),target)
    np.testing.assert_allclose(scaler.mean_,model['mean'],rtol=0,atol=1e-12)
    np.testing.assert_allclose(scaler.scale_,model['scale'],rtol=0,atol=1e-12)
    np.testing.assert_allclose(fit.coef_[0],model['coef'],rtol=0,atol=1e-10)
    np.testing.assert_allclose(fit.predict_proba(scaler.transform(features))[:,1],
                               predict_model(model,features),rtol=0,atol=1e-12)
    return 0


def main():
    sources, inputs = read(ROOT/'sources.json'), read(ROOT/'inputs.json')
    for path, expected in sources.items():
        if Path(path).name=='run.py':
            fixed = Path(path).read_text(encoding='utf-8')
            old = fixed.replace("(ROOT/'sources.json').read_text(encoding='utf-8')", "(ROOT/'sources.json').read_text()")
            assert hashlib.sha256(old.encode('utf-8')).hexdigest()==expected
        else:
            assert runner.sha(path)==expected, path
    for path, expected in inputs.items(): assert runner.sha(path)==expected, path
    source_change = dict(original=sources[str(ROOT/'run.py')],updated=runner.sha(ROOT/'run.py'),
                         change='Explicit UTF-8 in final source-manifest reader only')
    runner.save(ROOT/'audit_amendment.json',source_change)
    old_selections = {(r['dataset'],r['condition']):r for r in read(ROOT.parent/'channel_routing_variance_20261023/L1_selections.json')}
    count = fallbacks = 0
    with threadpool_limits(limits=2):
        for s in read(ROOT/'selections.json'):
            name, cond = s['dataset'],s['condition']; folder=ROOT/'cells'/name/cond
            base=OLD/'fits/E4'/name/'seed0'/cond
            a=runner.probability(base/'A300/probabilities.pt'); c=runner.probability(base/'C_head/probabilities.pt')
            cache=np.load(folder/'routing.npz'); vi=cache['valid']; y=cache['valid_y']
            x=student_features(a,c); xv=x[vi]; av=a[vi].argmax(1); cv=c[vi].argmax(1)
            assert s['global_weight']==old_selections[name,cond]['weight']
            hits=np.zeros(len(vi),int)
            for k, fold in enumerate(s['fold_models']):
                tr=np.array(fold['train']); ho=np.array(fold['hold'])
                assert len(set(tr)&set(ho))==0 and len(tr)+len(ho)==len(vi)
                hits[ho]+=1
                inner_scores=np.zeros(len(tr)); inner_hits=np.zeros(len(tr),int)
                for inner in fold['inner_models']:
                    it=np.array(inner['train']); ih=np.array(inner['hold'])
                    assert len(set(it)&set(ih))==0 and len(it)+len(ih)==len(tr)
                    inner_hits[ih]+=1
                    fallbacks+=check_model(inner['model'],xv[tr[it]],av[tr[it]],cv[tr[it]],y[tr[it]])
                    count+=1
                    inner_scores[ih]=predict_model(inner['model'],xv[tr[ih]])
                assert np.all(inner_hits==1)
                threshold=choose_threshold(y[tr],av[tr],cv[tr],inner_scores)
                assert (None if np.isinf(threshold) else threshold)==fold['threshold']
                fallbacks+=check_model(fold['model'],xv[tr],av[tr],cv[tr],y[tr]);count+=1
                score=predict_model(fold['model'],xv[ho])
                np.testing.assert_array_equal(apply_route(av[ho],cv[ho],score,fold['threshold']),cache['nested_pred'][ho])
                assert np.all(cache['fold_ids'][ho]==k)
                assert choose_weight(y[tr],a[vi[tr]],c[vi[tr]])==fold['global_weight']
                w=fold['global_weight']
                np.testing.assert_array_equal((w*a[vi[ho]]+(1-w)*c[vi[ho]]).argmax(1),cache['global_nested'][ho])
            assert np.all(hits==1)
            for full in s['full_oof_models']:
                tr=np.array(full['train']);ho=np.array(full['hold'])
                fallbacks+=check_model(full['model'],xv[tr],av[tr],cv[tr],y[tr]);count+=1
                np.testing.assert_array_equal(predict_model(full['model'],xv[ho]),cache['full_oof'][ho])
            chosen=choose_threshold(y,av,cv,cache['full_oof'])
            assert s['threshold']==(None if np.isinf(chosen) else chosen)
            fallbacks+=check_model(s['model'],xv,av,cv,y);count+=1
            np.testing.assert_array_equal(predict_model(s['model'],x),cache['score'])
    with (ROOT/'results.csv').open(encoding='utf-8-sig') as stream: rows=list(csv.DictReader(stream))
    idx={(r['dataset'],r['condition'],r['method']):r for r in rows}
    for name in CONFIG['datasets']:
        y=runner.legacy.r2.raw_data(name).y.cpu().numpy()
        for cond in CONFIG['conditions']:
            pred=np.load(ROOT/'cells'/name/cond/'predictions.npz'); ti=pred['test']
            ap,cp=pred['A'][ti],pred['C'][ti]; route=pred['ROUTER'][ti]
            num_classes=len(np.unique(y))
            for method in ['A','C','MEAN','GLOBAL','ROUTER','ROUTER05','TCONF']:
                met=accuracy_f1(y[ti],pred[method][ti],num_classes)
                row=idx[name,cond,method]
                assert abs(met['accuracy']-float(row['test_accuracy']))<1e-12
                assert abs(met['macro_f1']-float(row['test_macro_f1']))<1e-12
            rescued=int(((route==y[ti])&(ap!=y[ti])).sum())
            harmed=int(((route!=y[ti])&(ap==y[ti])).sum())
            assert rescued==int(idx[name,cond,'ROUTER']['rescued'])
            assert harmed==int(idx[name,cond,'ROUTER']['harmed'])
    decision=read(ROOT/'decision.json')
    noisy={n:float(np.mean([100*(float(idx[n,c,'ROUTER']['valid_accuracy'])-float(idx[n,c,'GLOBAL']['valid_accuracy'])) for c in CONFIG['conditions'] if c!='clean'])) for n in CONFIG['datasets']}
    for n in noisy: assert abs(noisy[n]-decision['noisy_gain_global_pp'][n])<1e-12
    runner.save(ROOT/'final_audit.json',dict(passed=True,source_files=len(sources),input_files=len(inputs),
        original_source_reconstructed=True,model_refits_verified=count,constant_fallbacks=fallbacks,
        outer_validation_predictions=20,all_test_metric_rows=140,legacy_global_weights_exact=20,
        finalizer_sha256=runner.sha(Path(__file__)),old_results_modified=False))
    runner.save(ROOT/'complete.json',dict(cells=20,neural_fits=0,router_fit_slots=620,
        audit_passed=True,completed=datetime.now(timezone.utc).isoformat(),
        status='nested validation gate passed; test replication of fusion advantage absent; local features pending'))
    print(json.dumps(read(ROOT/'final_audit.json'),indent=2))


if __name__=='__main__': main()
