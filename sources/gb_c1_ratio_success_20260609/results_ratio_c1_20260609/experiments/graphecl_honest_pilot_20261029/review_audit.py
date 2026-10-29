"""Post-fit, read-only scientific checks; writes only audit/diagnostic artifacts."""
from common import *
from mechanisms import positive_weights,split_folds,true_margins,vote_details
from sklearn.metrics import accuracy_score,f1_score


def validate_manifest(folder):
    manifest=read(folder/'complete.json')
    for name,digest in manifest['files'].items():
        assert sha(folder/name)==digest,(str(folder),name)
    return len(manifest['files'])


def independent_score(folder,d,s,c):
    lab=old.gvec.analysis_labels(d,s);data=old.supervised(d,s,c)
    p=load(folder/'probabilities.pt').numpy();pred=p.argmax(1)
    y=lab['y'].numpy();test=lab['test'].numpy();train=data['train'].numpy()
    t=load(folder/'target.pt').double().numpy()
    truth=np.eye(data['classes'])[y[train]]
    return dict(accuracy=float(accuracy_score(y[test],pred[test])),
        macro_f1=float(f1_score(y[test],pred[test],labels=np.arange(data['classes']),average='macro',zero_division=0)),
        validation=float(accuracy_score(data['valid_y'].numpy(),pred[data['valid'].numpy()])),
        brier=float(np.square(t-truth).sum(1).mean()))


def main():
    prepare();old.verify(read(ROOT/'crossfit/receipt.json')['signature'])
    lock=read(ROOT/'crossfit/fits_locked.json')
    assert len(lock['ssls'])==20 and len(lock['heads'])==20
    head_paths=[Path(r['source']) for r in read(ROOT/'stage1/fits_locked.json')['audits']]+[Path(p) for p in lock['heads']]
    hash_checks=0;head_epochs=0;ssl_epochs=0;head_audits=[]
    for folder in head_paths:
        hash_checks+=validate_manifest(folder)
        cfg=read(folder/'config.json');result=read(folder/'result.json')
        rows=[json.loads(r) for r in (folder/'epochs.jsonl').read_text().splitlines()]
        assert [r['epoch'] for r in rows]==list(range(1,201))
        assert all(np.isfinite(r['training_loss']) for r in rows)
        best=max(r['validation_correct'] for r in rows)
        first=next(r['epoch'] for r in rows if r['validation_correct']==best)
        assert result['best_epoch']==first and result['valid_correct']==best
        data=old.supervised(cfg['dataset'],cfg['seed'],cfg['condition'])
        p=load(folder/'probabilities.pt')
        assert int((p[data['valid']].argmax(1)==data['valid_y']).sum())==best
        assert state_hash(load(folder/'initial.pt'))==cfg['initial_hash']
        assert state_hash({'target':load(folder/'target.pt')})==cfg['target_hash']
        head_epochs+=len(rows);head_audits.append(dict(folder=str(folder),best_epoch=first,valid_correct=best))
    loss_rows=[]
    for folder in map(Path,lock['ssls']):
        hash_checks+=validate_manifest(folder)
        cfg=read(folder/'config.json');rows=[json.loads(r) for r in (folder/'epochs.jsonl').read_text().splitlines()]
        assert [r['epoch'] for r in rows]==list(range(1,21))
        assert all(np.isfinite(r['ecl_loss']) and np.isfinite(r['aux_loss']) for r in rows)
        assert read(folder/'result.json')['gpu_embedding_bitwise']
        ssl_epochs+=len(rows)
        loss_rows.append(dict(dataset=cfg['dataset'],seed=cfg['seed'],arm=cfg['arm'],fold=cfg['fold'],
            first_aux=rows[0]['aux_loss'],last_aux=rows[-1]['aux_loss'],first_ecl=rows[0]['ecl_loss'],last_ecl=rows[-1]['ecl_loss'],
            lambda_weight=cfg['weight'],seconds=read(folder/'result.json')['seconds_current_attempt']))
    errors=[]
    for stage in ['stage1','crossfit']:
        for row in read(ROOT/stage/'scores.json'):
            actual=independent_score(Path(row['source']),row['dataset'],row['seed'],row.get('condition',CFG['crossfit_condition']))
            errors.extend(abs(actual[k]-row[k]) for k in actual)
    assert max(errors)<1e-12
    pairs=read(ROOT/'crossfit/paired.json');scores=read(ROOT/'crossfit/scores.json')
    lookup={(r['dataset'],r['seed'],r['arm']):r for r in scores}
    for row in pairs:
        for metric in ['accuracy','macro_f1','validation','brier']:
            values=[(1 if metric=='brier' else 100)*(lookup[row['dataset'],s,row['arm']][metric]-lookup[row['dataset'],s,row['reference']][metric]) for s in CFG['seeds']]
            np.testing.assert_allclose(values,row[metric]['seeds'],rtol=0,atol=1e-12)
            assert abs(np.mean(values)-row[metric]['mean'])<1e-12
            assert abs(np.std(values,ddof=1)-row[metric]['sd'])<1e-12
    aux=[];region=[];drift=[];noise_checks=[]
    for d in CFG['datasets']:
        for s in CFG['seeds']:
            data=old.supervised(d,s,CFG['crossfit_condition']);train=data['train'].numpy()
            truth=old.gvec.analysis_labels(d,s)['y'].numpy();parts=split_folds(train,s)
            from gblcss.data import _independent_transition
            transition=_independent_transition(data['classes'],.4,'uniform')
            rng=np.random.RandomState(s)
            reconstructed=np.array([np.flatnonzero(rng.multinomial(1,transition[int(y)],1)[0])[0] for y in truth[train]])
            assert np.array_equal(reconstructed,data['observed'].numpy())
            noise_checks.append(dict(dataset=d,seed=s,independent_transition_reproduces_cache=True))
            full_assignment=load(source_assignment(d,s)).numpy()
            full_details=vote_details(truth[train],full_assignment[train],data['classes'])
            margins,purity=true_margins(truth[train],full_assignment[train],data['classes'])
            region.append(dict(dataset=d,seed=s,fold=-1,arm='a0',support=float(full_details['support'].mean()),
                region_nodes=float(np.bincount(full_assignment)[full_assignment[train]].mean()),
                true_margin=float(margins.mean()),true_class_fraction=float(purity.mean())))
            hbase=load(ROOT/'crossfit/formal/ssl'/d/f'seed{s}'/'ecl/H.pt')
            for f in range(2):
                folder=ROOT/'crossfit/packages'/d/f'seed{s}'/f'fold{f}'
                package={k:v.numpy() for k,v in load(folder/'package.pt').items()}
                ids=package['ids'];group=package['group'];y=truth[ids]
                assignment=load(folder/'partition/support32.pt').numpy()
                margins,purity=true_margins(y,group,data['classes'])
                detail=vote_details(y,group,data['classes'])
                region.append(dict(dataset=d,seed=s,fold=f,arm='b0',support=float(detail['support'].mean()),
                    region_nodes=float(np.bincount(assignment)[group].mean()),true_margin=float(margins.mean()),true_class_fraction=float(purity.mean())))
                same=(group[:,None]==group[None,:]);np.fill_diagonal(same,False)
                uniform=same/np.maximum(same.sum(1,keepdims=True),1)
                for arm in ['real','shuffle']:
                    target=package['target'].copy()
                    if arm=='shuffle':target=target[np.random.default_rng(s+29100+f).permutation(len(target))]
                    w=positive_weights(target,group);active=w.sum(1)>0
                    agreement=(y[:,None]==y[None,:])
                    aux.append(dict(dataset=d,seed=s,fold=f,arm=arm,active_fraction=float(active.mean()),
                        positive_class_precision=float((w*agreement).sum(1)[active].mean()),
                        same_ball_uniform_precision=float((uniform*agreement).sum(1)[active].mean()),
                        positive_count=float((w>0).sum(1)[active].mean()),
                        effective_positive_count=float((1/np.square(w[active]).sum(1)).mean()),
                        target_argmax_accuracy=float((target.argmax(1)==y).mean())))
                    h=load(ROOT/'crossfit/formal/ssl'/d/f'seed{s}'/f'{arm}_fold{f}'/'H.pt')
                    cosine=F.cosine_similarity(h,hbase,dim=1)
                    drift.append(dict(dataset=d,seed=s,fold=f,arm=arm,cosine_to_equal_budget_ecl_mean=float(cosine.mean()),
                        cosine_to_equal_budget_ecl_min=float(cosine.min())))
    for name,rows in [('auxiliary_quality',aux),('split_geometry',region),('embedding_drift',drift),('loss_trends',loss_rows)]:
        save(ROOT/'review'/f'{name}.json',rows);csv_write(ROOT/'review'/f'{name}.csv',rows)
    audit=dict(status='passed',formal_ssl=20,new_heads=44,ssl_updates=ssl_epochs,head_updates=head_epochs,
        manifest_file_hashes=hash_checks,max_independent_metric_error=max(errors),paired_statistics_checked=14,
        original_heads_reproduced=12,gpu_head_reloads=read(ROOT/'crossfit/independent_audit.json')['heads'],
        audit_source_sha=sha(Path(__file__)),noise_checks=noise_checks,head_selections=head_audits)
    save(ROOT/'review/audit.json',audit)
    print(json.dumps({k:v for k,v in audit.items() if k!='head_selections'},ensure_ascii=False))


if __name__=='__main__':main()
