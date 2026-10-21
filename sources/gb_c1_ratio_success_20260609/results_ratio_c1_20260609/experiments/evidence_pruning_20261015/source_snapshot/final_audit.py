"""Reopen persisted targets/decisions and logits; never trust summary CSV alone."""
import logging
import numpy as np
import torch
from sklearn.metrics import f1_score
import common as c
from pruning_audit import verify_output


def read_output(folder):
    out=dict(folds=np.load(folder/'folds.npy'),decisions=c.read(folder/'decisions.json'),bundles={},components={})
    for arm in c.CONFIG['arms']:
        with np.load(folder/(arm+'.npz')) as data:
            out['bundles'][arm]={k:data[k].copy() for k in data.files if not k.startswith('component_')}
            out['components'][arm]={k[len('component_'):]:data[k].copy() for k in data.files if k.startswith('component_')}
    return out


def audit_all():
    counts=dict(cases=0,new_targets=0,components=0,logits_results=0,historical_results=0,new_training_executed=0,new_alias_records=0)
    identities=set()
    for name in c.CONFIG['datasets']:
        for seed in c.CONFIG['seeds']:
            tree=c.load_tree(name,seed)
            for noise in c.CONFIG['conditions']:
                folder=c.case_path(name,seed,noise);graph,protocol=c.dt.load_case(name,seed,noise)
                assert c.read(folder/'protocol.json')==protocol
                tr=graph.train_index.numpy();z=graph.noisy_y[tr].numpy();truth=graph.y[tr].numpy()
                np.testing.assert_array_equal(z,np.load(folder/'observed_train.npy'))
                out=read_output(folder);folds=np.empty(tree.n,dtype=int)
                for k,indices in enumerate(np.array_split(np.random.default_rng(seed+c.CONFIG['fold_seed_offset']).permutation(tree.n),3)):folds[indices]=k
                np.testing.assert_array_equal(folds,out['folds'])
                for decision in out['decisions']:
                    q=decision['query_fold'];orientation=decision['orientation'];other=[k for k in range(3) if k!=q]
                    np.testing.assert_array_equal(decision['query'],np.flatnonzero(folds==q))
                    np.testing.assert_array_equal(decision['selector'],np.flatnonzero(folds==other[orientation]))
                    np.testing.assert_array_equal(decision['voters'],np.flatnonzero(folds==other[1-orientation]))
                    shuffled=np.random.default_rng(seed+c.CONFIG['fold_seed_offset']+1009+q*101+orientation).permutation(decision['selector'])
                    aa,dd=np.array_split(shuffled,2)
                    np.testing.assert_array_equal(decision['fit'],aa);np.testing.assert_array_equal(decision['score'],dd)
                checked=verify_output(tree,z,graph.num_classes,out,min_voters=c.CONFIG['min_voters'],delta=c.CONFIG['delta'])
                counts['components']+=checked['components_checked'];counts['cases']+=1
                frozen=c.read(folder/'frozen_targets.json');metrics={r['arm']:r for r in c.read(folder/'metrics.json')}
                for arm in c.CONFIG['arms']+c.CONFIG['historical_arms']:
                    record=c.read(c.ROOT/'students'/name/f'seed{seed}'/noise/arm/'result.json')
                    assert all(record[k]==v for k,v in dict(dataset=name,seed=seed,noise=noise,arm=arm).items())
                    if arm in c.CONFIG['arms']:
                        assert c.sha(folder/(arm+'.npz'))==frozen[arm]==record['target_hash']
                        target=out['bundles'][arm]['target'];assert c.target_key(target)==record['target_semantic_hash']
                        brier=float(np.square(target-np.eye(graph.num_classes)[truth]).sum(1).mean())
                        assert abs(brier-metrics[arm]['target_brier'])<2e-12
                        counts['new_targets']+=1
                        counts['new_training_executed']+=int(record['training_executed'])
                        counts['new_alias_records']+=int(not record['training_executed'])
                    else:
                        prior=c.read(record['source_result'])
                        for key in ['test_accuracy','validation_accuracy','train_accuracy','test_macro_f1','target_hash','target_semantic_hash','logits_path']:
                            assert record[key]==prior[key]
                        counts['historical_results']+=1
                    logits=torch.load(record['logits_path'],map_location='cpu',weights_only=True)
                    assert logits.shape==(graph.num_nodes,graph.num_classes) and bool(torch.isfinite(logits).all())
                    pred=logits.argmax(1)
                    for key,index,labels in [('train_accuracy',graph.train_index,graph.noisy_y),('validation_accuracy',graph.val_index,graph.y),('test_accuracy',graph.test_index,graph.y)]:
                        actual=float((pred[index]==labels[index]).float().mean())
                        assert abs(actual-record[key])<1e-7,(name,seed,noise,arm,key)
                    macro=float(f1_score(graph.y[graph.test_index].numpy(),pred[graph.test_index].numpy(),labels=list(range(graph.num_classes)),average='macro',zero_division=0))
                    assert abs(macro-record['test_macro_f1'])<1e-12
                    identities.add(record['unique_model_id']);counts['logits_results']+=1
        logging.info('AUDIT %s done',name)
    manifest=c.read(c.ROOT/'source_manifest.json')
    for p,digest in manifest.items():assert c.sha(p)==digest,'Changed frozen source: '+p
    assert counts['cases']==54 and counts['new_targets']==324 and counts['historical_results']==216 and counts['logits_results']==540
    c.write(c.ROOT/'audit.json',dict(status='passed',**counts,unique_model_ids_including_history=len(identities),source_files_checked=len(manifest),errors=[],
        uncertainty='sample SD across three protocol seeds; one SSL seed; not CI or independent replication'))
    from report import make_report
    make_report()


if __name__=='__main__':audit_all()
