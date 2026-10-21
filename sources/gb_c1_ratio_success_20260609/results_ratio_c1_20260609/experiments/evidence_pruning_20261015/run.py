"""Frozen-target experiment; no clean-label efficacy gate or test selection."""
import argparse
import logging
import os
import time
import traceback
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits
import common as c
from gblcss.granular_ball_pruning import crossfit_targets
from gblcss.training import train_x_only_mlp
from gblcss.data import set_seed
from pruning_audit import verify_output


def status(stage,**extra):
    c.write(c.ROOT/'status.json',dict(status='running',stage=stage,pid=os.getpid(),updated_local=datetime.now().isoformat(timespec='seconds'),**extra))


def freeze():
    files=list(c.ROOT.glob('*.py'))+[c.ROOT/'PLAN.md',c.ROOT/'config.json',c.WORKSPACE/'GB-LCSS_code/src/gblcss/granular_ball_pruning.py']
    original=c.read(c.PREVIOUS/'source_manifest.json')
    for path,digest in original.items():
        if c.sha(path)!=digest:raise ValueError('Old frozen input changed: '+path)
    files+=[Path(p) for p in original]+[c.PREVIOUS/'source_manifest.json',c.PREVIOUS/'audit.json']
    for name in c.CONFIG['datasets']:
        for seed in c.CONFIG['seeds']:
            part=c.PREVIOUS/'partitions'/name/f'seed{seed}'/'GREET'
            files += [part/'assignments.npz',part/'trajectory_trace.json']
            for noise in c.CONFIG['conditions']:
                old=c.old_case(name,seed,noise)
                files += [old/'observed_train.npy',old/'protocol.json',old/'metrics.json']
                for arm in c.CONFIG['historical_arms']+['GREET_B1']:
                    result=c.PREVIOUS/'students'/name/f'seed{seed}'/noise/arm/'result.json'
                    row=c.read(result);files += [old/(arm+'.npz'),result,Path(row['logits_path'])]
    manifest={str(p.resolve()):c.sha(p) for p in set(files)}
    path=c.ROOT/'source_manifest.json'
    if path.exists() and c.read(path)!=manifest:raise ValueError('Frozen sources/inputs differ; do not resume silently')
    c.write(path,manifest)
    for p in c.ROOT.glob('*.py'):
        dest=c.ROOT/'source_snapshot'/p.name;dest.parent.mkdir(exist_ok=True)
        if dest.exists() and c.sha(dest)!=c.sha(p):raise ValueError('Snapshot mismatch')
        if not dest.exists():dest.write_bytes(p.read_bytes())


def verify_sources():
    inputs=c.read(c.ROOT/'source_manifest.json')
    for path,digest in inputs.items():
        if c.sha(path)!=digest:raise ValueError('Frozen input changed: '+path)
    return len(inputs)


def summarize():
    for kind,pattern in [('target','targets/*/seed*/*/metrics.json'),('student','students/*/seed*/*/*/result.json')]:
        rows=[]
        for p in c.ROOT.glob(pattern):
            value=c.read(p);rows.extend(value if isinstance(value,list) else [value])
        if rows:pd.DataFrame(rows).to_csv(c.ROOT/(kind+'_results.csv'),index=False,encoding='utf-8-sig')


def create_targets(name,seed):
    tree=c.load_tree(name,seed)
    for noise in c.CONFIG['conditions']:
        folder=c.case_path(name,seed,noise)
        if (folder/'complete.json').exists():continue
        graph,protocol=c.dt.load_case(name,seed,noise);tr=graph.train_index.numpy()
        observed=graph.noisy_y[tr].numpy().copy();classes=graph.num_classes
        assert protocol==c.read(c.old_case(name,seed,noise)/'protocol.json')
        assert np.array_equal(observed,np.load(c.old_case(name,seed,noise)/'observed_train.npy'))
        folder.mkdir(parents=True,exist_ok=True)
        status('targets',dataset=name,seed=seed,noise=noise)
        out=crossfit_targets(tree,observed,classes,seed=seed+c.CONFIG['fold_seed_offset'],
            min_voters=c.CONFIG['min_voters'],delta=c.CONFIG['delta'])
        audit=verify_output(tree,observed,classes,out,min_voters=c.CONFIG['min_voters'],delta=c.CONFIG['delta'])
        # A real-data leave-one-out intervention, separate from clean diagnostics.
        index=seed%len(observed);altered=observed.copy();altered[index]=(altered[index]+1)%classes
        other=crossfit_targets(tree,altered,classes,seed=seed+c.CONFIG['fold_seed_offset'],
            min_voters=c.CONFIG['min_voters'],delta=c.CONFIG['delta'])
        for arm in c.CONFIG['arms']:
            for key in ['q','alpha','r','support']:
                np.testing.assert_allclose(out['bundles'][arm][key][index],other['bundles'][arm][key][index],atol=2e-12,rtol=0)
            np.testing.assert_array_equal(out['components'][arm]['region'][:,index],other['components'][arm]['region'][:,index])
        np.save(folder/'observed_train.npy',observed);np.save(folder/'folds.npy',out['folds'])
        c.write(folder/'decisions.json',out['decisions']);c.write(folder/'protocol.json',protocol)
        hashes={}
        for arm in c.CONFIG['arms']:
            arrays=dict(out['bundles'][arm]);arrays.update({'component_'+k:v for k,v in out['components'][arm].items()})
            path=folder/(arm+'.npz');np.savez_compressed(path,**arrays);hashes[arm]=c.sha(path)
        # Truth appears only AFTER all deployable targets have been saved.
        truth=graph.y[tr].numpy();rows=[]
        for arm,bundle in out['bundles'].items():
            m=c.dt.d.target_metrics(bundle['target'],bundle['q'],bundle['alpha'],truth,observed)
            comp=out['components'][arm]
            rows.append(dict(dataset=name,seed=seed,noise=noise,arm=arm,origin='new',**m,
                mean_support=float(bundle['support'].mean()),min_support=float(comp['support'].min()),
                region_count_mean=float(np.mean([len(np.unique(comp['region'][o,out['folds']==q])) for o in range(2) for q in range(3)])),
                support_coverage=float((comp['support']>=c.CONFIG['min_voters']).mean())))
        old_metrics={r['arm']:r for r in c.read(c.old_case(name,seed,noise)/'metrics.json')}
        for arm in c.CONFIG['historical_arms']:
            rows.append(dict(old_metrics[arm],origin='historical'))
        c.write(folder/'metrics.json',rows);c.write(folder/'frozen_targets.json',hashes)
        c.write(folder/'complete.json',dict(status='passed',audit=audit,own_label_intervention_index=int(index),
            probability_targets_frozen_before_truth_metrics=True,split_hash=protocol['split_hash'],noise_hash=protocol['noise_hash']))
        summarize();logging.info('TARGETS %s seed%d %s 6 new / 4 historical',name,seed,noise)


def train_case(name,seed,noise,device):
    graph,protocol=c.dt.load_case(name,seed,noise);tr=graph.train_index
    folder=c.case_path(name,seed,noise);assert protocol==c.read(folder/'protocol.json')
    frozen=c.read(folder/'frozen_targets.json');cache={}
    for arm in c.CONFIG['historical_arms']+['GREET_B1']:
        old=c.read(c.PREVIOUS/'students'/name/f'seed{seed}'/noise/arm/'result.json')
        with np.load(c.old_case(name,seed,noise)/(arm+'.npz')) as data:semantic=c.target_key(data['target'])
        assert semantic==old['target_semantic_hash']
        cache.setdefault(semantic,dict(old,source_result=str(c.PREVIOUS/'students'/name/f'seed{seed}'/noise/arm/'result.json'),model_origin='historical'))
        if arm in c.CONFIG['historical_arms']:
            record=dict(old,origin='historical',model_origin='historical',training_executed=False,
                source_training_executed=old['training_executed'],source_result=str(c.PREVIOUS/'students'/name/f'seed{seed}'/noise/arm/'result.json'))
            c.write(c.ROOT/'students'/name/f'seed{seed}'/noise/arm/'result.json',record)
    for arm in c.CONFIG['arms']:
        target=folder/(arm+'.npz');assert c.sha(target)==frozen[arm]
        with np.load(target) as data:local=data['target'].astype(np.float32)
        semantic=c.target_key(local);dest=c.ROOT/'students'/name/f'seed{seed}'/noise/arm
        if (dest/'result.json').exists():
            old=c.read(dest/'result.json');assert old['target_hash']==frozen[arm] and old['target_semantic_hash']==semantic
            cache.setdefault(semantic,old);continue
        status('students',dataset=name,seed=seed,noise=noise,arm=arm)
        c.write(dest/'config.json',dict(dataset=name,seed=seed,noise=noise,arm=arm,protocol=protocol,
            training=c.CONFIG['training'],target_hash=frozen[arm],target_semantic_hash=semantic,device=device))
        if semantic in cache:
            prior=cache[semantic]
            keys=['train_accuracy','validation_accuracy','test_accuracy','test_macro_f1','best_epoch','logits_path','unique_model_id']
            row={k:prior[k] for k in keys}
            row.update(training_executed=False,model_origin=prior['model_origin'],alias_of=prior['arm'],elapsed_seconds=0.)
            logging.info('ALIAS %s seed%d %s %s -> %s',name,seed,noise,arm,prior['arm'])
        else:
            full=torch.nn.functional.one_hot(graph.noisy_y,graph.num_classes).float();full[tr]=torch.from_numpy(local)
            set_seed(seed);started=time.monotonic()
            result=train_x_only_mlp(graph,full,device=device,**c.CONFIG['training'])
            logits=dest/'logits.pt';torch.save(result.logits,logits)
            row=dict(train_accuracy=result.train_accuracy,validation_accuracy=result.validation_accuracy,test_accuracy=result.test_accuracy,
                test_macro_f1=float(f1_score(graph.y[graph.test_index].numpy(),result.logits[graph.test_index].argmax(1).numpy(),
                    labels=list(range(graph.num_classes)),average='macro',zero_division=0)),
                best_epoch=result.best_epoch,logits_path=str(logits.resolve()),unique_model_id=f'evidence_pruning/{name}/seed{seed}/{noise}/{arm}',
                elapsed_seconds=time.monotonic()-started,training_executed=True,model_origin='new')
            logging.info('STUDENT %s seed%d %s %s test%.6f',name,seed,noise,arm,result.test_accuracy)
        row.update(dataset=name,seed=seed,noise=noise,arm=arm,origin='new',target_hash=frozen[arm],target_semantic_hash=semantic)
        c.write(dest/'result.json',row);cache.setdefault(semantic,row);summarize()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cuda');args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.StreamHandler(),logging.FileHandler(c.ROOT/'raw.log',encoding='utf-8')])
    torch.set_num_threads(c.CONFIG['threads'])
    try:
        freeze();status('preparing_targets')
        with threadpool_limits(limits=c.CONFIG['threads']):
            for name in c.CONFIG['datasets']:
                for seed in c.CONFIG['seeds']:create_targets(name,seed)
            assert len(list(c.ROOT.glob('targets/*/seed*/*/complete.json')))==54
            verify_sources();status('targets_audited',target_cases=54,new_targets=324)
            c.write(c.ROOT/'target_gate.json',dict(status='passed',type='correctness only',cases=54,new_targets=324,
                clean_metric_used_to_gate=False,test_used_to_gate=False))
            for name in c.CONFIG['datasets']:
                for seed in c.CONFIG['seeds']:
                    for noise in c.CONFIG['conditions']:train_case(name,seed,noise,args.device)
        summarize();verify_sources()
        from final_audit import audit_all
        audit_all();status('finished')
        finished=c.read(c.ROOT/'status.json');finished['status']='completed';c.write(c.ROOT/'status.json',finished)
    except Exception:
        c.write(c.ROOT/'status.json',dict(status='failed',error=traceback.format_exc(),updated_local=datetime.now().isoformat(timespec='seconds')))
        logging.exception('FAILED');raise


if __name__=='__main__':main()
