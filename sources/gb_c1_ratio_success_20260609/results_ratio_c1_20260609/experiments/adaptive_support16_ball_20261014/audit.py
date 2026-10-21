"""Independent post-hoc artifact, relation, formal-target and saved-logit audit."""
from pathlib import Path
import argparse
import json
import math
import sys
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import run_screen as screen
sys.path.insert(0,str(screen.MECHANISM))
from audit_diagnostic import reference_vote
from gblcss.granular_ball_adaptive import _normalized_array
from gblcss.targets import build_leave_one_out_targets
from gblcss.data import inject_train_label_noise

ROOT=screen.ROOT;d=screen.d


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def close(actual,expected):
    assert actual.shape==expected.shape
    assert np.allclose(actual,expected,atol=2e-6,rtol=0,equal_nan=True)


def verify_case(folder):
    name=folder.parent.name;seed=int(folder.name.removeprefix('seed'))
    graph=screen.load_case(name,seed);tr=graph.train_index.numpy()
    with np.load(folder/'split.npz') as a:
        for key,expected in [('train',tr),('val',graph.val_index.numpy()),('test',graph.test_index.numpy())]:
            assert np.array_equal(a[key],expected)
    with np.load(folder/'assignments.npz') as a:assignments={k:a[k].copy() for k in a.files}
    stats=read(folder/'partition_stats.json');metadata=read(folder/'relations.json')
    for arm,assignment in assignments.items():
        assert assignment.shape==(graph.num_nodes,) and assignment.min()>=0
        labels,sizes=np.unique(assignment,return_counts=True)
        assert len(labels)==stats[arm]['actual_balls'] and sizes.sum()==graph.num_nodes
        ballots=np.unique(assignment[tr],return_counts=True)[1]
        if arm=='SQRT_SUPPORT':
            assert stats[arm]['requested_target']==math.ceil(math.sqrt(graph.num_nodes))
            assert ballots.min()>=16 and stats[arm]['min_train_support']>=16
        if arm in ['ADAPT64','SQRT_GEOM','SQRT_SUPPORT']:
            assert sizes.min()>=stats[arm]['min_leaf_size']
            balls=read(folder/(arm+'_balls.json'))
            assert sum(b['size'] for b in balls)==graph.num_nodes
            for b in balls:
                assert b['size']==int((assignment==b['ball_id']).sum())
                assert b['train_count']==int((assignment[tr]==b['ball_id']).sum())
            trace=read(folder/(arm+'_split_trace.json'))
            accepted=[r for r in trace if r['accepted']]
            assert len(accepted)==len(labels)-1
            for row in accepted:
                assert sum(row['child_sizes'])==row['parent_size']
                assert row['sse_gain']>max(1e-12,1e-8*row['parent_sse'])
                assert abs(row['parent_sse']-sum(row['child_sse'])-row['sse_gain'])<1e-8
    relations=None
    if seed==0:
        h=d.load_frozen_representation(screen.H[name],graph.num_nodes).numpy()
        z=_normalized_array(h)
        for arm in ['ADAPT64','SQRT_GEOM','SQRT_SUPPORT']:
            total=0.
            for ball in read(folder/(arm+'_balls.json')):
                x=z[assignments[arm]==ball['ball_id']];centered=x-x.mean(0)
                cost=float(np.einsum('ij,ij->',centered,centered));total+=cost
                assert abs(cost-ball['sse'])<1e-7
            assert abs(total-stats[arm]['total_sse'])<1e-7
        relations={arm:d.group_weights(a[tr]) for arm,a in assignments.items()}
        counts=np.diff(relations['SQRT_SUPPORT'].indptr)
        relations['KNN_MATCH']=d.knn_weights(h[tr],counts,'cuda')
        for arm,w in relations.items():
            assert not np.any(w.diagonal())
            assert d.fingerprint(w.indptr,w.indices,w.data)==metadata[arm]['hash']
        assert np.array_equal(np.diff(relations['KNN_MATCH'].indptr),counts)
    rows=read(folder/'metrics.json')
    for noise,rate in d.CONFIG['noise_conditions'].items():
        data=inject_train_label_noise(graph,'uniform' if noise=='clean' else noise,rate,seed)
        observed=data.noisy_y[tr].numpy();clean=data.y[tr].numpy();source=folder/noise
        protocol=read(source/'protocol.json');frozen=read(source/'frozen_targets.json')
        assert protocol['split_hash']==d.fingerprint(tr,data.val_index.numpy(),data.test_index.numpy())
        assert protocol['noisy_hash']==d.fingerprint(observed)
        assert torch.equal(data.noisy_y[data.val_mask],data.y[data.val_mask])
        assert torch.equal(data.noisy_y[data.test_mask],data.y[data.test_mask])
        for arm,digest in frozen.items():
            file=source/(arm+'.npz');assert d.sha(file)==digest
            with np.load(file) as artifact:a={k:artifact[k].copy() for k in artifact.files}
            assert a['target'].shape==(len(tr),data.num_classes)
            assert np.isfinite(a['target']).all() and a['target'].min()>=0
            close(a['target'].sum(1),np.ones(len(tr)))
            close(a['target'],d.mix(a['q'],a['alpha'],observed))
            metric=d.target_metrics(a['target'],a['q'],a['alpha'],clean,observed)
            row=[r for r in rows if r['noise']==noise and r['arm']==arm];assert len(row)==1
            assert abs(metric['target_brier']-row[0]['target_brier'])<2e-6
            if arm in ['OLD64','SQRT_SUPPORT']:
                formal=build_leave_one_out_targets(data.noisy_y,assignments[arm],data.train_index,data.num_classes,
                    min_train_per_ball=2,beta=.5).soft_target[tr].numpy()
                close(a['target'],formal)
            if relations is not None and arm!='CE':
                r,q,alpha,count,effective=reference_vote(relations[arm],observed,data.num_classes)
                for key,values in [('r',r),('q',q),('alpha',alpha),('support',count),('effective',effective)]:
                    close(a[key],values)
    return dict(dataset=name,seed=seed,status='passed',deep_relations=seed==0)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--students-only',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(4)
    for manifest in ['source_manifest.json','pilot_source_manifest.json']:
        if (ROOT/manifest).exists():
            for path,digest in read(ROOT/manifest).items():assert d.sha(path)==digest,path
    cases=[];prior_audit_hash=None
    if args.students_only:
        prior=read(ROOT/'audit/report.json');assert prior['status']=='passed'
        cases=prior['cases'];prior_audit_hash=d.sha(ROOT/'audit/report.json')
    else:
        with threadpool_limits(limits=4):
            for file in sorted((ROOT/'runs').glob('*/seed*/metrics.json')):
                print('AUDIT',file.parent.relative_to(ROOT),flush=True)
                cases.append(verify_case(file.parent))
    assert len(cases)==5
    students=[]
    for file in sorted((ROOT/'students').glob('*/seed*/*/*/result.json')):
        row=read(file);cfg=read(file.parent/'config.json');graph=screen.load_case(row['dataset'],row['seed'])
        assert d.sha(cfg['target_path'])==cfg['target_hash']==row['target_hash']
        logits=torch.load(file.parent/'logits.pt',map_location='cpu',weights_only=True)
        for idx,key in [(graph.val_index,'validation_accuracy'),(graph.test_index,'test_accuracy')]:
            actual=(logits[idx].argmax(1)==graph.y[idx]).float().mean().item()
            assert abs(actual-row[key])<1e-8
        assert 1<=row['best_epoch']<=200
        students.append(dict(dataset=row['dataset'],noise=row['noise'],arm=row['arm'],status='passed'))
    if args.students_only:
        assert len(students)==75
        expected={(name,noise,arm) for name in screen.NAMES for noise in d.CONFIG['noise_conditions']
                  for arm in ['CE','OLD64','SQRT_SUPPORT','KM_MATCH','KNN_MATCH']}
        assert {(r['dataset'],r['noise'],r['arm']) for r in students}==expected
    d.write(ROOT/'audit/report.json',dict(status='passed',protocol_runs=len(cases),student_runs=len(students),cases=cases,
        reused_target_audit_sha256=prior_audit_hash,
        coverage='All source/frozen hashes, splits/noise, support/size/trace invariants, probabilities/Brier/formal OLD64+primary targets; seed0 per graph additionally SSE/relations/independent vote formulas; saved-logit validation/test accuracies.',
        limitations='No refitting of all partitions or repeated model training; single SSL seed and seed0 pilot remain exploratory.'))
    print(json.dumps(dict(status='passed',protocol_runs=len(cases),student_runs=len(students))),flush=True)


if __name__=='__main__':main()
