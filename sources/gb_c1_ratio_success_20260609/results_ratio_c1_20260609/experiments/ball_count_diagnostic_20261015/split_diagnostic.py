"""Post-hoc split utilities; clean labels are diagnostics only, never training inputs."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits
from audit_report import audit_target
import data_tools as dt


def diagnose_tree(final,trace,train,observed,truth,classes):
    accepted=[x for x in trace if x['accepted']]
    leaves={0}
    for row in accepted:
        leaves.remove(row['parent_id']);leaves.update(row['child_ids'])
    final_ids=np.asarray(sorted(leaves));node_leaf=final_ids[np.asarray(final)]
    train=np.asarray(train);total=len(train)
    members={leaf:np.flatnonzero(node_leaf[train]==leaf) for leaf in final_ids}
    for row in reversed(accepted):
        members[row['parent_id']]=np.concatenate([members[c] for c in row['child_ids']])
    costs={}
    for node,positions in members.items():
        target=audit_target(np.zeros(len(positions),dtype=int),observed[positions],classes)['target']
        costs[node]=float(np.square(target-np.eye(classes)[truth[positions]]).sum())
    root=costs[0]/total;current=root;prefixes=[(1,current)];splits=[]
    for row in accepted:
        parent=row['parent_id'];children=row['child_ids'];gain=(costs[parent]-sum(costs[c] for c in children))/total
        current-=gain;prefixes.append((row['actual_count_after'],current))
        splits.append(dict(parent_id=parent,actual_count_after=row['actual_count_after'],
            parent_train_count=len(members[parent]),min_child_train_count=min(len(members[c]) for c in children),
            relative_sse_gain=float(row['sse_gain']/row['parent_sse']),global_brier_gain=gain,
            brier_gain_per_parent_train=gain*total/len(members[parent])))
    final_cost=sum(costs[leaf] for leaf in leaves)/total
    if abs(final_cost-current)>2e-10:raise ValueError('Local split gains do not telescope')
    optimum={leaf:(costs[leaf],1) for leaf in leaves}
    for row in reversed(accepted):
        parent=row['parent_id'];child_cost=sum(optimum[c][0] for c in row['child_ids'])
        optimum[parent]=(costs[parent],1) if costs[parent]<=child_cost+1e-12 else (child_cost,sum(optimum[c][1] for c in row['child_ids']))
    best=min(prefixes,key=lambda x:(x[1],x[0]));gains=np.array([x['global_brier_gain'] for x in splits])
    relative=np.array([x['relative_sse_gain'] for x in splits])
    rho=None
    if len(gains)>=3 and np.ptp(gains)>0 and np.ptp(relative)>0:
        rho=float(spearmanr(relative,gains).statistic)
    return dict(root_brier=root,final_brier=final_cost,best_prefix_brier=best[1],best_prefix_balls=best[0],
        oracle_brier=optimum[0][0]/total,oracle_balls=optimum[0][1],splits=splits,
        positive_split_fraction=float((gains>1e-12).mean()) if len(gains) else None,
        negative_split_fraction=float((gains < -1e-12).mean()) if len(gains) else None,
        spearman_relative_sse_vs_brier_gain=rho)


def main():
    root=dt.ROOT;rows=[];summaries=[];inputs={}
    source=[Path(__file__),root/'test_split_diagnostic.py',root/'data_tools.py',root/'audit_report.py',root/'source_manifest.json']
    inputs.update({str(p.resolve()):dt.d.sha(p) for p in source})
    with threadpool_limits(limits=1):
        for name in dt.CONFIG['datasets']:
            for seed in dt.CONFIG['protocol_seeds']:
                graph,_=dt.load_case(name,seed,'clean');train=graph.train_index.numpy().copy()
                truth=graph.y[graph.train_index].numpy().copy();classes=graph.num_classes;del graph
                for space in dt.CONFIG['spaces']:
                    partition=root/'partitions'/name/f'seed{seed}'/space
                    for path in [partition/'assignments.npz',partition/'trajectory_trace.json']:
                        inputs[str(path.resolve())]=dt.d.sha(path)
                    with np.load(partition/'assignments.npz') as a:final=a['B256'].copy()
                    trace=json.loads((partition/'trajectory_trace.json').read_text(encoding='utf-8'))
                    for noise in dt.CONFIG['conditions']:
                        folder=root/'targets'/name/f'seed{seed}'/noise
                        observed=np.load(folder/'observed_train.npy')
                        for path in [folder/'observed_train.npy',folder/f'{space}_B1.npz',folder/f'{space}_B256.npz']:
                            inputs[str(path.resolve())]=dt.d.sha(path)
                        diagnostic=diagnose_tree(final,trace,train,observed,truth,classes)
                        for arm,key in [('B1','root_brier'),('B256','final_brier')]:
                            with np.load(folder/f'{space}_{arm}.npz') as a:
                                actual=float(np.square(a['target']-np.eye(classes)[truth]).sum(1).mean())
                            if abs(actual-diagnostic[key])>2e-10:raise ValueError('Frozen endpoint mismatch')
                        meta=dict(dataset=name,seed=seed,space=space,noise=noise)
                        rows.extend(dict(**meta,**x) for x in diagnostic.pop('splits'))
                        summaries.append(dict(**meta,**diagnostic,oracle_not_used_for_training=True))
    out=root/'split_diagnostic';out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'per_split_utility.csv',index=False,encoding='utf-8-sig')
    frame=pd.DataFrame(summaries);frame.to_csv(out/'tree_summary.csv',index=False,encoding='utf-8-sig')
    metrics=['root_brier','final_brier','best_prefix_brier','oracle_brier','best_prefix_balls','oracle_balls',
             'negative_split_fraction','spearman_relative_sse_vs_brier_gain']
    summary=frame.groupby(['dataset','space','noise'])[metrics].agg(['mean','std'])
    summary.columns=['_'.join(x) for x in summary.columns]
    summary.reset_index().to_csv(out/'summary.csv',index=False,encoding='utf-8-sig')
    for path,digest in inputs.items():
        if dt.d.sha(path)!=digest:raise ValueError('Diagnostic input changed')
    dt.d.write(out/'manifest.json',dict(status='passed',analysis_type='post-hoc; not preregistered training intervention',
        source_inputs=inputs,tree_condition_cells=len(summaries),split_condition_rows=len(rows),
        frozen_endpoint_checks=2*len(summaries),clean_label_usage='diagnostics and oracle pruning only; never used to train or select deployed model',
        telescoping_gain_check='passed; final minus root equals summed local split effects',
        oracle_warning='clean-label oracle is a target-quality bound within this tree, not a practical algorithm or an accuracy bound'))
    print(json.dumps(dict(status='passed',tree_condition_cells=len(summaries),split_condition_rows=len(rows)),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
