"""Post-hoc training-target diagnostics; never changes the frozen gate or trains."""
from pathlib import Path
import numpy as np
import pandas as pd
import common as c
from group_targets import group_vote


def main():
    if c.read(c.ROOT/'audit/targets.json')['status']!='passed':raise ValueError('Complete audited targets required')
    rows=[]
    for name in c.CONFIG['datasets']:
        for seed in c.SEEDS:
            for noise in c.conditions(name):
                graph,protocol=c.load_case(name,seed,noise);tr=graph.train_index.numpy()
                clean=graph.y[tr].numpy();observed=graph.noisy_y[tr].numpy();classes=graph.num_classes
                folder=c.ROOT/'targets'/name/f'seed{seed}'/noise
                if protocol!=c.read(folder/'protocol.json'):raise ValueError('Protocol changed')
                c.verify_manifest(folder/'frozen_targets.json')
                truth=np.eye(classes)[clean]
                predictions={}
                for arm in c.ARMS:
                    with np.load(folder/(arm+'.npz')) as a:predictions[arm]=a['target'].copy()
                _,q,alpha,_,_=group_vote(np.zeros(len(tr),dtype=np.int64),observed,classes)
                predictions['GLOBAL_POSTHOC']=c.d.mix(q,alpha,observed).astype(np.float32)
                for arm,target in predictions.items():
                    error=((target-truth)**2).sum(1)
                    per_class=[float(error[clean==k].mean()) for k in range(classes)]
                    row=dict(dataset=name,seed=seed,noise=noise,arm=arm,
                        micro_brier=float(error.mean()),balanced_class_brier=float(np.mean(per_class)),
                        worst_class_brier=float(np.max(per_class)),
                        clean_majority_fraction=float(np.bincount(clean,minlength=classes).max()/len(clean)),
                        observed_majority_fraction=float(np.bincount(observed,minlength=classes).max()/len(observed)),
                        posthoc=True,new_student_training=False)
                    rows.append(row)
                    c.write(c.ROOT/'posthoc_prior_details'/name/f'seed{seed}'/noise/(arm+'.json'),dict(**row,per_class_brier=per_class,clean_class_counts=np.bincount(clean,minlength=classes).tolist()))
    c.d.csv_write(c.ROOT/'posthoc_prior_diagnostic.csv',rows)
    frame=pd.DataFrame(rows);cols=['micro_brier','balanced_class_brier','worst_class_brier','clean_majority_fraction','observed_majority_fraction']
    summary=frame.groupby(['dataset','noise','arm'])[cols].agg(['mean','std','count']);summary.columns=['_'.join(x) for x in summary.columns]
    summary.reset_index().to_csv(c.ROOT/'posthoc_prior_summary.csv',index=False)
    c.write(c.ROOT/'posthoc_prior_manifest.json',dict(status='completed',posthoc=True,not_a_gate_or_trained_arm=True,rows=len(rows),script_sha256=c.sha(__file__),target_audit_sha256=c.sha(c.ROOT/'audit/targets.json'),table_sha256=c.sha(c.ROOT/'posthoc_prior_diagnostic.csv')))
    print('Posthoc prior diagnostic rows:',len(rows))


if __name__=='__main__':main()
