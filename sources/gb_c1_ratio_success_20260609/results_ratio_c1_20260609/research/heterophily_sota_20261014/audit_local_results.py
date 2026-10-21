"""Recompute existing clean-label scores and inspect protocols, without training."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]


def main():
    first = WORKSPACE/'experiments/heterophily_original_baseline_20261014/student_results.csv'
    second = WORKSPACE/'experiments/adaptive_support16_ball_20261014/student_results.csv'
    rows = []
    for path, datasets, aliases in [
        (first, ['texas','wisconsin','roman'], {'CE':'MLP_CE','GB64':'GB_LCSS_original'}),
        (second, ['corafull','coauthorcs'], {'CE':'MLP_CE','OLD64':'GB_LCSS_original','SQRT_SUPPORT':'GB_LCSS_support16'}),
    ]:
        frame = pd.read_csv(path)
        frame = frame[(frame.noise=='clean') & frame.dataset.isin(datasets)]
        for (name,arm),group in frame.groupby(['dataset','arm']):
            if arm not in aliases:
                continue
            v = group.test_accuracy * 100
            rows.append(dict(dataset=name, method=aliases[arm],n=len(v),mean_percent=v.mean(),
                sample_std_percent=v.std(ddof=1) if len(v)>1 else None,
                source=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                split_family='official_first5' if path==first else 'NoisyGL_seed0_20_10_70'))
    output = pd.DataFrame(rows)
    output.to_csv(ROOT/'local_clean_summary.csv',index=False,encoding='utf-8')
    protocols = []
    for name in ['texas','wisconsin','roman']:
        path = WORKSPACE/'experiments/heterophily_original_baseline_20261014/data'/f'{name}.npz'
        if name=='roman':
            path = WORKSPACE/'experiments/heterophily_mechanism_20261014/data/roman_empire.npz'
        with np.load(path) as a:
            counts = {k:int(a[k][0].sum()) for k in ['train_masks','val_masks','test_masks']}
            n = len(a['node_labels'])
            assert sum(counts.values()) == n
            protocols.append(dict(dataset=name,num_nodes=n,split_count=len(a['train_masks']),
                seed0_counts=counts,raw_edges=len(a['edges']),graph_preprocessing='to_undirected',
                source=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    for name in ['corafull','coauthorcs']:
        folder=WORKSPACE/'experiments/adaptive_support16_ball_20261014/runs'/name/'seed0'
        with np.load(folder/'assignments.npz') as a:
            n=len(a['OLD64'])
        with np.load(folder/'split.npz') as a:
            counts={k:len(a[k]) for k in a.files}
        protocols.append(dict(dataset=name,num_nodes=n,seed0_counts=counts,
            nodes_outside_split=n-sum(counts.values()),graph_preprocessing='existing NoisyGL graph',
            source=str(folder/'split.npz'),sha256=hashlib.sha256((folder/'split.npz').read_bytes()).hexdigest()))
    (ROOT/'protocol_audit.json').write_text(json.dumps(dict(status='passed',training_started=False,
        protocols=protocols,local_score_rows=len(rows)),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(output[['dataset','method','n','mean_percent','sample_std_percent']].to_string(index=False))


if __name__=='__main__':
    main()
