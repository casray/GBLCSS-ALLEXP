"""Fixed experiment protocol and source integrity helpers."""
from pathlib import Path
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parent
WORKSPACE=ROOT.parents[1]
MECHANISM=ROOT.parent/'heterophily_mechanism_20261014'
sys.path.insert(0,str(MECHANISM))
sys.path.insert(0,str(WORKSPACE/'GB-LCSS_code/src'))
import diagnostic_v2 as d
from gblcss.data import inject_train_label_noise

CONFIG=json.loads((ROOT/'config.json').read_text(encoding='utf-8'))
DATA=dict(d.DATA,**{n:ROOT/'data'/(n+'.npz') for n in CONFIG['new_datasets']})
H=dict(d.CACHED_H,roman=MECHANISM/'representations/Roman-empire/grace/H.pt',
       **{n:ROOT/'representations'/n/'H.pt' for n in CONFIG['new_datasets']})
SEEDS=CONFIG['protocol_seeds']
ARMS=CONFIG['baseline_arms']+CONFIG['candidate_arms']
sha=d.sha
write=d.write
fingerprint=d.fingerprint


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def conditions(name):
    return CONFIG['binary_conditions' if name=='tolokers' else 'multiclass_conditions']


def load_graph(name,seed):
    if name not in CONFIG['new_datasets']:return d.load_graph(name,seed)
    from torch_geometric.utils import to_undirected
    import torch
    from gblcss.data import GraphData
    with np.load(DATA[name],allow_pickle=False) as raw:
        x=np.asarray(raw['node_features'],dtype=np.float32).copy()
        y=np.asarray(raw['node_labels'],dtype=np.int64).copy()
        edges=np.asarray(raw['edges'],dtype=np.int64).copy()
        masks=[np.asarray(raw[k][seed],dtype=bool).copy() for k in ('train_masks','val_masks','test_masks')]
    if x.ndim!=2 or y.shape!=(len(x),) or not np.isfinite(x).all():raise ValueError('Invalid data arrays')
    if not np.array_equal(np.unique(y),np.arange(len(np.unique(y)))):raise ValueError('Non-contiguous classes')
    if edges.ndim!=2 or edges.shape[1]!=2 or np.any(edges<0) or np.any(edges>=len(x)):raise ValueError('Invalid edges')
    if any(m.shape!=(len(x),) for m in masks) or not np.all(np.stack(masks).sum(0)==1):raise ValueError('Invalid masks')
    idx=[torch.from_numpy(np.flatnonzero(m)) for m in masks]
    if any(len(np.unique(y[i.numpy()]))<2 for i in idx):raise ValueError('Single-class split')
    edge_index=to_undirected(torch.from_numpy(edges).T,num_nodes=len(x))
    edge_index=edge_index[:,edge_index[0]!=edge_index[1]]
    return GraphData(name,torch.from_numpy(x),edge_index,torch.from_numpy(y),
        *[torch.from_numpy(m) for m in masks],train_index=idx[0],val_index=idx[1],test_index=idx[2])


def load_case(name,seed,noise):
    g=load_graph(name,seed)
    kind,rate=conditions(name)[noise]
    inject_train_label_noise(g,kind,rate,seed)
    tr=g.train_index.numpy()
    return g,dict(dataset=name,seed=seed,noise=noise,kind=kind,rate=rate,
        split_hash=fingerprint(tr,g.val_index.numpy(),g.test_index.numpy()),
        noise_hash=fingerprint(g.noisy_y[tr].numpy()),
        data_sha256=sha(DATA[name]),val_clean=bool((g.y[g.val_index]==g.noisy_y[g.val_index]).all()),
        test_clean=bool((g.y[g.test_index]==g.noisy_y[g.test_index]).all()))


def freeze_manifest(path,files):
    expected={str(Path(p).resolve()):sha(p) for p in files}
    if Path(path).exists() and read(path)!=expected:raise ValueError('Frozen sources changed: '+str(path))
    write(path,expected)
    return expected


def verify_manifest(path):
    for p,h in read(path).items():
        if sha(p)!=h:raise ValueError('Frozen file changed: '+p)


def compute_gate(rows):
    lookup={(r['dataset'],int(r['seed']),r['noise'],r['arm']):r for r in rows}
    if len(lookup)!=len(rows):raise ValueError('Duplicate target rows')
    result={}
    for name in sorted({r['dataset'] for r in rows}):
        checks=[]
        noises=sorted({r['noise'] for r in rows if r['dataset']==name and r['noise']!='clean'})
        if len(noises)!=2:raise ValueError('Gate requires two noisy conditions')
        for noise in noises:
            for ref in CONFIG['gate']['references']:
                try:
                    delta=np.array([lookup[name,s,noise,'R_TREE']['target_brier']-lookup[name,s,noise,ref]['target_brier'] for s in SEEDS])
                except KeyError as exc:raise ValueError('Incomplete target gate') from exc
                passed=bool(delta.mean()<=CONFIG['gate']['mean_brier_delta_max'] and (delta<0).sum()>=CONFIG['gate']['minimum_improved_seeds'])
                checks.append(dict(noise=noise,reference=ref,mean_delta=float(delta.mean()),std=float(delta.std(ddof=1)),improved_seeds=int((delta<0).sum()),passed=passed))
        result[name]=dict(passed=all(x['passed'] for x in checks),checks=checks)
    return result
