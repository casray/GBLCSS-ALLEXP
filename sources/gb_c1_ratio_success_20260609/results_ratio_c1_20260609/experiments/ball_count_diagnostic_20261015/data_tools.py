"""Fixed dataset/representation paths and audited official split adapters."""
from pathlib import Path
import json
import math
import sys
import numpy as np
import torch
from torch_geometric.utils import to_undirected

ROOT=Path(__file__).resolve().parent
WORKSPACE=ROOT.parents[1]
MECHANISM=ROOT.parent/'heterophily_mechanism_20261014'
PREVIOUS=ROOT.parent/'greet_adaptive_ball_20261015'
GREET_ROOT=ROOT.parent/'greet_half_pilot_20261015'
sys.path.insert(0,str(MECHANISM))
import diagnostic_v2 as d
from gblcss.data import GraphData,inject_train_label_noise

CONFIG=json.loads((ROOT/'config.json').read_text(encoding='utf-8'))
DATA=dict(d.DATA,**{n:ROOT/'data'/(n+'.npz') for n in CONFIG['new_datasets']})
H_PATHS=dict(GRACE=dict(d.CACHED_H,roman=MECHANISM/'representations/Roman-empire/grace/H.pt'),
             GREET={n:GREET_ROOT/'representations'/n/'H.pt' for n in ['roman','corafull','coauthorcs']})
for space in CONFIG['spaces']:
    H_PATHS[space].update({n:ROOT/'representations'/n/space/'H.pt' for n in CONFIG['new_datasets']})


def budgets_for(n):
    return sorted(set(CONFIG['budgets']+[math.ceil(math.sqrt(n))]))


def arms_for(n):
    return ['CE']+[space+'_'+kind for space in CONFIG['spaces']
                   for kind in ['OLD64']+['B'+str(b) for b in budgets_for(n)]]


def load_new_graph(path,name,seed):
    with np.load(path,allow_pickle=False) as raw:
        x=np.asarray(raw['node_features'],dtype=np.float32).copy()
        y=np.asarray(raw['node_labels'],dtype=np.int64).copy()
        edges=np.asarray(raw['edges'],dtype=np.int64).copy()
        masks=[np.asarray(raw[k][seed],dtype=bool).copy() for k in ['train_masks','val_masks','test_masks']]
    if x.ndim!=2 or y.shape!=(len(x),) or not np.isfinite(x).all():
        raise ValueError('Invalid ordered features or labels')
    if not np.array_equal(np.unique(y),np.arange(len(np.unique(y)))):
        raise ValueError('Class IDs must already be contiguous, without remapping')
    if edges.ndim!=2 or edges.shape[1]!=2 or np.any(edges<0) or np.any(edges>=len(x)):
        raise ValueError('Invalid edge endpoints')
    if any(m.shape!=(len(x),) for m in masks) or not np.all(np.stack(masks).sum(0)==1):
        raise ValueError('Official masks must be a disjoint full partition')
    indices=[torch.from_numpy(np.flatnonzero(m)) for m in masks]
    edge_index=to_undirected(torch.from_numpy(edges).T,num_nodes=len(x))
    return GraphData(name,torch.from_numpy(x),edge_index,torch.from_numpy(y),
                     *[torch.from_numpy(m) for m in masks],train_index=indices[0],val_index=indices[1],test_index=indices[2])


def load_case(name,seed,condition):
    graph=load_new_graph(DATA[name],name,seed) if name in CONFIG['new_datasets'] else d.load_graph(name,seed)
    graph=inject_train_label_noise(graph,'uniform' if condition=='clean' else condition,CONFIG['conditions'][condition],seed)
    tr=graph.train_index.numpy(); observed=graph.noisy_y[tr].numpy()
    protocol=dict(dataset=name,seed=seed,noise=condition,
        split_hash=d.fingerprint(tr,graph.val_index.numpy(),graph.test_index.numpy()),noise_hash=d.fingerprint(observed),
        val_clean=bool(torch.equal(graph.noisy_y[graph.val_index],graph.y[graph.val_index])),
        test_clean=bool(torch.equal(graph.noisy_y[graph.test_index],graph.y[graph.test_index])))
    if name not in CONFIG['new_datasets']:
        old=json.loads((PREVIOUS/'targets'/name/f'seed{seed}'/condition/'protocol.json').read_text(encoding='utf-8'))
        if old['split_hash']!=protocol['split_hash'] or old['noise_hash']!=protocol['noise_hash']:
            raise ValueError('Historical split or noise mismatch')
    return graph,protocol


def describe(graph):
    metrics=d.graph_metrics(graph) if hasattr(d,'graph_metrics') else {}
    u,v=graph.edge_index.numpy(); selected=u<v
    u,v=u[selected],v[selected]; y=graph.y.numpy()
    homophily=float((y[u]==y[v]).mean())
    degree=np.bincount(np.concatenate([u,v]),minlength=len(y))
    prior=np.bincount(y,weights=degree,minlength=graph.num_classes)/degree.sum()
    chance=float(np.square(prior).sum())
    return dict(**metrics,nodes=graph.num_nodes,edges_undirected=len(u),feature_dim=graph.x.shape[1],classes=graph.num_classes,
        edge_homophily=homophily,degree_prior_agreement=chance,adjusted_homophily=(homophily-chance)/(1-chance),
        class_counts=np.bincount(y,minlength=graph.num_classes).tolist(),
        split_sizes=[len(graph.train_index),len(graph.val_index),len(graph.test_index)],budgets=budgets_for(graph.num_nodes))
