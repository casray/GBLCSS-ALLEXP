"""Qualified label-free partitions plus untouched source edges for TFR."""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import Data

from experiment import atomic_json, file_hash


def load_verified_edges(source, digest, x, y, p):
    if file_hash(source)!=digest:
        raise ValueError('original source digest changed')
    stored=torch.load(source,map_location='cpu',weights_only=False)
    raw=Data.from_dict(stored[0]) if isinstance(stored[0],dict) else stored[0]
    if not np.array_equal(raw.x.numpy().astype(np.float32),x):
        raise ValueError('original source features differ from cache')
    if not np.array_equal(raw.y.numpy().astype(np.int64),y):
        raise ValueError('original source labels differ from cache')
    edges=raw.edge_index.numpy().astype(np.int64)
    n=len(y)
    if edges.ndim!=2 or edges.shape[0]!=2 or edges.min()<0 or edges.max()>=n:
        raise ValueError('invalid original source edges')
    a=sp.coo_matrix((np.ones(edges.shape[1],np.float32),(edges[0],edges[1])),shape=(n,n)).tocsr()
    a=((a+a.T)>0).astype(np.float32);a.setdiag(0);a.eliminate_zeros()
    a+=sp.eye(n,dtype=np.float32,format='csr')
    inv=sp.diags(1/np.sqrt(np.asarray(a.sum(1)).ravel()))
    delta=((inv@a@inv).tocsr()-p).tocsr()
    if delta.nnz and np.max(np.abs(delta.data))>1e-7:
        raise ValueError('original source propagation differs from cache')
    return edges


def prepare(config_path,root,dataset):
    started=time.perf_counter()
    here=Path(__file__).parent
    path=here/'prior'/'prepare_data.py'
    spec=importlib.util.spec_from_file_location('_qualified_partition_prepare',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    meta=module.prepare(config_path,root,dataset)
    original=meta['original_metadata'] if 'original_metadata' in meta else None
    if original is None:
        raise ValueError('partition metadata lacks original source record')
    folder=Path(root)/'cache'/dataset
    if 'edge_index.npy' in meta['new_files']:
        if file_hash(original['source'])!=original['source_sha256']:
            raise ValueError('original raw source changed')
        return meta
    old=Path(meta['old_directory'])
    edges=load_verified_edges(original['source'],original['source_sha256'],
        np.load(old/'x.npy',mmap_mode='r'),np.load(old/'y.npy'),sp.load_npz(old/'p.npz'))
    np.save(folder/'edge_index.npy',edges)
    meta['new_files']['edge_index.npy']=file_hash(folder/'edge_index.npy')
    meta['tfr_raw_edges']=edges.shape[1]
    meta['edge_scope']='Unmodified source edges; TFR GCNConv performs native normalization and self loops'
    meta['extension_preparation_wall_s']=time.perf_counter()-started
    atomic_json(folder/'metadata.json',meta)
    print(json.dumps(dict(dataset=dataset,raw_edges=edges.shape[1],source_and_propagation_verified=True)),flush=True)
    return meta


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    parser.add_argument('--run-root',required=True);parser.add_argument('--dataset',required=True)
    args=parser.parse_args();prepare(args.config,args.run_root,args.dataset)
