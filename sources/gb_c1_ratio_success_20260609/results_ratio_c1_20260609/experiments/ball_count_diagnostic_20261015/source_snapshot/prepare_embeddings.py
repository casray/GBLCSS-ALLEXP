"""Run the previously audited label-free trainers on the new official data only."""
from pathlib import Path
import argparse
import json
import logging
import sys
import time
import torch
import numpy as np
from threadpoolctl import threadpool_limits
import data_tools as dt
from data_tools import ROOT,WORKSPACE,DATA,CONFIG,d
import prepare_representations as grace
sys.path.insert(0,str(dt.GREET_ROOT))
import prepare_greet as greet
from methods import return_features


def freeze():
    files=[Path(__file__),ROOT/'data_tools.py',ROOT/'prepare_data.py',ROOT/'config.json',ROOT/'PLAN.md',ROOT/'dataset_manifest.json',
           Path(grace.__file__),Path(greet.__file__),dt.GREET_ROOT/'methods.py',
           WORKSPACE/'GB-LCSS_code/src/gblcss/grace.py']
    manifest={str(p.resolve()):d.sha(p) for p in files+list(DATA[n] for n in CONFIG['new_datasets'])}
    path=ROOT/'embedding_source_manifest.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=manifest:raise ValueError('Embedding sources changed')
    d.write(path,manifest)


def status(name,space,phase):
    d.write(ROOT/'embedding_status.json',dict(status='running',dataset=name,space=space,phase=phase,
            updated_local=time.strftime('%Y-%m-%d %H:%M:%S')))


def train_grace(name,device):
    folder=ROOT/'representations'/name/'GRACE'
    if (folder/'summary.json').exists():
        s=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
        if d.sha(folder/'H.pt')!=s['representation_sha256']:raise ValueError('GRACE H changed')
        return
    # The inherited loader still reads only X/A and verifies this exact official NPZ hash.
    grace.ROMAN_SHA256=d.sha(DATA[name]); status(name,'GRACE','training')
    grace.run(argparse.Namespace(action='train-grace',data=str(DATA[name]),output=str(folder),
        epochs=CONFIG['grace']['epochs'],device=device,seed=0,chunk_size=CONFIG['grace']['chunk_size'],threads=CONFIG['threads']))


def train_greet(name,device):
    folder=ROOT/'representations'/name/'GREET'
    if (folder/'summary.json').exists():
        s=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
        if d.sha(folder/'H.pt')!=s['H_sha256']:raise ValueError('GREET H changed')
        return
    folder.mkdir(parents=True,exist_ok=True); config=CONFIG['greet']
    d.write(folder/'config.json',dict(config=config,data_sha256=d.sha(DATA[name]),
        label_usage='none: only X/A enter train_representation',embedding_manifest_sha256=d.sha(ROOT/'embedding_source_manifest.json')))
    # Read features and edges only; no truth or split arrays enter this preparation path.
    with np.load(DATA[name],allow_pickle=False) as raw:
        x=torch.from_numpy(raw['node_features'].copy()).float()
        edges=torch.from_numpy(raw['edges'].T.copy()).long()
    from torch_geometric.utils import to_undirected
    edges=to_undirected(edges,num_nodes=len(x));edges=edges[:,edges[0]!=edges[1]]
    torch.save(edges,folder/'edges.pt');status(name,'GREET','structural_encoding')
    se=return_features(edges.numpy(),len(x),hops=config['se_hops'],walks=config['se_walks'],seed=0,block=config['se_block'])
    np.save(folder/'structural_encoding.npy',se)
    x=x.to(device);edges=edges.to(device);se=torch.from_numpy(se).to(device)
    knn=greet.feature_knn(x,config['k']);torch.save(knn.cpu(),folder/'knn.pt')
    status(name,'GREET','training');greet.train_representation(x,edges,se,knn,config,folder)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cuda');args=parser.parse_args()
    torch.set_num_threads(CONFIG['threads']);freeze()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.StreamHandler(),
        logging.FileHandler(ROOT/'embedding_raw.log',encoding='utf-8')])
    try:
        with threadpool_limits(limits=CONFIG['threads']):
            for name in CONFIG['new_datasets']:
                train_grace(name,args.device);train_greet(name,args.device)
                if torch.cuda.is_available():torch.cuda.empty_cache()
        for p,digest in json.loads((ROOT/'embedding_source_manifest.json').read_text(encoding='utf-8')).items():
            if d.sha(p)!=digest:raise ValueError('Embedding input changed: '+p)
        d.write(ROOT/'embedding_status.json',dict(status='completed',new_representations=6,
                updated_local=time.strftime('%Y-%m-%d %H:%M:%S')))
    except Exception as exc:
        d.write(ROOT/'embedding_status.json',dict(status='failed',error=repr(exc)));raise


if __name__=='__main__':main()
