"""Pin official new data and train two label-free GRACE representations."""
import argparse
import logging
from pathlib import Path
import time
import urllib.request
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import common as c
import prepare_representations as grace


def prepare_data():
    manifest=c.ROOT/'dataset_manifest.json'
    if manifest.exists():
        for n,r in c.read(manifest)['datasets'].items():
            if c.sha(c.DATA[n])!=r['sha256']:raise ValueError('Data changed: '+n)
        return
    records={}
    for name in c.CONFIG['datasets']:
        source=None
        if name in c.CONFIG['new_datasets']:
            source=f"https://raw.githubusercontent.com/yandex-research/heterophilous-graphs/{c.CONFIG['official_commit']}/data/{name}.npz"
            path=c.DATA[name];path.parent.mkdir(parents=True,exist_ok=True)
            if not path.exists():
                req=urllib.request.Request(source,headers={'User-Agent':'research-reproducibility'})
                with urllib.request.urlopen(req,timeout=60) as response:payload=response.read()
                temp=path.with_suffix('.download');temp.write_bytes(payload);temp.replace(path)
        g=c.load_graph(name,0)
        splits=[]
        for seed in c.SEEDS:
            v=c.load_graph(name,seed)
            splits.append(dict(seed=seed,hash=c.fingerprint(v.train_index.numpy(),v.val_index.numpy(),v.test_index.numpy()),sizes=[len(v.train_index),len(v.val_index),len(v.test_index)]))
        src,dst=g.edge_index.numpy();keep=src<dst;y=g.y.numpy()
        deg=np.bincount(np.concatenate([src[keep],dst[keep]]),minlength=len(y))
        chance=float(np.square(np.bincount(y,weights=deg)/deg.sum()).sum())
        hom=float(np.mean(y[src[keep]]==y[dst[keep]]))
        records[name]=dict(path=str(c.DATA[name]),source_url=source,sha256=c.sha(c.DATA[name]),nodes=g.num_nodes,features=g.x.shape[1],classes=g.num_classes,
            edges_undirected=int(keep.sum()),class_counts=np.bincount(y).tolist(),edge_homophily=hom,adjusted_homophily=(hom-chance)/(1-chance),splits=splits)
        logging.info('DATA %s %s',name,records[name])
    c.write(manifest,dict(commit=c.CONFIG['official_commit'],datasets=records))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cuda');parser.add_argument('--data-only',action='store_true');a=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.StreamHandler(),logging.FileHandler(c.ROOT/'prepare_raw.log',encoding='utf-8')])
    torch.set_num_threads(c.CONFIG['threads'])
    c.write(c.ROOT/'prepare_status.json',dict(status='running',stage='data',started=time.time()))
    try:
        prepare_data()
        files=[Path(__file__),c.ROOT/'common.py',c.ROOT/'PLAN.md',c.ROOT/'config.json',c.ROOT/'dataset_manifest.json',Path(grace.__file__),c.WORKSPACE/'GB-LCSS_code/src/gblcss/grace.py',c.WORKSPACE/'GB-LCSS_code/src/gblcss/data.py']+list(c.DATA[n] for n in c.CONFIG['datasets'])
        c.freeze_manifest(c.ROOT/'preparation_manifest.json',files)
        if a.data_only:return
        with threadpool_limits(limits=c.CONFIG['threads']):
            for name in c.CONFIG['new_datasets']:
                folder=c.H[name].parent
                if (folder/'summary.json').exists():
                    if c.sha(c.H[name])!=c.read(folder/'summary.json')['representation_sha256']:raise ValueError('H changed')
                    continue
                c.write(c.ROOT/'prepare_status.json',dict(status='running',stage='grace',dataset=name))
                grace.ROMAN_SHA256=c.sha(c.DATA[name])
                grace.run(argparse.Namespace(action='train-grace',data=str(c.DATA[name]),output=str(folder),epochs=c.CONFIG['grace']['epochs'],device=a.device,seed=0,chunk_size=c.CONFIG['grace']['chunk_size'],threads=c.CONFIG['threads']))
                torch.cuda.empty_cache()
        c.verify_manifest(c.ROOT/'preparation_manifest.json')
        c.write(c.ROOT/'representation_manifest.json',{n:dict(path=str(c.H[n]),sha256=c.sha(c.H[n]),newly_trained=n in c.CONFIG['new_datasets']) for n in c.CONFIG['datasets']})
        c.write(c.ROOT/'prepare_status.json',dict(status='completed',new_representations=2))
    except Exception as exc:
        c.write(c.ROOT/'prepare_status.json',dict(status='failed',error=repr(exc)));raise


if __name__=='__main__':main()
