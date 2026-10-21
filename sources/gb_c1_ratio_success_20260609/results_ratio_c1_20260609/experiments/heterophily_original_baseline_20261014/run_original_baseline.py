"""Unmodified GB64/LCSS/MLP across official heterophilic graph splits."""
from pathlib import Path
from datetime import datetime
import argparse
import json
import logging
import sys
import time
import inspect
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from torch_geometric.utils import to_undirected

HERE=Path(__file__).resolve().parent
PREVIOUS=HERE.parent/'heterophily_mechanism_20261014'
WORKSPACE=HERE.parents[1]
sys.path.insert(0,str(PREVIOUS))
import diagnostic_v2 as d
import prepare_representations as prep
from gblcss.data import GraphData, inject_train_label_noise, set_seed
from gblcss.targets import build_leave_one_out_targets
from gblcss.training import train_x_only_mlp
from run_students import TRAINING, full_targets, aggregate

FILES={name:HERE/'data'/f'{name}.npz' for name in ['texas','wisconsin']}
FILES['roman']=PREVIOUS/'data/roman_empire.npz'


def load_npz_graph(path,name,seed):
    with np.load(path,allow_pickle=False) as raw:
        x=torch.from_numpy(raw['node_features'].copy()).float()
        y=torch.from_numpy(raw['node_labels'].copy()).long()
        edges=to_undirected(torch.from_numpy(raw['edges'].copy()).long().T,num_nodes=len(y))
        masks=[torch.from_numpy(raw[k][seed].copy()).bool() for k in ['train_masks','val_masks','test_masks']]
    if any(len(m)!=len(y) for m in masks) or not torch.all(torch.stack(masks).sum(0)==1):
        raise ValueError('Official masks must be a disjoint full node partition')
    indices=[torch.nonzero(m,as_tuple=False).flatten() for m in masks]
    return GraphData(name,x,edges,y,*masks,train_index=indices[0],val_index=indices[1],test_index=indices[2])


def load_graph(name,seed):
    return load_npz_graph(FILES[name],name,seed)


def prepare_webkb():
    from torch_geometric.datasets import WebKB
    for name in ['texas','wisconsin']:
        if FILES[name].exists():continue
        dataset=WebKB(str(HERE/'data'/'webkb'),name=name)
        raw=dataset[0]
        node_lines=Path(dataset.raw_paths[0]).read_text().splitlines()[1:]
        identifiers=[int(line.split('\t')[0]) for line in node_lines if line.strip()]
        if identifiers!=list(range(raw.num_nodes)):
            raise ValueError('Raw node IDs are not in PyG row order')
        FILES[name].parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(FILES[name],node_features=raw.x.numpy(),node_labels=raw.y.numpy(),
            edges=raw.edge_index.T.numpy(),train_masks=raw.train_mask.T.numpy(),
            val_masks=raw.val_mask.T.numpy(),test_masks=raw.test_mask.T.numpy())
        d.write(HERE/'data'/f'{name}_source.json',dict(source_repository=dataset.url,
            raw_hashes={str(Path(f).resolve()):d.sha(f) for f in dataset.raw_paths},
            loader_source=inspect.getsourcefile(WebKB),loader_sha256=d.sha(inspect.getsourcefile(WebKB)),
            packed_sha256=d.sha(FILES[name]),node_order='verified raw ID == row index',
            split_count=raw.train_mask.shape[1]))


def configure_diagnostic():
    d.ROOT=HERE
    d.DATA=FILES
    d.load_graph=load_graph


def prepare_h(name,device):
    output=HERE/'representations'/name/'grace'
    if (output/'summary.json').exists():
        s=json.loads((output/'summary.json').read_text(encoding='utf-8'))
        if d.sha(output/'H.pt')!=s['representation_sha256']:raise ValueError('H hash changed')
        return output/'H.pt'
    # Reuse the frozen exact GRACE trainer; only its audited input hash changes.
    prep.ROMAN_SHA256=d.sha(FILES[name])
    d.write(output/'input_adapter.json',dict(dataset=name,adapter_sha256=d.sha(__file__),
        source_sha256=prep.ROMAN_SHA256,label_usage='none: inherited loader reads only features/edges'))
    prep.run(argparse.Namespace(action='train-grace',data=str(FILES[name]),output=str(output),
        epochs=1000,device=device,seed=0,chunk_size=512,threads=4))
    return output/'H.pt'


def summarize():
    rows=[json.loads(f.read_text(encoding='utf-8')) for f in (HERE/'students').glob('*/seed*/*/*/result.json')]
    d.csv_write(HERE/'student_results.csv',rows)
    if not rows:return
    summaries=aggregate(rows)
    d.csv_write(HERE/'student_summary.csv',summaries)
    d.write(HERE/'student_summary.json',summaries)
    lookup={(r['dataset'],r['seed'],r['noise'],r['arm']):r for r in rows}
    pairs=[]
    for key,row in lookup.items():
        ce=lookup.get((*key[:3],'CE'))
        if ce and key[3]!='CE':pairs.append(dict(dataset=key[0],seed=key[1],noise=key[2],arm=key[3],
            accuracy_minus_ce_pp=100*(row['test_accuracy']-ce['test_accuracy'])))
    d.csv_write(HERE/'student_paired_deltas.csv',pairs)


def run_students(name,seeds,device):
    for seed in seeds:
        graph=load_graph(name,seed)
        base=(PREVIOUS if name=='roman' else HERE)/'runs'/name/'grace'/f'seed{seed}'
        with np.load(base/'assignments.npz') as a:groups=a['GB64'].copy()
        for noise,rate in d.CONFIG['noise_conditions'].items():
            graph=inject_train_label_noise(graph,'uniform' if noise=='clean' else noise,rate,seed)
            tr=graph.train_index.numpy();source=base/noise
            protocol=json.loads((source/'protocol.json').read_text(encoding='utf-8'))
            frozen=json.loads((source/'frozen_targets.json').read_text(encoding='utf-8'))
            if d.fingerprint(graph.noisy_y[tr].numpy())!=protocol['noisy_train_fingerprint']:raise ValueError('Noise mismatch')
            if d.fingerprint(tr,graph.val_index.numpy(),graph.test_index.numpy())!=protocol['split_fingerprint']:raise ValueError('Split mismatch')
            formal=build_leave_one_out_targets(graph.noisy_y,groups,graph.train_index,graph.num_classes,
                min_train_per_ball=2,beta=.5).soft_target[tr].numpy()
            with np.load(source/'GB64.npz') as artifact:
                if not np.allclose(formal,artifact['target'],atol=2e-6,rtol=0):raise ValueError('Original GB64 formula mismatch')
            for arm in ['CE','GB64','KNN64']:
                folder=HERE/'students'/name/f'seed{seed}'/noise/arm
                if (folder/'result.json').exists():continue
                file=source/f'{arm}.npz'
                if d.sha(file)!=frozen[arm]:raise ValueError('Frozen target hash changed')
                with np.load(file) as artifact:local=artifact['target'].copy()
                targets=full_targets(graph.noisy_y,graph.train_index,graph.num_classes,local)
                config=dict(dataset=name,seed=seed,noise=noise,arm=arm,training=TRAINING,
                    input_hash=d.sha(FILES[name]),target_path=str(file),target_hash=frozen[arm],
                    split_hash=protocol['split_fingerprint'],noise_hash=protocol['noisy_train_fingerprint'],
                    script_hash=d.sha(__file__),trainer_hash=d.sha(WORKSPACE/'GB-LCSS_code/src/gblcss/training.py'),
                    model_hash=d.sha(WORKSPACE/'GB-LCSS_code/src/gblcss/models.py'),device=device)
                d.write(folder/'config.json',config)
                logging.info('START %s seed%d %s %s',name,seed,noise,arm)
                start=time.time();set_seed(seed)
                result=train_x_only_mlp(graph,targets,device=device,**TRAINING)
                torch.save(result.logits,folder/'logits.pt')
                # Independent post-training accuracy reconstruction.
                for index,key in [(graph.val_index,'validation_accuracy'),(graph.test_index,'test_accuracy')]:
                    actual=(result.logits[index].argmax(1)==graph.y[index]).float().mean().item()
                    if actual!=getattr(result,key):raise ValueError('Saved-logit accuracy mismatch')
                row=dict(dataset=name,representation='grace',seed=seed,noise=noise,arm=arm,
                    train_accuracy=result.train_accuracy,validation_accuracy=result.validation_accuracy,
                    test_accuracy=result.test_accuracy,best_epoch=result.best_epoch,
                    elapsed_seconds=time.time()-start,test_nodes=len(graph.test_index),
                    target_hash=frozen[arm],original_formula_check='passed',logit_check='passed')
                d.write(folder/'result.json',row);summarize()
                logging.info('DONE %s seed%d %s %s test=%.6f',name,seed,noise,arm,result.test_accuracy)
                del targets,result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--datasets',nargs='+',default=['texas','wisconsin','roman'])
    parser.add_argument('--seeds',type=int,nargs='+',default=list(range(5)))
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args()
    torch.set_num_threads(4)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',
        handlers=[logging.StreamHandler(),logging.FileHandler(HERE/'raw.log',encoding='utf-8')])
    status=dict(status='running',started_at=datetime.now().astimezone().isoformat(),datasets=args.datasets,seeds=args.seeds)
    d.write(HERE/'status.json',status)
    try:
        prepare_webkb();configure_diagnostic()
        sources={str(Path(f).resolve()):d.sha(f) for f in [__file__,HERE/'PLAN.md',d.__file__,prep.__file__,
            WORKSPACE/'GB-LCSS_code/src/gblcss/training.py',WORKSPACE/'GB-LCSS_code/src/gblcss/models.py',
            WORKSPACE/'GB-LCSS_code/src/gblcss/targets.py',*FILES.values()]}
        d.write(HERE/'adapter_manifest.json',sources)
        with threadpool_limits(limits=4):
            for name in args.datasets:
                status['current_dataset']=name;d.write(HERE/'status.json',status)
                if name!='roman':
                    h=prepare_h(name,args.device)
                    d.run_dataset(name,'grace',h,args.seeds,args.device)
                run_students(name,args.seeds,args.device)
        for path,digest in sources.items():
            if d.sha(path)!=digest:raise ValueError('Source/input changed during run: '+path)
        status.update(status='completed',completed_at=datetime.now().astimezone().isoformat(),
            current_dataset=None,student_runs=len(list((HERE/'students').glob('*/seed*/*/*/result.json'))))
    except Exception as exc:
        status.update(status='failed',error=repr(exc));d.write(HERE/'status.json',status);raise
    d.write(HERE/'status.json',status)
    print(json.dumps(status),flush=True)


if __name__=='__main__':main()
