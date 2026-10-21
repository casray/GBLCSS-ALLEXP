"""Compare a completed short native worker with the actual legacy train_tfr AST."""
import argparse
import ast
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import dropout_edge
from threadpoolctl import threadpool_limits
import yaml


def main(root,source):
    root,source=Path(root),Path(source)
    sys.path.insert(0,str(source))
    from tfr_core import load_reference_classes
    from legacy_api import protocol_labels,set_seed
    from experiment import atomic_json
    config=json.loads((root/'config.json').read_text(encoding='utf8'))
    path=root/'results/citeseer__u0p6__seed0__tfr_P_ind.json'
    result=json.loads(path.read_text(encoding='utf8'))
    meta=json.loads((root/'cache/citeseer/metadata.json').read_text(encoding='utf8'))
    old=Path(meta['old_directory'])
    y=np.load(old/'y.npy');x=np.load(old/'x.npy',mmap_mode='r')
    tr,va,te,noisy=protocol_labels(y,0,'u0.6',meta['C'])
    data=SimpleNamespace(name='citeseer',x=torch.from_numpy(np.array(x)).cuda(),
        y=torch.from_numpy(y).cuda(),noisy_y=torch.from_numpy(noisy).cuda(),
        edge_index=torch.from_numpy(np.load(root/'cache/citeseer/edge_index.npy')).cuda(),
        train_index=torch.from_numpy(tr).cuda(),val_index=torch.from_numpy(va).cuda(),
        num_nodes=len(y),num_classes=meta['C'])
    refpath=source/'robust/run_screen.py'
    tree=ast.parse(refpath.read_text(encoding='utf8'))
    train=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train_tfr')
    history=[]
    def log_epoch(handle,phase,epoch,loss,valid):
        history.append(dict(epoch=epoch,loss=loss,val=valid))
    namespace=dict(torch=torch,F=F,np=np,yaml=yaml,ROOT=source/'robust',
        tfr_classes=load_reference_classes,deepcopy=deepcopy,dropout_edge=dropout_edge,log_epoch=log_epoch)
    exec(compile(ast.Module(body=[train],type_ignores=[]),str(refpath),'exec'),namespace)
    torch.set_num_threads(config['cpu_threads']);set_seed(0)
    model,probabilities,best_epoch,_=namespace['train_tfr'](data,config['tfr']['epochs'],None)
    with np.load(root/'results'/result['prediction_file']) as saved:
        max_probability_delta=float(np.max(np.abs(probabilities.detach().cpu().numpy()[te]-saved['probabilities'])))
    state=torch.load(root/result['checkpoint_file'],map_location='cpu',weights_only=False)['state_dict']
    max_parameter_delta=max(float((value.detach().cpu()-state['base.'+name]).abs().max()) for name,value in model.state_dict().items())
    loss_delta=max(abs(a['loss']-b['train_loss']) for a,b in zip(history,result['validation_history']))
    val_delta=max(abs(a['val']-b['primary_val']) for a,b in zip(history,result['validation_history']))
    report=dict(dataset='citeseer',noise='u0.6',seed=0,epochs=config['tfr']['epochs'],
        actual_legacy_train_function=str(refpath),best_epochs=[best_epoch,result['best_primary_epoch']],
        max_probability_delta=max_probability_delta,max_parameter_delta=max_parameter_delta,
        max_loss_delta=loss_delta,max_val_delta=val_delta,
        scope='Short full-GPU native worker versus actual legacy training function; not formal performance evidence')
    passed=(best_epoch==result['best_primary_epoch'] and len(history)==len(result['validation_history'])
            and max_probability_delta<2e-5 and max_parameter_delta<2e-4 and loss_delta<2e-5 and val_delta<1e-7)
    report['status']='passed' if passed else 'failed'
    atomic_json(root/'NATIVE_TFR_REFERENCE_CHECK.json',report)
    print(json.dumps(report));assert passed,'Native worker comparison failed'


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-root',required=True);parser.add_argument('--source',required=True)
    args=parser.parse_args()
    with threadpool_limits(limits=4):main(args.run_root,args.source)
