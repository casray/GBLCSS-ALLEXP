"""Confined artifacts and read-only bridges to the frozen experiment inputs."""
from pathlib import Path
import importlib.util
import json,sys,os,time,hashlib,random,copy
from contextlib import contextmanager
import numpy as np
import torch

CODE=Path(__file__).resolve().parent
W=Path('C:/Users/Admin/Desktop/实验8去噪5.19-/gb_c1_ratio_success_20260609/results_ratio_c1_20260609')
CFG=json.loads((CODE/'config.json').read_text(encoding='utf-8-sig'));ROOT=Path(CFG['runtime'])
SCREEN=W/'experiments/ssl_backbone_screen_20261026';CSCODE=W/'experiments/ssl_backbone_csphoto_20261027'
sys.path[1:1]=[str(SCREEN),str(W/'experiments/ssl_readout_factorial_20261027')]
import artifacts as art
import downstream as down
import data_bridge as legacy_bridge
from fit import fit_linear
art.ROOT=ROOT;down.ROOT=ROOT
read,save,save_tensor,sha=art.read,art.save,art.save_tensor,art.sha
gvec=legacy_bridge._old
seed_all,state_hash=gvec.seed_all,gvec.state_hash
ge=gvec.import_file('context_ecl_model',CSCODE/'graphecl.py')
tree=gvec.import_file('context_tree',W/'experiments/neighbor_preserving_ball_20261022/generator.py')
ROMAN=W/'experiments/heterophily_mechanism_20261014/data/roman_empire.npz'
P1=W/'research/ssl_backbone_selection_20261026/protocol.json'
P2=W/'research/ssl_backbone_csphoto_20261027/protocol.json'

def load(path):return torch.load(path,map_location='cpu',weights_only=True)
def dataset_config(d):
    if d=='roman':return dict(CFG['roman'])
    p=read(P2 if d in ('coauthorcs','amazonpho') else P1)['graphecl']
    cfg={k:p[k] for k in ['temperature','lam','lambda_loss','num_mlp']};cfg.update(p['dataset_configs'][d]);return cfg
def loss_kwargs(cfg):return dict(tau=cfg['temperature'],lam=cfg['lam'],lambda_loss=cfg['lambda_loss'])
def old_root(d):return Path('D:/codex_temp')/('ssl_backbone_csphoto_20261027' if d in ('coauthorcs','amazonpho') else 'ssl_backbone_screen_20261026')
def old_ssl(d,s):return old_root(d)/'formal/ssl'/d/f'seed{s}'/'graphecl'
def old_head(d,s,c,head):return old_root(d)/'formal/linear'/d/f'seed{s}'/'graphecl'/c/head
def ssl_dir(d,s,arm,stage='formal'):return ROOT/stage/'ssl'/d/f'seed{s}'/arm
def head_dir(d,s,arm,c,head,stage='formal'):return ROOT/stage/'heads'/d/f'seed{s}'/arm/c/head

def features(d,s):
    if d=='roman':
        from torch_geometric.utils import to_undirected
        with np.load(ROMAN,allow_pickle=False) as f:
            x=torch.from_numpy(f['node_features'].copy()).float()
            edge=to_undirected(torch.from_numpy(f['edges'].copy()).long().T,num_nodes=len(x))
    else:
        data=gvec.load_features(d,s);x,edge=data['x'],data['edge_index']
    return dict(x=x/x.abs().sum(1,keepdim=True).clamp_min(1e-12),edge_index=edge)

def supervised(d,s,c):
    if d!='roman':return gvec.load_supervised(d,s,c)
    from gblcss.data import _independent_transition
    with np.load(ROMAN,allow_pickle=False) as f:
        train=torch.from_numpy(np.flatnonzero(f['train_masks'][s]));valid=torch.from_numpy(np.flatnonzero(f['val_masks'][s]))
        labels=f['node_labels'];y=torch.from_numpy(labels[train.numpy()].copy()).long()
        vy=torch.from_numpy(labels[valid.numpy()].copy()).long()
    classes=int(y.max())+1;kind='uniform' if c.startswith('uniform') else 'pair';rate=float(c[len(kind):])
    transition=_independent_transition(classes,rate,kind);rng=np.random.RandomState(s)
    observed=torch.tensor([int(np.flatnonzero(rng.multinomial(1,transition[int(v)],1)[0])[0]) for v in y],dtype=torch.long)
    return dict(train=train,valid=valid,classes=classes,observed=observed,valid_y=vy)

def labels_after_lock(d,s):
    if not (ROOT/'formal/fits_locked.json').exists():raise RuntimeError('Test labels sealed until fitting lock')
    if d!='roman':return gvec.analysis_labels(d,s)
    with np.load(ROMAN,allow_pickle=False) as f:
        return dict(y=torch.from_numpy(f['node_labels'].copy()).long(),**{key:torch.from_numpy(np.flatnonzero(f[src][s])) for key,src in [('train','train_masks'),('valid','val_masks'),('test','test_masks')]})

def verify(signature):
    for p,h in signature.items():
        if sha(p)!=h:raise RuntimeError('Source changed: '+p)

def prepare():
    import shutil
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'receipt.json').exists():
        r=read(ROOT/'receipt.json');verify(r['signature']);return r
    if shutil.disk_usage(ROOT).free<CFG['minimum_free_gib']*1024**3:raise RuntimeError('Insufficient runtime disk space')
    files=set(CODE.glob('*.py'))|{CODE/'config.json',P1,P2,ROMAN,Path(tree.__file__),Path(gvec.__file__),Path(down.__file__),Path(ge.__file__),W/'experiments/ssl_readout_factorial_20261027/fit.py'}
    references=[]
    for d in CFG['primary_datasets']:
        for s in CFG['seeds']:
            data=gvec.load_features(d,s)
            for key,value in data['metadata']['sources'].items():
                if not key.endswith('_sha256') and Path(value).is_file():files.add(Path(value))
            for c in CFG['conditions']:files.add(gvec.E1/d/f'seed{s}'/(c+'.pt'))
            base=old_ssl(d,s);m=read(base/'complete.json');files.add(base/'complete.json')
            for name,h in m['files'].items():
                p=base/name
                if sha(p)!=h:raise RuntimeError('Legacy SSL changed: '+str(p))
                files.add(p)
            for c in CFG['conditions']:
                for head in ['CE','SUPPORT32']:
                    p=old_head(d,s,c,head);rec=read(p/'complete.json');files.add(p/'complete.json')
                    for name,h in rec['files'].items():
                        if sha(p/name)!=h:raise RuntimeError('Legacy head changed')
                        files.add(p/name)
            references.append(dict(dataset=d,seed=s,old_ssl=str(base),input_hash=state_hash({'x':data['x'],'edge':data['edge_index']})))
    # Imported source dependencies are included even when loaded from the parent code tree.
    for module in list(sys.modules.values()):
        p=getattr(module,'__file__',None)
        if p and Path(p).suffix=='.py' and Path(p).is_relative_to(W.parents[1]):files.add(Path(p))
    r=dict(config=CFG,references=references,signature={str(p.resolve()):sha(p) for p in sorted(files)},created=time.strftime('%Y-%m-%d %H:%M:%S'))
    save(ROOT/'receipt.json',r);return r

def rng_state():return dict(cpu=torch.get_rng_state(),gpu=torch.cuda.get_rng_state_all(),numpy=np.random.get_state(),python=random.getstate())
def restore_rng(r):
    torch.set_rng_state(r['cpu']);torch.cuda.set_rng_state_all(r['gpu']);np.random.set_state(r['numpy']);random.setstate(r['python'])
def restore_snapshot(model,optimizer,checkpoint):
    model.load_state_dict(checkpoint['model'])
    # Adam's CPU step tensors can alias the supplied state after load_state_dict.
    optimizer.load_state_dict(copy.deepcopy(checkpoint['optimizer']))
    restore_rng(checkpoint['rng'])
@contextmanager
def single_instance():
    import msvcrt
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'runner.lock').open('a+b') as f:
        f.seek(0);f.write(b'0');f.flush();f.seek(0)
        try:msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        except OSError as exc:raise RuntimeError('Existing runner holds experiment lock') from exc
        try:yield
        finally:f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
