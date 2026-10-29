"""Read-only bridge and new-root-only artifact helpers."""
from pathlib import Path
import sys,os,time,json,hashlib,shutil,importlib.util,csv,gc
import numpy as np
import torch
import torch.nn.functional as F

CODE=Path(__file__).resolve().parent
W=Path('C:/Users/Admin/Desktop/实验8去噪5.19-/gb_c1_ratio_success_20260609/results_ratio_c1_20260609')
sys.path.insert(1,str(W/'experiments/graphecl_gb_context_20261028'))
import runtime as old
CFG=json.loads((CODE/'config.json').read_text(encoding='utf-8-sig'))
ROOT=Path(CFG['runtime']);OLD=old.ROOT
old.art.ROOT=ROOT;old.down.ROOT=ROOT
read,save,save_tensor,sha=old.read,old.save,old.save_tensor,old.sha
load,seed_all,state_hash,ge=old.load,old.seed_all,old.state_hash,old.ge
torch.set_num_threads(CFG['threads'])


def status(stage,**kw):
    data=dict(stage=stage,pid=os.getpid(),time=time.strftime('%Y-%m-%d %H:%M:%S'),**kw)
    save(ROOT/'status.json',data);print(json.dumps(data,ensure_ascii=False),flush=True)


def source_ssl(d,s,arm='ecl'):return OLD/'formal/ssl'/d/f'seed{s}'/arm
def source_head(d,s,c,arm='ecl',head='fixed'):return OLD/'formal/heads'/d/f'seed{s}'/arm/c/head
def source_assignment(d,s,arm='ecl'):return OLD/'formal/targets'/d/f'seed{s}'/arm/'support32.pt'


def prepare():
    ROOT.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(ROOT).free<CFG['minimum_free_gib']*1024**3:raise RuntimeError('Insufficient free disk')
    if (ROOT/'receipt.json').exists():
        receipt=read(ROOT/'receipt.json');old.verify(receipt['signature']);return receipt
    old_receipt=read(OLD/'receipt.json');old.verify(old_receipt['signature'])
    files=set(CODE.glob('*.py'))|{CODE/'config.json',CODE/'PLAN.md',OLD/'receipt.json'}
    files.update(Path(p) for p in old_receipt['signature'])
    for d in CFG['datasets']:
        for s in CFG['seeds']:
            for arm in ['ecl','gb100']:
                folder=source_ssl(d,s,arm)
                for name,digest in read(folder/'complete.json')['files'].items():
                    if sha(folder/name)!=digest:raise RuntimeError('Old SSL artifact mismatch')
                files.update([folder/'complete.json',folder/'model.pt',folder/'H.pt',source_assignment(d,s,arm)])
                for c in CFG['diagnostic_conditions']:
                    for head in (['fixed','CE'] if arm=='ecl' else ['fixed','own']):
                        folder=source_head(d,s,c,arm,head)
                        for name,digest in read(folder/'complete.json')['files'].items():
                            if sha(folder/name)!=digest:raise RuntimeError('Old head artifact mismatch')
                            files.add(folder/name)
                        files.add(folder/'complete.json')
    r=dict(config=CFG,signature={str(p.resolve()):sha(p) for p in sorted(files)},created=time.strftime('%Y-%m-%d %H:%M:%S'))
    save(ROOT/'receipt.json',r);return r


def model_data(d,s):
    data=old.features(d,s);cfg=old.dataset_config(d);x=data['x'].cuda()
    edge=ge.with_loops(data['edge_index'].cuda(),len(x));adj=ge.adjacency(edge,len(x),x.dtype)
    return cfg,x,edge,adj


def original_model(d,s,cfg,x):
    model=ge.Model(x.shape[1],cfg).cuda();model.load_state_dict(load(source_ssl(d,s)/'model.pt'));return model


def support(folder,h,ids,d,s):
    return old.down._support_assignment(folder,F.normalize(h.float(),dim=1),torch.as_tensor(ids).long(),d,s,CFG['support']).numpy()


def fit_head(folder,h,target,data,d,s,c,arm):
    h=F.normalize(h.float(),dim=1);target=torch.as_tensor(target).float()
    seed_all(s);model=torch.nn.Linear(h.shape[1],data['classes']);initial={k:v.clone() for k,v in model.state_dict().items()}
    expected=dict(dataset=d,seed=s,condition=c,arm=arm,initial_hash=state_hash(initial),h_hash=state_hash({'H':h}),target_hash=state_hash({'target':target}),
                  epochs=CFG['head_epochs'],lr=CFG['head_lr'],weight_decay=CFG['head_weight_decay'],receipt_sha256=sha(ROOT/'receipt.json'))
    return old.fit_linear(folder,h,target,data,initial,expected)


def score(folder,lab,data):
    p=load(folder/'probabilities.pt').double().numpy();pred=p.argmax(1);te=lab['test'].numpy();y=lab['y'].numpy();classes=data['classes']
    conf=np.bincount(y[te]*classes+pred[te],minlength=classes*classes).reshape(classes,classes)
    den=conf.sum(0)+conf.sum(1);f1=np.divide(2*np.diag(conf),den,out=np.zeros(classes,dtype=float),where=den>0).mean()
    t=load(folder/'target.pt').double().numpy();yt=y[data['train'].numpy()]
    return dict(accuracy=float((pred[te]==y[te]).mean()),macro_f1=float(f1),validation=float((pred[data['valid'].numpy()]==data['valid_y'].numpy()).mean()),
                brier=float(np.mean((t*t).sum(1)-2*t[np.arange(len(t)),yt]+1)),source=str(folder))


def csv_write(path,rows):
    if not rows:return
    old.art.inside(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def verify_head(folder,h):
    p=load(folder/'probabilities.pt');w=load(folder/'student.pt');m=torch.nn.Linear(h.shape[1],p.shape[1]).cuda();m.load_state_dict(w)
    with torch.no_grad():q=m(F.normalize(h.float(),dim=1).cuda()).softmax(1).cpu()
    if not torch.equal(p,q):raise RuntimeError('GPU head replay differs: '+str(folder))
    return dict(source=str(folder),gpu_bitwise=True)
