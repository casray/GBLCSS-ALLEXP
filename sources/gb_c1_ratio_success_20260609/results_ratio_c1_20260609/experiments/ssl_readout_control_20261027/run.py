"""Unified readout of frozen GraphECL, PCA64 and attribute R2 features."""
import argparse,gc,json,os,sys,time,traceback,shutil
from pathlib import Path
from contextlib import contextmanager
import torch
import torch.nn.functional as F
CODE=Path(__file__).resolve().parent
W=Path('C:/Users/Admin/Desktop/实验8去噪5.19-/gb_c1_ratio_success_20260609/results_ratio_c1_20260609')
CFG=json.loads((CODE/'config.json').read_text(encoding='utf-8-sig'))
ROOT=Path(CFG['runtime']);PARENT=Path(CFG['parent_runtime'])
sys.path.insert(1,str(W/CFG['parent_code']))
import artifacts
import downstream as down
import probe
from data_bridge import load_features,load_supervised,state_hash
from operations import tasks,normalize_rows,fit_pca,project_pca
read,sha=artifacts.read,artifacts.sha
artifacts.ROOT=ROOT;probe.ROOT=ROOT
save,save_tensor=artifacts.save,artifacts.save_tensor

def load(p):return torch.load(p,map_location='cpu',weights_only=True)
def ref(receipt,ds,seed):return next(r for r in receipt['references'] if (r['dataset'],r['seed'])==(ds,seed))
def feature_dir(ds,seed):return ROOT/'features'/ds/f'seed{seed}'
def folder(task,prefix='formal'):
    return ROOT/prefix/'linear'/task['dataset']/f"seed{task['seed']}"/task['representation']/task['condition']/task['head']

def verify_signature(receipt):
    for p,digest in receipt['signature'].items():
        if sha(p)!=digest:raise RuntimeError('Frozen artifact changed: '+p)

def prepare():
    ROOT.mkdir(parents=True,exist_ok=True);dest=ROOT/'receipt.json'
    if dest.exists():
        receipt=read(dest);verify_signature(receipt);return receipt
    if shutil.disk_usage(ROOT).free<3*1024**3:raise RuntimeError('Insufficient free disk')
    assert artifacts.PROTOCOL['probe']['epochs']==CFG['epochs']==200
    assert CFG['lr']==.01 and CFG['weight_decay']==.0005 and CFG['bias']
    assert read(PARENT/'formal/final.json')['status']=='complete'
    parent=read(PARENT/'receipt.json');verify_signature(parent)
    files={str(p.resolve()) for p in CODE.glob('*.py')}|{str(CODE/'config.json'),str(PARENT/'receipt.json')}
    files.update(parent['signature'])
    for item in read(W/'research/ssl_backbone_csphoto_20261027/input_manifest.json')['files']:
        if sha(item['path'])!=item['sha256']:raise RuntimeError('Input changed')
        files.add(item['path'])
    refs=[]
    for ds in CFG['datasets']:
        for seed in CFG['seeds']:
            base=PARENT/'downstream'/ds/f'seed{seed}'/'graphecl'
            native=PARENT/'formal/linear'/ds/f'seed{seed}'/'graphecl'
            r=read(base/'r2/result.json');ep=900 if ds=='amazonpho' else 1000
            h=PARENT/'formal/ssl'/ds/f'seed{seed}'/'graphecl'/f'H_epoch{ep}.pt'
            encoder=base/'r2/encoder.pt';assignment=base/'r2/support32.pt'
            assert sha(h)==r['h_sha256'] and sha(encoder)==r['checkpoint_sha256']
            verify_signature({'signature':r['source_hashes']});files.update(r['source_hashes'])
            assert torch.equal(load(assignment),load(native/'support32.pt'))
            files.update(map(str,[h,encoder,assignment,native/'support32.pt',base/'r2/result.json']))
            rr=dict(dataset=ds,seed=seed,h=str(h),encoder=str(encoder),assignment=str(assignment),classes=r['classes'],native={},raw_r2={})
            for c in CFG['conditions']:
                rr['native'][c]={};rr['raw_r2'][c]={}
                for head in CFG['heads']:
                    n=native/c/head;b=base/'conditions'/c/head
                    record=read(n/'complete.json')
                    for name,digest in record['files'].items():
                        assert sha(n/name)==digest;files.add(str(n/name))
                    files.add(str(n/'complete.json'));result=read(b/'result.json')
                    for name,key in [('config.json','config.json_sha256'),('epochs.jsonl','epochs.jsonl_sha256'),('student.pt','student_sha256'),('probabilities.pt','probabilities_sha256'),('target.pt','target_file_sha256')]:
                        assert sha(b/name)==result[key];files.add(str(b/name))
                    files.add(str(b/'result.json'))
                    assert torch.equal(load(n/'target.pt'),load(b/'target.pt'))
                    rr['native'][c][head]=str(n);rr['raw_r2'][c][head]=str(b)
            refs.append(rr)
    receipt=dict(config=CFG,references=refs,tasks=tasks(CFG),signature={p:sha(p) for p in sorted(files)},created=time.strftime('%Y-%m-%d %H:%M:%S'))
    assert len(receipt['tasks'])==64 and len(refs)==4
    save(dest,receipt);return receipt

def features(receipt):
    for rr in receipt['references']:
        ds,seed=rr['dataset'],rr['seed'];dest=feature_dir(ds,seed)
        expected=dict(receipt_sha256=sha(ROOT/'receipt.json'),dataset=ds,seed=seed)
        if artifacts.reusable(dest,expected):continue
        if dest.exists() and any(dest.iterdir()):raise RuntimeError('Incomplete features retained: '+str(dest))
        h=normalize_rows(load(rr['h'])['H'].float())
        state=fit_pca(h,CFG['dimension']);pca=normalize_rows(project_pca(h,state).float())
        source=load_features(ds,seed);x=source['x']
        model=down._CORE.AlignedMLP(x.shape[1],rr['classes'],h.shape[1],hidden=64,dropout=.5).cuda().eval()
        enc=load(rr['encoder']);model.load_state_dict(enc);before=state_hash(model.state_dict())
        with torch.no_grad():raw=model.encode(x.cuda()).cpu()
        assert before==state_hash(model.state_dict())==state_hash(enc)
        unit=normalize_rows(raw)
        old=Path(rr['raw_r2']['uniform0.4']['CE']);oldstate=load(old/'student.pt')
        with torch.no_grad():replay=F.linear(raw.cuda(),oldstate['layers.1.weight'].cuda(),oldstate['layers.1.bias'].cuda()).softmax(1).cpu()
        error=float((replay-load(old/'probabilities.pt')).abs().max())
        if error>2e-6:raise RuntimeError('R2 extraction replay mismatch: '+str(error))
        stats=dict(dataset=ds,seed=seed,nodes=len(h),native_dimension=h.shape[1],dimension=64,pca_retained_fraction=state['retained_fraction'],r2_extraction_replay_error=error,encoder_hash=before,
                   raw_r2_norm=dict(min=float(raw.norm(dim=1).min()),mean=float(raw.norm(dim=1).mean()),max=float(raw.norm(dim=1).max())))
        for name,value in [('native',h),('pca64',pca),('r2_unit64',unit),('r2_raw64',raw),('pca_state',state)]:save_tensor(dest/(name+'.pt'),value)
        assert torch.equal(normalize_rows(project_pca(h,load(dest/'pca_state.pt')).float()),load(dest/'pca64.pt'))
        for z in [h,pca,unit]:
            norms=z.norm(dim=1);assert torch.isfinite(z).all() and torch.all((norms<1e-8)|((norms-1).abs()<1e-5))
        save(dest/'stats.json',stats)
        artifacts.complete(dest,expected,['native.pt','pca64.pt','r2_unit64.pt','r2_raw64.pt','pca_state.pt','stats.json'])
        print('FEATURES',json.dumps(stats),flush=True)
        del model,x,raw,replay,h,pca,unit,source;gc.collect();torch.cuda.empty_cache()
    hashes={str(p):sha(p) for rr in receipt['references'] for p in feature_dir(rr['dataset'],rr['seed']).iterdir() if p.is_file()}
    value=dict(receipt_sha256=sha(ROOT/'receipt.json'),signature=hashes)
    locked=ROOT/'features_locked.json'
    if locked.exists():assert read(locked)==value
    else:save(locked,value)

def fit(task,receipt,engineering=False):
    ds,seed,rep,c,head=(task[k] for k in ['dataset','seed','representation','condition','head'])
    rr=ref(receipt,ds,seed);fp=feature_dir(ds,seed)/(rep+'.pt');h=load(fp)
    assignment=load(rr['assignment']);data=load_supervised(ds,seed,c)
    assert assignment.shape==(len(h),) and torch.equal(data['train'],load_features(ds,seed)['train'])
    def fixed_assignment(base,features,train,dataset,seed_value,support):
        assert dataset==ds and seed_value==seed and support==32 and len(features)==len(assignment) and torch.equal(train,data['train'])
        return assignment
    original_callback,original_root=probe._support_assignment,probe.ROOT
    probe._support_assignment=fixed_assignment
    if engineering:probe.ROOT=ROOT/'engineering'
    try:dst=probe.fit(h,fp,ds,seed,rep,c,head,smoke=False)
    finally:probe._support_assignment=original_callback;probe.ROOT=original_root
    target_source=Path(rr['native'][c][head])/'target.pt'
    assert torch.equal(load(dst/'target.pt'),load(target_source))
    bind=dict(task=task,receipt_sha256=sha(ROOT/'receipt.json'),feature_path=str(fp),feature_sha256=sha(fp),target_source=str(target_source),target_source_sha256=sha(target_source),result_sha256=sha(dst/'result.json'))
    if (dst/'source_binding.json').exists():assert read(dst/'source_binding.json')==bind
    else:save(dst/'source_binding.json',bind)
    return dst

def smoke(receipt):
    features(receipt);checks=[]
    for ds in CFG['datasets']:
        for head in CFG['heads']:
            task=dict(dataset=ds,seed=0,representation='native',condition='uniform0.4',head=head)
            dst=fit(task,receipt,engineering=True);source=Path(ref(receipt,ds,0)['native']['uniform0.4'][head])
            error=float((load(dst/'probabilities.pt')-load(source/'probabilities.pt')).abs().max())
            a,b=read(dst/'result.json'),read(source/'result.json')
            assert error<=2e-6 and a['best_epoch']==b['best_epoch'] and a['initial_hash']==b['initial_hash']
            match=state_hash(load(dst/'student.pt'))==state_hash(load(source/'student.pt'))
            if not match:raise RuntimeError('Native model replay hash mismatch')
            checks.append(dict(task=task,probability_max_error=error,model_hash_match=match,best_epoch=a['best_epoch']))
            print('NATIVE_REPLAY',json.dumps(checks[-1]),flush=True)
    save(ROOT/'engineering/smoke_complete.json',dict(receipt_sha256=sha(ROOT/'receipt.json'),features_lock_sha256=sha(ROOT/'features_locked.json'),checks=checks))

def execute(receipt):
    gate=read(ROOT/'engineering/smoke_complete.json')
    assert gate['receipt_sha256']==sha(ROOT/'receipt.json') and gate['features_lock_sha256']==sha(ROOT/'features_locked.json')
    verify_signature(read(ROOT/'features_locked.json'));done=[];failures=[]
    for task in tasks(CFG):
        save(ROOT/'formal/status.json',dict(status='running',task=task,completed=len(done),total=64,failures=failures,pid=os.getpid()))
        try:
            dst=fit(task,receipt);done.append(task)
            print('COMPLETE',json.dumps(dict(task=task,result=read(dst/'result.json'))),flush=True)
        except Exception:
            error=dict(task=task,traceback=traceback.format_exc());failures.append(error)
            save(folder(task)/'failure.json',error);print(error['traceback'],flush=True)
        finally:gc.collect();torch.cuda.empty_cache()
    verify_signature(receipt);verify_signature(read(ROOT/'features_locked.json'))
    save(ROOT/'formal/fits_locked.json',dict(receipt_sha256=sha(ROOT/'receipt.json'),completed=done,failures=failures))
    from analyze import analyze
    analyze()

@contextmanager
def lock():
    import msvcrt
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'runner.lock').open('a+b') as f:
        f.seek(0);f.write(b'0');f.flush();f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        try:yield
        finally:f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','smoke','all']);args=p.parse_args()
    with lock():
        receipt=prepare();torch.set_num_threads(CFG['threads']);torch.use_deterministic_algorithms(True)
        if args.command=='prepare':print(json.dumps(dict(references=len(receipt['references']),tasks=len(receipt['tasks']),hashes=len(receipt['signature']))));return
        if not torch.cuda.is_available():raise RuntimeError('GPU unavailable')
        if args.command=='smoke':smoke(receipt)
        else:execute(receipt)

if __name__=='__main__':
    sys.modules['run']=sys.modules[__name__]
    main()
