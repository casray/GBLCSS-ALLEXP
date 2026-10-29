"""Registered R2 feature scale x classifier initialization experiment."""
import argparse,gc,importlib.util,json,os,sys,time,traceback
from pathlib import Path
from contextlib import contextmanager
import torch
CODE=Path(__file__).resolve().parent
W=Path('C:/Users/Admin/Desktop/实验8去噪5.19-/gb_c1_ratio_success_20260609/results_ratio_c1_20260609')
CFG=json.loads((CODE/'config.json').read_text(encoding='utf-8-sig'))
ROOT=Path(CFG['runtime']);PARENT=Path(CFG['parent_runtime']);BCODE=W/CFG['parent_code']
sys.path.insert(1,str(BCODE))
spec=importlib.util.spec_from_file_location('_readout_parent',BCODE/'run.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
import artifacts
from data_bridge import load_supervised,seed_all,state_hash
from contract import tasks
from fit import fit_linear
artifacts.ROOT=ROOT
read,sha,save,save_tensor,load=artifacts.read,artifacts.sha,artifacts.save,artifacts.save_tensor,b.load
verify_signature=b.verify_signature;down=b.down

def ref(receipt,ds,seed):return next(r for r in receipt['references'] if (r['dataset'],r['seed'])==(ds,seed))
def init_dir(ds,seed):return ROOT/'initializations'/ds/f'seed{seed}'
def folder(task,prefix='formal'):
    return ROOT/prefix/task['dataset']/f"seed{task['seed']}"/(task['scale']+'__'+task['initialization'])/task['condition']/task['head']

def prepare():
    ROOT.mkdir(parents=True,exist_ok=True);dest=ROOT/'receipt.json'
    if dest.exists():
        receipt=read(dest);verify_signature(receipt);return receipt
    assert read(PARENT/'formal/final.json')['status']=='complete'
    parent=read(PARENT/'receipt.json');flock=read(PARENT/'features_locked.json');verify_signature(parent);verify_signature(flock)
    files=set(parent['signature'])|set(flock['signature'])|{str(p.resolve()) for p in CODE.glob('*.py')}
    files.update([str(CODE/'config.json'),str(PARENT/'receipt.json'),str(PARENT/'features_locked.json'),str(PARENT/'formal/final.json')])
    refs=[]
    for rr in parent['references']:
        ds,seed=rr['dataset'],rr['seed'];r=dict(rr)
        r['features']={scale:str(PARENT/'features'/ds/f'seed{seed}'/name) for scale,name in [('raw','r2_raw64.pt'),('unit','r2_unit64.pt')]}
        r['unit_unified']={}
        assert torch.equal(torch.nn.functional.normalize(load(r['features']['raw']),dim=1),load(r['features']['unit']))
        for c in CFG['conditions']:
            r['unit_unified'][c]={}
            for head in CFG['heads']:
                path=PARENT/'formal/linear'/ds/f'seed{seed}'/'r2_unit64'/c/head
                record=read(path/'complete.json')
                for name,digest in record['files'].items():assert sha(path/name)==digest;files.add(str(path/name))
                files.update([str(path/'complete.json'),str(path/'source_binding.json')])
                assert torch.equal(load(path/'target.pt'),load(Path(rr['raw_r2'][c][head])/'target.pt'))
                r['unit_unified'][c][head]=str(path)
        refs.append(r)
    receipt=dict(config=CFG,references=refs,tasks=tasks(CFG),signature={p:sha(p) for p in sorted(files)},created=time.strftime('%Y-%m-%d %H:%M:%S'))
    assert len(receipt['tasks'])==64 and len(refs)==4
    save(dest,receipt);return receipt

def initializations(receipt):
    for rr in receipt['references']:
        ds,seed=rr['dataset'],rr['seed'];dest=init_dir(ds,seed)
        expected=dict(receipt_sha256=sha(ROOT/'receipt.json'),dataset=ds,seed=seed)
        if artifacts.reusable(dest,expected):continue
        if dest.exists() and any(dest.iterdir()):raise RuntimeError('Incomplete initialization retained')
        enc=load(rr['encoder']);features=enc['backbone.layers.0.weight'].shape[1]
        seed_all(seed)
        legacy=down._CORE.AlignedMLP(features,rr['classes'],enc['projection.weight'].shape[0],64,.5)
        down._copy_r2_encoder(legacy,enc)
        full_hash=state_hash(legacy.deployment_state());old=legacy.backbone.layers[-1].state_dict()
        seed_all(seed);unified=torch.nn.Linear(64,rr['classes']);new=unified.state_dict()
        for c in CFG['conditions']:
            for head in CFG['heads']:
                assert full_hash==read(Path(rr['raw_r2'][c][head])/'result.json')['initial_hash']
                assert state_hash(new)==read(Path(rr['unit_unified'][c][head])/'result.json')['initial_hash']
        save_tensor(dest/'legacy.pt',{k:v.detach().cpu().clone() for k,v in old.items()})
        save_tensor(dest/'unified.pt',{k:v.detach().cpu().clone() for k,v in new.items()})
        stats=dict(dataset=ds,seed=seed,legacy_deployment_initial_hash=full_hash,legacy_head_hash=state_hash(old),unified_head_hash=state_hash(new),max_initial_weight_difference=max(float((old[k]-new[k]).abs().max()) for k in old))
        assert stats['legacy_head_hash']!=stats['unified_head_hash']
        save(dest/'stats.json',stats);artifacts.complete(dest,expected,['legacy.pt','unified.pt','stats.json'])
        print('INITIALIZATIONS',json.dumps(stats),flush=True)
    hashes={str(p):sha(p) for rr in receipt['references'] for p in init_dir(rr['dataset'],rr['seed']).iterdir() if p.is_file()}
    value=dict(receipt_sha256=sha(ROOT/'receipt.json'),signature=hashes);path=ROOT/'initializations_locked.json'
    if path.exists():assert read(path)==value
    else:save(path,value)

def fit(task,receipt,prefix='formal'):
    ds,seed,scale,init,c,head=(task[k] for k in ['dataset','seed','scale','initialization','condition','head'])
    rr=ref(receipt,ds,seed);hpath=Path(rr['features'][scale]);h=load(hpath)
    initial_path=init_dir(ds,seed)/(init+'.pt');initial=load(initial_path)
    target_path=Path(rr['raw_r2'][c][head])/'target.pt';target=load(target_path)
    data=load_supervised(ds,seed,c)
    expected_target=torch.nn.functional.one_hot(data['observed'],data['classes']).float()
    if head=='SUPPORT32':expected_target=down.build_support32_targets(data['observed'],load(rr['assignment'])[data['train']],classes=data['classes'])
    assert torch.equal(target,expected_target)
    expected=dict(task,receipt_sha256=sha(ROOT/'receipt.json'),feature_path=str(hpath),feature_sha256=sha(hpath),initial_path=str(initial_path),initial_file_sha256=sha(initial_path),initial_hash=state_hash(initial),target_source=str(target_path),target_source_sha256=sha(target_path),epochs=CFG['epochs'],lr=CFG['lr'],weight_decay=CFG['weight_decay'],matmul_scope=CFG['matmul_scope'])
    return fit_linear(folder(task,prefix),h,target,data,initial,expected)

def smoke(receipt):
    initializations(receipt);checks=[]
    for rr in receipt['references']:
        for scale,init,key in [('raw','legacy','raw_r2'),('unit','unified','unit_unified')]:
            for head in CFG['heads']:
                task=dict(dataset=rr['dataset'],seed=rr['seed'],scale=scale,initialization=init,condition='uniform0.4',head=head)
                result=fit(task,receipt,'engineering');dst=folder(task,'engineering');src=Path(rr[key]['uniform0.4'][head])
                error=float((load(dst/'probabilities.pt')-load(src/'probabilities.pt')).abs().max())
                old=read(src/'result.json');prior=load(src/'student.pt');current=load(dst/'student.pt')
                if init=='legacy':prior={'weight':prior['layers.1.weight'],'bias':prior['layers.1.bias']}
                weight_error=max(float((current[k]-prior[k]).abs().max()) for k in current)
                a=[json.loads(x)['validation_correct'] for x in (dst/'epochs.jsonl').read_text().splitlines()]
                c=[json.loads(x)['validation_correct'] for x in (src/'epochs.jsonl').read_text().splitlines()]
                check=dict(task=task,probability_max_error=error,weight_max_error=weight_error,all_validation_counts_match=a==c,best_epoch_match=result['best_epoch']==old['best_epoch'])
                checks.append(check);save(ROOT/'engineering/replay_progress.json',checks);print('REPLAY',json.dumps(check),flush=True)
                if error>CFG['replay_probability_atol'] or a!=c or not check['best_epoch_match']:raise RuntimeError('Historical diagonal replay failed: '+json.dumps(check))
    save(ROOT/'engineering/smoke_complete.json',dict(receipt_sha256=sha(ROOT/'receipt.json'),initializations_lock_sha256=sha(ROOT/'initializations_locked.json'),checks=checks))

def execute(receipt):
    gate=read(ROOT/'engineering/smoke_complete.json')
    assert gate['receipt_sha256']==sha(ROOT/'receipt.json') and gate['initializations_lock_sha256']==sha(ROOT/'initializations_locked.json') and len(gate['checks'])==16
    verify_signature(read(ROOT/'initializations_locked.json'));done=[];failures=[]
    for task in tasks(CFG):
        save(ROOT/'formal/status.json',dict(status='running',completed=len(done),total=64,task=task,failures=failures,pid=os.getpid()))
        try:
            result=fit(task,receipt);done.append(task);print('COMPLETE',json.dumps(result),flush=True)
        except Exception:
            error=dict(task=task,traceback=traceback.format_exc());failures.append(error);save(folder(task)/'failure.json',error);print(error['traceback'],flush=True)
        finally:gc.collect();torch.cuda.empty_cache()
    verify_signature(receipt);verify_signature(read(ROOT/'initializations_locked.json'))
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
        if args.command=='prepare':print(json.dumps(dict(tasks=len(receipt['tasks']),references=len(receipt['references']),hashes=len(receipt['signature']))));return
        if not torch.cuda.is_available():raise RuntimeError('GPU unavailable')
        if args.command=='smoke':smoke(receipt)
        else:execute(receipt)

if __name__=='__main__':
    sys.modules['run']=sys.modules[__name__]
    main()
