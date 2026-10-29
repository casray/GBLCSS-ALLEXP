"""Frozen-R2 feature/target crossing. Reuse the signed parent fitting kernel."""
import argparse, gc, json, os, sys, time, traceback, shutil
from pathlib import Path
from contextlib import contextmanager
import torch
CODE=Path(__file__).resolve().parent
W=Path('C:/Users/Admin/Desktop/实验8去噪5.19-/gb_c1_ratio_success_20260609/results_ratio_c1_20260609')
CFG=json.loads((CODE/'config.json').read_text())
ROOT=Path(CFG['runtime']);PARENT=Path(CFG['parent_runtime'])
sys.path.insert(1,str(W/CFG['parent_code']))
import artifacts
import downstream as down
from data_bridge import load_features,load_supervised,state_hash
from contract import tasks,validate_target
read,sha=artifacts.read,artifacts.sha
# Only output ownership changes; all optimizer/model/selection code is reused.
artifacts.ROOT=ROOT
down.ROOT=ROOT
save=artifacts.save
def load(p):return torch.load(p,map_location='cpu',weights_only=True)
def arm(f,t):return f+'__target_'+t
def folder(task,prefix='formal'):
    return ROOT/prefix/task['dataset']/f"seed{task['seed']}"/arm(task['features'],task['targets'])/task['condition']
def ref(receipt,ds,seed,a):
    return next(r for r in receipt['references'] if (r['dataset'],r['seed'],r['arm'])==(ds,seed,a))

def verify_signature(receipt):
    for path,digest in receipt['signature'].items():
        if sha(path)!=digest:raise RuntimeError('Frozen source changed: '+path)

def prepare():
    ROOT.mkdir(parents=True,exist_ok=True)
    dest=ROOT/'receipt.json'
    if dest.exists():
        receipt=read(dest);verify_signature(receipt);return receipt
    if shutil.disk_usage(ROOT).free<3*1024**3:raise RuntimeError('Insufficient free disk')
    for key,ours in [('hidden','hidden'),('dropout','dropout'),('learning_rate','lr'),('weight_decay','weight_decay'),('head_epochs','epochs')]:
        if down.CONFIG['r2'][key]!=CFG[ours]:raise RuntimeError('Registered head protocol mismatch')
    if read(PARENT/'formal/final.json')['status']!='complete':raise RuntimeError('Parent fit incomplete')
    parent=read(PARENT/'receipt.json');verify_signature(parent)
    files={str(p.resolve()) for p in CODE.glob('*.py')}|{str((CODE/'config.json').resolve()),str(PARENT/'receipt.json')}
    files.update(parent['signature'])
    for entry in read(W/'research/ssl_backbone_csphoto_20261027/input_manifest.json')['files']:
        if sha(entry['path'])!=entry['sha256']:raise RuntimeError('Parent input changed')
        files.add(entry['path'])
    refs=[]
    for ds,pair in CFG['pairs'].items():
        for seed in CFG['seeds']:
            for a in pair:
                base=PARENT/'downstream'/ds/f'seed{seed}'/a
                result=read(base/'r2/result.json');ep=900 if ds=='amazonpho' and a=='graphecl' else 1000
                h=PARENT/'formal/ssl'/ds/f'seed{seed}'/a/f'H_epoch{ep}.pt'
                encoder=base/'r2/encoder.pt';assignment=base/'r2/support32.pt'
                if sha(encoder)!=result['checkpoint_sha256'] or sha(h)!=result['h_sha256']:raise RuntimeError('Parent encoder changed')
                record=dict(dataset=ds,seed=seed,arm=a,encoder=str(encoder),h=str(h),assignment=str(assignment),conditions={})
                files.update([str(encoder),str(h),str(assignment),str(base/'r2/result.json')])
                files.update(result['source_hashes'])
                for path,digest in result['source_hashes'].items():
                    if sha(path)!=digest:raise RuntimeError('Parent dependency changed: '+path)
                for c in CFG['conditions']:
                    record['conditions'][c]={}
                    for head in ['CE','SUPPORT32']:
                        d=base/'conditions'/c/head;r=read(d/'result.json')
                        for filename,key in [('config.json','config.json_sha256'),('epochs.jsonl','epochs.jsonl_sha256'),('student.pt','student_sha256'),('probabilities.pt','probabilities_sha256'),('target.pt','target_file_sha256')]:
                            if sha(d/filename)!=r[key]:raise RuntimeError('Parent classifier artifact changed')
                            files.add(str(d/filename))
                        files.add(str(d/'result.json'));record['conditions'][c][head]=str(d)
                refs.append(record)
    receipt=dict(config=CFG,references=refs,tasks=tasks(CFG),signature={p:sha(p) for p in sorted(files)},created=time.strftime('%Y-%m-%d %H:%M:%S'))
    assert len(receipt['tasks'])==32 and len(refs)==8
    save(dest,receipt);return receipt

def fit(task,receipt,prefix='formal'):
    ds,seed,f,t,c=(task[k] for k in ['dataset','seed','features','targets','condition'])
    fr,tr=ref(receipt,ds,seed,f),ref(receipt,ds,seed,t)
    data=load_supervised(ds,seed,c);x=load_features(ds,seed)['x']
    target=load(Path(tr['conditions'][c]['SUPPORT32'])/'target.pt')
    assignment=load(tr['assignment'])
    expected=down.build_support32_targets(data['observed'],assignment[data['train']],classes=data['classes'])
    validate_target(target,expected,len(data['train']),data['classes'])
    encoder=load(fr['encoder']);dst=folder(task,prefix)
    r=down._fit_classifier(dst,x=x,train=data['train'],valid=data['valid'],observed=data['observed'],valid_y=data['valid_y'],classes=data['classes'],encoder_state=encoder,target=target,name=ds,seed=seed,arm=arm(f,t),condition=c,head='SUPPORT32',h_path=Path(fr['h']),r2_sha=sha(fr['encoder']),hidden=64,dropout=.5,epochs=200,device=torch.device('cuda'))
    base=read(Path(fr['conditions'][c]['SUPPORT32'])/'result.json')
    for key in ['initial_hash','initial_rng_hash','frozen_encoder_hash_before','frozen_encoder_hash_after']:
        if r[key]!=base[key]:raise RuntimeError('Feature/init mismatch: '+key)
    save(dst/'source_binding.json',dict(task=task,feature_checkpoint=fr['encoder'],feature_sha256=sha(fr['encoder']),target_source=str(Path(tr['conditions'][c]['SUPPORT32'])/'target.pt'),target_sha256=sha(dst/'target.pt'),target_source_file_sha256=sha(Path(tr['conditions'][c]['SUPPORT32'])/'target.pt'),result_sha256=sha(dst/'result.json')))
    return r

def smoke(receipt):
    checks=[]
    for ds,pair in CFG['pairs'].items():
        for a in pair:
            task=dict(dataset=ds,seed=0,features=a,targets=a,condition='uniform0.4')
            fit(task,receipt,'engineering');dst=folder(task,'engineering');fr=ref(receipt,ds,0,a)
            source=Path(fr['conditions']['uniform0.4']['SUPPORT32'])
            diff=float((load(dst/'probabilities.pt')-load(source/'probabilities.pt')).abs().max())
            if diff>2e-6:raise RuntimeError('Diagonal prediction replay mismatch')
            if read(dst/'result.json')['best_epoch']!=read(source/'result.json')['best_epoch']:raise RuntimeError('Diagonal selection mismatch')
            model_match=state_hash(load(dst/'student.pt'))==state_hash(load(source/'student.pt'))
            checks.append(dict(task=task,probability_max_error=diff,model_hash_match=model_match))
            print('DIAGONAL_REPLAY',json.dumps(checks[-1]),flush=True)
    save(ROOT/'engineering/smoke_complete.json',dict(receipt_sha256=sha(ROOT/'receipt.json'),checks=checks))

def execute(receipt):
    gate=read(ROOT/'engineering/smoke_complete.json')
    if gate['receipt_sha256']!=sha(ROOT/'receipt.json'):raise RuntimeError('Smoke signature changed')
    done=[];failures=[]
    for task in tasks(CFG):
        save(ROOT/'formal/status.json',dict(status='running',task=task,completed=len(done),total=32,failures=failures,pid=os.getpid()))
        try:
            result=fit(task,receipt);done.append(task)
            print('COMPLETE',json.dumps(dict(task=task,valid=result['validation_accuracy'],seconds=result['seconds'])),flush=True)
        except Exception:
            error=dict(task=task,traceback=traceback.format_exc());failures.append(error)
            save(folder(task)/'failure.json',error);print(error['traceback'],flush=True)
        finally:gc.collect();torch.cuda.empty_cache()
    verify_signature(receipt)
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
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','smoke','all']);args=parser.parse_args()
    with lock():
        receipt=prepare();torch.set_num_threads(CFG['threads']);torch.use_deterministic_algorithms(True)
        if args.command=='prepare':print(json.dumps(dict(references=len(receipt['references']),tasks=len(receipt['tasks']),hashes=len(receipt['signature']))));return
        if not torch.cuda.is_available():raise RuntimeError('GPU unavailable')
        if args.command=='smoke':smoke(receipt)
        else:execute(receipt)

if __name__=='__main__':
    # Keep one shared module instance when analyze imports run after fitting.
    sys.modules['run']=sys.modules[__name__]
    main()
