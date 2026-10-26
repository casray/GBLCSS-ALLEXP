"""Fixed-protocol student-initialization controls, with immutable old inputs."""
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT/'config.json').read_text(encoding='utf-8'))
OLD = ROOT.parent/CONFIG['source_experiment']
sys.path.insert(0, str(OLD))
import fit_pipeline as f
import torch

# All writes by the reused fit function are redirected; protocol lookup stays at seed0.
f.ROOT = ROOT


def fixed_cell(name, student_seed, condition):
    base = OLD/'diagnostics/e1'/name/'seed0'
    return f.load(base/'geometry.pt'), f.load(base/f'{condition}.pt')


f.cell = fixed_cell


def old_folder(name, condition, arm):
    return OLD/'fits/E4'/name/'seed0'/condition/arm


def setup():
    torch.set_num_threads(CONFIG['training']['threads'])
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required, no CPU fallback')
    free,total = torch.cuda.mem_get_info()
    f.save_json(ROOT/'environment.json', dict(time=datetime.now().astimezone().isoformat(),pid=os.getpid(),
        python=sys.executable,torch=torch.__version__,device=torch.cuda.get_device_name(0),free_bytes=free,total_bytes=total))
    if free < CONFIG['training']['min_free_gpu_gb']*1024**3:
        raise RuntimeError('GPU memory occupied; leave pending, do not terminate other tasks')
    sources = [*ROOT.glob('*.py'),ROOT/'PLAN.md',ROOT/'config.json',Path(f.__file__),
        Path(f.r2.__file__),Path(sys.modules['core'].__file__),Path(sys.modules['gblcss.models'].__file__),
        Path(sys.modules['gblcss.data'].__file__),Path(sys.modules['run_diagnostics'].__file__)]
    manifest = {str(p.resolve()):f.sha(p) for p in sources}
    dest=ROOT/'fit_sources.json'
    if dest.exists() and f.read(dest)!=manifest:
        raise RuntimeError('Frozen source manifest differs; record amendment before resuming')
    if not dest.exists():
        f.save_json(dest,manifest)


def inputs(name, condition, arm):
    source = old_folder(name,condition,arm)
    record = f.read(source/'result.json')
    f.record_input(source/'result.json')
    target=f.load(source/'target.pt')
    if f.state_hash(dict(target=target))!=record['target_hash']:
        raise RuntimeError('Old target hash differs')
    encoder=Path(record['encoder']) if record['encoder'] else None
    if encoder and f.sha(encoder)!=record['encoder_sha256']:
        raise RuntimeError('Old encoder differs')
    return target,encoder,record


def smoke():
    if (ROOT/'smoke_complete.json').exists():
        return
    checks=[]
    for arm in CONFIG['arms']:
        t,enc,ref=inputs('cora','uniform0.4',arm)
        dest=ROOT/'smoke'/arm
        result=f.fit(dest,'cora',0,'uniform0.4',arm,t,3,encoder=enc,
                     frozen=arm=='C_head',all_nodes=arm=='A300')
        for key in ('initial_hash','initial_rng_hash','target_hash'):
            if result[key]!=ref[key]:
                raise RuntimeError('Historical bridge failed: '+key)
        old=[json.loads(s) for s in (old_folder('cora','uniform0.4',arm)/'epochs.jsonl').read_text().splitlines()][:3]
        new=[json.loads(s) for s in (dest/'epochs.jsonl').read_text().splitlines()]
        error=max(abs(a['loss']-b['loss']) for a,b in zip(old,new))
        if error>1e-6 or [r['validation_correct'] for r in old]!=[r['validation_correct'] for r in new]:
            raise RuntimeError('Historical loss/validation bridge failed')
        checks.append(dict(arm=arm,initialization_exact=True,max_loss_error=error,validation_exact=True))
    f.save_json(ROOT/'smoke_complete.json',dict(passed=True,checks=checks))


def main():
    if (ROOT/'fits_complete.json').exists():
        print('Fits already complete; do not restart',flush=True)
        return
    lock=ROOT/'fit.lock'
    with lock.open('x') as h:
        h.write(str(os.getpid()))
    try:
        setup()
        smoke()
        paired=[]
        for name in CONFIG['datasets']:
            for condition in CONFIG['conditions']:
                for arm,epochs in CONFIG['arms'].items():
                    target,encoder,ref=inputs(name,condition,arm)
                    dest=ROOT/'fits'/name/'protocol0/student1'/condition/arm
                    f.save_json(ROOT/'status.json',dict(status='running',dataset=name,condition=condition,arm=arm,pid=os.getpid()))
                    out=f.fit(dest,name,1,condition,arm,target,epochs,encoder=encoder,
                              frozen=arm=='C_head',all_nodes=arm=='A300')
                    if out['initial_hash']==ref['initial_hash']:
                        raise RuntimeError('Student initialization did not change')
                    paired.append(dict(dataset=name,condition=condition,arm=arm,protocol_seed=0,student_seed=1,
                        old_result=str(old_folder(name,condition,arm)/'result.json'),new_result=str(dest/'result.json'),
                        same_target=out['target_hash']==ref['target_hash'],same_encoder=out['encoder_sha256']==ref['encoder_sha256'],
                        changed_initialization=True))
        if not all(r['same_target'] and r['same_encoder'] for r in paired):
            raise RuntimeError('Pairing violated')
        f.save_json(ROOT/'pairing_audit.json',dict(passed=True,records=paired))
        f.save_json(ROOT/'fits_complete.json',dict(time=datetime.now().astimezone().isoformat(),fits=len(paired),
            protocol_seed=0,student_seed=1,results={r['new_result']:f.sha(Path(r['new_result'])) for r in paired}))
        f.save_json(ROOT/'status.json',dict(status='fits_complete',fits=len(paired)))
        print('40 FITS COMPLETE',flush=True)
    except Exception:
        failure=dict(time=datetime.now().astimezone().isoformat(),traceback=traceback.format_exc())
        f.save_json(ROOT/f'failure_{datetime.now():%Y%m%d_%H%M%S}.json',failure)
        f.save_json(ROOT/'status.json',dict(status='failed',**failure))
        raise
    finally:
        lock.unlink()


if __name__=='__main__':
    main()
