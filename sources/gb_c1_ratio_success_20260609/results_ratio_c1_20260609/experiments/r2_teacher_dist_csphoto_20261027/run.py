"""Append CS/Photo using unchanged five-seed DIST data and fitting kernels."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode=True
CODE=Path(__file__).resolve().parent
CONFIG=json.loads((CODE/'config.json').read_text(encoding='utf-8'))
ROOT=Path(CONFIG['runtime'])
spec=importlib.util.spec_from_file_location('csphoto_dist_base',CODE.parent/'r2_teacher_dist_five_seed_20261026/run.py')
base=importlib.util.module_from_spec(spec);sys.modules[spec.name]=base;spec.loader.exec_module(base)
base.CODE,base.CONFIG,base.ROOT=CODE,CONFIG,ROOT
p=base.p;p.ROOT,p.CONFIG=ROOT,CONFIG
base.fitter.ROOT=ROOT
torch=base.torch


def sources():
    files=[*CODE.glob('*.py'),CODE/'config.json',CODE/'PLAN.md',*CODE.glob('tests/*.py'),
        Path(base.__file__),CODE.parent/'r2_teacher_dist_five_seed_20261026/analysis.py',
        base.OLD/'protocol.py',base.OLD/'fit.py',base.OLD/'kernels.py',Path(sys.modules['core'].__file__),
        Path(p.common.__file__),Path(sys.modules['gblcss.models'].__file__),Path(sys.modules['gblcss.data'].__file__),
        base.EXP/'gb_strong_kd_screen_20261007/losses.py',*list((base.EXP/'gb_strong_kd_screen_20261007/vendor').glob('*.py'))]
    value={str(f.resolve()):p.sha(f) for f in files}
    if (ROOT/'sources.json').exists():assert p.read(ROOT/'sources.json')==value,'Frozen source drift'
    else:p.save(ROOT/'sources.json',value)


def prepare():
    if (ROOT/'teachers_frozen.json').exists():base.check_teachers();return
    refs,bridges={},[]
    for n in CONFIG['datasets']:
        for s in CONFIG['seeds']:
            src=base.source(s);assert (src/'complete.json').exists()
            freeze=p.read(p.track(src/'selections_frozen.json'))
            manifest=p.read(p.track(src/'reuse_manifest.json'))
            assert p.sha(src/'reuse_manifest.json')==freeze['reuse_manifest_sha256']
            for c in CONFIG['conditions']:
                v,geo=base.data(n,s,c)
                selection=src/'selections'/n/f'seed{s}'/f'{c}.json'
                chosen=p.read(p.track(selection))
                assert p.sha(selection)==freeze['choices'][selection.relative_to(src).as_posix()]
                for route,method in CONFIG['teacher_routes'].items():
                    key=chosen['choices'][method];ref=manifest[key];folder=Path(ref['folder'])
                    result=p.read(p.track(folder/'result.json'))
                    assert p.sha(folder/'result.json')==ref['result_sha256']
                    assert (ref['dataset'],ref['seed'],ref['condition'])==(n,s,c)
                    assert result['status']=='complete' and result['epoch']==200
                    assert result['input_hash']==p.state_hash({k:x for k,x in v.items() if isinstance(x,torch.Tensor)})
                    for file,field in [('probabilities.pt','probabilities_sha256'),('checkpoint.pt','checkpoint_sha256')]:
                        assert p.sha(p.track(folder/file))==result[field]
                    prob=p.load(folder/'probabilities.pt')
                    assert prob.shape==(len(v['x']),v['classes']) and torch.isfinite(prob).all() and (prob>=0).all()
                    assert torch.allclose(prob.sum(1),torch.ones(len(prob)),atol=1e-6)
                    model=base.reload_teacher(folder,v,device='cpu')
                    with torch.inference_mode():actual=model(v['x']).softmax(1)
                    diff=float((actual-prob).abs().max())
                    assert diff<3e-6 and torch.equal(actual.argmax(1),prob.argmax(1))
                    assert int((prob[v['valid']].argmax(1)==v['valid_y']).sum())==result['valid_correct']
                    refs[f'{base.cid(n,s,c)}/{route}']=dict(dataset=n,seed=s,condition=c,route=route,
                        source=str(src),selected_key=key,selected_arm=ref['arm'],folder=str(folder.resolve()),
                        selection_path=str(selection.resolve()),selection_sha256=p.sha(selection),
                        result_sha256=p.sha(folder/'result.json'),checkpoint_sha256=result['checkpoint_sha256'],
                        probabilities_sha256=result['probabilities_sha256'])
                    bridges.append(dict(dataset=n,seed=s,condition=c,route=route,cpu_reload_max_error=diff))
                    del model,actual
            base.status('CPU_TEACHERS_VERIFIED',dataset=n,seed=s,count=len(refs))
    assert len(refs)==60
    p.save(ROOT/'teacher_refs.json',refs);p.csv_save(ROOT/'teacher_bridges.csv',bridges)
    p.save(ROOT/'historical_students.json',[])
    p.save(ROOT/'teachers_frozen.json',dict(time=p.now(),refs_sha256=p.sha(ROOT/'teacher_refs.json'),count=60))
    p.save(ROOT/'preflight_complete.json',dict(time=p.now(),teachers=60,cpu_only=True,test_metrics_read=False))


def smoke():
    if (ROOT/'smoke_complete.json').exists():return
    checks=[]
    for n in CONFIG['datasets']:
        cpu,_=base.data(n,0,'uniform0.4');v=base.gpu(cpu)
        ref=p.read(ROOT/'teacher_refs.json')[f'{n}/seed0/uniform0.4/R2_FULL']
        probs=p.load(Path(ref['folder'])/'probabilities.pt').cuda().detach()
        for label,steps in [('continuous',[2]),('resumed',[1,2])]:
            folder=ROOT/'smoke'/n/label
            for ep in steps:
                base.fitter.fit(folder,'DIST',v,base.own(cpu),dict(teacher=probs),seed=0,until=ep,
                    metadata=dict(dataset=n,condition='uniform0.4',arm='smoke',teacher_probability_sha256=ref['probabilities_sha256']))
        a,b=[torch.load(ROOT/'smoke'/n/k/'resume.pt',map_location='cpu',weights_only=False) for k in ('continuous','resumed')]
        assert all(p.tree_hash(a[k])==p.tree_hash(b[k]) for k in ('model','optimizer','rng','best_state'))
        checks.append(dict(dataset=n,resume_exact=True))
    p.save(ROOT/'smoke_complete.json',dict(time=p.now(),checks=checks,trajectories=4,updates=8,
        device=torch.cuda.get_device_name(),free_total=list(torch.cuda.mem_get_info())))


def train():
    if (ROOT/'selections_frozen.json').exists():base.check_freeze();return
    assert (ROOT/'smoke_complete.json').exists();base.check_teachers()
    teachers=p.read(ROOT/'teacher_refs.json');students=[]
    for n in CONFIG['datasets']:
        for s in CONFIG['seeds']:
            for c in CONFIG['conditions']:
                cpu,_=base.data(n,s,c);v=base.gpu(cpu);choices={}
                for route in CONFIG['teacher_routes']:
                    ref=teachers[f'{base.cid(n,s,c)}/{route}'];folder=ROOT/'fits'/base.cid(n,s,c)/route
                    prob=p.load(Path(ref['folder'])/'probabilities.pt').cuda().detach()
                    base.status('DIST_STUDENT',dataset=n,seed=s,condition=c,route=route,complete=len(students))
                    result=base.fitter.fit(folder,'DIST',v,base.own(cpu),dict(teacher=prob),seed=s,until=200,
                        metadata=dict(dataset=n,condition=c,arm=route,stage='STUDENT',partition='TEACHER_FIXED',
                            teacher_folder=ref['folder'],teacher_probability_sha256=ref['probabilities_sha256']))
                    students.append(dict(dataset=n,seed=s,condition=c,route=route,folder=str(folder.resolve()),historical=False,
                        result_sha256=p.sha(folder/'result.json'),checkpoint_sha256=result['checkpoint_sha256'],
                        probabilities_sha256=result['probabilities_sha256'],initial_hash=result['initial_hash'],initial_rng_hash=result['initial_rng_hash']))
                    choices[route]=str(folder.resolve())
                cp=ROOT/'selections'/n/f'seed{s}'/f'{c}.json'
                if cp.exists():assert p.read(cp)['choices']==choices
                else:p.save(cp,dict(time=p.now(),choices=choices))
            torch.cuda.empty_cache()
    assert len(students)==60
    p.save(ROOT/'student_refs.json',students)
    p.save(ROOT/'selections_frozen.json',dict(time=p.now(),teacher_freeze_sha256=p.sha(ROOT/'teachers_frozen.json'),
        student_refs_sha256=p.sha(ROOT/'student_refs.json'),choices={p.canonical(x):p.sha(x) for x in (ROOT/'selections').rglob('*.json')}))
    base.status('ALL_CHOICES_FROZEN',students=60)


base.freeze_sources=sources
base.prepare=prepare
base.smoke=smoke
base.train=train


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['preflight','all'],default='all')
    args=parser.parse_args();ROOT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1 if args.stage=='preflight' else 4)
    if (ROOT/'complete.json').exists():print('Already complete');return
    lock=ROOT/'run.lock'
    with lock.open('x') as f:json.dump(dict(pid=__import__('os').getpid(),time=p.now()),f)
    try:
        if (ROOT/'inputs.json').exists():p.INPUTS.update(p.read(ROOT/'inputs.json'))
        sources();p.disk_guard(1024**3);prepare()
        if args.stage=='preflight':
            assert not torch.cuda.is_initialized(),'CPU preflight must not initialize CUDA'
            return
        assert torch.cuda.is_available() and torch.cuda.mem_get_info()[0]>=CONFIG['gpu_queue']['minimum_free_gib']*1024**3
        smoke();train();base.analyze()
        p.save(ROOT/'fits_complete.json',dict(time=p.now(),students=60,updates=12000,status='ANALYZED_AUDIT_PENDING'))
    except Exception:
        import traceback
        failure=dict(time=p.now(),traceback=traceback.format_exc())
        p.save(ROOT/'failure_latest.json',failure);p.append(ROOT/'failures.jsonl',failure);raise
    finally:lock.unlink()


if __name__=='__main__':main()
