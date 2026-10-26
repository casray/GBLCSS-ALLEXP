"""Resume external controls; preserve completed R2 fits and original source hashes."""
import importlib.util
import os
import traceback
import torch
import run as r

if __name__=='__main__':
    if (r.ROOT/'complete.json').exists():raise SystemExit('Already complete')
    assert (r.ROOT/'r2_complete.json').exists()
    torch.set_num_threads(4)
    if torch.cuda.mem_get_info()[0]<4*2**30:raise RuntimeError('GPU busy')
    with (r.ROOT/'run.lock').open('x') as lock:lock.write(str(os.getpid()))
    try:
        r.INPUTS.update(r.f.read(r.ROOT/'inputs.json'))
        frozen=r.f.read(r.ROOT/'sources.json')
        for path,digest in frozen.items():assert r.sha(path)==digest,path
        r.save(r.ROOT/'recovery_sources.json',{str(p):r.sha(p) for p in [r.ROOT/'external_v2.py',r.ROOT/'recover.py']})
        spec=importlib.util.spec_from_file_location('external_recovered',r.ROOT/'external_v2.py')
        external=importlib.util.module_from_spec(spec);spec.loader.exec_module(external)
        external.run_external()
        r.save(r.ROOT/'inputs.json',r.INPUTS)
        from analyze import analyze
        analyze()
    except Exception:
        r.save(r.ROOT/'recovery_failure.json',dict(traceback=traceback.format_exc()));raise
    finally:(r.ROOT/'run.lock').unlink(missing_ok=True)
