"""Reduced schedule; all retained scientific updates use the frozen V2 kernel."""
import argparse
import gc
import traceback
import copy
import time
import numpy as np
from reuse import *
from schedule import PLAN, tasks

BUDGET_CODE = Path(__file__).resolve().parent
REV = ROOT/'budget_revision'


def prepare_budget():
    original.prepare()
    assert len(tasks()) == PLAN['ssl_endpoints']
    assert sum(6 if v['arm']=='ecl' else 9 for v in tasks()) == PLAN['heads']
    files = [*sorted(BUDGET_CODE.glob('*.py')), BUDGET_CODE/'plan.json']
    sig = {str(p): sha(p) for p in files}
    receipt = dict(parent_receipt_sha256=sha(ROOT/'receipt.json'), signature=sig, plan=PLAN)
    path = REV/'receipt.json'
    if path.exists():
        assert read(path) == receipt, 'Budget source/config changed after freeze'
    else:
        save(path, receipt)
    return receipt


def recursive_equal(a, b):
    if torch.is_tensor(a): return torch.is_tensor(b) and torch.equal(a, b)
    if isinstance(a, np.ndarray): return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict): return a.keys()==b.keys() and all(recursive_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)): return type(a)==type(b) and len(a)==len(b) and all(recursive_equal(x,y) for x,y in zip(a,b))
    return a == b


def smoke():
    prepare_budget()
    shared = ensure_midpoint('cora', 0, testing=True)
    reference = ROOT/'formal/warmup/cora/seed0'
    ck = torch.load(shared/'checkpoint.pt', map_location='cpu', weights_only=False)
    ref = torch.load(reference/'checkpoint.pt', map_location='cpu', weights_only=False)
    for key in ['expected','epoch','model','optimizer','rng']:
        assert recursive_equal(ck[key], ref[key]), 'Midpoint mismatch: '+key
    assert torch.equal(load(shared/'H.pt'), load(reference/'H.pt'))
    assert recursive_equal(load(shared/'context/context.pt'), load(reference/'context/context.pt'))
    # Exercise the actual cache-reuse path, including Photo's two-layer projector.
    imported = import_baseline('amazonpho', 0, testing=True)
    legacy = old_ssl('amazonpho', 0); terminal = dataset_config('amazonpho')['epochs']
    assert state_hash(load(imported/'model.pt')) == state_hash(load(legacy/f'model_epoch{terminal}.pt'))
    assert torch.equal(load(imported/'H.pt'), load(legacy/f'H_epoch{terminal}.pt')['H'])
    # A real next update checks that optimizer moments and random state are compatible.
    data=features('cora',0);cfg=dataset_config('cora');x=data['x'].cuda();edge=ge.with_loops(data['edge_index'].cuda(),len(x));adj=ge.adjacency(edge,len(x),x.dtype)
    model=ge.Model(x.shape[1],cfg).cuda();optimizer=torch.optim.Adam(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);hashes=[]
    for saved in [ck,ref]:
        restore_snapshot(model,optimizer,saved);model.train();optimizer.zero_grad(set_to_none=True);g,t=model(x,adj)
        ge.backward_loss(g,t,edge,chunk=CFG['chunk'],**loss_kwargs(cfg));optimizer.step();hashes.append(state_hash(model.state_dict()))
    assert hashes[0]==hashes[1]
    save(REV/'smoke_complete.json',dict(status='passed',budget_receipt_sha256=sha(REV/'receipt.json'),
        midpoint_model_optimizer_rng_exact=True,midpoint_H_context_exact=True,next_step_exact=True,imported_final_model_H_exact=True))
    original.status('BUDGET_SMOKE_PASSED')


def execute():
    prepare_budget();check=read(REV/'smoke_complete.json')
    assert check['status']=='passed' and check['budget_receipt_sha256']==sha(REV/'receipt.json')
    completed=[];failures=[]
    for task in tasks():
        d,s,arm=task['dataset'],task['seed'],task['arm']
        try:
            if arm=='ecl' and d in PLAN['cached_baseline_datasets']:
                ensure_midpoint(d,s);import_baseline(d,s)
            else:
                original.train_endpoint(d,s,arm)
            original.fit_heads(d,s,arm);completed.append(task)
            save(ROOT/'formal/completed.json',dict(completed=completed,failures=failures,revision=PLAN['id'],planned=PLAN['ssl_endpoints']))
            original.status('BUDGET_ENDPOINT_COMPLETE',**task,count=len(completed),total=PLAN['ssl_endpoints'])
        except Exception:
            error=dict(task=task,traceback=traceback.format_exc(),time=time.strftime('%Y-%m-%d %H:%M:%S'))
            save(REV/'failures'/f'{d}_{s}_{arm}_{time.time_ns()}.json',error)
            original.status('BUDGET_FAILED',**task,error=error['traceback']);raise
        finally:
            gc.collect();torch.cuda.empty_cache()
    prepare_budget()
    save(ROOT/'formal/fits_locked.json',dict(status='complete',completed=completed,failures=failures,revision=PLAN['id'],
        receipt_sha256=sha(ROOT/'receipt.json'),budget_receipt_sha256=sha(REV/'receipt.json'),heads=PLAN['heads'],deferred=PLAN['deferred']))
    from analyze_budget import analyze
    analyze()
    original.status('FINISHED_BUDGET_REVISION',endpoints=len(completed),heads=PLAN['heads'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','smoke','all']);args=p.parse_args()
    torch.set_num_threads(CFG['threads']);torch.use_deterministic_algorithms(True)
    with single_instance():
        if args.command=='prepare':prepare_budget()
        elif args.command=='smoke':smoke()
        else:execute()
