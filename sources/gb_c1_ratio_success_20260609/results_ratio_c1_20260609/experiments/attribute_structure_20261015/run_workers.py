"""Schedule independent unchanged protocol seeds in bounded CPU processes."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import logging
from pathlib import Path
import time
import torch
from threadpoolctl import threadpool_limits
import common as c
import run


def work(name,seed):
    torch.set_num_threads(c.CONFIG['threads'])
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.StreamHandler(),logging.FileHandler(c.ROOT/'worker_raw.log',encoding='utf-8')])
    with threadpool_limits(limits=c.CONFIG['threads']):
        run.partitions(name,seed);run.targets(name,seed)
    return dict(dataset=name,seed=seed,finished=time.time())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',required=True,choices=c.CONFIG['datasets']);parser.add_argument('--workers',type=int,default=2);a=parser.parse_args()
    if a.workers not in (1,2):raise ValueError('Bounded to at most two CPU processes')
    c.verify_manifest(c.ROOT/'source_manifest.json');c.verify_manifest(c.ROOT/'preparation_manifest.json')
    c.verify_manifest(c.ROOT/'views'/a.dataset/'manifest.json')
    c.freeze_manifest(c.ROOT/'worker_source_manifest.json',[Path(__file__),c.ROOT/'source_manifest.json',c.ROOT/'config.json',c.ROOT/'PLAN.md'])
    # Only this scheduler may own these seeds; the original serial CPU runner
    # must be stopped at a saved boundary before this command is launched.
    lock=c.ROOT/('workers_'+a.dataset+'.lock')
    with lock.open('x',encoding='utf-8') as f:f.write(str(__import__('os').getpid()))
    status=dict(status='running',dataset=a.dataset,workers=a.workers,completed=[])
    c.write(c.ROOT/'worker_status.json',status)
    try:
        pending=[]
        for seed in c.SEEDS:
            if (c.ROOT/'targets'/a.dataset/f'seed{seed}'/'metrics.json').exists():
                c.verify_manifest(c.ROOT/'partitions'/a.dataset/f'seed{seed}'/'manifest.json')
                for noise in c.conditions(a.dataset):c.verify_manifest(c.ROOT/'targets'/a.dataset/f'seed{seed}'/noise/'frozen_targets.json')
                status['completed'].append(dict(dataset=a.dataset,seed=seed,cached=True))
            else:pending.append(seed)
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            jobs=[pool.submit(work,a.dataset,s) for s in pending]
            for job in as_completed(jobs):
                status['completed'].append(job.result());run.summarize();c.write(c.ROOT/'worker_status.json',status)
        rows=run.summarize();c.verify_manifest(c.ROOT/'source_manifest.json');c.verify_manifest(c.ROOT/'worker_source_manifest.json')
        status.update(status='completed',target_rows=len(rows));c.write(c.ROOT/'worker_status.json',status)
        c.write(c.ROOT/'target_status.json',dict(status='completed' if len(rows)==420 else 'partial',target_rows=len(rows)))
    except Exception as exc:
        status.update(status='failed',error=repr(exc));c.write(c.ROOT/'worker_status.json',status);raise
    finally:lock.unlink()


if __name__=='__main__':main()
