"""Sequential resumable queue; each arm owns a fresh GPU process."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil

from experiment import atomic_json, file_hash, source_signature, build_jobs, summarize, now


SOURCE=Path(__file__).resolve().parent


def owner_alive(pid, created):
    if not isinstance(pid,int) or pid<=0:
        return False
    try:
        return abs(psutil.Process(pid).create_time()-created)<.01
    except psutil.NoSuchProcess:
        return False
    except psutil.AccessDenied:
        return True


def claim_lock(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    path=root/'queue.lock'
    if path.exists():
        previous=json.loads(path.read_text(encoding='utf8'))
        if owner_alive(previous.get('pid'),previous.get('created',0)):
            raise RuntimeError('another queue owns this run-root')
        if owner_alive(previous.get('child_pid'),previous.get('child_created',0)):
            raise RuntimeError('orphan worker still owns this run-root; stop it before resuming')
        path.unlink()
    with path.open('x',encoding='utf8') as stream:
        json.dump(dict(pid=os.getpid(),created=psutil.Process().create_time()),stream)


def freeze(config_path, root):
    raw=Path(config_path).read_bytes();config=json.loads(raw.decode('utf8'))
    hashes,aggregate=source_signature(SOURCE)
    signature=dict(config_sha256=__import__('hashlib').sha256(raw).hexdigest(),
                   source_sha256=aggregate,source_hashes=hashes)
    path=root/'protocol.json'
    if path.exists():
        old=json.loads(path.read_text(encoding='utf8'))
        if any(old.get(k)!=v for k,v in signature.items()):
            raise ValueError('frozen code or configuration drift; use a new run-root')
        if file_hash(root/'config.json')!=signature['config_sha256']:
            raise ValueError('frozen config bytes changed')
    else:
        root.mkdir(parents=True,exist_ok=True)
        (root/'config.json').write_bytes(raw)
        atomic_json(path,dict(signature,created_at=now(),source_directory=str(SOURCE),python=sys.executable))
    return config,signature


def validate(result, job, protocol, root):
    if result.get('status')!='completed' or result.get('job')!=job:
        raise ValueError('result/manifest mismatch')
    for key in ['config_sha256','source_sha256']:
        if result.get(key)!=protocol[key]:
            raise ValueError('result provenance mismatch '+key)
    for path_key,hash_key,folder in [('prediction_file','prediction_sha256','results'),
                                    ('checkpoint_file','checkpoint_sha256',''),
                                    ('trace_file','trace_sha256','')]:
        path=(root/folder/result[path_key]).resolve()
        path.relative_to(root.resolve())
        if file_hash(path)!=result[hash_key]:
            raise ValueError('artifact mismatch '+path_key)


def child(command, logfile, timeout, root, status):
    start=time.perf_counter()
    with logfile.open('a',encoding='utf8') as stream:
        stream.write('\nSTART '+now()+'\n'+subprocess.list2cmdline(command)+'\n');stream.flush()
        process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT)
        status.update(pid=process.pid)
        lock=json.loads((root/'queue.lock').read_text(encoding='utf8'))
        lock.update(child_pid=process.pid,child_created=psutil.Process(process.pid).create_time())
        atomic_json(root/'queue.lock',lock)
        try:
            while process.poll() is None:
                status.update(child_wall_s=time.perf_counter()-start,updated_at=now())
                atomic_json(root/'status.json',status)
                if (root/'PAUSE.request').exists():
                    process.terminate();process.wait();return None
                if time.perf_counter()-start>timeout:
                    process.kill();process.wait();raise TimeoutError(str(command))
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
        except BaseException:
            if process.poll() is None:
                process.kill();process.wait()
            raise
        finally:
            lock=json.loads((root/'queue.lock').read_text(encoding='utf8'))
            if process.poll() is not None and lock.get('child_pid')==process.pid:
                lock.pop('child_pid',None);lock.pop('child_created',None)
                atomic_json(root/'queue.lock',lock)
    if process.returncode!=0:
        raise RuntimeError(f'child failed {process.returncode}; log={logfile}')
    return time.perf_counter()-start


def run(config_path, root, limit=None, prepare_only=False):
    root=Path(root).resolve()
    claim_lock(root)
    status=dict(status='initializing',completed=0,total=0,parent_pid=os.getpid(),started_at=now(),updated_at=now())
    try:
        config,protocol=freeze(config_path,root);jobs=build_jobs(config)
        if limit is not None:
            if limit<1:
                raise ValueError('limit must be positive and use a separate run-root')
            jobs=jobs[:limit]
        for name in ['jobs','logs','results','checkpoints','cache']:
            (root/name).mkdir(exist_ok=True)
        path=root/'jobs.json'
        if path.exists() and json.loads(path.read_text(encoding='utf8'))!=jobs:
            raise ValueError('frozen manifest drift')
        atomic_json(path,jobs);status['total']=len(jobs)
        done=set()
        for j in jobs:
            result=root/'results'/f"{j['id']}.json"
            atomic_json(root/'jobs'/f"{j['id']}.json",j)
            if result.exists():
                rr=json.loads(result.read_text(encoding='utf8'));validate(rr,j,protocol,root)
                if 'process_wall_s' not in rr:
                    raise ValueError('unmeasured completed child; inspect interrupted queue before resume')
                done.add(j['id'])
        status['completed']=len(done)
        for dataset in dict.fromkeys(j['dataset'] for j in jobs):
            if (root/'PAUSE.request').exists():
                status['status']='paused';atomic_json(root/'status.json',status);return
            status.update(status='preparing',current={'dataset':dataset})
            command=[sys.executable,str(SOURCE/'prepare_data.py'),'--config',str(root/'config.json'),
                     '--run-root',str(root),'--dataset',dataset]
            wall=child(command,root/'logs'/f'prepare_{dataset}.txt',config['job_timeout_s'],root,status)
            if wall is None:
                status['status']='paused';atomic_json(root/'status.json',status);return
        if prepare_only:
            status.update(status='prepared',current=None,pid=None,updated_at=now())
            atomic_json(root/'status.json',status);return
        for j in jobs:
            if j['id'] in done:
                continue
            if (root/'PAUSE.request').exists():
                status['status']='paused';atomic_json(root/'status.json',status);return
            freeze(config_path,root)
            status.update(status='running',current=j,pid=None)
            command=[sys.executable,str(SOURCE/'run_job.py'),'--config',str(root/'config.json'),
                     '--job',str(root/'jobs'/f"{j['id']}.json"),'--run-root',str(root)]
            wall=child(command,root/'logs'/f"{j['id']}.txt",config['job_timeout_s'],root,status)
            if wall is None:
                status['status']='paused';atomic_json(root/'status.json',status);return
            result_path=root/'results'/f"{j['id']}.json"
            rr=json.loads(result_path.read_text(encoding='utf8'));validate(rr,j,protocol,root)
            rr['process_wall_s']=wall
            rr['reconstructed_cold_process_s']=wall+rr['required_historical_preparation_s']
            atomic_json(result_path,rr);done.add(j['id'])
            status.update(completed=len(done),pid=None,updated_at=now());atomic_json(root/'status.json',status)
            summarize(root);print(f"{len(done)}/{len(jobs)} completed {j['id']}",flush=True)
        freeze(config_path,root)
        for dataset in dict.fromkeys(j['dataset'] for j in jobs):
            meta=json.loads((root/'cache'/dataset/'metadata.json').read_text(encoding='utf8'))
            for name,record in meta['original_cache_files'].items():
                if file_hash(Path(meta['old_directory'])/name)!=record['sha256']:
                    raise ValueError('original inputs changed during run')
        status.update(status='completed',current=None,pid=None,finished_at=now(),updated_at=now())
        atomic_json(root/'status.json',status);summarize(root)
    except BaseException as exc:
        if root.exists():
            status.update(status='failed',reason=f'{type(exc).__name__}: {exc}',pid=None,updated_at=now())
            atomic_json(root/'status.json',status)
        raise
    finally:
        lock=root/'queue.lock'
        if lock.exists() and json.loads(lock.read_text(encoding='utf8')).get('pid')==os.getpid():
            lock.unlink()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    parser.add_argument('--run-root',required=True);parser.add_argument('--limit',type=int)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args();run(args.config,args.run_root,args.limit,args.prepare_only)
