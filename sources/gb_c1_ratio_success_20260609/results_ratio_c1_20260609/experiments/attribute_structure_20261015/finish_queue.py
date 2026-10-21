"""Continue the approved finite experiment once the owned CPU workers finish."""
from pathlib import Path
import logging
import subprocess
import sys
import time
import common as c


def main():
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.StreamHandler(),logging.FileHandler(c.ROOT/'queue_raw.log',encoding='utf-8')])
    files=[Path(__file__),c.ROOT/'run_workers.py',c.ROOT/'run_students.py',c.ROOT/'metric_training.py',c.ROOT/'audit.py',c.ROOT/'finish_report.py',c.ROOT/'diagnose_priors.py',c.ROOT/'source_manifest.json',c.ROOT/'PLAN.md',c.ROOT/'config.json']
    c.freeze_manifest(c.ROOT/'queue_source_manifest.json',files)
    status=dict(status='waiting_for_cpu',started=time.time(),completed_stages=[])
    c.write(c.ROOT/'queue_status.json',status)
    try:
        while True:
            worker=c.read(c.ROOT/'worker_status.json')
            if worker['status']=='failed':raise RuntimeError('Owned CPU worker failed: '+str(worker))
            if worker['status']=='completed':break
            time.sleep(5)
        if c.read(c.ROOT/'prepare_status.json')['status']!='completed':raise RuntimeError('Preparation incomplete')
        c.verify_manifest(c.ROOT/'queue_source_manifest.json')
        import run
        rows=run.summarize()
        if len(rows)!=420:raise RuntimeError('Expected 420 original target records')
        stages=[('targets_audit','audit.py',['--targets-only']),('prior_diagnostic','diagnose_priors.py',[]),('students','run_students.py',['--device','cuda']),('final_audit','audit.py',[]),('report','finish_report.py',[])]
        for stage,script,args in stages:
            c.verify_manifest(c.ROOT/'queue_source_manifest.json')
            status.update(status='running',stage=stage);c.write(c.ROOT/'queue_status.json',status)
            logging.info('START STAGE %s',stage)
            subprocess.run([sys.executable,'-X','utf8',str(c.ROOT/script)]+args,cwd=c.WORKSPACE,check=True)
            status['completed_stages'].append(stage);logging.info('DONE STAGE %s',stage)
        c.verify_manifest(c.ROOT/'queue_source_manifest.json')
        status.update(status='completed',elapsed_seconds=time.time()-status['started']);c.write(c.ROOT/'queue_status.json',status)
    except Exception as exc:
        status.update(status='failed',error=repr(exc));c.write(c.ROOT/'queue_status.json',status);raise


if __name__=='__main__':main()
