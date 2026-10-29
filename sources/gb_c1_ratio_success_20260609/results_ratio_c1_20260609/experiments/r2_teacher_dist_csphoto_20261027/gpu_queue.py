"""One-shot serial queue, standard library only until another GPU job exits."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

CODE=Path(__file__).resolve().parent
CFG=json.loads((CODE/'config.json').read_text(encoding='utf-8'))
ROOT=Path(CFG['runtime'])
PYTHON='D:/codex_temp/noisygl-rocm721/Scripts/python.exe'


def now():return datetime.now(timezone.utc).isoformat()
def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2),encoding='utf-8');temp.replace(path)
def state(stage,**kw):
    value=dict(stage=stage,time=now(),pid=os.getpid(),**kw)
    save(ROOT/'queue_status.json',value);save(CODE/'queue_status.json',value)
    with (ROOT/'queue_events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(value)+'\n')


def other_owners(rows,pid):
    byid={int(r['ProcessId']):r for r in rows};own={pid}
    # Only Python ancestors and descendants of this queue process belong to it.
    current=pid
    while current in byid:
        parent=int(byid[current].get('ParentProcessId',-1))
        if parent not in byid or parent in own:break
        own.add(parent);current=parent
    changed=True
    while changed:
        changed=False
        for r in rows:
            ident=int(r['ProcessId'])
            if int(r.get('ParentProcessId',-1)) in own and ident not in own:
                own.add(ident);changed=True
    return [r for r in rows if int(r['ProcessId']) not in own]


def inventory():
    command="Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?(\\.exe)?$' } | Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Compress"
    result=subprocess.run(['powershell','-NoProfile','-Command',command],capture_output=True,text=True)
    if result.returncode:raise RuntimeError('Read-only inventory failed: '+result.stderr)
    rows=json.loads(result.stdout or '[]');return [rows] if isinstance(rows,dict) else rows


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'complete.json').exists():state('ALREADY_COMPLETE');return
    lock=ROOT/'queue.lock'
    with lock.open('x',encoding='utf-8') as f:json.dump(dict(pid=os.getpid(),time=now()),f)
    try:
        previous=None
        while True:
            others=other_owners(inventory(),os.getpid())
            if not others:break
            ids=sorted(int(r['ProcessId']) for r in others)
            if ids!=previous:state('WAITING_GPU',blocked_processes=others,training_started=False);previous=ids
            time.sleep(CFG['gpu_queue']['poll_seconds'])
        state('RUNNING',training_started=True)
        with (ROOT/'background_stdout.log').open('a',encoding='utf-8') as out, (ROOT/'background_stderr.log').open('a',encoding='utf-8') as err:
            for script,args,stage in [('run.py',['--stage','all'],'FITTING'),('audit.py',[],'CPU_AUDIT')]:
                state(stage,training_started=True)
                result=subprocess.run([PYTHON,'-B',str(CODE/script),*args],cwd=CODE.parents[1],stdout=out,stderr=err)
                if result.returncode:raise RuntimeError(script+' exit '+str(result.returncode))
        assert json.loads((ROOT/'complete.json').read_text())['status']=='COMPLETE_AUDITED'
        state('COMPLETE',training_started=True,report=str(CODE/'REPORT_zh.md'))
    except BaseException:
        state('FAILED',traceback=traceback.format_exc());raise
    finally:lock.unlink()


if __name__=='__main__':main()
