"""One-shot durable serial GPU launcher and explicit postprocessing pipeline."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

CODE=Path(__file__).resolve().parent
CONFIG=json.loads((CODE/'config.json').read_text(encoding='utf-8'))
ROOT=Path(CONFIG['runtime'])
PYTHON='D:/codex_temp/noisygl-rocm721/Scripts/python.exe'


def write(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2),encoding='utf-8');temp.replace(path)
def status(stage,**kw):
    row=dict(stage=stage,time=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),**kw)
    write(ROOT/'launcher_status.json',row);write(CODE/'launcher_status.json',row)
    with (ROOT/'launcher_events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')


def others():
    command="Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?(\\.exe)?$' } | Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Compress"
    r=subprocess.run(['powershell','-NoProfile','-Command',command],capture_output=True,text=True)
    if r.returncode:raise RuntimeError('Process inventory failed: '+r.stderr)
    rows=json.loads(r.stdout or '[]');rows=[rows] if isinstance(rows,dict) else rows
    own={os.getpid()};byid={int(x['ProcessId']):x for x in rows};current=os.getpid()
    while current in byid:
        parent=int(byid[current].get('ParentProcessId',-1))
        if parent not in byid or parent in own:break
        own.add(parent);current=parent
    return [x for x in rows if int(x['ProcessId']) not in own]


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'complete.json').exists():status('ALREADY_COMPLETE');return
    lock=ROOT/'launcher.lock'
    with lock.open('x') as f:json.dump(dict(pid=os.getpid()),f)
    try:
        previous=None
        while True:
            owners=others()
            if not owners:break
            ids=sorted(int(x['ProcessId']) for x in owners)
            if previous!=ids:status('WAITING_GPU',owners=owners,training_started=False);previous=ids
            time.sleep(60)
        for file in ('analysis.py','audit.py'):
            if not (CODE/file).exists():raise FileNotFoundError('Postprocessing adapter missing before launch: '+file)
        with (ROOT/'background_stdout.log').open('a',encoding='utf-8') as out,(ROOT/'background_stderr.log').open('a',encoding='utf-8') as err:
            for file,args,stage in [('run.py',['--stage','fit'],'RUNNING_GPU'),('analysis.py',[],'ANALYSIS'),('audit.py',[],'CPU_AUDIT')]:
                status(stage,training_started=True)
                result=subprocess.run([PYTHON,'-B',str(CODE/file),*args],cwd=CODE.parents[1],stdout=out,stderr=err)
                if result.returncode:raise RuntimeError(file+' exit '+str(result.returncode))
        assert (ROOT/'complete.json').exists()
        status('COMPLETE',training_started=True,report=str(CODE/'REPORT_zh.md'))
    except BaseException:status('FAILED',traceback=traceback.format_exc());raise
    finally:lock.unlink()


if __name__=='__main__':main()
