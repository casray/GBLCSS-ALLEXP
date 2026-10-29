"""Single queue owner; waits for other Python jobs, then smoke and formal fits."""
import json, os, subprocess, sys, time, traceback
from pathlib import Path
from contextlib import contextmanager
CODE=Path(__file__).resolve().parent
ROOT=Path('D:/codex_temp/ssl_readout_factorial_20261027')
PYTHON='D:/codex_temp/noisygl-rocm721/Scripts/python.exe'

def save(stage,**extra):
    row=dict(stage=stage,pid=os.getpid(),time=time.strftime('%Y-%m-%d %H:%M:%S'),**extra)
    dest=ROOT/'queue_status.json';tmp=dest.with_suffix('.tmp')
    tmp.write_text(json.dumps(row,indent=2),encoding='utf-8');tmp.replace(dest)
    with (ROOT/'queue_events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')

def others():
    command="Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?(\\.exe)?$' } | Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Compress"
    r=subprocess.run(['powershell','-NoProfile','-Command',command],capture_output=True,text=True)
    if r.returncode:raise RuntimeError('Process inventory failed: '+r.stderr)
    rows=json.loads(r.stdout or '[]');rows=[rows] if isinstance(rows,dict) else rows
    byid={int(r['ProcessId']):r for r in rows};own={os.getpid()};current=os.getpid()
    while current in byid:
        parent=int(byid[current]['ParentProcessId'])
        if parent not in byid or parent in own:break
        own.add(parent);current=parent
    changed=True
    while changed:
        changed=False
        for row in rows:
            ident=int(row['ProcessId'])
            if int(row['ParentProcessId']) in own and ident not in own:own.add(ident);changed=True
    return [r for r in rows if int(r['ProcessId']) not in own]

@contextmanager
def queue_lock():
    import msvcrt
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'queue.lock').open('a+b') as f:
        f.seek(0);f.write(b'0');f.flush();f.seek(0)
        msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        try:yield
        finally:f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)

def main():
    with queue_lock():
        if (ROOT/'formal/final.json').exists():save('ALREADY_FINISHED');return
        try:
            prior=None
            while True:
                blockers=others()
                if not blockers:break
                ids=sorted(int(r['ProcessId']) for r in blockers)
                if ids!=prior:save('WAITING_GPU',training_started=False,blocked_processes=blockers);prior=ids
                time.sleep(30)
            env=os.environ.copy();env['PYTHONIOENCODING']='utf-8'
            with (ROOT/'background_stdout.log').open('a',encoding='utf-8') as out,(ROOT/'background_stderr.log').open('a',encoding='utf-8') as err:
                for stage,command in [('SMOKE','smoke'),('FORMAL','all')]:
                    save(stage,training_started=stage=='FORMAL')
                    child=subprocess.Popen([PYTHON,'-B',str(CODE/'run.py'),command],cwd=CODE.parents[1],env=env,stdout=out,stderr=err)
                    save(stage,training_started=stage=='FORMAL',worker_pid=child.pid)
                    code=child.wait()
                    if code:raise RuntimeError(command+' exit '+str(code))
            final=json.loads((ROOT/'formal/final.json').read_text())
            save('FINISHED',training_started=True,result_status=final['status'],scored=final['scored'],report=str(ROOT/'analysis/REPORT_zh.md'))
        except BaseException:
            save('FAILED',traceback=traceback.format_exc());raise

if __name__=='__main__':main()
