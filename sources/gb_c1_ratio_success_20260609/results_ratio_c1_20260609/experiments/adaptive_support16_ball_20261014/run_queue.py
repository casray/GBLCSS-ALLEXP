"""Serial fixed support16 screen, audits, pilot and report; no automatic retry."""
from pathlib import Path
import subprocess
import json
import sys
from datetime import datetime

ROOT=Path(__file__).resolve().parent


def main():
    sys.stdout.reconfigure(encoding='utf-8',errors='replace')
    state_file=ROOT/'queue_status.json'
    if state_file.exists():raise RuntimeError('Inspect existing queue status before explicit recovery')
    state=dict(status='running',started_at=datetime.now().astimezone().isoformat(),stages=[])
    def save():state_file.write_text(json.dumps(state,indent=2),encoding='utf-8')
    save()
    with (ROOT/'queue_raw.log').open('x',encoding='utf-8') as log:
        try:
            for name,args in [('run_screen.py',[]),('audit.py',[]),('run_pilot.py',[]),
                              ('audit.py',['--students-only']),('analyze.py',[]),('compare.py',[])]:
                command=[sys.executable,'-X','utf8','-u',str(ROOT/name),*args]
                row=dict(command=command,status='running',started_at=datetime.now().astimezone().isoformat())
                state['stages'].append(row);save();print('START',name,*args,flush=True)
                with subprocess.Popen(command,cwd=ROOT.parents[1],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                      text=True,encoding='utf-8',errors='replace') as child:
                    row['pid']=child.pid;save()
                    for line in child.stdout:log.write(line);log.flush();print(line,end='',flush=True)
                    code=child.wait()
                row.update(status='completed' if code==0 else 'failed',return_code=code,
                           completed_at=datetime.now().astimezone().isoformat());save()
                if code:raise RuntimeError(f'{name} failed with exit {code}')
            state.update(status='completed',completed_at=datetime.now().astimezone().isoformat());save()
        except BaseException as exc:
            state.update(status='failed',error=repr(exc));save();raise


if __name__=='__main__':main()
