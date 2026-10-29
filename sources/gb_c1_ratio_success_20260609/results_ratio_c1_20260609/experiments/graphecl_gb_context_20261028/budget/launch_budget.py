"""Launch the reduced, single-instance worker without a visible console."""
import json
import os
import subprocess
import sys
from pathlib import Path

CODE=Path(__file__).resolve().parent
ROOT=Path(json.loads((CODE.parent/'config.json').read_text())['runtime'])/'budget_revision'
if __name__=='__main__':
    ROOT.mkdir(parents=True,exist_ok=True)
    startup=subprocess.STARTUPINFO();startup.dwFlags|=subprocess.STARTF_USESHOWWINDOW;startup.wShowWindow=0
    env=os.environ.copy();env['PYTHONIOENCODING']='utf-8';env['PYTHONUNBUFFERED']='1'
    with (ROOT/'stdout.log').open('ab') as out,(ROOT/'stderr.log').open('ab') as err:
        command=[sys.executable,'-B',str(CODE/'run_budget.py'),'all']
        worker=subprocess.Popen(command,cwd=CODE,stdout=out,stderr=err,stdin=subprocess.DEVNULL,env=env,startupinfo=startup,creationflags=subprocess.CREATE_NO_WINDOW)
    (ROOT/'launch.json').write_text(json.dumps(dict(pid=worker.pid,command=command)),encoding='utf-8')
    print('Started reduced schedule launcher PID',worker.pid)
