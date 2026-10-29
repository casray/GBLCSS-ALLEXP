"""Hidden Windows launch; no shell command interpolation."""
import json,subprocess,sys,os
from pathlib import Path
CODE=Path(__file__).resolve().parent
CFG=json.loads((CODE/'config.json').read_text(encoding='utf-8-sig'));ROOT=Path(CFG['runtime'])
if __name__=='__main__':
    ROOT.mkdir(parents=True,exist_ok=True)
    startup=subprocess.STARTUPINFO();startup.dwFlags|=subprocess.STARTF_USESHOWWINDOW;startup.wShowWindow=0
    env=os.environ.copy();env['PYTHONIOENCODING']='utf-8';env['PYTHONUNBUFFERED']='1'
    with (ROOT/'background_stdout.log').open('ab') as out,(ROOT/'background_stderr.log').open('ab') as err:
        p=subprocess.Popen([sys.executable,'-B',str(CODE/'run.py'),'all'],cwd=CODE,stdout=out,stderr=err,stdin=subprocess.DEVNULL,env=env,startupinfo=startup,creationflags=subprocess.CREATE_NO_WINDOW)
    (ROOT/'launch.json').write_text(json.dumps(dict(pid=p.pid,command=[sys.executable,'-B',str(CODE/'run.py'),'all'])),encoding='utf-8')
    print('Started hidden runner PID',p.pid)
