"""Durable serial external queue followed by scoring and deployment replay audit."""
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parent
def write(name,value):
    (ROOT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
    with (ROOT/'background.lock').open('x') as lock:lock.write(str(os.getpid()))
    try:
        write('background_status.json',dict(status='running',pid=os.getpid(),started=datetime.now(timezone.utc).isoformat()))
        with (ROOT/'background_console.log').open('a',encoding='utf-8') as console:
            for script in ['recover.py','verify_and_interpret.py']:
                if script=='recover.py' and (ROOT/'complete.json').exists():continue
                process=subprocess.run([sys.executable,'-X','utf8','-u',str(ROOT/script)],stdout=console,stderr=subprocess.STDOUT,
                                       cwd=ROOT.parents[1],env={**os.environ,'PYTHONUNBUFFERED':'1'})
                if process.returncode:raise RuntimeError(f'{script} exited {process.returncode}; inspect background_console.log')
        write('background_status.json',dict(status='complete',pid=os.getpid(),completed=datetime.now(timezone.utc).isoformat()))
        (ROOT/'HANDOFF.md').write_text('# 本轮已完成\n\n读取REPORT_zh.md、summary.csv、matched_seed0.csv、decision.json和delivery_audit.json。\n480个R2分类头、40个外部结果（5个历史GSSC复用）及480模型重载审计完成。\n原始未完成GSSC单元在interrupted目录留存，来源及恢复说明见ENGINEERING.md。\n',encoding='utf-8')
    except Exception:
        write('background_status.json',dict(status='failed',pid=os.getpid(),traceback=traceback.format_exc()))
        raise
    finally:(ROOT/'background.lock').unlink(missing_ok=True)
if __name__=='__main__':main()
