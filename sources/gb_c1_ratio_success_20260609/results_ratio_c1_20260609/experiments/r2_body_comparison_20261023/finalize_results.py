"""Recover postprocessing through absolute imports; no training restarts."""
import importlib.util
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import run as r

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec)
    sys.modules[name]=value
    spec.loader.exec_module(value)
    return value

if __name__=='__main__':
    assert (ROOT/'r2_complete.json').exists() and (ROOT/'external_complete.json').exists()
    analysis=module('analyze',ROOT/'analyze.py')
    analysis.analyze()
    verification=module('body_final_verification',ROOT/'verify_and_interpret.py')
    verification.main()
    r.save(ROOT/'postprocess_recovery.json',dict(passed=True,entry_sha256=r.sha(Path(__file__)),training_restarted=False))
    r.save(ROOT/'background_status.json',dict(status='complete',postprocessing='absolute-import recovery',training_restarted=False))
