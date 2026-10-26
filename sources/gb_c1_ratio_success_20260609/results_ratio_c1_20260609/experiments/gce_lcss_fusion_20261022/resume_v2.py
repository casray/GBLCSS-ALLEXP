"""Recorded numerical recovery; frozen v1 sources remain untouched."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import run
from stable_losses import stable_objective


def main():
    run.initialize()
    run.verify()
    files = [ROOT/'stable_losses.py', ROOT/'resume_v2.py', ROOT/'test_stable.py']
    manifest_path = ROOT/'numerical_recovery_v2.json'
    expected = {str(p):run.sha(p) for p in files}
    if manifest_path.exists():
        if run.read(manifest_path)['sha256'] != expected:
            raise RuntimeError('Numerical recovery source changed')
    else:
        run.write(manifest_path, dict(reason='exp(log_p).pow(q) underflow caused nonfinite gradient',
                  change='exp(q*log_p), algebraically identical; no q/rate/target/budget/gate changes',
                  rerun_rules=['GCE','SOFT_GCE'], unchanged_rules=['CE','LCSS','W_LCSS','W_MASS','MASS'],
                  sha256=expected, original_frozen_manifest_sha256=run.sha(ROOT/'frozen_inputs.json')))
    run.objective = stable_objective
    original_fit = run.fit
    def fit(*args, **kwargs):
        result = original_fit(*args, **kwargs)
        if result['rule'] in ('GCE','SOFT_GCE'):
            result['numerical_implementation'] = 'v2_exp_q_log_p'
            result['numerical_source_sha256'] = run.sha(ROOT/'stable_losses.py')
            run.write(Path(result['folder'])/'result.json',result)
        return result
    run.fit = fit
    run.run(False)
    run.verify()
    for p,h in expected.items():
        if run.sha(p) != h:
            raise RuntimeError('Recovery source modified during execution')


if __name__=='__main__':
    try:
        main()
    except Exception:
        import traceback
        run.write(ROOT/'failure_v2.json',dict(traceback=traceback.format_exc()))
        raise
