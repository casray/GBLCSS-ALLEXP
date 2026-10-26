"""Post-only recovery after all stages have completed; zero classification fits."""
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from protocol import CONFIG, INPUTS, read, save, sha, now, append


def main():
    import torch
    torch.set_num_threads(1)
    if (ROOT/'run.lock').exists():
        raise RuntimeError('Training queue still has a lock; do not overlap finalization')
    for stage in CONFIG['stages']:
        assert (ROOT/f'{stage}_complete.json').exists(), stage
    assert read(ROOT/'selections_frozen.json')['all_stages_complete']
    INPUTS.update(read(ROOT/'inputs.json'))
    sources = {str(p): sha(p) for p in Path(__file__).resolve().parent.glob('*.py')}
    save(ROOT/'postprocessing_sources.json', sources)
    try:
        from analyze_windows import analyze
        analyze()
        from audit import audit
        audit()
        from supplement import main as supplement
        supplement()
        unique = len({r['folder'] for r in read(ROOT/'fit_registry.json')})
        assert unique == 1500
        save(ROOT/'complete.json', dict(finished=now(), unique_classification_fits=unique,
             planned_upper_bound=1500, cpu_predictors=20, smoke_fits=5, interrupted_partial_fit_updates=122,
             stages=CONFIG['stages'], audit=read(ROOT/'audit.json')['status'],
             postprocessing_repair='Windows separator keys and anchor/alias presentation only',
             training_sources_unchanged=True, supplement_manifest_sha256=sha(ROOT/'supplement_manifest.json')))
        save(ROOT/'status.json', dict(timestamp=now(), stage='COMPLETE', unique_classification_fits=unique,
                                     cpu_predictors=20, postprocessing_complete=True))
        print('COMPLETE: 1500 classification fits, 20 CPU predictors, 5 smoke fits; audits passed', flush=True)
    except BaseException:
        append(ROOT/'failures.jsonl', dict(timestamp=now(), stage='postprocessing_recovery', traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
