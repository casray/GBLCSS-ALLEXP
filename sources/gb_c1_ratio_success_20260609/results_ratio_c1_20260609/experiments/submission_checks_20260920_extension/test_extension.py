import json
import importlib.util
from run_experiment import HERE, OLD, runner, check_sources


def test_original_unit_checks():
    spec = importlib.util.spec_from_file_location('old_checks', OLD/'test_experiment.py')
    module = importlib.util.module_from_spec(spec)
    # The old test imports these public functions from run_experiment.
    import run_experiment
    for name in ['arms','brier_decomposition','fingerprint','validate_pairs']:
        setattr(run_experiment,name,getattr(runner,name))
    spec.loader.exec_module(module)
    for name in dir(module):
        if name.startswith('test_'):
            getattr(module,name)()


def test_only_datasets_change():
    current = json.loads((HERE/'config.json').read_text(encoding='utf8'))
    original = json.loads((OLD/'config.json').read_text(encoding='utf8'))
    assert current.pop('datasets') == ['coauthorcs','amazonphoto']
    assert current.pop('enforce_archive_brier') is False
    original.pop('datasets')
    assert current == original
    check_sources()
