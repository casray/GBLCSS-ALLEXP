"""Explicit component-version entry point; original training files stay frozen."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import protocol as p
import fit as fit_module
import run as queue
from static_repair import static_features, gate_features


def main():
    fit_module.static_features = static_features
    fit_module.gate_features = gate_features
    sources = {str(path): p.sha(path) for path in Path(__file__).resolve().parent.glob('*.py')}
    manifest = ROOT/'repair_sources.json'
    record = dict(training_manifest_sha256=p.sha(ROOT/'sources.json'), sources=sources,
                  correction='D2 static observed-class frequency scaled using training-node mean/std',
                  affects='D2 gates only; no D2 fits existed before repair', historical_sources_modified=False)
    if manifest.exists():
        old = p.read(manifest)
        # Report-only supplement may evolve without changing any effective fit component.
        for name in ['static_repair.py', 'run_repaired.py']:
            path = str(Path(__file__).resolve().parent/name)
            if old['sources'][path] != record['sources'][path]:
                raise RuntimeError('Repaired fit component changed')
    p.save(manifest, record)
    original_fit = queue.fit

    def versioned_fit(*args, **kwargs):
        mode = kwargs.get('mode', 'STANDARD')
        if mode.startswith('GATE_'):
            metadata = dict(kwargs.get('metadata') or {})
            metadata.update(static_feature_version='train_scaled_frequency_v2',
                            repair_component_sha256=p.sha(Path(__file__).resolve().parent/'static_repair.py'))
            kwargs['metadata'] = metadata
        return original_fit(*args, **kwargs)

    queue.fit = versioned_fit
    queue.main()


if __name__ == '__main__':
    main()
