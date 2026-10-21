"""Expand paper experiments into dataset, method, noise and seed combinations."""
from itertools import product

from .experiment import load_config


def build_plan(root, suites, datasets=None, seeds=None):
    catalog = load_config(root / 'configs/experiments.yaml')['suites']
    defaults = load_config(root / 'configs/main/gblcss.yaml')
    if suites == ['all']:
        suites = list(catalog)
    if not suites or len(suites) != len(set(suites)) or set(suites) - catalog.keys():
        raise ValueError('select unique suite names from --list')
    if seeds is not None and (not seeds or len(seeds) != len(set(seeds)) or min(seeds) < 0):
        raise ValueError('seeds must be unique nonnegative integers')
    plan = []
    for name in suites:
        spec = catalog[name]
        selected = spec['datasets'] if datasets is None else datasets
        if not selected or len(selected) != len(set(selected)):
            raise ValueError('select unique datasets')
        for dataset in selected:
            if not (root / 'configs/datasets' / f'{dataset}.yaml').is_file():
                raise ValueError(f'unknown dataset: {dataset}')
        variants = spec.get('variants', {'default': {}})
        for dataset, seed, noise, validation, variant, method in product(
            selected,
            seeds if seeds is not None else spec.get('seeds', defaults['seeds']),
            spec.get('noise', [['uniform', .4]]),
            spec.get('validation', ['clean']), variants, spec['methods'],
        ):
            plan.append(dict(suite=name, task=spec.get('task', 'train'), dataset=dataset,
                             method=method, noise_type=noise[0], noise_rate=noise[1],
                             seed=seed, validation_protocol=validation, variant=variant,
                             overrides=variants[variant]))
    return plan
