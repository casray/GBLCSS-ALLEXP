"""Predetermined selectors, CE fallback, support dedup and paid refinement."""
import hashlib


def choose(pool, selector):
    if selector not in ('accuracy','balanced'):
        raise ValueError(selector)
    def key(item):
        r = item[1]
        primary = r['validation'] if selector=='accuracy' else (r['validation']+r['valid_macro_f1'])/2
        return (-primary,-r['validation'],r['valid_nll'],r['best_epoch'])
    return min(pool,key=key)[0]


def support_from_probe(arm):
    return None if arm=='CE' else int(arm.split('_')[0][1:])


def distinct_supports(probe):
    s = support_from_probe(probe)
    return [32] if s in (None,32) else [32,s]


def refinement_cost(probe, support):
    stages = {} if support is None else {f'S{support}_L{l}':25 for l in (.25,.5,1.)}
    for arm in stages:
        if arm in probe['rounds'][0]['survivors']:
            stages[arm] = 75
        if arm in probe['rounds'][1]['survivors']:
            stages[arm] = 200
    extra = sum(200-v for v in stages.values())
    return dict(upstream_probe_with_ce=1200,prefix_epochs=stages,additional_global_updates=extra,
                additional_group_updates=800 if support is not None else 0,
                global_total_updates=1200+extra,expanded_total_updates=1200+extra+(800 if support is not None else 0))


def freeze_required(path):
    if not path.exists():
        raise RuntimeError('Selection freeze required before test scoring')
    return hashlib.sha256(path.read_bytes()).hexdigest()
