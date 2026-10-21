"""Frozen variants, provenance helpers and paired descriptive analysis."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics


def variant(family, rho=0., mask='shared', partition='granule'):
    return dict(family=family, rho=rho, mask=mask, partition=partition, target='A')


VARIANTS = {
    'tfr_P_ind': variant('tfr', mask='independent'),
    'tfr_P_shared': variant('tfr'),
    'tfr_S_shared': variant('tfr', rho=1.),
    'ce_P_ind': variant('ce', mask='independent'),
    'ce_P_shared': variant('ce'),
    'ce_M_shared': variant('ce', rho=.5),
    'ce_S_shared': variant('ce', rho=1.),
    'ce_KM_shared': variant('ce', rho=1., partition='kmeans'),
}


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.partial')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')
    tmp.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def cache_key(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def source_signature(folder):
    folder = Path(folder)
    files = {p.relative_to(folder).as_posix(): file_hash(p)
             for p in sorted(folder.rglob('*.py')) if not p.name.startswith('test_')}
    return files, cache_key(files)


def build_jobs(config):
    jobs = []
    for seed in config['seeds']:
        for dataset, noise in config['conditions']:
            for name in VARIANTS:
                identifier = f'{dataset}__{noise.replace(".", "p")}__seed{seed}__{name}'
                jobs.append(dict(id=identifier, dataset=dataset, noise=noise, seed=seed, variant=name))
    if len({j['id'] for j in jobs}) != len(jobs):
        raise ValueError('duplicate job specification')
    return jobs


PAIRS = [
    ('tfr_S_shared', 'tfr_P_ind', True),
    ('tfr_S_shared', 'tfr_P_shared', True),
    ('tfr_P_shared', 'tfr_P_ind', True),
    ('ce_S_shared', 'ce_P_shared', True),
    ('ce_M_shared', 'ce_P_shared', True),
    ('ce_M_shared', 'ce_S_shared', True),
    ('ce_S_shared', 'ce_P_ind', True),
    ('ce_P_shared', 'ce_P_ind', True),
    ('ce_KM_shared', 'ce_S_shared', False),
]


def paired_summary(rows):
    lookup = {(r['job']['dataset'], r['job']['noise'], r['job']['seed'], r['job']['variant']): r for r in rows}
    out = []
    for dataset, noise in sorted({k[:2] for k in lookup}):
        for a, b, match_partition in PAIRS:
            differences, seeds = [], []
            for seed in sorted({k[2] for k in lookup}):
                ra, rb = lookup.get((dataset, noise, seed, a)), lookup.get((dataset, noise, seed, b))
                if ra is None or rb is None:
                    continue
                keys = ['split_sha256', 'noise_sha256', 'config_sha256', 'source_sha256']
                if match_partition:
                    keys.append('assignment_sha256')
                if any(ra.get(k) != rb.get(k) for k in keys):
                    continue
                differences.append(100 * (ra['test_accuracy'] - rb['test_accuracy']))
                seeds.append(seed)
            if differences:
                out.append(dict(dataset=dataset, noise=noise, comparison=a+'-'+b, n_pairs=len(seeds),
                    seeds=seeds, delta_pp= differences, delta_pp_mean=statistics.mean(differences),
                    delta_pp_std=statistics.stdev(differences) if len(differences)>1 else None))
    return out


def summarize(root):
    import csv
    root = Path(root)
    rows = [json.loads(p.read_text(encoding='utf8')) for p in sorted((root/'results').glob('*.json'))]
    groups = {}
    for r in rows:
        j = r['job']; groups.setdefault((j['dataset'], j['noise'], j['variant']), []).append(r)
    table = []
    for (dataset, noise, name), rr in sorted(groups.items()):
        vals = [100*r['test_accuracy'] for r in rr]
        table.append(dict(dataset=dataset, noise=noise, variant=name, n=len(vals),
            accuracy_mean=statistics.mean(vals), accuracy_std=statistics.stdev(vals) if len(vals)>1 else None,
            macro_f1_mean=statistics.mean(100*r['macro_f1'] for r in rr),
            process_wall_mean=statistics.mean(r.get('process_wall_s', r['worker_wall_s']) for r in rr)))
    for filename, data in [('summary.csv', table), ('paired.csv', paired_summary(rows))]:
        if not data:
            continue
        with (root/filename).open('w', encoding='utf8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
    atomic_json(root/'summary.json', dict(completed=len(rows), aggregates=table, pairs=paired_summary(rows)))
    lines = ['# 强方法插件与固定剂量进度', '', f'更新时间UTC：{now()}；完成{len(rows)}项。',
             '未满三个seed的数值只作进度，不作为最终结论。', '',
             '|图|噪声|臂|n|Accuracy%|std|', '|---|---|---|---|---|---|']
    lines += [f"|{r['dataset']}|{r['noise']}|{r['variant']}|{r['n']}|{r['accuracy_mean']:.3f}|{r['accuracy_std']}|" for r in table]
    (root/'PROGRESS.md').write_text('\n'.join(lines)+'\n', encoding='utf8')
