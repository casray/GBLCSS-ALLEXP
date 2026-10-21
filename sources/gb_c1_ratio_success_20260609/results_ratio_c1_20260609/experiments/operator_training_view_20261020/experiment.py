"""Frozen variants, provenance helpers and paired descriptive analysis."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics


def variant(mode='P', mask='independent', target='B', loss='ce', partition='granule', coarse=False):
    return dict(mode=mode, mask=mask, target=target, loss=loss, partition=partition, coarse=coarse)


VARIANTS = {
    'P_ind_A': variant(target='A', partition='none'),
    'P_ind_B': variant(),
    'P_shared_B': variant(mask='shared'),
    'PPI_shared_B': variant(mode='PPI', mask='shared'),
    'PIP_shared_B': variant(mode='PIP', mask='shared'),
    'S_shared_B': variant(mode='PIPPI', mask='shared'),
    'S_ind_B': variant(mode='PIPPI'),
    'coarse_B': variant(mask='shared', coarse=True),
    'P_ind_GCE': variant(target='A', loss='gce', partition='none'),
    'P_shared_GCE': variant(mask='shared', target='A', loss='gce'),
    'S_shared_GCE': variant(mode='PIPPI', mask='shared', target='A', loss='gce'),
    'P_ind_LS': variant(target='A', loss='label_smoothing', partition='none'),
    'P_ind_mixup': variant(target='A', loss='postprop_mixup', partition='none'),
    'KM_shared_B': variant(mode='PIPPI', mask='shared', partition='kmeans'),
    'random_shared_B': variant(mode='PIPPI', mask='shared', partition='random'),
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
    ('S_shared_B', 'P_shared_B', True), ('S_ind_B', 'P_ind_B', True),
    ('P_shared_B', 'P_ind_B', True), ('S_shared_B', 'S_ind_B', True),
    ('PPI_shared_B', 'P_shared_B', True), ('PIP_shared_B', 'P_shared_B', True),
    ('S_shared_B', 'coarse_B', True), ('S_shared_GCE', 'P_shared_GCE', True),
    ('S_shared_GCE', 'P_ind_GCE', False), ('S_shared_B', 'P_ind_A', False),
    ('S_shared_B', 'P_ind_GCE', False), ('S_shared_B', 'P_ind_LS', False),
    ('S_shared_B', 'P_ind_mixup', False), ('KM_shared_B', 'S_shared_B', False),
    ('random_shared_B', 'S_shared_B', False),
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
    lines = ['# 算子训练视图进度', '', f'更新时间UTC：{now()}；完成{len(rows)}项。',
             '未满三个seed的数值只作进度，不作为最终结论。', '',
             '|图|噪声|臂|n|Accuracy%|std|', '|---|---|---|---|---|---|']
    lines += [f"|{r['dataset']}|{r['noise']}|{r['variant']}|{r['n']}|{r['accuracy_mean']:.3f}|{r['accuracy_std']}|" for r in table]
    (root/'PROGRESS.md').write_text('\n'.join(lines)+'\n', encoding='utf8')
