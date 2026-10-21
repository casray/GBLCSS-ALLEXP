from pathlib import Path
import csv
import io
import hashlib
import json
import os
import statistics
import subprocess
import time

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[3]
NOISYGL = PROJECT / 'external/NoisyGL'
ARCHIVE = NOISYGL / 'runs/tie_safe_full_grid_20260727_v2_immutable'
PYTHON = Path('D:/codex_temp/noisygl-rocm721/Scripts/python.exe')


def rows(path):
    raw = path.read_bytes()
    try:
        content = raw.decode('utf-8-sig')
    except UnicodeDecodeError:
        content = raw.decode('gb18030')
    return list(csv.DictReader(io.StringIO(content)))


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def main():
    assert NOISYGL.is_dir() and PYTHON.is_file(), (NOISYGL, PYTHON)
    specs = [r for r in rows(ARCHIVE / 'command_manifest.csv')
             if r['regime'] == 'uniform0.4' and r['group'] == 'softdist']
    assert len(specs) == 5
    (HERE / 'gate').mkdir(exist_ok=True)
    comparisons = []
    for spec in specs:
        dataset = spec['dataset']
        command = json.loads(spec['command_json'])
        command[0] = str(PYTHON)
        output = HERE / 'gate' / (dataset + '.csv')
        command[command.index('--output') + 1] = str(output)
        env = os.environ.copy()
        flags = json.loads(spec['deterministic_env_json'])
        env.update(flags)
        record = dict(command=command, environment_flags=flags)
        manifest = HERE / 'gate' / (dataset + '_command.json')
        if manifest.exists():
            assert json.loads(manifest.read_text(encoding='utf-8')) == record
        else:
            save(manifest, record)
        if not output.exists():
            print('START ' + dataset, flush=True)
            start = time.perf_counter()
            with (HERE / 'gate' / (dataset + '.stdout.log')).open('w', encoding='utf-8') as out, \
                 (HERE / 'gate' / (dataset + '.stderr.log')).open('w', encoding='utf-8') as err:
                result = subprocess.run(command, env=env, cwd=NOISYGL, stdout=out, stderr=err)
            assert result.returncode == 0, (dataset, result.returncode)
            print(f'FINISH {dataset} {time.perf_counter()-start:.1f}s', flush=True)
        actual = rows(output)
        expected = rows(Path(spec['immutable_raw_path']))
        key = lambda r: (r['base_method'], int(r['seed']))
        reference = {key(r): r for r in expected}
        assert len(actual) == len(reference) == 10
        for r in actual:
            e = reference[key(r)]
            comparison = dict(dataset=dataset, method=r['base_method'], seed=int(r['seed']), status=r['status'])
            for field in ('split_hash', 'noise_hash', 'config_id', 'ball_assignment_hash'):
                comparison[field + '_matches'] = r[field] == e[field]
            comparison['test'] = float(r['test'])
            comparison['archived_test'] = float(e['test'])
            comparison['delta_pp'] = 100 * (float(r['test']) - float(e['test']))
            comparison['valid_matches'] = abs(float(r['valid'])-float(e['valid'])) < 1e-7
            comparisons.append(comparison)
        save(HERE / 'gate_comparison.json', comparisons)
        subset = [r for r in comparisons if r['dataset'] == dataset and not r['method'].endswith('shuffled')]
        print(json.dumps(dict(dataset=dataset,
            mean=100*statistics.mean(r['test'] for r in subset),
            sd=100*statistics.stdev(r['test'] for r in subset),
            max_abs_delta_pp=max(abs(r['delta_pp']) for r in subset),
            assignment_matches=sum(r['ball_assignment_hash_matches'] for r in subset))), flush=True)
    passed = all(r['status'] == 'ok' and abs(r['delta_pp']) < 1e-5
                 and all(r[k] for k in ('split_hash_matches','noise_hash_matches','config_id_matches','ball_assignment_hash_matches'))
                 for r in comparisons)
    save(HERE / 'gate_status.json', dict(passed=passed, rows=len(comparisons)))
    print('GATE ' + ('PASSED' if passed else 'MISMATCH: investigate before matrix'), flush=True)


if __name__ == '__main__':
    main()
