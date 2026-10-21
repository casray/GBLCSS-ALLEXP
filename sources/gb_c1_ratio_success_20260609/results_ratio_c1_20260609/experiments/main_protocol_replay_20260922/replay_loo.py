import json
import os
import statistics
import subprocess
from replay_gate import HERE, NOISYGL, ARCHIVE, PYTHON, rows, save


def main():
    folder = HERE / 'formal_loo'
    folder.mkdir(exist_ok=True)
    reference = rows(NOISYGL / 'runs/v31_tiesafe_affected_20260728/loo_uniform0p4.csv')
    specs = [r for r in rows(ARCHIVE / 'command_manifest.csv')
             if r['regime'] == 'uniform0.4' and r['group'] == 'softdist']
    comparisons = []
    for spec in specs:
        dataset = spec['dataset']
        command = json.loads(spec['command_json'])
        command[0] = str(PYTHON)
        command[command.index('--base_methods') + 1] = 'frozengrace_gb_mlp_softdist_beta05_loo,frozengrace_gb_mlp_softdist_beta05_loo_shuffled'
        output = folder / (dataset + '.csv')
        command[command.index('--output') + 1] = str(output)
        env = os.environ.copy()
        flags = json.loads(spec['deterministic_env_json'])
        env.update(flags)
        save(folder / (dataset + '_command.json'), dict(command=command, flags=flags))
        if not output.exists():
            print('START formal LOO ' + dataset, flush=True)
            with (folder / (dataset + '.stdout.log')).open('w') as out, (folder / (dataset + '.stderr.log')).open('w') as err:
                subprocess.run(command, cwd=NOISYGL, env=env, stdout=out, stderr=err, check=True)
        actual = rows(output)
        assert len(actual) == 10
        for r in actual:
            e = next(e for e in reference if e['dataset'] == dataset and e['method'] == r['base_method'] and e['seed'] == r['seed'])
            assert r['status'] == 'ok', r
            entry = dict(dataset=dataset, method=r['base_method'], seed=int(r['seed']), test=float(r['test']), archived_test=float(e['test']))
            entry['delta_pp'] = 100 * (entry['test']-entry['archived_test'])
            for field in ('split_hash', 'noise_hash', 'config_id', 'ball_assignment_hash'):
                entry[field + '_matches'] = r[field] == e[field]
            comparisons.append(entry)
        save(folder / 'comparison.json', comparisons)
        current = [r for r in comparisons if r['dataset'] == dataset and not r['method'].endswith('shuffled')]
        print(json.dumps(dict(dataset=dataset, mean=100*statistics.mean(r['test'] for r in current), std=100*statistics.stdev(r['test'] for r in current), max_delta_pp=max(abs(r['delta_pp']) for r in current), assignment_matches=sum(r['ball_assignment_hash_matches'] for r in current))), flush=True)
    save(folder / 'status.json', dict(rows=len(comparisons), exact_tests=all(abs(r['delta_pp']) < 1e-5 for r in comparisons)))


if __name__ == '__main__':
    main()
