"""Independent stdlib statistics verification and final delivery manifest."""
import csv
import hashlib
import json
import math
from collections import defaultdict, Counter
from datetime import datetime, timezone
from pathlib import Path
import shutil
import statistics

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(name):
    with (ROOT / name).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def writecsv(path, values):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(values[0]))
        w.writeheader()
        w.writerows(values)


def verify_summary(base, target, ids, metrics, seeds):
    expected = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in base:
        seed = int(r['seed'])
        if seed not in seeds:
            continue
        variants = [r]
        if r['condition'] == 'clean':
            variants.append(dict(r, dataset='ALL'))
        else:
            variants.extend([dict(r, condition='NOISY_MACRO'), dict(r, condition='NOISY_MACRO', dataset='ALL')])
        for v in variants:
            key = tuple(v[k] for k in ids)
            for m in metrics:
                expected[key][m][seed].append(float(v[m]))
    assert len(target) == len(expected)
    checks = 0
    for r in target:
        group = expected[tuple(r[k] for k in ids)]
        assert int(r['n_seeds']) == len(seeds)
        for m in metrics:
            averages = {s: statistics.mean(group[m][s]) for s in seeds}
            for suffix, value in [('mean', statistics.mean(averages.values())),
                                  ('std_ddof1', statistics.stdev(averages.values()))]:
                assert math.isclose(float(r[m+'_'+suffix]), value, abs_tol=1e-10)
                checks += 1
            for s, value in averages.items():
                assert math.isclose(float(r[m+f'_seed{s}']), value, abs_tol=1e-10)
                checks += 1
    return checks


def main():
    assert read(ROOT / 'complete.json')['audit'] == 'passed'
    assert not (ROOT / 'run.lock').exists()
    assert not (OUT / 'final_verification.json').exists()
    assert 'Ran 17 tests' in (OUT / 'final_tests.log').read_text() and '\nOK' in (OUT / 'final_tests.log').read_text()
    sources = read(ROOT / 'sources.json')
    assert all(sha(Path(path)) == expected for path, expected in sources.items())
    selected, pairs = rows('selected_results.csv'), rows('paired_differences.csv')
    keys = ['stage', 'dataset', 'seed', 'backbone', 'condition']
    lookup = {tuple(r[k] for k in keys) + (r['arm'],): r for r in selected}
    metrics = ['test_accuracy', 'validation', 'test_macro_f1']
    for r in pairs:
        key = tuple(r[k] for k in keys)
        a, b = lookup[key + (r['selected'],)], lookup[key + (r['reference'],)]
        for m in metrics:
            assert math.isclose(float(r[m+'_diff_pp']), 100*(float(a[m])-float(b[m])), abs_tol=1e-10)
    checks = 0
    for prefix, seeds in [('', (0, 1, 2)), ('seeds12_', (1, 2))]:
        checks += verify_summary(selected, rows(prefix+'summary.csv'),
                                 ['stage', 'dataset', 'backbone', 'condition', 'arm'], metrics, seeds)
        checks += verify_summary(pairs, rows(prefix+'paired_summary.csv'),
                                 ['stage', 'dataset', 'backbone', 'condition', 'selected', 'reference'],
                                 [m+'_diff_pp' for m in metrics], seeds)
    choice_counts = []
    for b in ['R0', 'R2']:
        cells = list((ROOT / 'selections/C1').glob(f'*/seed*/{b}/*_full.json'))
        for subset in [(0, 1, 2), (1, 2)]:
            for condition in ['clean', 'uniform0.4', 'uniform0.6', 'pair0.4', 'NOISY_MACRO']:
                chosen = [p for p in cells if int(p.parts[-3][4:]) in subset and
                          (p.stem[:-5] == condition or condition == 'NOISY_MACRO' and p.stem != 'clean_full')]
                for method in ['PROBE', 'FULL_VAL']:
                    counts = Counter(read(p if method == 'FULL_VAL' else p.with_name(p.name.replace('_full.json', '.json')))[method] for p in chosen)
                    for arm, count in sorted(counts.items()):
                        choice_counts.append(dict(backbone=b, seeds=','.join(map(str, subset)), condition=condition,
                                                  method=method, arm=arm, count=count, total=len(chosen)))
    writecsv(OUT / 'c1_selection_counts.csv', choice_counts)
    costs = rows('c1_costs.csv')
    for r in costs:
        extra = 88 if r['dataset']=='coauthorcs' and r['seed']=='0' and r['backbone']=='R0' and r['condition']=='pair0.4' else 0
        r['observed_interrupted_extra_updates'] = extra
        r['actual_candidate_updates_lower_bound'] = 3600 + extra
        r['unlogged_updates_and_extra_seconds'] = 'unknown' if extra else 'none recorded'
    writecsv(OUT / 'c1_actual_cost_ledger.csv', costs)
    report_path = ROOT / 'REPORT_zh.md'
    content = report_path.read_text(encoding='utf-8')
    content = content.replace('外部进程中断发生在', '训练进程退出发生在')
    content = content.replace('锁PID退出码为1。', '锁PID退出码为1，未落盘异常栈，退出原因未确定。恢复receipt中的external_interruption为恢复分类标签。')
    content = content.replace('### C1 新 Seed 与最差格',
        '新增seed1/2的带噪30格中，R0选择(0,1)/(.25,1)的数量为17/13，R2为25/5；三个噪声型均只出现这两类规则。clean中R0有6/10格、R2有9/10格选择(1,0)或(1,.25)。这支持带噪时重点调整冲突样本、减少同意样本扰动的解释；clean仍有F1代价，后续需专门验证改进。\n\n### C1 新 Seed 与最差格')
    content = content.replace('### 中断恢复与实际成本',
        '新增seed1/2的带噪FULL_VAL中，R0为30/30格、R2为29/30格选lambda1。R0的PROBE与FULL_VAL选定support32均为6/30格；R2分别为13/30与7/30格（后者含1格lambda.5）。当前支持量选择更活跃，力度搜索的额外贡献有限；逐格与分条件计数见postprocess/c1_selection_counts.csv。\n\n### 中断恢复与实际成本')
    report_path.write_text(content, encoding='utf-8')
    verification = dict(finished=datetime.now(timezone.utc).isoformat(), status='passed',
        frozen_sources=len(sources), paired_rows_verified=len(pairs), summary_numeric_checks=checks,
        tests=17, full_fits=3600, nominal_updates=720000, actual_updates_lower_bound=720088,
        source_manifest_sha256=sha(ROOT / 'sources.json'), free_bytes=shutil.disk_usage(ROOT).free,
        parent_independent_audit='performed separately by parent; not claimed here')
    (OUT / 'final_verification.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
    (OUT / 'aliases.json').write_text(json.dumps(dict(historical_reuse=False, aliases=[], unique_fits=3600), indent=2), encoding='utf-8')
    (ROOT / 'HANDOFF.md').write_text(
        '# D1/C1 confirmation: COMPLETE\n\n'
        '3600/3600 scientific fits complete (D1 1440, C1 2160); all 200 epochs. Six two-epoch smoke fits.\n'
        'Original audit passed: 3600 artifacts, 120 races, 120 exact-mass controls, 60 reloads, max error 0.\n'
        'Postprocess/final_verification.json: independent stdlib paired/statistical checks and 17 fresh tests passed.\n'
        'Frozen runtime sources unchanged. Training process exited normally after recovery. No active run.lock.\n\n'
        'Read REPORT_zh.md, postprocess/primary_control_warnings.csv, d1_winner_counts.csv, c1_selection_counts.csv.\n'
        'R0 clean macro-F1 cost vs CELL_GLOBAL_VAL on Cora/CiteSeer is prominent; newseed breakdown included.\n'
        'Report generator decision thresholds are descriptive suggestions, not additional PLAN hard gates.\n\n'
        'Interruption: recovery/interruption_01/ retains original files and stale lock. Own epoch75 prefix restored;\n'
        'logged epochs76-163 replay byte-identically. Actual updates >=720088, including 88 observed extra updates;\n'
        'unlogged updates and interrupted segment seconds unknown. Corrected cost ledger under postprocess/.\n'
        'Parent audit remains independently owned in research/ and was not edited or invoked here.\n', encoding='utf-8')
    # Include final artifacts only after all writes; exclude the manifest's self-reference.
    files = sorted(p for p in ROOT.rglob('*') if p.is_file() and p != ROOT / 'DELIVERY_SHA256.csv')
    manifest = [dict(path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size, sha256=sha(p)) for p in files]
    writecsv(ROOT / 'DELIVERY_SHA256.csv', manifest)
    assert all((ROOT / r['path']).stat().st_size == r['bytes'] for r in manifest)
    print(json.dumps(dict(verification, delivery_files=len(manifest)), indent=2))


if __name__ == '__main__':
    main()
