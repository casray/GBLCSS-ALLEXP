"""Incremental descriptive summaries; absent observations are never imputed."""
import argparse
import csv
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics


GROUP_FIELDS = ['dataset', 'noise', 'backbone', 'arm', 'ratio', 'rule']
METRICS = ['test_accuracy', 'cold_total_s', 'run_total_s', 'gpu_peak_allocated_gib', 'parent_process_wall_s']


def metric_value(result, metric):
    if metric.startswith('amortized_total_s_K'):
        return result.get('amortized_total_s', {}).get(metric.split('_K', 1)[1])
    return result.get(metric)


def number(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


def stats(values):
    valid = [v for value in values if (v := number(value)) is not None]
    return (len(valid), statistics.mean(valid) if valid else 'NA',
            statistics.stdev(valid) if len(valid) > 1 else 'NA')


def paired_rows(results, metrics=None):
    metrics = metrics or METRICS
    index = {}
    for result in results:
        job = result['job']
        key = tuple(job.get(f) for f in ['dataset', 'noise', 'seed', 'backbone', 'ratio', 'arm', 'rule'])
        if key in index:
            raise ValueError('Duplicate completed result pairing key: ' + repr(key))
        index[key] = result
    groups = defaultdict(list)
    for key, candidate in index.items():
        dataset, noise, seed, backbone, ratio, arm, rule = key
        if arm == 'C' and rule == 'R0':
            references = [(other, 'R0') for other in ['A', 'B', 'Bprime', 'LCSS', 'D']]
        elif arm == 'C':
            references = [('C', 'R0')]
        elif arm == 'D' and rule == 'R0':
            references = [('A', 'R0')]
        else:
            continue
        for other, other_rule in references:
            reference_ratio = .1 if other == 'A' else ratio
            reference = index.get((dataset, noise, seed, backbone, reference_ratio, other, other_rule))
            if reference is None:
                continue
            candidate_accuracy, reference_accuracy = number(candidate.get('test_accuracy')), number(reference.get('test_accuracy'))
            if candidate_accuracy is None or reference_accuracy is None:
                continue
            comparison = arm + '_' + rule + '-' + other + '_' + other_rule
            delta = {'seed': seed, 'delta_pp': 100 * (candidate_accuracy - reference_accuracy)}
            for metric in metrics[1:]:
                a, b = number(metric_value(candidate, metric)), number(metric_value(reference, metric))
                delta[metric + '_delta'] = a - b if a is not None and b is not None else None
            groups[(dataset, noise, backbone, ratio, reference_ratio, comparison)].append(delta)
    rows = []
    for key, deltas in sorted(groups.items(), key=lambda item: repr(item[0])):
        row = dict(zip(['dataset', 'noise', 'backbone', 'ratio', 'reference_ratio', 'comparison'], key))
        deltas.sort(key=lambda d: d['seed'])
        n, mean, std = stats([d['delta_pp'] for d in deltas])
        row.update(n_pairs=n, delta_pp_mean=mean, delta_pp_std=std,
                   paired_seeds=json.dumps([d['seed'] for d in deltas]),
                   per_seed_deltas=json.dumps(deltas, allow_nan=False),
                   interpretation='3-seed descriptive screen only' if n >= 3 else 'partial; fewer than 3 matched seeds')
        for metric in metrics[1:]:
            count, mean, std = stats([d[metric + '_delta'] for d in deltas])
            row.update({metric + '_delta_n': count, metric + '_delta_mean': mean, metric + '_delta_std': std})
        rows.append(row)
    return rows


def _read_json(path, fallback=None):
    if not path.exists():
        return fallback
    return json.loads(path.read_text(encoding='utf-8'))


def _csv(path, rows, fields):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def summarize(run_root):
    root = Path(run_root)
    jobs = _read_json(root / 'jobs.json', [])
    status = _read_json(root / 'status.json', {})
    protocol = _read_json(root / 'protocol.json', {})
    config = _read_json(root / 'config.json', {})
    metrics = METRICS + ['amortized_total_s_K' + str(k) for k in config.get('amortization_K', [1, 3, 10, 30])]
    completed, runs, invalid = [], [], []
    for job in jobs:
        path = root / 'results' / (job['id'] + '.json')
        try:
            result = _read_json(path, {})
            result_status = result.get('status', 'missing')
            reason = result.get('reason', result.get('error', ''))
            if (result_status == 'missing' and status.get('status') == 'failed'
                    and isinstance(status.get('current'), dict) and status['current'].get('id') == job['id']):
                result_status, reason = 'failed', status.get('reason', 'unknown worker failure')
            if result_status == 'completed':
                if result.get('job') != job:
                    raise ValueError('result job mismatch')
                for field in ['config_sha256', 'source_sha256']:
                    if field in protocol and result.get(field) != protocol[field]:
                        raise ValueError(field + ' mismatch')
                from run_suite import validate_prediction_artifact
                validate_prediction_artifact(result, root)
                completed.append(result)
        except (ValueError, OSError, TypeError) as exc:
            result, result_status, reason = {}, 'invalid', str(exc)
            invalid.append(job['id'] + ': ' + reason)
        row = dict(job, status=result_status, reason=reason)
        for metric in metrics:
            value = metric_value(result, metric)
            row[metric] = value if number(value) is not None else 'NA'
        runs.append(row)
    groups = defaultdict(list)
    for result in completed:
        groups[tuple(result['job'].get(f) for f in GROUP_FIELDS)].append(result)
    summary_rows = []
    for key, values in sorted(groups.items(), key=lambda item: repr(item[0])):
        row = dict(zip(GROUP_FIELDS, key))
        seeds = sorted({r['job']['seed'] for r in values})
        row.update(n_completed=len(values), n_seeds=len(seeds), seeds=json.dumps(seeds),
                   interpretation='3-seed descriptive screen only' if len(seeds) >= 3 else 'partial; fewer than 3 seeds')
        for metric in metrics:
            n, mean, std = stats([metric_value(r, metric) for r in values])
            row.update({metric + '_n': n, metric + '_mean': mean, metric + '_std': std})
        summary_rows.append(row)
    summary_fields = GROUP_FIELDS + ['n_completed', 'n_seeds', 'seeds', 'interpretation']
    summary_fields += [m + suffix for m in metrics for suffix in ['_n', '_mean', '_std']]
    _csv(root / 'summary.csv', summary_rows, summary_fields)
    _csv(root / 'runs.csv', runs, ['id', 'dataset', 'noise', 'seed', 'backbone', 'arm', 'ratio', 'rule', 'experiment', 'status', 'reason'] + metrics)
    pairs = paired_rows(completed, metrics)
    pair_fields = ['dataset', 'noise', 'backbone', 'ratio', 'reference_ratio', 'comparison', 'n_pairs', 'delta_pp_mean', 'delta_pp_std', 'paired_seeds', 'per_seed_deltas', 'interpretation']
    pair_fields += [m + '_delta' + suffix for m in metrics[1:] for suffix in ['_n', '_mean', '_std']]
    _csv(root / 'paired_deltas.csv', pairs, pair_fields)
    counts = {name: sum(r['status'] == name for r in runs) for name in ['missing', 'failed', 'invalid']}
    full = len(completed) == len(jobs) and bool(jobs)
    lines = ['# 实验进度（描述性汇总）', '',
             '更新时间：' + datetime.now(timezone.utc).isoformat(), '',
             f"完成 {len(completed)}/{len(jobs)}；队列状态：{status.get('status', 'unknown')}。",
             '当前为完整清单。' if full else '当前结果不完整，未完成组合不得视为零收益或成功。', '',
             f"结果缺失 {counts['missing']}；显式失败 {counts['failed']}；来源/格式无效 {counts['invalid']}。",
             '失败原因：' + str(status.get('reason', '无已记录的队列失败；未运行作业的结果未知。')), '',
             'summary.csv 保留所有已完成臂的均值和样本标准差，精度为 [0,1] 比例；缺失指标记 NA。',
             'runs.csv 保留每个计划作业的状态；原始 JSON 和日志为追溯依据。',
             'paired_deltas.csv 按相同数据集、噪声、seed、骨干和 ratio 精确配对；delta_pp 为前项减后项的百分点差。',
             'C/D 对 A 的比较复用同条件 ratio=.1 的 A，敏感性预算明确标注 reference_ratio=.1；其余比较保持 ratio 相同。',
             'R3/O2 的 C 只与已有 R0 C 配对；R0 不重复训练。负差与负结果全部保留。', '',
             '不足 3 个匹配 seed 仅为部分读数，不宣称稳定。3 seed 仅用于描述性筛查，不能据此宣称统计显著或等价。',
             '±0.5pp 仅为实际差异描述带；进入该范围不证明等价。O2 是 clean-label 切面诊断，不能称性能上界。',
             '完整成本、执行耗时、显存与精度分别报告；未知或失败的组合不作结论。', '']
    lines += ['cold_total_s 为必要阶段成本重建总和，包含准备公共开销和 worker 启动开销；它不是独立无缓存端到端冷跑的直接测量。',
              '摊销 K=' + str(config.get('amortization_K', [1, 3, 10, 30])) + ' 是同图、同表示和同候选树的假设性多任务成本；标签相关切面逐任务计费。原节点对照也按自身依赖允许复用。',
              '三个 seed 的成本汇总不等于同一条件的三次独立 timing repeat。', '']
    if invalid:
        lines += ['无效结果：'] + ['- ' + item for item in invalid] + ['']
    temporary = root / 'PROGRESS.md.tmp'
    temporary.write_text('\n'.join(lines), encoding='utf-8')
    temporary.replace(root / 'PROGRESS.md')
    return dict(completed=len(completed), total=len(jobs), paired_groups=len(pairs), **counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.run_root), ensure_ascii=False))


if __name__ == '__main__':
    main()
