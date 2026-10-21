"""One entry point for paper tables, target diagnostics and grouping costs."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path

import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from .data import inject_train_label_noise, load_public_dataset, unique_undirected_edge_statistics
from .experiment import load_config
from .granular_ball import build_granular_balls
from .representation import load_frozen_representation
from .runner import REPRESENTATION_METHODS, run_method, validate_main_config
from .suites import build_plan
from .targets import build_leave_one_out_targets

ROOT = Path(__file__).resolve().parents[2]
KEY = ['suite', 'dataset', 'method', 'noise_type', 'noise_rate', 'seed', 'validation_protocol', 'variant']


def save_tables(records, destination):
    from .reporting import summarize
    frame = pd.DataFrame(records)
    if frame.empty:
        return
    frame.to_csv(destination / 'per_seed.csv', index=False)
    trained = frame.loc[frame.task == 'train'].dropna(axis=1, how='all')
    if not trained.empty:
        summarize(trained, destination / 'tables')
    for task in ['targets', 'statistics', 'cost']:
        subset = frame.loc[frame.task == task].dropna(axis=1, how='all')
        if subset.empty:
            continue
        subset.to_csv(destination / f'{task}_per_seed.csv', index=False)
        group = ['dataset', 'method', 'noise_type', 'noise_rate', 'variant']
        metrics = [col for col in subset.select_dtypes('number') if col not in KEY]
        subset.groupby(group)[metrics].agg(['mean', 'std']).to_csv(destination / f'{task}_summary.csv')
        if task == 'targets':
            selected = subset.loc[(subset.noise_type == 'uniform') & (subset.noise_rate == .4)]
            columns = ['noisy_onehot_brier', 'soft_target_brier', 'corrupted_brier_reduction',
                       'correct_brier_increase', 'net_brier_reduction']
            selected.groupby('dataset')[columns].mean().to_csv(destination / 'table_i2.csv')


def execute(job, graph, representation, config, adaptations, device):
    settings = config['method']
    if job['task'] == 'statistics':
        edges, homophily = unique_undirected_edge_statistics(graph.edge_index, graph.y)
        return dict(nodes=graph.num_nodes, edges=edges, features=graph.x.shape[1],
                    classes=graph.num_classes, homophily=homophily)
    if job['task'] == 'cost':
        from .benchmark import benchmark
        return benchmark(representation, graph.train_index, job['method'], settings, job['seed'])
    graph = inject_train_label_noise(graph, job['noise_type'], job['noise_rate'], job['seed'],
        noise_scope='train_val' if job['validation_protocol'] == 'noisy' else 'train')
    if job['task'] == 'targets':
        from .diagnostics import target_metrics
        balls = build_granular_balls(representation, max_balls=settings['max_balls'],
            min_ball_size=settings['min_ball_size'], split_improvement=settings['split_improvement'],
            random_state=job['seed'])
        targets = build_leave_one_out_targets(graph.noisy_y.to(device), balls.assignment,
            graph.train_index, graph.num_classes, min_train_per_ball=settings['min_train_per_ball'],
            beta=settings['beta'])
        return target_metrics(graph, balls.assignment, targets, min_support=settings['min_train_per_ball'])
    result, diagnostics = run_method(job['method'], graph, representation, config,
        seed=job['seed'], noise_rate=job['noise_rate'], device=device, adaptations=adaptations)
    details = {key: value for key, value in diagnostics.items()
               if isinstance(value, (float, int))}
    return dict(details, test_accuracy=result.test_accuracy, validation_accuracy=result.validation_accuracy,
                best_epoch=result.best_epoch)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', default='matched', help='Comma-separated names, or all; see --list.')
    parser.add_argument('--list', action='store_true', help='List suites and corresponding paper tables.')
    parser.add_argument('--datasets', help='Optional comma-separated dataset subset.')
    parser.add_argument('--seeds', help='Optional comma-separated seeds; default 0,1,2,3,4.')
    parser.add_argument('--device', default='cpu', help='cpu or cuda (also used by PyTorch on ROCm).')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--epochs', type=int, help='Override the 200-epoch training budget for a short run.')
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data')
    parser.add_argument('--representation', help='Alternative representation path, with optional {dataset}, {seed}.')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/paper')
    parser.add_argument('--dry-run', action='store_true', help='Print the experiment plan without downloading or training.')
    parser.add_argument('--resume', action='store_true', help='Continue the same run, skipping completed records.')
    parser.add_argument('--summarize', action='store_true', help='Regenerate tables from this output directory.')
    args = parser.parse_args()
    catalog = load_config(ROOT / 'configs/experiments.yaml')
    if args.list:
        for name, spec in catalog['suites'].items():
            print(f"{name:18} {spec['tables']}")
        return
    journal = args.output / 'runs.jsonl'
    if args.summarize:
        records = [json.loads(line) for line in journal.read_text(encoding='utf-8').splitlines()]
        save_tables(records, args.output)
        return
    if args.threads < 1:
        parser.error('threads must be positive')
    config = load_config(ROOT / 'configs/main/gblcss.yaml')
    config['training']['device'] = args.device
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    validate_main_config(config)
    datasets = [x.strip() for x in args.datasets.split(',')] if args.datasets else None
    seeds = list(map(int, args.seeds.split(','))) if args.seeds else None
    plan = build_plan(ROOT, args.suite.split(','), datasets, seeds)
    if args.dry_run:
        print(pd.DataFrame(plan).groupby(['suite', 'dataset', 'task']).size().to_string())
        print(f"Total: {len(plan)} runs; {config['training']['epochs']} epochs per training stage")
        return
    adaptations = load_config(ROOT / 'configs/main/protocol_adaptations.yaml')
    manifest = dict(config=config, adaptations=adaptations, plan=plan, threads=args.threads,
                    representation=args.representation, data_root=str(args.data_root.resolve()))
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_file = args.output / 'config.json'
    records = []
    if journal.exists() or manifest_file.exists():
        if not args.resume:
            parser.error('output exists; choose another --output or use --resume')
        if json.loads(manifest_file.read_text(encoding='utf-8')) != manifest:
            parser.error('resume requires the same configuration and experiment selection')
        if journal.exists():
            records = [json.loads(line) for line in journal.read_text(encoding='utf-8').splitlines()]
    else:
        manifest_file.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    completed = {tuple(row[key] for key in KEY) for row in records}
    torch.set_num_threads(args.threads)
    current = None
    graph = representation = None
    computed = {}
    with threadpool_limits(limits=args.threads), journal.open('a', encoding='utf-8') as handle:
        for number, job in enumerate(plan, 1):
            if tuple(job[key] for key in KEY) in completed:
                continue
            identity = (job['dataset'], job['seed'])
            if identity != current:
                dataset = load_config(ROOT / 'configs/datasets' / f"{job['dataset']}.yaml")
                graph = load_public_dataset(dataset['name'], args.data_root, job['seed'])
                representation = None
                current = identity
            needs_representation = (job['method'] in REPRESENTATION_METHODS or job['task'] in {'targets', 'cost'})
            if needs_representation and representation is None:
                path = (Path(args.representation.format(dataset=job['dataset'], seed=job['seed']))
                        if args.representation else ROOT / dataset['representation'])
                representation = load_frozen_representation(path, graph.num_nodes)
            settings = deepcopy(config)
            settings['method'].update(job['overrides'])
            # Several paper tables share a run. Compute it once within this invocation.
            cache_key = (job['task'], job['dataset'], job['seed'], job['method'],
                         job['noise_type'], job['noise_rate'], job['validation_protocol'],
                         json.dumps(settings, sort_keys=True))
            if cache_key not in computed:
                computed[cache_key] = execute(job, graph, representation, settings, adaptations, args.device)
            outcome = computed[cache_key]
            row = {key: value for key, value in job.items() if key != 'overrides'}
            row.update(outcome, epochs=config['training']['epochs'])
            handle.write(json.dumps(row, allow_nan=False) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
            records.append(row)
            score = f" accuracy={100 * row['test_accuracy']:.2f}%" if 'test_accuracy' in row else ''
            print(f"[{number}/{len(plan)}] {job['suite']} {identity} {job['method']} {job['variant']}{score}", flush=True)
    save_tables(records, args.output)
    print(f'Results: {args.output.resolve()}')
