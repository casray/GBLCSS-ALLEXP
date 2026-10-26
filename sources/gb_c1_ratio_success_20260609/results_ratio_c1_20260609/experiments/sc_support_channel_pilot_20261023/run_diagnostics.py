"""Scheduled E1 candidate diagnostics and E2 historical alignment audit.

No neural-network fitting occurs here. The time guard is checked before loading
scientific dependencies or project data. GPU stages follow the frozen runbook.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import logging
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
CONFIG = json.loads((ROOT/'config.json').read_text(encoding='utf-8'))
SOURCES = {}


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def save_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def track(path):
    path = Path(path).resolve()
    digest = sha(path)
    if str(path) in SOURCES and SOURCES[str(path)] != digest:
        raise RuntimeError('Source changed during diagnostics: '+str(path))
    SOURCES[str(path)] = digest
    return digest


def import_file(name, path):
    track(path)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def array_hash(values):
    import numpy as np
    return hashlib.sha256(np.asarray(values, dtype=np.int64).tobytes()).hexdigest()


def e1():
    import numpy as np
    import torch
    from scipy.stats import spearmanr

    local = import_file('sc_local_kernels', ROOT/'kernels.py')
    parent = WORKSPACE/'experiments'
    r2 = import_file('sc_r2_source', parent/'r2_support_5dataset_20261022/run.py')
    support = import_file('sc_support_source', parent/'neighbor_support_ablation_20261022/kernels.py')
    g = r2.generator
    for path in [Path(g.__file__), parent/'representation_supervision_20261021/core.py',
                 parent/'new_graph_mlp_seed0_20261009/common.py',
                 WORKSPACE/'GB-LCSS_code/src/gblcss/data.py']:
        track(path)
    torch.set_num_threads(CONFIG['training']['threads'])
    profiles, scores, selections, analysis_inputs = [], [], [], []
    destination = ROOT/'diagnostics/e1'
    for name in CONFIG['datasets']:
        raw = r2.raw_data(name)
        track(r2.DATA_FILES[name])
        h, h_path, meta_path = r2.load_h(name, raw)
        track(h_path)
        track(meta_path)
        classes = int(raw.y.max())+1
        for seed in CONFIG['seeds']:
            train, valid, test = [torch.as_tensor(v).long() for v in r2.percent_split_indices(len(raw.x), seed)]
            old_root = (parent/'neighbor_support_ablation_20261022' if name in ('cora','citeseer','corafull')
                        else parent/'r2_support_5dataset_20261022')
            old_path = old_root/'inputs'/name/f'seed{seed}'/'geometry.pt'
            track(old_path)
            old = torch.load(old_path, map_location='cpu', weights_only=True)
            for key, value in [('train', train), ('valid', valid), ('test', test)]:
                if not torch.equal(old[key], value):
                    raise RuntimeError(f'Historical split differs: {name}/{seed}/{key}')
            legacy = (old['partitions']['SUPPORT'] if 'partitions' in old else old['assignment']).numpy()
            tree = g.build_tree(h.numpy(), train.numpy(), seed=seed, **CONFIG['tree'])
            nodes, partitions = tree['nodes'], {}
            for s in CONFIG['support_grid']:
                leaves, objective = g.select_cut(nodes, support.support_costs(nodes, s))
                a = g.assignment(nodes, leaves, len(raw.x))
                partitions[s] = a
                _, inverse, counts = np.unique(a[train.numpy()], return_inverse=True, return_counts=True)
                votes = counts[inverse]-1
                if votes.min() < CONFIG['target']['min_votes']:
                    raise RuntimeError('Candidate has insufficient donors; do not omit queries')
                profiles.append(dict(dataset=name, seed=seed, support_requested=s, balls=len(leaves),
                    actual_min=int(votes.min()), actual_q25=float(np.quantile(votes,.25)),
                    actual_median=float(np.median(votes)), actual_q75=float(np.quantile(votes,.75)),
                    actual_mean=float(votes.mean()), actual_max=int(votes.max()),
                    fraction_within_factor2=float(((votes >= s/2)&(votes <= 2*s)).mean()),
                    fine_leaves=len(tree['fine_leaves']), cap_reached=len(tree['fine_leaves']) >= CONFIG['tree']['max_leaves'],
                    assignment_sha256=array_hash(a), objective=objective,
                    legacy32_sha256=array_hash(legacy)))
            folder = destination/name/f'seed{seed}'
            folder.mkdir(parents=True, exist_ok=True)
            torch.save(dict(train=train, valid=valid, test=test,
                partitions={s:torch.from_numpy(a) for s,a in partitions.items()},
                legacy32=torch.from_numpy(legacy), h_path=str(h_path), h_sha256=sha(h_path),
                nodes=[{**node, 'members':torch.from_numpy(node['members'])} for node in nodes]), folder/'geometry.pt')
            masks = []
            for ids in (train, valid, test):
                mask = torch.zeros(len(raw.x), dtype=torch.bool)
                mask[ids] = True
                masks.append(mask)
            for condition, (kind, rate) in CONFIG['conditions'].items():
                graph = r2.GraphData(name, raw.x.float(), raw.edge_index.long(), raw.y.long(), *masks,
                    train_index=train, val_index=valid, test_index=test)
                r2.inject_train_label_noise(graph, kind, rate, seed)
                z = graph.noisy_y[train].cpu().long()
                if not torch.equal(graph.y[~masks[0]], graph.noisy_y[~masks[0]]):
                    raise RuntimeError('Noise escaped training split')
                cached = old_root/'inputs'/name/f'seed{seed}'/f'{condition}.pt'
                if cached.exists():
                    track(cached)
                    previous = torch.load(cached, map_location='cpu', weights_only=True)
                    if not torch.equal(previous['observed'], z):
                        raise RuntimeError('Historical noise differs')
                elif condition != 'clean':
                    raise FileNotFoundError('Missing required historical noisy cache: '+str(cached))
                targets, alphas, votes_by_s, cell_scores = {}, {}, {}, {}
                for s, a in partitions.items():
                    counts = local.loo_counts(z.numpy(), a[train.numpy()], classes)
                    winner_prob = local.majority_probabilities(counts)
                    score = local.agreement(winner_prob, z.numpy())
                    target, alpha = g.targets_from_counts(z, torch.from_numpy(counts).float(), **CONFIG['target'])
                    if not torch.isfinite(target).all() or not torch.allclose(target.sum(1), torch.ones(len(z)), atol=3e-7):
                        raise RuntimeError('Invalid target')
                    cell_scores[s] = score
                    targets[s], alphas[s], votes_by_s[s] = target, alpha, torch.from_numpy(counts.sum(1))
                    scores.append(dict(dataset=name,seed=seed,condition=condition,support_requested=s,
                        sc=score,tied_fraction=float(((counts == counts.max(1,keepdims=True)).sum(1)>1).mean())))
                selected = local.choose_support(cell_scores, CONFIG['selector']['tie_tolerance'])
                selections.append(dict(dataset=name,seed=seed,condition=condition,selected_support=selected,
                    sc=cell_scores[selected], selection_source='observed training labels only'))
                target_path = folder/f'{condition}.pt'
                torch.save(dict(observed=z,valid_y=raw.y[valid].cpu().long(),targets=targets,
                    alphas=alphas,supports=votes_by_s,selected_support=selected), target_path)
                analysis_inputs.append((name,seed,condition,train.numpy(),partitions,target_path))
            logging.info('E1 prepared %s seed%s: %s unique candidate cuts',name,seed,
                         len({array_hash(a) for a in partitions.values()}))
    # Freeze every label-only choice before the separate clean-label analysis.
    save_csv(destination/'geometry_profiles.csv', profiles)
    save_csv(destination/'candidate_scores.csv', scores)
    save_json(destination/'selected_support.json', selections)
    selection_digest = sha(destination/'selected_support.json')
    diagnostic_rows, rank_rows = [], []
    for name,seed,condition,train,partitions,path in analysis_inputs:
        clean = r2.raw_data(name).y[train].cpu().numpy()
        cache = torch.load(path, map_location='cpu', weights_only=True)
        z = cache['observed'].numpy()
        classes = cache['targets'][CONFIG['support_grid'][0]].shape[1]
        onehot = np.eye(classes)[clean]
        clean_scores, observed_scores = [], []
        for s, a in partitions.items():
            counts = local.loo_counts(z,a[train],classes)
            vote = local.majority_probabilities(counts)
            per_node = vote[np.arange(len(z)),clean]
            target = cache['targets'][s].numpy()
            brier = ((target-onehot)**2).sum(1)
            clean_scores.append(float(per_node.mean()))
            observed_scores.append(local.agreement(vote,z))
            for stratum, mask in [('all',np.ones(len(z),bool)),('observed_clean',z==clean),('corrupted',z!=clean)]:
                diagnostic_rows.append(dict(dataset=name,seed=seed,condition=condition,support_requested=s,
                    stratum=stratum,nodes=int(mask.sum()),
                    vote_accuracy=float(per_node[mask].mean()) if mask.any() else None,
                    target_brier=float(brier[mask].mean()) if mask.any() else None,
                    class_balanced_vote_accuracy=float(np.mean([per_node[mask&(clean==c)].mean()
                        for c in np.unique(clean[mask])])) if mask.any() else None))
        correlation = (float(spearmanr(clean_scores,observed_scores).statistic)
                       if np.ptp(clean_scores)>0 and np.ptp(observed_scores)>0 else None)
        rank_rows.append(dict(dataset=name,seed=seed,condition=condition,spearman=correlation))
    if sha(destination/'selected_support.json') != selection_digest:
        raise RuntimeError('Posthoc analysis changed selections')
    save_csv(destination/'posthoc_vote_diagnostics.csv',diagnostic_rows)
    save_csv(destination/'score_rank_diagnostics.csv',rank_rows)
    save_json(destination/'audit.json',dict(score_rows=len(scores),selection_rows=len(selections),
        all_historical_splits_and_noise_match=True,selection_sha256=selection_digest,
        new_neural_network_fits=0,clean_truth_used_only_for_corruption_and_posthoc_analysis=True))


def e2():
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    parent = WORKSPACE/'experiments'
    rows, curves = [], []
    fig, axes = plt.subplots(3,2,figsize=(10,10),layout='constrained')
    for index,name in enumerate(CONFIG['datasets']):
        ax = axes.flat[index]
        for seed in CONFIG['seeds']:
            if name in ('coauthorcs','amazonpho'):
                directory = parent/'r2_support_5dataset_20261022/pretraining'/name/f'seed{seed}'
            elif seed == 0:
                directory = parent/'representation_supervision_20261021/pretraining'/name
            else:
                directory = parent/'gce_lcss_fusion_20261022/encoders'/name/f'seed{seed}'
            path = directory/'epochs.jsonl'
            track(path)
            track(directory/'result.json')
            history = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
            epochs = np.array([v['epoch'] for v in history])
            values = np.array([v['evaluation_alignment'] for v in history],dtype=float)
            if not np.array_equal(epochs,np.arange(1,101)) or not np.isfinite(values).all():
                raise RuntimeError('Expected complete finite 100-epoch alignment log: '+str(path))
            best = int(values.argmin())
            rows.append(dict(dataset=name,seed=seed,epoch100_loss=float(values[-1]),
                best_epoch=int(epochs[best]),best_loss=float(values[best]),
                last_minus_best=float(values[-1]-values[best]),
                last10_slope=float(np.polyfit(epochs[-10:],values[-10:],1)[0]),
                source=str(path),interpretation='in-sample all-node alignment, not unseen-node predictability'))
            curves.extend(dict(dataset=name,seed=seed,**v) for v in history)
            ax.plot(epochs,values,label=f'seed {seed}',linewidth=1.1)
        ax.set(title=name,xlabel='Alignment epoch',ylabel='Evaluation alignment loss')
        ax.legend(frameon=False)
        ax.grid(alpha=.2)
    axes.flat[-1].set_visible(False)
    output = ROOT/'diagnostics/e2'
    output.mkdir(parents=True,exist_ok=True)
    fig.savefig(output/'alignment_curves.png',dpi=180)
    fig.savefig(output/'alignment_curves.pdf')
    plt.close(fig)
    save_csv(output/'alignment_summary.csv',rows)
    save_csv(output/'alignment_curves.csv',curves)
    logging.info('E2 audited %s historical alignment logs',len(rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['e1','e2','all'],default='all')
    args = parser.parse_args()
    due = datetime.fromisoformat(CONFIG['not_before'])
    if datetime.now(timezone.utc) < due:
        raise SystemExit('Not started: user requested execution no earlier than '+due.isoformat())
    lock = ROOT/'.diagnostics.lock'
    try:
        lock.mkdir()
    except FileExistsError:
        raise SystemExit('Diagnostic lock already exists; inspect its owner before resuming')
    log_handler = logging.FileHandler(ROOT/'diagnostics_raw.log',encoding='utf-8')
    logging.basicConfig(level=logging.INFO,handlers=[log_handler,logging.StreamHandler(sys.stdout)],
                        format='%(asctime)s %(levelname)s %(message)s')
    try:
        for path in [ROOT/'config.json',ROOT/'PLAN.md',ROOT/'run_diagnostics.py',ROOT/'kernels.py']:
            track(path)
        manifest = ROOT/'diagnostic_sources.json'
        if manifest.exists():
            previous = json.loads(manifest.read_text(encoding='utf-8'))
            for path,digest in previous.items():
                if sha(Path(path)) != digest:
                    raise RuntimeError('Frozen diagnostic source changed: '+path)
            SOURCES.update(previous)
        stages = ['e2','e1'] if args.stage == 'all' else [args.stage]
        for stage in stages:
            done = ROOT/'diagnostics'/f'{stage}_completed.json'
            if done.exists():
                record = json.loads(done.read_text(encoding='utf-8'))
                for path,digest in record['outputs'].items():
                    if sha(Path(path)) != digest:
                        raise RuntimeError('Completed output changed: '+path)
                logging.info('Skip verified completed stage %s',stage)
                continue
            destination = ROOT/'diagnostics'/stage
            if destination.exists() and any(destination.iterdir()):
                raise RuntimeError('Partial stage exists; record a recovery plan before resuming: '+stage)
            save_json(ROOT/'status.json',dict(status='running',stage=stage,started_at=datetime.now(timezone.utc).isoformat()))
            {'e1':e1,'e2':e2}[stage]()
            save_json(manifest,SOURCES)
            outputs = {str(path.resolve()):sha(path) for path in destination.rglob('*') if path.is_file()}
            save_json(done,dict(status='complete',outputs=outputs,completed_at=datetime.now(timezone.utc).isoformat()))
        save_json(ROOT/'status.json',dict(status='diagnostics_complete',
            training_status='not_implemented_or_started_by_this_entry',next='Follow RUNBOOK.md for gated GPU fitting'))
    except Exception:
        error = traceback.format_exc()
        logging.error(error)
        save_json(ROOT/'status.json',dict(status='failed',error=error))
        raise
    finally:
        log_handler.close()
        lock.rmdir()


if __name__ == '__main__':
    main()
