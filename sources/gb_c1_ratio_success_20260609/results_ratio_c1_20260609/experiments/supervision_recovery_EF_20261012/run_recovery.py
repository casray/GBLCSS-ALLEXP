from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import gc
import json
import logging
import math
from pathlib import Path
import platform
import sys
import time
import traceback

import torch

ROOT = Path(__file__).resolve().parent
ORACLE = ROOT.parent / 'oracle_teacher_transfer_20261012'
sys.path.insert(0, str(ORACLE))
import run_oracle as oracle

common, previous = oracle.common, oracle.previous
read, write, sha = oracle.read, oracle.write, oracle.sha
DATASETS = ['cora', 'citeseer', 'corafull']
RATES = [.4, .6]
SEEDS = [1, 2, 3]
ARMS = ['A_noisy', 'B_all_clean', 'C_oracle_filter', 'E_correct', 'F_delete']
SETTINGS = dict(oracle.SETTINGS, datasets=DATASETS, rates=RATES, seeds=SEEDS,
                stage='three-seed realistic recovery E/F confirmation', coverage=.2,
                ranking='1 - A probability of observed label', seconds_budget=7200)
for obsolete in ('protocol_seed', 'teacher_seed', 'student_seed', 'random_subset_seed'):
    SETTINGS.pop(obsolete, None)


def select_suspects(train, observed, probabilities, coverage=.2):
    if not 0 < coverage < 1 or len(train) == 0:
        raise ValueError('coverage must be strictly between zero and one on nonempty training data')
    local = probabilities[train]
    labels = observed[train]
    if not torch.isfinite(local).all() or (local < 0).any():
        raise ValueError('invalid training probabilities')
    if not torch.allclose(local.sum(1), torch.ones(len(train)), atol=1e-5):
        raise ValueError('probabilities must sum to one')
    if labels.min() < 0 or labels.max() >= local.shape[1]:
        raise ValueError('training label outside class range')
    score = 1-local[torch.arange(len(train)), labels]
    order = torch.argsort(score, descending=True, stable=True)
    return train[order[:math.ceil(coverage*len(train))]]


def realistic_intervention(train, observed, probabilities, arm, coverage=.2):
    marked = select_suspects(train, observed, probabilities, coverage)
    if arm == 'E_correct':
        selected = train.clone()
        permitted = torch.zeros_like(observed)
        permitted[train] = observed[train]
        permitted[marked] = probabilities[marked].argmax(1)
    elif arm == 'F_delete':
        keep = torch.ones(len(observed), dtype=torch.bool)
        keep[marked] = False
        selected = train[keep[train]]
        permitted = torch.zeros_like(observed)
        permitted[selected] = observed[selected]
    else:
        raise ValueError(arm)
    return selected, permitted, marked


def cells():
    for dataset in DATASETS:
        for rate in RATES:
            for seed in SEEDS:
                yield dataset, rate, seed


def base_folder(dataset, rate, seed):
    return ROOT/'protocols'/dataset/f'uniform{rate}'/f'seed{seed}'


def folder_for(dataset, rate, seed, arm):
    rate = .4 if arm == 'B_all_clean' else rate
    return ROOT/'runs'/dataset/f'uniform{rate}'/f'seed{seed}'/arm


def pair_location(dataset, rate, seed, arm):
    folder = folder_for(dataset, rate, seed, arm)
    if (folder/'reuse.json').exists():
        return Path(read(folder/'reuse.json')['source'])
    return folder


def load_protocol(dataset, rate, seed):
    return torch.load(base_folder(dataset, rate, seed)/'original.pt', weights_only=True)


def make_data(dataset, rate, seed, arm, probabilities=None):
    protocol = load_protocol(dataset, rate, seed)
    train, clean, noisy = [protocol[k] for k in ('train', 'clean', 'noisy')]
    classes = read(base_folder(dataset, rate, seed)/'original.json')['classes']
    marked = torch.empty(0, dtype=torch.long)
    if arm in ('E_correct', 'F_delete'):
        if probabilities is None:
            probabilities = torch.load(pair_location(dataset, rate, seed, 'A_noisy')/'teacher/probabilities.pt', weights_only=True)
        selected, permitted, marked = realistic_intervention(train, noisy, probabilities, arm)
    else:
        selected, permitted = oracle.intervention(train, clean, noisy, classes, arm)
    counts = oracle.class_counts(permitted, selected, classes)
    if counts.min() == 0:
        raise ValueError(f'{dataset} {rate} seed{seed} {arm}: missing prototype class {counts.tolist()}')
    visible = oracle.sanitized_targets(selected, permitted, protocol['val'], clean)
    x, edge = oracle.raw_graph(dataset)
    masks = []
    for indices in (selected, protocol['val'], protocol['test']):
        mask = torch.zeros(len(x), dtype=torch.bool)
        mask[indices] = True
        masks.append(mask)
    data = previous.GraphData(dataset, x, edge, visible, *masks, noisy_y=permitted,
              train_index=selected, val_index=protocol['val'], test_index=protocol['test'])
    assert data.num_classes == classes
    assert (data.y[data.test_index] == -1).all()
    assert data.noisy_y[~data.train_mask].sum() == 0
    info = dict(dataset=dataset, rate=rate, seed=seed, arm=arm, classes=classes,
           original_train=len(train), supervised=len(selected), retained_class_counts=counts.tolist(),
           effective_input_hash=oracle.effective_hash(selected, permitted, protocol['val'], visible),
           split_hash=read(base_folder(dataset, rate, seed)/'original.json')['split_hash'])
    if arm in ('E_correct', 'F_delete'):
        info.update(marked_count=len(marked), marked_hash=previous.fingerprint(marked),
                    ranking_teacher_hash=sha(pair_location(dataset, rate, seed, 'A_noisy')/'teacher/probabilities.pt'),
                    coverage=.2, ranking=SETTINGS['ranking'])
    return data, info, marked


def save_intervention(dataset, rate, seed, arm, data, info, marked):
    folder = base_folder(dataset, rate, seed)
    path = folder/f'{arm}.pt'
    content = dict(selected=data.train_index, noisy_y=data.noisy_y, visible_y=data.y,
                   train_mask=data.train_mask, val=data.val_index, test=data.test_index, marked=marked)
    if path.exists():
        assert read(path.with_suffix('.json')) == info
        previous_content = torch.load(path, weights_only=True)
        assert all(torch.equal(value, previous_content[key]) for key, value in content.items())
        manifest = read(path.with_suffix('.freeze.json'))
        assert sha(path) == manifest['protocol_sha256']
    else:
        torch.save(content, path)
        write(path.with_suffix('.json'), info)
        write(path.with_suffix('.freeze.json'), dict(protocol_sha256=sha(path), config_sha256=sha(path.with_suffix('.json'))))
    manifest = read(path.with_suffix('.freeze.json'))
    assert sha(path.with_suffix('.json')) == manifest['config_sha256']


def verify():
    for path, expected in read(ROOT/'frozen_inputs.json')['hashes'].items():
        if sha(path) != expected:
            raise RuntimeError(f'frozen source changed: {path}')


def prepare():
    if (ROOT/'frozen_inputs.json').exists():
        verify()
        return
    oracle.verify()
    common.old.require_gpu()
    paths = list(ROOT.glob('*.py'))+[ROOT/'PLAN.md', ORACLE/'frozen_inputs.json']
    paths += [Path(path) for path in read(ORACLE/'frozen_inputs.json')['hashes']]
    checks = []
    for dataset, rate, seed in cells():
        data, source_info, _ = common.load_data(dataset, rate, seed)
        folder = base_folder(dataset, rate, seed)
        folder.mkdir(parents=True, exist_ok=True)
        protocol = dict(train=data.train_index, val=data.val_index, test=data.test_index,
                        clean=data.y, noisy=data.noisy_y)
        torch.save(protocol, folder/'original.pt')
        info = dict(source_info, classes=data.num_classes, training_truth_used_for_EF=False)
        write(folder/'original.json', info)
        paths += [folder/'original.pt', folder/'original.json', previous.DATA_FILES[dataset],
                  common.MATRIX/dataset/'default'/f'seed{seed}.json',
                  Path(source_info['partition_source'])]
        for arm in ARMS[:3]:
            intervention, meta, marked = make_data(dataset, rate, seed, arm)
            save_intervention(dataset, rate, seed, arm, intervention, meta, marked)
            gpu = previous.gpu_data(intervention)
            conf = previous.erase_settings(dataset)
            labels, _ = previous.erase_preprocess(gpu, conf['alpha'], conf['T'])
            counts = torch.bincount(labels[gpu.train_mask].argmax(1), minlength=meta['classes'])
            assert counts.min() > 0, (dataset, rate, seed, arm, counts)
            checks.append(dict(**meta, preprocessed_class_counts=counts.cpu().tolist()))
            path = folder/f'{arm}.pt'
            paths += [path, path.with_suffix('.json'), path.with_suffix('.freeze.json')]
            if seed == 1:
                old_folder = oracle.run_folder(dataset, rate, arm)
                for role in ('teacher', 'student'):
                    config = read(old_folder/role/'config.json')
                    assert config['effective_input_hash'] == meta['effective_input_hash']
                    assert read(old_folder/role/'result.json')['status'] == 'ok'
                    paths += list((old_folder/role).glob('*'))
                new_folder = folder_for(dataset, rate, seed, arm)
                new_folder.mkdir(parents=True, exist_ok=True)
                write(new_folder/'reuse.json', dict(source=str(old_folder),
                      reason='identical sanitized inputs, seed and unchanged training implementation',
                      effective_input_hash=meta['effective_input_hash']))
                paths.append(new_folder/'reuse.json')
            del gpu, labels
            torch.cuda.empty_cache()
    write(ROOT/'environment.json', dict(executable=sys.executable, python=sys.version,
          torch=torch.__version__, hip=torch.version.hip, gpu=torch.cuda.get_device_name(0),
          platform=platform.platform(), settings=SETTINGS))
    paths.append(ROOT/'environment.json')
    write(ROOT/'preflight.json', dict(passed=True, original_cells=18, oracle_interventions=checks,
          expected_new_pairs=66, expected_reused_pairs=15, test_labels_hidden_from_fit=True))
    write(ROOT/'frozen_inputs.json', dict(settings=SETTINGS,
          hashes={str(path.resolve()): sha(path) for path in sorted(set(paths)) if path.is_file()}))
    logging.info('PREPARED 18 cells, 66 new and 15 reused teacher/student pairs')


def fit_pair(dataset, rate, seed, arm, data, info, folder):
    if (folder/'reuse.json').exists():
        return
    if (folder/'student/result.json').exists() and read(folder/'student/result.json')['status'] == 'ok':
        assert read(folder/'teacher/config.json')['effective_input_hash'] == info['effective_input_hash']
        return
    oracle.SEED = seed
    oracle.SETTINGS = dict(SETTINGS, protocol_seed=seed, teacher_seed=seed, student_seed=seed)
    gpu = previous.gpu_data(data)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder/'console.log').open('a', encoding='utf-8') as log, redirect_stdout(log):
        teacher = oracle.teacher_fit(gpu, info, folder/'teacher')
        oracle.student_fit(gpu, teacher, info, folder/'student')
    del teacher, gpu
    gc.collect()
    torch.cuda.empty_cache()


def grid():
    verify()
    if (ROOT/'execution.json').exists():
        raise RuntimeError('completed grid cannot be overwritten')
    started = time.perf_counter()
    pairs_done = 0
    for dataset, rate, seed in cells():
        for arm in ARMS:
            if time.perf_counter()-started > SETTINGS['seconds_budget']:
                raise TimeoutError('registered grid budget exceeded')
            data, info, marked = make_data(dataset, rate, seed, arm)
            save_intervention(dataset, rate, seed, arm, data, info, marked)
            folder = folder_for(dataset, rate, seed, arm)
            if arm == 'B_all_clean' and rate == .6:
                actual = pair_location(dataset, rate, seed, arm)
                assert read(actual/'teacher/config.json')['effective_input_hash'] == info['effective_input_hash']
                logging.info('REUSE B %s seed%s for noise .6', dataset, seed)
                continue
            if (folder/'reuse.json').exists():
                logging.info('REUSE historical %s %.1f seed%s %s', dataset, rate, seed, arm)
                continue
            pairs_done += 1
            logging.info('START %s/66 %s %.1f seed%s %s supervised=%s marked=%s',
                         pairs_done, dataset, rate, seed, arm, info['supervised'], len(marked))
            fit_pair(dataset, rate, seed, arm, data, info, folder)
            logging.info('DONE %s/66 %s %.1f seed%s %s teacher_epoch=%s student_epoch=%s',
                  pairs_done, dataset, rate, seed, arm,
                  read(folder/'teacher/result.json')['best_epoch'], read(folder/'student/result.json')['best_epoch'])
            del data
    verify()
    write(ROOT/'execution.json', dict(status='ok', new_pairs=pairs_done, reused_pairs=15,
          logical_pairs=90, seconds=time.perf_counter()-started))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'grid'])
    arguments = parser.parse_args()
    torch.set_num_threads(4)
    common.old.require_gpu()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
          handlers=[logging.StreamHandler(), logging.FileHandler(ROOT/'raw_log.txt', encoding='utf-8')])
    try:
        prepare() if arguments.mode == 'prepare' else grid()
    except Exception:
        write(ROOT/'failure.json', dict(mode=arguments.mode, traceback=traceback.format_exc()))
        raise
