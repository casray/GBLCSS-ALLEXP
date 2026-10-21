from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from copy import deepcopy
import gc
import hashlib
import json
import logging
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'gb_dist_protocol_seeds_20261007'
sys.path.insert(0, str(SOURCE))
import run_protocol as common

previous = common.previous
read, write, sha = common.read, common.write, common.sha
DATASETS = ['cora', 'citeseer', 'corafull']
RATES = [.4, .6]
SEED = 1
ARMS = ['A_noisy', 'B_all_clean', 'C_oracle_filter', 'D_matched_clean']
SETTINGS = dict(datasets=DATASETS, rates=RATES, protocol_seed=SEED, teacher_seed=SEED,
                student_seed=SEED, random_subset_seed=101, teacher_epochs=400,
                teacher_patience=150, student_epochs=200, DIST_temperature=1.,
                DIST_beta=1., DIST_gamma=1., student_hidden=64, student_layers=2,
                student_dropout=.5, student_lr=.01, student_weight_decay=.0005,
                stage='one-seed oracle intervention screen', seconds_budget=5400)
RAW_CACHE = {}


def cells():
    for dataset in DATASETS:
        for rate in RATES:
            yield dataset, rate


def source_folder(dataset, rate):
    return SOURCE / 'runs' / dataset / f'uniform{rate}' / f'seed{SEED}'


def run_folder(dataset, rate, arm):
    canonical_rate = .4 if arm == 'B_all_clean' else rate
    return ROOT / 'runs' / dataset / f'uniform{canonical_rate}' / arm


def protocol_path(dataset, rate, arm):
    return ROOT / 'protocols' / dataset / f'uniform{rate}' / f'{arm}.pt'


def state_hash(state):
    return common.old.state_hash(state)


def class_counts(labels, selected, classes):
    return torch.bincount(labels[selected], minlength=classes)


def intervention(train, clean, noisy, classes, arm):
    train = train.long()
    good = train[clean[train] == noisy[train]]
    if arm in ('A_noisy', 'B_all_clean'):
        selected = train.clone()
    elif arm == 'C_oracle_filter':
        selected = good.clone()
    elif arm == 'D_matched_clean':
        rng = np.random.RandomState(SETTINGS['random_subset_seed'])
        picked = []
        required = class_counts(clean, good, classes)
        for c in range(classes):
            candidates = train[clean[train] == c].numpy()
            picked.extend(rng.choice(candidates, int(required[c]), replace=False).tolist())
        keep = torch.zeros(len(clean), dtype=torch.bool)
        keep[torch.tensor(picked, dtype=torch.long)] = True
        selected = train[keep[train]]
    else:
        raise ValueError(arm)
    permitted = torch.zeros(len(clean), dtype=torch.long)
    permitted[selected] = noisy[selected] if arm == 'A_noisy' else clean[selected]
    if class_counts(permitted, selected, classes).min() == 0:
        raise ValueError(f'{arm} has missing prototype classes')
    return selected, permitted


def sanitized_targets(selected, permitted, validation, clean):
    visible = torch.full_like(clean, -1)
    visible[selected] = permitted[selected]
    visible[validation] = clean[validation]
    return visible


def effective_hash(selected, permitted, validation, visible):
    return previous.fingerprint(selected, permitted, validation, visible[validation])


def raw_graph(dataset):
    if dataset not in RAW_CACHE:
        stored = torch.load(previous.DATA_FILES[dataset], map_location='cpu', weights_only=False)
        raw = stored[0]
        x = raw['x'] if isinstance(raw, dict) else raw.x
        edge = raw['edge_index'] if isinstance(raw, dict) else raw.edge_index
        RAW_CACHE[dataset] = (x.float(), edge.long())
    return RAW_CACHE[dataset]


def data_for(dataset, rate, arm):
    proto = torch.load(source_folder(dataset, rate) / 'protocol.pt', weights_only=True)
    teacher = torch.load(source_folder(dataset, rate) / 'teacher/probabilities.pt', weights_only=True)
    classes = teacher.shape[1]
    selected, permitted = intervention(proto['train'], proto['clean'], proto['noisy'], classes, arm)
    visible = sanitized_targets(selected, permitted, proto['val'], proto['clean'])
    x, edge = raw_graph(dataset)
    n = len(x)
    masks = []
    for index in (selected, proto['val'], proto['test']):
        mask = torch.zeros(n, dtype=torch.bool)
        mask[index] = True
        masks.append(mask)
    data = previous.GraphData(dataset, x, edge, visible, *masks, noisy_y=permitted,
                              train_index=selected, val_index=proto['val'], test_index=proto['test'])
    assert data.num_classes == classes
    assert (data.y[data.test_index] == -1).all()
    assert (data.noisy_y[~data.train_mask] == 0).all()
    metadata = dict(dataset=dataset, rate=rate, seed=SEED, arm=arm, original_train=len(proto['train']),
                    supervised=len(selected), retained_class_counts=class_counts(permitted, selected, classes).tolist(),
                    effective_input_hash=effective_hash(selected, permitted, proto['val'], visible),
                    classes=classes, split_hash=read(source_folder(dataset, rate) / 'protocol.json')['split_hash'])
    return data, metadata


def verify():
    for path, expected in read(ROOT / 'frozen_inputs.json')['hashes'].items():
        if sha(path) != expected:
            raise RuntimeError(f'frozen source/input changed: {path}')


def prepare():
    if (ROOT / 'frozen_inputs.json').exists():
        verify()
        return
    common.old.require_gpu()
    torch.set_num_threads(4)
    paths = list(ROOT.glob('*.py')) + [ROOT / 'PLAN.md', SOURCE / 'run_protocol.py',
             common.screen.ROOT / 'run_challenge.py', common.screen.ROOT / 'losses.py',
             common.stable.ROOT / 'run_stability.py', previous.ROOT / 'run_screen.py',
             previous.ROOT / 'adapters.py']
    paths += list((previous.ROOT / 'vendor/erase/scripts').glob('*.py'))
    paths += list((common.screen.ROOT / 'vendor').glob('*.py'))
    paths += list((previous.WORKSPACE / 'GB-LCSS_GitHub/src/gblcss').glob('*.py'))
    checks = []
    original_hashes = read(SOURCE / 'frozen_inputs.json')['hashes']
    for dataset, rate in cells():
        source = source_folder(dataset, rate)
        proto = torch.load(source / 'protocol.pt', weights_only=True)
        meta = read(source / 'protocol.json')
        assert previous.fingerprint(*(proto[k] for k in ('train', 'val', 'test'))) == meta['split_hash']
        assert previous.fingerprint(proto['clean'], proto['noisy']) == meta['noise_hash']
        for arm in ARMS:
            data, info = data_for(dataset, rate, arm)
            out = protocol_path(dataset, rate, arm)
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(dict(selected=data.train_index, noisy_y=data.noisy_y, visible_y=data.y,
                            train_mask=data.train_mask, val=data.val_index, test=data.test_index), out)
            write(out.with_suffix('.json'), info)
            with torch.no_grad():
                gpu = previous.gpu_data(data)
                labels, _ = previous.erase_preprocess(gpu, previous.erase_settings(dataset)['alpha'],
                                                       previous.erase_settings(dataset)['T'])
                prototype_counts = torch.bincount(labels[gpu.train_mask].argmax(1), minlength=info['classes'])
                assert prototype_counts.min() > 0, (dataset, rate, arm, prototype_counts)
            info['preprocessed_prototype_counts'] = prototype_counts.cpu().tolist()
            checks.append(info)
            paths += [out, out.with_suffix('.json')]
            del gpu, data, labels
        c = checks[-2]['retained_class_counts']
        d = checks[-1]['retained_class_counts']
        assert c == d
        paths += [previous.DATA_FILES[dataset], source / 'protocol.pt', source / 'protocol.json']
        paths += [source / 'teacher' / name for name in ('teacher.pt', 'probabilities.pt', 'result.json', 'config.json', 'epochs.jsonl')]
        paths += [source / 'GB0' / name for name in ('student.pt', 'probabilities.pt', 'result.json', 'config.json', 'epochs.jsonl')]
        torch.cuda.empty_cache()
    for dataset in DATASETS:
        b = [c for c in checks if c['dataset'] == dataset and c['arm'] == 'B_all_clean']
        assert len(b) == 2 and b[0]['effective_input_hash'] == b[1]['effective_input_hash']
    write(ROOT / 'preflight.json', dict(passed=True, protocol_cells=6, interventions=checks,
          expected_physical_teacher_fits=21, expected_physical_student_fits=21,
          test_labels_hidden_from_fit=True, B_rate_equivalence=True))
    write(ROOT / 'environment.json', dict(executable=sys.executable, python=sys.version,
          platform=platform.platform(), torch=torch.__version__, hip=torch.version.hip,
          gpu=torch.cuda.get_device_name(0), numpy=np.__version__, settings=SETTINGS))
    hashes = {}
    for path in sorted(set(p.resolve() for p in paths)):
        value = sha(path)
        if str(path) in original_hashes:
            assert original_hashes[str(path)] == value, path
        hashes[str(path)] = value
    write(ROOT / 'frozen_inputs.json', dict(hashes=hashes, settings=SETTINGS))
    logging.info('PREPARED 6 protocols, 21 distinct teacher/student pairs, %s frozen files', len(hashes))


def teacher_fit(data, info, folder, replay=False):
    folder.mkdir(parents=True, exist_ok=True)
    result_path = folder / 'result.json'
    if result_path.exists() and read(result_path)['status'] == 'ok':
        assert read(folder / 'config.json')['effective_input_hash'] == info['effective_input_hash']
        return torch.load(folder / 'probabilities.pt', weights_only=True).cuda()
    if result_path.exists():
        raise RuntimeError('incomplete fit requires an explicit recorded repair; refusing silent retry')
    previous.set_seed(SEED)
    conf = previous.erase_settings(data.name)
    Model, _ = previous.erase_classes()
    model = Model.GAT.get_model(data.x.shape[1], conf['hidden'], conf['embedding'], conf['layers'],
                               conf['heads'], conf['dropout'], 'cuda', False, False, False, False).cuda()
    model.reset_parameters()
    initial_hash = state_hash(model.state_dict())
    initial_rng_hash = state_hash({'cpu': torch.get_rng_state(), 'gpu': torch.cuda.get_rng_state()})
    del model
    previous.set_seed(SEED)
    write(folder / 'config.json', dict(**info, teacher=conf, epochs=400, replay=replay,
          initial_state_hash=initial_hash, initial_rng_hash=initial_rng_hash,
          test_labels_hidden=True, label_preprocess_and_probe_use_selected_mask=True))
    write(result_path, dict(status='running'))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with (folder / 'epochs.jsonl').open('w', encoding='utf-8') as handle:
        model, probabilities, epoch, settings = previous.train_erase(data, 400, handle)
    assert torch.isfinite(probabilities).all() and (probabilities.sum(1)-1).abs().max() < 1e-5
    torch.save(dict(state=model.state_dict(), settings=settings), folder / 'teacher.pt')
    torch.save(probabilities.cpu(), folder / 'probabilities.pt')
    torch.cuda.synchronize()
    write(result_path, dict(status='ok', seconds=time.perf_counter()-started, best_epoch=epoch,
          executed_epochs=settings['executed_epochs'],
          valid_correct=int((probabilities[data.val_index].argmax(1) == data.y[data.val_index]).sum()),
          peak_allocated_bytes=torch.cuda.max_memory_allocated()))
    del model
    return probabilities.detach()


def student_fit(data, teacher, info, folder, replay=False):
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'result.json').exists():
        result = read(folder / 'result.json')
        if result['status'] != 'ok':
            raise RuntimeError('refusing to silently restart incomplete student')
        assert read(folder / 'config.json')['teacher_hash'] == state_hash({'probabilities': teacher.cpu()})
        return
    previous.set_seed(SEED)
    model = previous.MLP(data.x.shape[1], 64, data.num_classes, 2, .5).cuda()
    initial_hash = state_hash(model.state_dict())
    initial_rng_hash = state_hash({'cpu': torch.get_rng_state(), 'gpu': torch.cuda.get_rng_state()})
    objective = common.screen.KDLoss('DIST', teacher, 1.)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=.0005)
    write(folder / 'config.json', dict(**info, replay=replay, settings=SETTINGS,
          initial_state_hash=initial_hash, initial_rng_hash=initial_rng_hash,
          teacher_hash=state_hash({'probabilities': teacher.cpu()}), test_labels_hidden=True))
    write(folder / 'result.json', dict(status='running'))
    best_valid, best_epoch, best_state = -1, -1, None
    torch.cuda.synchronize()
    started = time.perf_counter()
    with (folder / 'epochs.jsonl').open('w', encoding='utf-8') as handle:
        for epoch in range(1, 201):
            model.train()
            optimizer.zero_grad()
            logits = model(data.x)
            loss = objective(logits)
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                logits = model(data.x)
                correct = int((logits[data.val_index].argmax(1) == data.y[data.val_index]).sum())
            if correct > best_valid:
                best_valid, best_epoch, best_state = correct, epoch, deepcopy(model.state_dict())
            handle.write(json.dumps(dict(epoch=epoch, loss=float(loss), valid_correct=correct)) + '\n')
            handle.flush()
        torch.save(logits.softmax(1).cpu(), folder / 'epoch200_probabilities.pt')
        torch.save(model.state_dict(), folder / 'epoch200.pt')
        model.load_state_dict(best_state)
        with torch.no_grad():
            probs = model(data.x).softmax(1)
    torch.save(model.state_dict(), folder / 'student.pt')
    torch.save(probs.cpu(), folder / 'probabilities.pt')
    torch.cuda.synchronize()
    write(folder / 'result.json', dict(status='ok', best_epoch=best_epoch, valid_correct=best_valid,
                                     seconds=time.perf_counter()-started))


def replay():
    verify()
    data, info = data_for('cora', .4, 'A_noisy')
    data = previous.gpu_data(data)
    output = ROOT / 'engineering' / 'cora_uniform0.4_seed1'
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'console.log').open('a', encoding='utf-8') as log, redirect_stdout(log):
        observed_teacher = teacher_fit(data, info, output / 'teacher', replay=True)
        original_teacher = torch.load(source_folder('cora', .4) / 'teacher/probabilities.pt', weights_only=True).cuda()
        assert torch.equal(observed_teacher.argmax(1), original_teacher.argmax(1))
        student_fit(data, original_teacher, info, output / 'student', replay=True)
    observed_student = torch.load(output / 'student/probabilities.pt', weights_only=True)
    original_student = torch.load(source_folder('cora', .4) / 'GB0/probabilities.pt', weights_only=True)
    new_trajectory = [read_line['valid_correct'] for read_line in map(json.loads, (output / 'student/epochs.jsonl').read_text().splitlines())]
    old_trajectory = [read_line['valid_correct'] for read_line in map(json.loads, (source_folder('cora', .4) / 'GB0/epochs.jsonl').read_text().splitlines())]
    check = dict(teacher_max_difference=float((observed_teacher-original_teacher).abs().max()),
          teacher_argmax_identical=torch.equal(observed_teacher.argmax(1), original_teacher.argmax(1)),
          teacher_best_epoch_same=read(output/'teacher/result.json')['best_epoch'] == read(source_folder('cora',.4)/'teacher/result.json')['best_epoch'],
          student_exact=torch.equal(observed_student, original_student),
          student_max_difference=float((observed_student-original_student).abs().max()),
          student_argmax_identical=torch.equal(observed_student.argmax(1),original_student.argmax(1)),
          student_validation_trajectory_equal=new_trajectory == old_trajectory,
          teacher_retraining_bit_exact=False,
          contemporary_noisy_baselines_required=True,
          short_repeat_evidence=str(ROOT/'engineering/short_repeat/result.json'))
    write(ROOT / 'replay_check.json', check)
    assert check['teacher_best_epoch_same'] and check['student_argmax_identical'] and check['student_validation_trajectory_equal']
    assert check['student_max_difference'] <= 1e-6
    del data, observed_teacher, original_teacher
    gc.collect()
    torch.cuda.empty_cache()
    logging.info('REPLAY passed %s', check)


def grid():
    verify()
    check = read(ROOT / 'replay_check.json')
    assert check['teacher_argmax_identical'] and check['student_validation_trajectory_equal']
    if (ROOT / 'execution.json').exists():
        raise RuntimeError('completed grid must not be overwritten')
    started = time.perf_counter()
    physical = 0
    for dataset, rate in cells():
        for arm in ARMS:
            if time.perf_counter()-started > SETTINGS['seconds_budget']:
                raise TimeoutError('registered 90-minute budget exceeded')
            data, info = data_for(dataset, rate, arm)
            frozen = read(protocol_path(dataset, rate, arm).with_suffix('.json'))
            assert info == frozen
            data = previous.gpu_data(data)
            folder = run_folder(dataset, rate, arm)
            folder.mkdir(parents=True, exist_ok=True)
            if arm == 'B_all_clean' and rate == .6:
                assert read(folder / 'teacher/config.json')['effective_input_hash'] == info['effective_input_hash']
                write(ROOT / 'protocols' / dataset / 'uniform0.6/B_reuse.json',
                      dict(source=str(folder), reason='identical effective training/validation inputs and RNG',
                           effective_input_hash=info['effective_input_hash']))
                logging.info('REUSE %s B_all_clean for noise .6', dataset)
                del data
                continue
            physical += 1
            logging.info('START %s/21 %s %.1f %s supervised=%s', physical, dataset, rate, arm, info['supervised'])
            with (folder / 'console.log').open('a', encoding='utf-8') as handle, redirect_stdout(handle):
                teacher = teacher_fit(data, info, folder / 'teacher')
                student_fit(data, teacher, info, folder / 'student')
            teacher_result = read(folder / 'teacher/result.json')
            logging.info('DONE %s %.1f %s teacher_epoch=%s epochs=%s seconds=%.1f', dataset, rate, arm,
                         teacher_result['best_epoch'], teacher_result['executed_epochs'], teacher_result['seconds'])
            del data, teacher
            gc.collect()
            torch.cuda.empty_cache()
    verify()
    write(ROOT / 'execution.json', dict(status='ok', protocol_cells=6, logical_pairs=24,
          logical_oracle_pairs=18, physical_teacher_fits=physical, physical_student_fits=physical,
          contemporary_noisy_baselines=6, baseline_pairs_reused=0,
          seconds=time.perf_counter()-started, old_input_hashes_unchanged=True,
          testing_not_performed_during_fit=True, stage='one-seed oracle intervention screen'))
    logging.info('GRID COMPLETE %s physical pairs %.1f seconds', physical, time.perf_counter()-started)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('prepare', 'replay', 'grid'):
        parser.add_argument('--'+key, action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
          handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(ROOT/'raw_log.txt',encoding='utf-8')])
    try:
        torch.set_num_threads(4)
        if args.prepare:
            prepare()
        if args.replay:
            replay()
        if args.grid:
            grid()
    except Exception:
        logging.exception('FAILED; partial records retained, no scientific settings changed')
        raise
