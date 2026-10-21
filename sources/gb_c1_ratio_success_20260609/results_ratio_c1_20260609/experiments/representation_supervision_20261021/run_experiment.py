from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import platform
import sys
import time
import traceback

import torch
import torch.nn.functional as F

from core import AlignedMLP, alignment_loss, reference_remember_count, keep_count, select_weight, state_hash, supervised_loss
from protocol import ROOT, WORKSPACE, PROJECT, prepare_inputs, load_dataset, load_condition, read, write, sha
from gblcss.data import set_seed
from gblcss.models import MLP

CONFIG = read(ROOT/'config.json')


def require_gpu():
    torch.set_num_threads(CONFIG['threads'])
    if not torch.cuda.is_available():
        raise RuntimeError('This experiment requires the verified GPU environment')


def verify_inputs():
    manifest = read(ROOT/'frozen_inputs.json')
    for source, expected in manifest['sha256'].items():
        if sha(source) != expected:
            raise RuntimeError('Frozen input changed: '+source)


def prepare():
    require_gpu()
    if (ROOT/'frozen_inputs.json').exists():
        raise RuntimeError('Inputs already frozen; do not overwrite')
    files, checks = prepare_inputs(CONFIG)
    schedule = reference_remember_count()
    for n in (1, 541, 665, 3958):
        for rate in (0., .4, .6):
            for epoch in range(200):
                if schedule(n, epoch, rate) != keep_count(n, epoch, rate):
                    raise RuntimeError('Reference selection schedule differs')
    write(ROOT/'environment.json', dict(executable=sys.executable, python=sys.version,
          torch=torch.__version__, hip=torch.version.hip, gpu=torch.cuda.get_device_name(0),
          platform=platform.platform(), threads=torch.get_num_threads()))
    files.update({str(path): sha(path) for path in ROOT.glob('*.py')})
    for path in (ROOT/'PLAN.md', ROOT/'config.json', ROOT/'environment.json',
                 WORKSPACE/'GB-LCSS_code/src/gblcss/models.py',
                 WORKSPACE/'GB-LCSS_code/src/gblcss/data.py',
                 WORKSPACE/'GB-LCSS_code/src/gblcss/targets.py',
                 PROJECT/'external/NoisyGL/utils/frozen_grace_targets.py',
                 PROJECT/'external/NoisyGL/predictor/MLPNoiseBaselines_Predictor.py'):
        files[str(path)] = sha(path)
    write(ROOT/'preflight.json', dict(protocols=checks, selection_schedule_reference_matches=True))
    write(ROOT/'frozen_inputs.json', dict(created=time.strftime('%Y-%m-%d %H:%M:%S'), sha256=files, config=CONFIG))
    print('PREPARED: 12 protocols verified; main 144, legacy 24, probes 24, tuning 4', flush=True)


def new_model(data):
    set_seed(CONFIG['seed'])
    return AlignedMLP(data['x'].shape[1], data['classes'], data['h'].shape[1],
                      CONFIG['hidden'], CONFIG['dropout']).cuda()


def log_line(handle, row):
    handle.write(json.dumps(row, allow_nan=False)+'\n')
    handle.flush()


def pretrain(name, engineering=False):
    folder = ROOT/('engineering_pretraining' if engineering else 'pretraining')/name
    if (folder/'result.json').exists():
        if read(folder/'result.json')['status'] != 'ok':
            raise RuntimeError('Failed pretraining requires an explicit engineering review')
        return folder
    folder.mkdir(parents=True, exist_ok=True)
    data = load_dataset(name)
    model = new_model(data)
    initial = state_hash(model.deployment_state())
    optimizer = torch.optim.Adam(model.encoder_parameters()+list(model.projection.parameters()),
                                 lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    epochs = 3 if engineering else CONFIG['alignment_epochs']
    best, selected, checkpoint = float('inf'), 0, None
    started = time.perf_counter()
    write(folder/'result.json', dict(status='running', dataset=name))
    with (folder/'epochs.jsonl').open('x', encoding='utf-8') as log:
        for epoch in range(1, epochs+1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = alignment_loss(model.projection(model.encode(data['x'])), data['h'])
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite pretraining loss')
            loss.backward(); optimizer.step()
            model.eval()
            with torch.no_grad():
                value = float(alignment_loss(model.projection(model.encode(data['x'])), data['h']))
            if value < best:
                best, selected, checkpoint = value, epoch, deepcopy(model.state_dict())
            log_line(log, dict(epoch=epoch, train_alignment=float(loss.detach()), evaluation_alignment=value))
    model.load_state_dict(checkpoint)
    model.eval()
    with torch.no_grad():
        h = model.encode(data['x'])
        variance = float(h.var(dim=0, unbiased=False).mean())
    torch.save({k:v.cpu() for k,v in checkpoint.items()}, folder/'model.pt')
    torch.cuda.synchronize()
    write(folder/'result.json', dict(status='ok', dataset=name, epochs=epochs, selected_epoch=selected,
          selected_by='minimum label-free evaluation alignment', labels_used=False,
          best_alignment=best, hidden_variance=variance, initial_hash=initial,
          encoder_hash=state_hash(model.encoder_state()), seconds=time.perf_counter()-started,
          checkpoint_sha256=sha(folder/'model.pt')))
    print(f'PRETRAIN {name}: loss={best:.5f}, epoch={selected}, variance={variance:.5f}', flush=True)
    return folder


def fit(name, condition, route, rule, weight=0., *, legacy=False, phase='main', engineering=False):
    tag = f'{route}_{rule}'+('_legacy' if legacy else '')
    if phase == 'tuning':
        tag += f'_lambda{weight:g}'
    folder = ROOT/('engineering' if engineering else phase)/name/condition/tag
    if (folder/'result.json').exists():
        saved = read(folder/'result.json')
        if saved['status'] != 'ok':
            raise RuntimeError('Failed fit requires an explicit recorded review: '+str(folder))
        return saved
    folder.mkdir(parents=True, exist_ok=True)
    data, cache = load_dataset(name), load_condition(name, condition)
    model = new_model(data)
    initial = state_hash(model.deployment_state())
    initial_rng = state_hash(dict(cpu=torch.get_rng_state(), gpu=torch.cuda.get_rng_state()))
    one_hot = F.one_hot(cache['observed'], data['classes']).float()
    target = (cache['legacy_target'] if legacy else cache['target']) if rule in ('T1','T3') else one_hot
    before_encoder, pretrain_source = None, None
    if route == 'R2':
        pretrain_source = pretrain(name, engineering)
        model.load_state_dict(torch.load(pretrain_source/'model.pt', map_location='cuda', weights_only=True))
        model.freeze_encoder()
        before_encoder = state_hash(model.encoder_state())
        with torch.no_grad():
            frozen_hidden = model.encode(data['x']).detach()
        parameters = model.classifier_parameters()
    else:
        parameters = list(model.backbone.parameters())
        if route == 'R1':
            parameters += list(model.projection.parameters())
    optimizer = torch.optim.Adam(parameters, lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    epochs = 3 if engineering else (CONFIG['head_epochs'] if route == 'R2' else CONFIG['epochs'])
    meta = dict(dataset=name, condition=condition, route=route, rule=rule, weight=weight, legacy=legacy,
                phase=phase, seed=CONFIG['seed'], engineering=engineering, supervised_epochs=epochs,
                total_updates=epochs+(3 if engineering else CONFIG['alignment_epochs']) if route=='R2' else epochs,
                projection_discarded=True, initial_hash=initial, initial_rng_hash=initial_rng,
                target_hash=state_hash(dict(target=target)), input_protocol=read(ROOT/'inputs'/name/(condition+'.json')),
                pretrained_source=str(pretrain_source) if pretrain_source else None)
    write(folder/'config.json', meta)
    write(folder/'result.json', dict(**meta, status='running'))
    best_correct, best_epoch, best_state = -1, 0, None
    rate = CONFIG['conditions'][condition][1]
    torch.cuda.synchronize(); started = time.perf_counter()
    try:
        with (folder/'epochs.jsonl').open('x', encoding='utf-8') as log:
            for epoch in range(epochs):
                if time.perf_counter()-started > CONFIG['timeout_seconds_per_fit']:
                    raise TimeoutError('Preregistered per-fit timeout')
                optimizer.zero_grad(set_to_none=True)
                if route == 'R2':
                    model.train_head()
                    logits = model.backbone.layers[-1](frozen_hidden)
                    feature_loss = logits.new_zeros(())
                else:
                    model.train()
                    logits, hidden = model(data['x'])
                    feature_loss = alignment_loss(model.projection(hidden), data['h']) if route == 'R1' else logits.new_zeros(())
                classification, selected = supervised_loss(logits[data['train']], target, rule, epoch, rate)
                loss = classification + weight*feature_loss
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training objective')
                loss.backward(); optimizer.step()
                model.eval()
                with torch.no_grad():
                    logits = (model.backbone.layers[-1](frozen_hidden) if route=='R2' else model(data['x'])[0])
                    correct = int((logits[data['valid']].argmax(1) == cache['valid_y']).sum())
                improved = correct > best_correct
                if improved:
                    best_correct, best_epoch, best_state = correct, epoch+1, deepcopy(model.state_dict())
                log_line(log, dict(epoch=epoch+1, loss=float(loss.detach()), classification=float(classification.detach()),
                                   alignment=float(feature_loss.detach()), selected_count=len(selected),
                                   valid_correct=correct, checkpoint_selected=improved))
            final_correct = correct
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            logits, hidden = model(data['x'])
            probabilities = logits.softmax(1).cpu()
            representation_loss = float(alignment_loss(model.projection(hidden), data['h'])) if route!='R0' else None
            hidden_variance = float(hidden.var(dim=0, unbiased=False).mean())
        if before_encoder and state_hash(model.encoder_state()) != before_encoder:
            raise RuntimeError('Frozen encoder or BN buffers changed')
        exported = model.deployment_state()
        deployed = MLP(data['x'].shape[1], CONFIG['hidden'], data['classes'], 2, CONFIG['dropout']).cuda().eval()
        deployed.load_state_dict(exported)
        with torch.no_grad():
            deployment_difference = float((deployed(data['x']).softmax(1).cpu()-probabilities).abs().max())
        if deployment_difference != 0:
            raise RuntimeError('X-only export does not reproduce predictions')
        torch.save(exported, folder/'student.pt')
        torch.save({k:v.cpu() for k,v in best_state.items()}, folder/'training_state.pt')
        torch.save(probabilities, folder/'probabilities.pt')
        torch.cuda.synchronize()
        result = dict(**meta, status='ok', best_epoch=best_epoch, valid_correct=best_correct,
                      valid_total=len(data['valid']), validation=best_correct/len(data['valid']),
                      final_validation=final_correct/len(data['valid']), alignment=representation_loss,
                      hidden_variance=hidden_variance, seconds=time.perf_counter()-started,
                      frozen_encoder_unchanged=before_encoder is not None,
                      deployment_max_difference=deployment_difference, student_sha256=sha(folder/'student.pt'),
                      probabilities_sha256=sha(folder/'probabilities.pt'), folder=str(folder))
        write(folder/'result.json', result)
        return result
    except Exception:
        write(folder/'result.json', dict(**meta, status='failed', traceback=traceback.format_exc()))
        raise


def probe(name, condition, kind):
    folder = ROOT/'probes'/name/condition/kind
    if (folder/'result.json').exists():
        if read(folder/'result.json')['status'] != 'ok':
            raise RuntimeError('Failed probe requires engineering review')
        return
    folder.mkdir(parents=True, exist_ok=True)
    data, cache = load_dataset(name), load_condition(name, condition)
    set_seed(CONFIG['seed'])
    model = MLP(data['h'].shape[1], CONFIG['hidden'], data['classes'], 1 if kind=='linear' else 2, CONFIG['dropout']).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    best_correct, best_state, best_epoch = -1, None, 0
    started = time.perf_counter()
    with (folder/'epochs.jsonl').open('x', encoding='utf-8') as log:
        for epoch in range(CONFIG['epochs']):
            model.train(); optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(data['h'])[data['train']], cache['observed'])
            loss.backward(); optimizer.step(); model.eval()
            with torch.no_grad():
                correct = int((model(data['h'])[data['valid']].argmax(1)==cache['valid_y']).sum())
            if correct > best_correct:
                best_correct, best_state, best_epoch = correct, deepcopy(model.state_dict()), epoch+1
            log_line(log, dict(epoch=epoch+1, loss=float(loss.detach()), valid_correct=correct))
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        probabilities = model(data['h']).softmax(1).cpu()
    torch.save({k:v.cpu() for k,v in best_state.items()}, folder/'model.pt')
    torch.save(probabilities, folder/'probabilities.pt')
    torch.cuda.synchronize()
    write(folder/'result.json', dict(status='ok', dataset=name, condition=condition, kind=kind,
          validation=best_correct/len(data['valid']), best_epoch=best_epoch, seed=0,
          seconds=time.perf_counter()-started, feature_input='cached normalized GRACE H',
          x_only_deployment=False, folder=str(folder)))


def smoke():
    require_gpu(); verify_inputs()
    results = []
    for route, rule in (('R0','T1'),('R1','T3'),('R2','T3')):
        result = fit('cora','uniform0.4',route,rule,1. if route=='R1' else 0.,engineering=True)
        results.append({k:result[k] for k in ('route','rule','validation','seconds','deployment_max_difference')})
    write(ROOT/'smoke.json', dict(passed=True, results=results))
    print(json.dumps(results), flush=True)


def tune():
    require_gpu(); verify_inputs()
    if not read(ROOT/'smoke.json')['passed']:
        raise RuntimeError('Engineering smoke must pass')
    rows = []
    for name in ('cora','citeseer'):
        for weight in CONFIG['lambda_candidates']:
            print(f'TUNE {name} lambda={weight}', flush=True)
            row = fit(name,'uniform0.4','R1','T0',weight,phase='tuning')
            rows.append(dict(dataset=name, weight=weight, validation=row['validation'], folder=row['folder']))
    selected = select_weight(rows)
    payload = dict(selected=selected, criteria='mean clean validation accuracy, T0, Cora/CiteSeer u0.4; tie=min lambda',
                   test_labels_accessed=False, rows=rows)
    if (ROOT/'selection.json').exists() and read(ROOT/'selection.json') != payload:
        raise RuntimeError('Refusing to change frozen selection')
    write(ROOT/'selection.json', payload)
    print(f'FROZEN lambda={selected}; test metrics not evaluated', flush=True)


def grid():
    require_gpu(); verify_inputs()
    weight = read(ROOT/'selection.json')['selected']
    completed, total = 0, CONFIG['expected_main']+CONFIG['expected_legacy']+CONFIG['expected_probes']
    started = time.time()
    for name in CONFIG['datasets']:
        for condition in CONFIG['conditions']:
            for route in CONFIG['representation_routes']:
                for rule in CONFIG['supervision_rules']:
                    row = fit(name,condition,route,rule,weight if route=='R1' else 0.)
                    completed += 1
                    write(ROOT/'progress.json',dict(status='running',completed=completed,total=total,
                          last=f'{name}/{condition}/{route}/{rule}',started_unix=started,updated_unix=time.time()))
                    print(f'DONE {completed}/{total} {name} {condition} {route}/{rule} val={row["validation"]:.4f} sec={row["seconds"]:.2f}',flush=True)
            if condition in ('uniform0.4','uniform0.6'):
                for route in ('R0','R1'):
                    for rule in ('T1','T3'):
                        row=fit(name,condition,route,rule,weight if route=='R1' else 0.,legacy=True,phase='legacy')
                        completed+=1
                        print(f'DONE {completed}/{total} bridge {name} {condition} {route}/{rule}',flush=True)
            for kind in ('linear','mlp'):
                probe(name,condition,kind); completed+=1
                print(f'DONE {completed}/{total} H probe {name} {condition} {kind}',flush=True)
        load_dataset.cache_clear()
        torch.cuda.empty_cache()
    verify_inputs()
    write(ROOT/'progress.json',dict(status='complete',completed=completed,total=total,seconds=time.time()-started))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare','smoke','tune','grid'])
    args=parser.parse_args()
    write(ROOT/('execution_'+args.command+'.json'),dict(command=sys.argv,executable=sys.executable,
          started=time.strftime('%Y-%m-%d %H:%M:%S'),pid=__import__('os').getpid()))
    try:
        globals()[args.command]()
    except Exception:
        write(ROOT/('failure_'+args.command+'.json'),dict(traceback=traceback.format_exc(),time=time.time()))
        raise


if __name__=='__main__':
    class Journal:
        def __init__(self, console, handle):
            self.console, self.handle = console, handle

        def write(self, value):
            self.console.write(value)
            self.handle.write(value)

        def flush(self):
            self.console.flush()
            self.handle.flush()

    with (ROOT/'raw_execution.log').open('a',encoding='utf-8') as journal:
        sys.stdout=Journal(sys.stdout,journal)
        try:
            main()
        finally:
            sys.stdout.flush()
            sys.stdout=sys.stdout.console
