import csv
import contextlib
import hashlib
import json
import os
import sys
from pathlib import Path

from replay_gate import HERE, NOISYGL, ARCHIVE, rows, save

for name in ('MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['MKL_DYNAMIC'] = os.environ['OMP_DYNAMIC'] = 'FALSE'
sys.path.insert(0, str(NOISYGL))
os.chdir(NOISYGL)
import torch
import numpy as np
from scripts import run_ssl_swap_downstream as driver
from predictor import FrozenGraceGB_GCN_Predictor as module
from utils import frozen_grace_targets
from threadpoolctl import threadpool_limits

torch.set_num_threads(4)
original_builder = frozen_grace_targets.build_ordinary_granular_balls


def grouping(*args, **kwargs):
    with threadpool_limits(limits=6):
        return original_builder(*args, **kwargs)


frozen_grace_targets.build_ordinary_granular_balls = grouping

OUT = HERE / 'matrix'
METHOD = 'frozengrace_gb_mlp_softdist_beta05_loo'
original_train = module._FrozenGraceGBGCNBase.train
original_prediction = module._FrozenGraceGBGCNBase.get_prediction
ACTIVE = None
LOSSES = []


def prediction(self, *args, **kwargs):
    result = original_prediction(self, *args, **kwargs)
    if self.model.training and result[1] is not None:
        LOSSES.append(float(result[1].detach().cpu()))
    return result


def train(self, *args, **kwargs):
    result = original_train(self, *args, **kwargs)
    if self.soft_target is None:
        save(ACTIVE.with_suffix('.diagnostics.json'), dict(best_epoch=int(self.best_epoch), losses=LOSSES))
        return result
    assert self.soft_target.device.type == 'cuda'
    index = torch.as_tensor(self.train_mask, device=self.device)
    if index.dtype == torch.bool:
        index = index.nonzero().flatten()
    target = self.soft_target[index].detach().cpu().double()
    clean = self.clean_label[index].detach().cpu().long()
    noisy = self.noisy_label[index].detach().cpu().long()
    truth = torch.nn.functional.one_hot(clean, target.shape[1]).double()
    observed = torch.nn.functional.one_hot(noisy, target.shape[1]).double()
    error = ((target-truth)**2).sum(1)
    old = ((observed-truth)**2).sum(1)
    corrupt = clean != noisy
    assignment = np.asarray(self.gb_assignment)
    np.savez_compressed(ACTIVE.with_suffix('.npz'), targets=target.numpy(), clean=clean.numpy(), noisy=noisy.numpy(), train_index=index.cpu().numpy(), assignment=assignment)
    save(ACTIVE.with_suffix('.diagnostics.json'), dict(
        target_brier=float(error.mean()), observed_brier=float(old.mean()),
        corrected_error_gain=float(((old-error)*corrupt).mean()),
        clean_error_cost=float((error*~corrupt).mean()),
        net_brier_gain=float((old-error).mean()),
        best_epoch=int(self.best_epoch), losses=LOSSES,
        target_hash=hashlib.sha256(target.numpy().tobytes()).hexdigest()))
    return result


module._FrozenGraceGBGCNBase.train = train
module._FrozenGraceGBGCNBase.get_prediction = prediction


def main():
    global ACTIVE
    OUT.mkdir(exist_ok=True)
    save(OUT / 'environment.json', dict(python=sys.version, torch=torch.__version__, gpu=torch.cuda.get_device_name(0), threads=torch.get_num_threads()))
    arms = [('default', {}), ('shuffled', {})]
    for name, field, values in (
        ('beta', 'target_beta', [0, .25, .75, 1]),
        ('Bmax', 'max_balls', [32, 128]),
        ('smin', 'min_train_per_ball', [5, 10]),
        ('nmin', 'min_ball_size', [1, 16, 64]),
        ('delta', 'split_improvement', [.01, .1])):
        arms += [(f'{name}_{value}', {'gb.'+field: value}) for value in values]
    arms += [('raw_X', {}), ('A2X', {}), ('mlp_ce', {})]
    if '--defaults-only' in sys.argv:
        arms = arms[:1]
    specs = [r for r in rows(ARCHIVE / 'command_manifest.csv') if r['regime']=='uniform0.4' and r['group']=='softdist']
    references = rows(NOISYGL / 'runs/v31_tiesafe_affected_20260728/loo_uniform0p4.csv')
    combined = []
    for spec in specs:
        dataset = spec['dataset']
        command = json.loads(spec['command_json'])
        parsed = driver.build_parser().parse_args(command[2:])
        parsed.config_overrides = {'gb.tie_mode': 'distribution_only'}
        # Only the partition input changes; the deployed MLP still receives X.
        representations = {}
        driver.setup_seed(0)
        with (OUT / f'{dataset}_data.log').open('w', encoding='utf-8') as log, contextlib.redirect_stdout(log):
            data = driver._load_dataset(dataset, 'cpu', 'percent', .2, .1, .7)
        for kind in ('raw_X', 'A2X'):
            path = OUT / f'{dataset}_{kind}.pt'
            values = data.feats.float()
            if kind == 'A2X':
                for _ in range(2):
                    values = torch.sparse.mm(data.adj, values)
            if not path.exists():
                torch.save(values, path)
            representations[kind] = str(path)
        del data
        for arm, overrides in arms:
            for seed in range(5):
                destination = OUT / dataset / arm
                destination.mkdir(parents=True, exist_ok=True)
                ACTIVE = destination / f'seed{seed}'
                result_path = ACTIVE.with_suffix('.json')
                recovered = HERE / 'assignment_recovery/result.json'
                if dataset == 'amazonpho' and arm == 'default' and seed == 1 and recovered.exists():
                    row = json.loads(recovered.read_text(encoding='utf-8'))
                    row['recovery_source'] = str(recovered)
                elif result_path.exists():
                    row = json.loads(result_path.read_text(encoding='utf-8'))
                else:
                    LOSSES.clear()
                    parsed.config_overrides = {'gb.tie_mode': 'distribution_only', **overrides}
                    if arm in representations:
                        parsed.config_overrides['gb.ball_embedding_path'] = representations[arm]
                    method = METHOD + '_shuffled' if arm == 'shuffled' else METHOD
                    if arm == 'mlp_ce':
                        method = 'mlp_ce'
                        parsed.config_overrides = {}
                    save(ACTIVE.with_suffix('.config.json'), dict(dataset=dataset, seed=seed, arm=arm, method=method, args=vars(parsed)))
                    with ACTIVE.with_suffix('.log').open('w', encoding='utf-8') as log, contextlib.redirect_stdout(log):
                        row = driver.run_one(parsed, dataset, 'grace', method, seed)
                    row['arm'] = arm
                    assert row['status'] == 'ok'
                    if arm == 'default':
                        ref = next(r for r in references if r['dataset']==dataset and r['method']==METHOD and int(r['seed'])==seed)
                        row['archived_test'] = float(ref['test'])
                        row['main_test_matches'] = abs(float(row['test'])-float(ref['test'])) < 1e-7
                        row['main_partition_matches'] = row['ball_assignment_hash']==ref['ball_assignment_hash']
                        assert all(row[f]==ref[f] for f in ('split_hash','noise_hash'))
                    save(result_path, row)
                combined.append(row)
                print(f"DONE {dataset} {arm} seed={seed} test={float(row['test']):.8f}", flush=True)
            driver.write_csv(OUT / 'per_seed.csv', combined)
    save(OUT / ('default_status.json' if '--defaults-only' in sys.argv else 'status.json'), dict(complete=True, rows=len(combined), arms=len(arms)))


if __name__ == '__main__':
    main()
