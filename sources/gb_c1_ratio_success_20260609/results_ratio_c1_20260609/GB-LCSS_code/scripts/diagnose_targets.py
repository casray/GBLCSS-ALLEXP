from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from gblcss.data import inject_train_label_noise, load_public_dataset
from gblcss.experiment import load_config
from gblcss.granular_ball import build_granular_balls
from gblcss.representation import load_frozen_representation
from gblcss.targets import build_leave_one_out_targets


def brier_scores(targets, noisy_labels, clean_labels, train_index, n_classes):
    """Class-summed multiclass Brier scores, averaged over training nodes."""
    truth = F.one_hot(clean_labels[train_index], n_classes).float()
    noisy = F.one_hot(noisy_labels[train_index], n_classes).float()
    soft_score = (targets[train_index] - truth).square().sum(dim=1).mean().item()
    noisy_score = (noisy - truth).square().sum(dim=1).mean().item()
    return soft_score, noisy_score


def main():
    parser = argparse.ArgumentParser(description='Recompute the target-quality data in Figure 4.')
    parser.add_argument('--datasets', default='cora,citeseer,corafull,coauthorcs,amazonphoto')
    parser.add_argument('--seeds', default='0,1,2,3,4')
    parser.add_argument('--data-root', default=str(ROOT / 'data'))
    parser.add_argument('--output', type=Path, default=ROOT / 'results/rerun_target_quality.csv')
    args = parser.parse_args()
    cfg = load_config(ROOT / 'configs/main/gblcss.yaml')['method']
    rows = []
    for key in args.datasets.split(','):
        key = key.strip().lower()
        dataset = load_config(ROOT / 'configs/datasets' / f'{key}.yaml')
        for seed in map(int, args.seeds.split(',')):
            data = load_public_dataset(dataset['name'], args.data_root, seed)
            h = load_frozen_representation(ROOT / dataset['representation'], data.num_nodes)
            balls = build_granular_balls(h, max_balls=cfg['max_balls'], min_ball_size=cfg['min_ball_size'],
                                        split_improvement=cfg['split_improvement'], random_state=seed)
            for noise_type, rate in [('uniform', .2), ('uniform', .4), ('uniform', .6), ('pair', .4), ('instance', .4)]:
                inject_train_label_noise(data, noise_type, rate, seed)
                targets = build_leave_one_out_targets(data.noisy_y, balls.assignment, data.train_index,
                    data.num_classes, min_train_per_ball=cfg['min_train_per_ball'], beta=cfg['beta'])
                soft, noisy = brier_scores(targets.soft_target, data.noisy_y, data.y, data.train_index, data.num_classes)
                rows.append(dict(dataset='amazonpho' if key == 'amazonphoto' else key,
                    noise_type=noise_type, noise_rate=rate, seed=seed,
                    soft_target_brier=soft, noisy_onehot_brier=noisy, soft_brier_gain=noisy-soft))
            print(f'{key} seed={seed}: target diagnostics complete', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
