"""Localize archived training differences without changing the experiment matrix."""
import json
from pathlib import Path
import sys

import numpy as np
import torch

from run_experiment import HERE, load_graph, snapshot, write_json


def main():
    cfg = json.loads((HERE/'config.json').read_text(encoding='utf8'))
    _, source = snapshot(cfg)
    sys.path.append(str(Path(cfg['data_root']).parent))
    from utils.frozen_grace_targets import apply_ball_soft_targets, _ball_assignment_hash
    from gblcss.experiment import load_config
    from gblcss.data import set_seed
    from gblcss.training import train_x_only_mlp
    base = load_config(source/'configs/main/gblcss.yaml')
    torch.set_num_threads(cfg['threads'])
    torch.set_num_interop_threads(1)
    rows = []
    for dataset, seed in [('cora',2),('corafull',4)]:
        graph = load_graph(dataset,seed,cfg)
        path = HERE/'gpu_reference_matched/runs'/dataset/f'seed{seed}/default.npz'
        saved = np.load(path)
        target, _, _ = apply_ball_soft_targets(
            graph.noisy_y.to('cuda:0'),saved['assignment'],graph.train_index,graph.num_classes,
            min_train_per_ball=2,alpha_baseline='multiclass',target_mode='distribution',target_beta=.5,
            leave_one_out=True,tie_mode='distribution_only')
        cpu = torch.from_numpy(saved['targets'])
        gpu = target.cpu()
        set_seed(seed)
        result = train_x_only_mlp(graph,target,**base['model'],
                                 learning_rate=base['training']['learning_rate'],
                                 weight_decay=base['training']['weight_decay'],
                                 epochs=cfg['epochs'],device='cuda:0')
        row = dict(dataset=dataset,seed=seed,
                   target_max_abs_difference=float((cpu-gpu).abs().max()),
                   different_target_entries=int((cpu!=gpu).sum()),
                   historical_assignment_hash=_ball_assignment_hash(saved['assignment']),
                   gpu_constructed_target_test_accuracy=result.test_accuracy,
                   best_epoch=result.best_epoch)
        rows.append(row)
        print(json.dumps(row),flush=True)
    write_json(HERE/'gpu_reference_matched/target_device_diagnostic.json',rows)


if __name__ == '__main__':
    main()
