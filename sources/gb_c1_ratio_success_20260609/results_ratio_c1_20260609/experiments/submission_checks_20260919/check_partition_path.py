"""Read-only localization of the historical CoraFull target discrepancy."""
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from threadpoolctl import threadpool_limits

from run_experiment import HERE, snapshot, load_graph, fingerprint, brier_decomposition, write_json

cfg = json.loads((HERE/'config.json').read_text(encoding='utf8'))
root, _ = snapshot(cfg)
sys.path.append(str(Path(cfg['data_root']).parent))
from gblcss.granular_ball import build_granular_balls
from gblcss.targets import build_leave_one_out_targets
from utils.ordinary_gb import build_ordinary_granular_balls
from utils.frozen_grace_targets import apply_ball_soft_targets, _ball_assignment_hash

torch.set_num_threads(4)
graph = load_graph('corafull', 0, cfg)
h = torch.load(root/'representations/corafull/grace/H.pt', map_location='cpu')
if isinstance(h, dict):
    h = h.get('H', h.get('embeddings'))
hn = h.numpy().copy()
hn /= np.maximum(np.linalg.norm(hn, axis=1, keepdims=True), 1e-12)
rows = []
for limit in [2, 6]:
    name, values, builder = f'original_threads_{limit}', F.normalize(h.float(),dim=1), build_ordinary_granular_balls
    with threadpool_limits(limits=limit):
        ball = builder(values,max_balls=64,min_ball_size=32,split_improvement=.05,random_state=0)
    target = build_leave_one_out_targets(graph.noisy_y,ball.assignment,graph.train_index,graph.num_classes)
    old, _, _ = apply_ball_soft_targets(graph.noisy_y,ball.assignment,graph.train_index,graph.num_classes,
        min_train_per_ball=2,alpha_baseline='multiclass',target_mode='distribution',target_beta=.5,
        leave_one_out=True,tie_mode='distribution_only')
    row = dict(path=name,assignment_hash=fingerprint(ball.assignment),balls=ball.summary['num_balls'],
               historical_assignment_hash=_ball_assignment_hash(ball.assignment),
               split_hash=__import__('run_experiment').historical_hash(graph.train_index, graph.val_index, graph.test_index),
               noise_hash=__import__('run_experiment').historical_hash(graph.y, graph.noisy_y),
               target_matches_original=torch.equal(old,target.soft_target),
               **brier_decomposition(target.soft_target,graph.noisy_y,graph.y,graph.train_index,graph.num_classes))
    rows.append(row)
    print(json.dumps(row),flush=True)
write_json(HERE/'partition_thread_check_2_6.json',rows)
