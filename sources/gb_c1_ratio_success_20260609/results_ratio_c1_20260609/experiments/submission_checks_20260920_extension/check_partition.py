"""Diagnose numerical grouping differences before any downstream evaluation."""
import json
import torch
from threadpoolctl import threadpool_limits
from run_experiment import HERE, runner, load_graph

cfg = json.loads((HERE/'config.json').read_text(encoding='utf8'))
root, _ = runner.snapshot(cfg)
from gblcss.granular_ball import build_granular_balls
from gblcss.targets import build_leave_one_out_targets
from gblcss.representation import load_frozen_representation

torch.set_num_threads(4)
graph = load_graph('amazonphoto',1,cfg)
h = load_frozen_representation(root/'representations/amazonphoto/grace/H.pt',graph.num_nodes)
rows = []
for threads in [6,6,1,2,4,8,12]:
    with threadpool_limits(limits=threads):
        balls = build_granular_balls(h, random_state=1)
    target = build_leave_one_out_targets(graph.noisy_y,balls.assignment,graph.train_index,graph.num_classes)
    row = dict(threads=threads, assignment_hash=runner.fingerprint(balls.assignment),
               **runner.brier_decomposition(target.soft_target,graph.noisy_y,graph.y,graph.train_index,graph.num_classes))
    rows.append(row)
    print(json.dumps(row),flush=True)
runner.write_json(HERE/'partition_diagnostic.json',rows)
