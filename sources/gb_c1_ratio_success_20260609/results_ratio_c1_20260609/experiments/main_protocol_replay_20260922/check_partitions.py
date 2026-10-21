import os
import sys
from replay_gate import HERE, NOISYGL, rows, save
sys.path.insert(0, str(NOISYGL))
import torch
from threadpoolctl import threadpool_limits
from utils.ordinary_gb import build_ordinary_granular_balls
from utils.frozen_grace_targets import load_frozen_embeddings, prepare_ball_features, _ball_assignment_hash

torch.set_num_threads(4)
reference = rows(NOISYGL / 'runs/v31_tiesafe_affected_20260728/loo_uniform0p4.csv')
results = []
for dataset, seed in [('coauthorcs', 0), ('corafull', 0), ('amazonpho', 1)]:
    row = next(r for r in reference if r['dataset']==dataset and int(r['seed'])==seed and r['method']=='frozengrace_gb_mlp_softdist_beta05_loo')
    path = (NOISYGL / row['ball_embedding_path']).resolve()
    values = prepare_ball_features(load_frozen_embeddings(path))
    for threads in [6, 4, 12]:
        with threadpool_limits(limits=threads):
            balls = build_ordinary_granular_balls(values, max_balls=64, min_ball_size=32, split_improvement=.05, random_state=seed)
        actual = _ball_assignment_hash(balls.assignment)
        result = dict(dataset=dataset, seed=seed, threads=threads, actual=actual, expected=row['ball_assignment_hash'], matches=actual==row['ball_assignment_hash'])
        results.append(result)
        print(result, flush=True)
        save(HERE / 'partition_thread_diagnostic.json', results)
        if result['matches']:
            break
