import contextlib
import json
import numpy as np
import run_matrix as runner
from replay_gate import HERE, save
from utils import ordinary_gb as gb

assignment = None


def fixed_groups(embeddings, **kwargs):
    assert kwargs['max_balls']==64 and kwargs['min_ball_size']==32 and kwargs['split_improvement']==.05
    z = gb._l2_normalize(gb._to_numpy_embeddings(embeddings))
    stats = [gb._measure_ball(z, np.flatnonzero(assignment==j), int(j)) for j in np.unique(assignment)]
    sizes = np.array([s['size'] for s in stats], dtype=np.float32)
    radii = np.array([s['radius'] for s in stats], dtype=np.float32)
    return gb.GranularBallResult(assignment.copy(), stats, dict(num_balls=len(stats),mean_ball_size=float(sizes.mean()),max_ball_size=int(sizes.max()),singleton_ratio=float(np.mean(sizes==1)),mean_radius=float(radii.mean())))


runner.frozen_grace_targets.build_ordinary_granular_balls = fixed_groups
manifest = HERE/'fixed_group_repairs/manifest.json'
repaired = json.loads(manifest.read_text()) if manifest.exists() else []
for path in (HERE/'matrix').glob('*/*/seed?.json'):
    row = json.loads(path.read_text(encoding='utf-8'))
    if not row['arm'].startswith(('beta_', 'smin_')) and row['arm'] != 'shuffled':
        continue
    dataset, seed = row['dataset'], int(row['seed'])
    default_path = HERE/'matrix'/dataset/'default'/path.name
    assignment_path = default_path.with_suffix('.npz')
    if dataset=='amazonpho' and seed==1:
        default_path = HERE/'assignment_recovery/result.json'
        assignment_path = HERE/'assignment_recovery/amazonpho_seed1.npz'
    default = json.loads(default_path.read_text(encoding='utf-8'))
    with np.load(assignment_path) as data:
        assignment = data['assignment'].copy()
    expected_hash = default['ball_assignment_hash']
    if row['arm']=='shuffled':
        expected_hash = runner.frozen_grace_targets._ball_assignment_hash(gb.build_random_size_matched_assignment(assignment, random_state=seed+17))
    if row['ball_assignment_hash']==expected_hash:
        continue
    destination = HERE/'fixed_group_repairs'/dataset/row['arm']
    destination.mkdir(parents=True, exist_ok=True)
    runner.ACTIVE = destination/path.stem
    if runner.ACTIVE.with_suffix('.json').exists():
        continue
    config = json.loads(path.with_suffix('.config.json').read_text(encoding='utf-8'))
    args = __import__('argparse').Namespace(**config['args'])
    runner.LOSSES.clear()
    with runner.ACTIVE.with_suffix('.log').open('w', encoding='utf-8') as log, contextlib.redirect_stdout(log):
        method = runner.METHOD+'_shuffled' if row['arm']=='shuffled' else runner.METHOD
        result = runner.driver.run_one(args, dataset, 'grace', method, seed)
    result['arm'] = row['arm']
    result['assignment_source'] = str(assignment_path)
    assert result['ball_assignment_hash']==expected_hash
    save(runner.ACTIVE.with_suffix('.json'), result)
    repaired.append(dict(dataset=dataset, arm=row['arm'], seed=seed, old_test=row['test'], test=result['test']))
    print(repaired[-1], flush=True)
save(HERE/'fixed_group_repairs/manifest.json', repaired)
