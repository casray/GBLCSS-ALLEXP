import contextlib
import json
import numpy as np
import run_matrix as runner
from replay_gate import HERE, NOISYGL, ARCHIVE, rows, save
from utils import ordinary_gb as gb
from utils.frozen_grace_targets import _ball_assignment_hash

cache = HERE.parent / 'submission_checks_20260920_extension/gpu_reference_matched_v2/runs/amazonphoto/seed1/split_improvement_0.1.npz'
reference = next(r for r in rows(NOISYGL/'runs/v31_tiesafe_affected_20260728/loo_uniform0p4.csv')
                 if r['dataset']=='amazonpho' and r['seed']=='1' and r['method']==runner.METHOD)
with np.load(cache) as data:
    assignment = data['assignment'].astype(np.int64)
assert assignment.shape == (7650,)
assert _ball_assignment_hash(assignment) == reference['ball_assignment_hash']
assert len(np.unique(assignment)) == 64


def cached_grouping(embeddings, **kwargs):
    assert kwargs == dict(max_balls=64, min_ball_size=32, split_improvement=.05, random_state=1), kwargs
    z = gb._l2_normalize(gb._to_numpy_embeddings(embeddings))
    stats = [gb._measure_ball(z, np.flatnonzero(assignment==j), j) for j in range(64)]
    sizes = np.array([s['size'] for s in stats], dtype=np.float32)
    radii = np.array([s['radius'] for s in stats], dtype=np.float32)
    return gb.GranularBallResult(assignment.copy(), stats, dict(num_balls=64,mean_ball_size=float(sizes.mean()),max_ball_size=int(sizes.max()),singleton_ratio=float(np.mean(sizes==1)),mean_radius=float(radii.mean())))


runner.frozen_grace_targets.build_ordinary_granular_balls = cached_grouping
out = HERE / 'assignment_recovery'
out.mkdir(exist_ok=True)
runner.ACTIVE = out / 'amazonpho_seed1'
spec = next(r for r in rows(ARCHIVE/'command_manifest.csv') if r['dataset']=='amazonpho' and r['regime']=='uniform0.4' and r['group']=='softdist')
command = json.loads(spec['command_json'])
args = runner.driver.build_parser().parse_args(command[2:])
args.config_overrides = {'gb.tie_mode':'distribution_only'}
save(out/'config.json', dict(args=vars(args), assignment_source=str(cache), assignment_hash=reference['ball_assignment_hash'], loaded_fields=['assignment']))
with (out/'run.log').open('w', encoding='utf-8') as log, contextlib.redirect_stdout(log):
    result = runner.driver.run_one(args, 'amazonpho', 'grace', runner.METHOD, 1)
result['arm'] = 'default'
save(out/'result.json', result)
checks = {field:result[field]==reference[field] for field in ('split_hash','noise_hash','ball_assignment_hash')}
checks['test_matches'] = abs(float(result['test'])-float(reference['test'])) < 1e-7
checks['test'] = result['test']
checks['archived_test'] = float(reference['test'])
checks['target_device'] = 'cuda'
save(out/'verification.json', checks)
print(json.dumps(checks, indent=2))
