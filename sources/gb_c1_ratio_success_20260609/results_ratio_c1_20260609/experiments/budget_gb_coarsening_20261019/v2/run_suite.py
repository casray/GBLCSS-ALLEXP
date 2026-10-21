"""Frozen, sequential experiment queue. This module imports no ML/GPU runtime."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


SOURCE_DIR = Path(__file__).resolve().parent


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def source_fingerprint(source_dir):
    source_dir = Path(source_dir).resolve()
    hashes = {p.name: sha256_bytes(p.read_bytes()) for p in sorted(source_dir.glob('*.py'))
              if not p.name.startswith('test_')}
    support_dir = source_dir.parents[2] / 'GB-LCSS_GitHub' / 'src' / 'gblcss'
    if support_dir.exists():
        for name in ['__init__.py', 'data.py', 'granular_ball.py', 'targets.py']:
            hashes['support/gblcss/' + name] = sha256_bytes((support_dir / name).read_bytes())
    aggregate = sha256_bytes(json.dumps(hashes, sort_keys=True, separators=(',', ':')).encode('utf-8'))
    return hashes, aggregate


def freeze_protocol(config_path, run_root, source_dir=SOURCE_DIR):
    """Reject drift before reusing any result; preserve exact input config bytes."""
    run_root = Path(run_root)
    config_bytes = Path(config_path).read_bytes()
    json.loads(config_bytes.decode('utf-8-sig'))
    hashes, aggregate = source_fingerprint(source_dir)
    expected = {'config_sha256': sha256_bytes(config_bytes), 'source_sha256': aggregate,
                'source_hashes': hashes}
    protocol_path = run_root / 'protocol.json'
    frozen_config = run_root / 'config.json'
    if protocol_path.exists():
        protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
        for key, value in expected.items():
            if protocol.get(key) != value:
                raise ValueError('Refusing run-root reuse: protocol mismatch for ' + key)
        if not frozen_config.exists() or sha256_bytes(frozen_config.read_bytes()) != expected['config_sha256']:
            raise ValueError('Refusing run-root reuse: frozen config missing or changed')
        return frozen_config, protocol
    if (run_root / 'results').exists() and any((run_root / 'results').glob('*.json')):
        raise ValueError('Refusing untracked results without protocol.json')
    run_root.mkdir(parents=True, exist_ok=True)
    frozen_config.write_bytes(config_bytes)
    protocol = dict(expected, created_at=now_iso(), source_directory=str(Path(source_dir).resolve()),
                    python=sys.executable, execution='sequential independent worker processes',
                    timeout_action='kill active child, retain logs/results, stop queue')
    atomic_json(protocol_path, protocol)
    return frozen_config, protocol


def _token(value):
    return str(value).replace('.', 'p')


def build_jobs(config):
    """Seed-first manifest; R0 E2 control aliases E0 C and is never retrained."""
    jobs = []
    preferred_noises = ['u0.4', 'pair0.4', 'clean', 'u0.6']
    noises = [n for n in preferred_noises if n in config['noises']]
    noises += [n for n in config['noises'] if n not in noises]
    backbones = [b for b in ['cached_mlp', 'gcn'] if b in config['backbones']]
    backbones += [b for b in config['backbones'] if b not in backbones]
    main_ratios = config.get('ratios', [.1])
    if not main_ratios or main_ratios[0] != .1:
        raise ValueError('Protocol requires primary ratio 0.1 first')

    def add(dataset, noise, seed, backbone, arm, ratio, rule='R0', experiment='E0'):
        parts = [experiment, dataset, noise, 'seed' + str(seed), backbone, arm, 'r' + _token(ratio), rule]
        identifier = '__'.join(parts)
        if not identifier.isascii():
            raise ValueError('Job IDs must be ASCII')
        jobs.append(dict(id=identifier, dataset=dataset, noise=noise, seed=seed,
                         backbone=backbone, arm=arm, ratio=ratio, rule=rule, experiment=experiment))

    for seed in config['seeds']:
        for dataset in config['datasets']:
            arms = (config['node_arms'] + config['coarse_arms'] if dataset in config['primary_datasets']
                    else config['cost_only_arms'])
            ratios = list(main_ratios)
            if dataset == 'corafull':
                ratios += [r for r in config.get('corafull_sensitivity', []) if r not in ratios]
            for noise in noises:
                for ratio in ratios:
                    for backbone in backbones:
                        for arm in arms:
                            if arm == 'A' and ratio != main_ratios[0]:
                                continue
                            add(dataset, noise, seed, backbone, arm, ratio)
                    if (ratio == main_ratios[0] and dataset in config.get('e2_datasets', [])
                            and noise in config.get('e2_noises', [])):
                        for backbone in backbones:
                            for rule in config.get('e2_rules', []):
                                if rule == 'R0':
                                    continue
                                add(dataset, noise, seed, backbone, 'C', ratio, rule, 'E2')
    if len({job['id'] for job in jobs}) != len(jobs):
        raise ValueError('Duplicate job IDs in configured manifest')
    return jobs


def validate_prediction_artifact(result, run_root):
    if 'prediction_file' not in result and 'prediction_sha256' not in result:
        return
    if run_root is None or not isinstance(result.get('prediction_file'), str) or not result.get('prediction_sha256'):
        raise ValueError('Prediction artifact requires path, SHA256 and run-root')
    results_dir = (Path(run_root) / 'results').resolve()
    prediction = (results_dir / result['prediction_file']).resolve()
    try:
        prediction.relative_to(results_dir)
    except ValueError:
        raise ValueError('Prediction artifact escapes run-root/results') from None
    if not prediction.is_file():
        raise ValueError('Prediction artifact is missing: ' + str(prediction))
    digest = hashlib.sha256()
    with prediction.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != result['prediction_sha256']:
        raise ValueError('Prediction artifact SHA256 mismatch: ' + str(prediction))


def validate_completed_result(result, job, config_sha256, source_sha256=None, run_root=None):
    if result.get('status') != 'completed':
        raise ValueError('Result is not completed: ' + job['id'])
    if result.get('job') != job:
        raise ValueError('Result job does not match manifest: ' + job['id'])
    if result.get('config_sha256') != config_sha256:
        raise ValueError('Result config hash does not match: ' + job['id'])
    if source_sha256 is not None and result.get('source_sha256') != source_sha256:
        raise ValueError('Result source hash does not match: ' + job['id'])
    validate_prediction_artifact(result, run_root)


def _freeze_jobs(root, jobs):
    path = root / 'jobs.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != jobs:
            raise ValueError('Manifest mismatch; debug --limit requires a separate run-root')
    else:
        atomic_json(path, jobs)
    for job in jobs:
        path = root / 'jobs' / (job['id'] + '.json')
        if path.exists() and json.loads(path.read_text(encoding='utf-8')) != job:
            raise ValueError('Frozen per-job specification changed: ' + job['id'])
        if not path.exists():
            atomic_json(path, job)


def _execute_child(command, log_path, timeout_s, status_path, status):
    """Poll the direct worker; it must not launch independent training children."""
    started = time.perf_counter()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a', encoding='utf-8') as log:
        log.write('\nSTART ' + now_iso() + '\n' + subprocess.list2cmdline(command) + '\n')
        log.flush()
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        status.update(pid=child.pid, updated_at=now_iso())
        atomic_json(status_path, status)
        try:
            while child.poll() is None:
                elapsed = time.perf_counter() - started
                if elapsed > timeout_s:
                    child.kill()
                    child.wait()
                    raise TimeoutError('Child exceeded timeout of ' + str(timeout_s) + ' seconds')
                status.update(child_wall_s=elapsed, updated_at=now_iso())
                atomic_json(status_path, status)
                try:
                    child.wait(timeout=min(1.0, max(.01, timeout_s - elapsed)))
                except subprocess.TimeoutExpired:
                    pass
        except BaseException:
            if child.poll() is None:
                child.kill()
                child.wait()
            raise
    if child.returncode != 0:
        raise RuntimeError('Child exited with code ' + str(child.returncode) + '; see ' + str(log_path))
    return time.perf_counter() - started


def run_suite(config_path, run_root, limit=None, skip_prepare=False, source_dir=SOURCE_DIR):
    root = Path(run_root).resolve()
    status_path = root / 'status.json'
    status = dict(status='initializing', completed=0, total=0, current=None,
                  parent_pid=__import__('os').getpid(), started_at=now_iso(), updated_at=now_iso())
    try:
        frozen, protocol = freeze_protocol(config_path, root, source_dir)
        config = json.loads(frozen.read_text(encoding='utf-8-sig'))
        jobs = build_jobs(config)
        if limit is not None:
            if limit < 1:
                raise ValueError('--limit must be positive')
            jobs = jobs[:limit]
        _freeze_jobs(root, jobs)
        (root / 'results').mkdir(exist_ok=True)
        status['total'] = len(jobs)
        completed = set()
        for job in jobs:
            result_path = root / 'results' / (job['id'] + '.json')
            if result_path.exists():
                result = json.loads(result_path.read_text(encoding='utf-8'))
                validate_completed_result(result, job, protocol['config_sha256'], protocol['source_sha256'], root)
                if 'parent_process_wall_s' not in result:
                    raise ValueError('Cannot reuse result without parent-measured complete process wall time: ' + job['id'])
                completed.add(job['id'])
        status['completed'] = len(completed)
        atomic_json(status_path, status)
        from summarize import summarize
        summarize(root)
        timeout = float(config.get('job_timeout_s', 1800))
        pending_datasets = {j['dataset'] for j in jobs if j['id'] not in completed}
        if not skip_prepare:
            for dataset in config['datasets']:
                if dataset not in pending_datasets:
                    continue
                freeze_protocol(config_path, root, source_dir)
                status.update(status='preparing', current={'dataset': dataset}, pid=None)
                prepare_wall = _execute_child([sys.executable, str(Path(source_dir) / 'prepare.py'), '--config', str(frozen),
                                '--dataset', dataset, '--run-root', str(root)],
                               root / 'logs' / ('prepare_' + dataset + '.txt'), timeout, status_path, status)
                metadata_path = root / 'cache' / dataset / 'metadata.json'
                if metadata_path.exists():
                    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
                    if 'preparation_process_wall_s' not in metadata:
                        times = metadata.get('times', {})
                        measured_stages = sum(value for value in times.values() if isinstance(value, (int, float)))
                        metadata['preparation_process_wall_s'] = prepare_wall
                        metadata['preparation_common_overhead_s'] = max(0.0, prepare_wall - measured_stages) + times.get('provenance_s', 0.0)
                        atomic_json(metadata_path, metadata)
        for job in jobs:
            if job['id'] in completed:
                continue
            freeze_protocol(config_path, root, source_dir)
            status.update(status='running', current=job, pid=None, child_wall_s=0)
            wall = _execute_child([sys.executable, str(Path(source_dir) / 'run_arm.py'), '--config', str(frozen),
                                   '--job', str(root / 'jobs' / (job['id'] + '.json')), '--run-root', str(root)],
                                  root / 'logs' / (job['id'] + '.txt'), timeout, status_path, status)
            result_path = root / 'results' / (job['id'] + '.json')
            if not result_path.exists():
                raise ValueError('Successful child wrote no result: ' + job['id'])
            result = json.loads(result_path.read_text(encoding='utf-8'))
            validate_completed_result(result, job, protocol['config_sha256'], protocol['source_sha256'], root)
            freeze_protocol(config_path, root, source_dir)
            result['parent_process_wall_s'] = wall
            result['process_wall_s'] = wall
            body_time = result.get('run_total_s')
            if isinstance(body_time, (int, float)):
                overhead = max(0.0, wall - body_time)
                result['startup_overhead_s'] = overhead
                if isinstance(result.get('cold_total_s'), (int, float)):
                    result['body_cold_total_s'] = result['cold_total_s']
                    result['cold_total_s'] += overhead
            if isinstance(result.get('cold_total_s'), (int, float)) and isinstance(result.get('shared_cold_s'), (int, float)):
                cold, shared = result['cold_total_s'], result['shared_cold_s']
                result['amortized_total_s'] = {str(k): cold - shared + shared / k
                                              for k in config.get('amortization_K', [1, 3, 10, 30])}
            atomic_json(result_path, result)
            completed.add(job['id'])
            status.update(completed=len(completed), updated_at=now_iso(), pid=None)
            atomic_json(status_path, status)
            summarize(root)
            print(str(len(completed)) + '/' + str(len(jobs)) + ' completed ' + job['id'], flush=True)
        status.update(status='completed', current=None, pid=None, updated_at=now_iso(), finished_at=now_iso())
        atomic_json(status_path, status)
        summarize(root)
        return status
    except BaseException as exc:
        # Preserve an existing protocol on mismatch; only status and diagnostic summaries change.
        if root.exists():
            status.update(status='failed', reason=type(exc).__name__ + ': ' + str(exc),
                          pid=None, updated_at=now_iso(), finished_at=now_iso())
            atomic_json(status_path, status)
            try:
                from summarize import summarize
                summarize(root)
            except Exception:
                pass
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--run-root', required=True, type=Path)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--skip-prepare', action='store_true')
    args = parser.parse_args()
    run_suite(args.config, args.run_root, args.limit, args.skip_prepare)


if __name__ == '__main__':
    main()
