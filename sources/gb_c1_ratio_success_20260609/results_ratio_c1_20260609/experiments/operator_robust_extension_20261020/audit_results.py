"""Recompute saved metrics and supervision from the original frozen inputs."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import f1_score


def audit(root, source):
    root = Path(root)
    sys.path.insert(0, str(Path(source)))
    from experiment import atomic_json, file_hash
    from legacy_api import protocol_labels, build_targets, array_hash

    config = json.loads((root / 'config.json').read_text(encoding='utf8'))
    protocol = json.loads((root / 'protocol.json').read_text(encoding='utf8'))
    manifest = json.loads((root / 'jobs.json').read_text(encoding='utf8'))
    assert file_hash(root / 'config.json') == protocol['config_sha256']
    expected_jobs = {j['id']: j for j in manifest}
    initial, auxiliary, reports = {}, {}, []
    for path in sorted((root / 'results').glob('*.json')):
        result = json.loads(path.read_text(encoding='utf8'))
        if 'process_wall_s' not in result:
            continue
        job = result['job']
        assert job == expected_jobs[job['id']]
        assert result['source_sha256'] == protocol['source_sha256']
        assert result['config_sha256'] == protocol['config_sha256']
        metadata = json.loads((root / 'cache' / job['dataset'] / 'metadata.json').read_text(encoding='utf8'))
        y = np.load(Path(metadata['old_directory']) / 'y.npy')
        tr, va, te, noisy = protocol_labels(y, job['seed'], job['noise'], metadata['C'])
        prediction = path.parent / result['prediction_file']
        assert file_hash(prediction) == result['prediction_sha256']
        assert file_hash(root / result['checkpoint_file']) == result['checkpoint_sha256']
        tracepath=root/result['trace_file']
        assert file_hash(tracepath)==result['trace_sha256']
        traces=json.loads(tracepath.read_text(encoding='utf8'))
        epochs={row['epoch'] for row in traces}
        assert 1 in epochs and result['epochs'] in epochs
        expected_epochs={1,result['epochs'],*range(config['trace_every'],result['epochs']+1,config['trace_every'])}
        assert epochs==expected_epochs
        assert len(traces)==2*len(epochs)
        flipped=int(np.sum(noisy[tr]!=y[tr]));unflipped=len(tr)-flipped
        for epoch in epochs:
            assert {row['view'] for row in traces if row['epoch']==epoch}=={'P','train'}
        for row in traces:
            assert row['flipped_count']==flipped and row['unflipped_count']==unflipped
            for name in ['fit_wrong_label','recover_true_label','unflipped_accuracy','train_clean_accuracy','train_noisy_accuracy']:
                assert row[name] is None or 0<=row[name]<=1
            if not flipped:
                assert row['fit_wrong_label'] is None and row['recover_true_label'] is None
            clean=(flipped*(row['recover_true_label'] or 0)+unflipped*(row['unflipped_accuracy'] or 0))/len(tr)
            noisyacc=(flipped*(row['fit_wrong_label'] or 0)+unflipped*(row['unflipped_accuracy'] or 0))/len(tr)
            assert abs(clean-row['train_clean_accuracy'])<1e-12
            assert abs(noisyacc-row['train_noisy_accuracy'])<1e-12

        with np.load(prediction) as saved:
            assert np.array_equal(te, saved['indices'])
            assert np.array_equal(tr, saved['train_indices'])
            assert np.array_equal(va, saved['validation_indices'])
            assert np.array_equal(noisy[tr], saved['train_noisy_labels'])
            assert array_hash(tr, va, te) == result['split_sha256']
            assert array_hash(noisy[tr]) == result['noise_sha256']
            assert array_hash(saved['assignment']) == result['assignment_sha256']
            targets, _, _, _ = build_targets(noisy, saved['assignment'], tr, metadata['C'], result['variant_spec']['target'])
            assert np.array_equal(targets, saved['targets'])
            assert array_hash(targets) == result['targets_sha256']
            p = saved['probabilities']
            assert np.isfinite(p).all() and (p >= 0).all()
            assert np.allclose(p.sum(axis=1), 1, atol=1e-5)
            assert np.array_equal(p.argmax(1), saved['prediction'])
            accuracy = float(np.mean(saved['prediction'] == y[te]))
            f1 = float(f1_score(y[te], saved['prediction'], labels=np.arange(metadata['C']), average='macro', zero_division=0))
            assert abs(accuracy - result['test_accuracy']) < 1e-12
            assert abs(f1 - result['macro_f1']) < 1e-12
            assert abs(float(np.mean(saved['validation_prediction'] == y[va])) - result['best_primary_val']) < 1e-12
            assert abs(float(np.mean(saved['train_view_prediction'] == y[te])) - result['same_checkpoint_train_view_accuracy']) < 1e-12
        key = (job['dataset'], job['seed'], result['variant_spec']['family'])
        assert initial.setdefault(key, result['initial_parameter_sha256']) == result['initial_parameter_sha256']
        if result['variant_spec']['family']=='tfr':
            for name in ['initial_decoder_parameter_sha256','initial_encoder_parameter_sha256']:
                assert result[name] is not None
                assert auxiliary.setdefault((key,name),result[name])==result[name]

        reports.append(dict(id=job['id'], accuracy=accuracy, macro_f1=f1, verified=True))
    status = json.loads((root / 'status.json').read_text(encoding='utf8'))
    if status['status'] == 'completed':
        assert len(reports) == len(manifest)
    report = dict(status='passed', audited=len(reports), expected=len(manifest),
                  queue_status=status['status'], runs=reports,
                  scope='Saved predictions/supervision/provenance plus trace digests, denominators and rate identities; per-epoch predictions not archived and not independently retrained')
    atomic_json(root / 'PREDICTION_AUDIT.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'runs'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--source', required=True)
    args = parser.parse_args()
    audit(args.run_root, args.source)
