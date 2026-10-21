"""Post-hoc descriptive diagnostics; never imported by the frozen trainer."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import statistics
import sys

import numpy as np


def main(root, source, output):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(source)))
    from experiment import build_jobs, file_hash, source_signature, atomic_json, paired_summary

    config = json.loads((root/'config.json').read_text(encoding='utf8'))
    protocol = json.loads((root/'protocol.json').read_text(encoding='utf8'))
    assert source_signature(source)[1] == protocol['source_sha256']
    jobs = json.loads((root/'jobs.json').read_text(encoding='utf8'))
    assert jobs == build_jobs(config)
    rows = [json.loads(p.read_text(encoding='utf8')) for p in sorted((root/'results').glob('*.json'))]
    assert len(rows) == len(jobs) == 180
    assert {r['job']['id'] for r in rows} == {j['id'] for j in jobs}
    audit = json.loads((root/'PREDICTION_AUDIT.json').read_text(encoding='utf8'))
    assert audit['status'] == 'passed' and audit['audited'] == 180
    index = {(r['job']['dataset'], r['job']['noise'], r['job']['seed'], r['job']['variant']): r for r in rows}
    groups = {}
    for r in rows:
        j = r['job']
        groups.setdefault((j['dataset'], j['noise'], j['variant']), []).append(r)
    with (root/'summary.csv').open(encoding='utf8', newline='') as handle:
        summary = list(csv.DictReader(handle))
    assert len(summary) == len(groups)
    for item in summary:
        rr = groups[item['dataset'], item['noise'], item['variant']]
        values = [100*r['test_accuracy'] for r in rr]
        assert len(rr) == int(item['n']) == 3
        assert abs(statistics.mean(values)-float(item['accuracy_mean'])) < 1e-10
        assert abs(statistics.stdev(values)-float(item['accuracy_std'])) < 1e-10
        assert abs(statistics.mean(100*r['macro_f1'] for r in rr)-float(item['macro_f1_mean'])) < 1e-10
    regenerated_pairs = paired_summary(rows)
    with (root/'paired.csv').open(encoding='utf8', newline='') as handle:
        stored_pairs = list(csv.DictReader(handle))
    assert len(stored_pairs) == len(regenerated_pairs)
    for a, b in zip(stored_pairs, regenerated_pairs):
        assert a['comparison'] == b['comparison'] and a['dataset'] == b['dataset'] and a['noise'] == b['noise']
        assert json.loads(a['delta_pp']) == b['delta_pp']

    source_checks, endpoint, gini_bins, partition_notes = [], [], [], []
    for ds in ['citeseer', 'corafull']:
        folder = root/'cache'/ds
        meta = json.loads((folder/'metadata.json').read_text(encoding='utf8'))
        old = Path(meta['old_directory'])
        assert file_hash(old/'metadata.json') == meta['original_metadata_sha256']
        for name, record in meta['original_cache_files'].items():
            assert file_hash(old/name) == record['sha256']
            source_checks.append(dict(dataset=ds, file=name, sha256=record['sha256']))
        for name, digest in meta['new_files'].items():
            assert file_hash(folder/name) == digest
        y = np.load(old/'y.npy')
        assignment = np.load(folder/'granule.npy')
        m = int(assignment.max())+1
        counts = np.zeros((m, meta['C']), dtype=np.int64)
        np.add.at(counts, (assignment, y), 1)
        sizes = counts.sum(1)
        gini = 1-((counts/sizes[:, None])**2).sum(1)
        partition_notes.append(dict(dataset=ds, M=m,
            all_node_weighted_clean_gini=float(np.average(gini, weights=sizes)),
            scope='All-node true labels are used only after training; descriptive, never fed into partitions or training'))
        noise = 'u0.6' if ds == 'citeseer' else 'u0.4'
        for seed in [0, 1, 2]:
            for variant in ['P_ind_A', 'P_ind_B', 'P_shared_B', 'S_shared_B', 'P_shared_GCE', 'S_shared_GCE']:
                row = index[ds, noise, seed, variant]
                with np.load(root/'results'/row['prediction_file']) as saved:
                    tr, pred, z = saved['train_indices'], saved['train_prediction'], saved['train_noisy_labels']
                    flipped = z != y[tr]
                    endpoint.append(dict(dataset=ds, noise=noise, seed=seed, variant=variant,
                        flipped_count=int(flipped.sum()), unflipped_count=int((~flipped).sum()),
                        fit_wrong_label=float(np.mean(pred[flipped] == z[flipped])),
                        recover_true_label=float(np.mean(pred[flipped] == y[tr][flipped])),
                        unflipped_accuracy=float(np.mean(pred[~flipped] == y[tr][~flipped])),
                        scope='Selected checkpoint evaluated with original P; not a training-view or per-epoch memorization measurement'))
            for condition in [noise, 'clean']:
                p = index[ds, condition, seed, 'P_shared_B']
                s = index[ds, condition, seed, 'S_shared_B']
                with np.load(root/'results'/p['prediction_file']) as fp, np.load(root/'results'/s['prediction_file']) as fs:
                    ids = fp['indices']
                    assert np.array_equal(ids, fs['indices'])
                    improvement = (fs['prediction'] == y[ids]).astype(float)-(fp['prediction'] == y[ids]).astype(float)
                    gg = gini[assignment[ids]]
                    edges = [0, .1, .3, .5, .7, 1.000001]
                    for lo, hi in zip(edges[:-1], edges[1:]):
                        mask = (gg >= lo) & (gg < hi)
                        if mask.any():
                            gini_bins.append(dict(dataset=ds, noise=condition, seed=seed,
                                lo=lo, hi=hi, n_test=int(mask.sum()),
                                delta_pp=float(100*improvement[mask].mean())))

    def difference(ds, noise, va, vb):
        values = [100*(index[ds, noise, seed, va]['test_accuracy']-index[ds, noise, seed, vb]['test_accuracy']) for seed in [0, 1, 2]]
        return dict(dataset=ds, noise=noise, comparison=va+'-'+vb, delta_pp=values,
                    mean=statistics.mean(values), std=statistics.stdev(values), positive=sum(x > 0 for x in values))

    contrasts = [difference(ds, no, a, b) for ds, no in config['conditions'] for a, b in [
        ('P_ind_B', 'P_ind_A'), ('P_shared_B', 'P_ind_B'), ('S_shared_B', 'P_shared_B'),
        ('S_shared_B', 'P_ind_A'), ('S_shared_GCE', 'P_shared_GCE'),
        ('S_shared_B', 'coarse_B'), ('P_ind_LS', 'P_ind_A'), ('P_ind_mixup', 'P_ind_A')]]
    gate_ce = [c for c in contrasts if c['noise'] != 'clean' and c['comparison'] == 'S_shared_B-P_shared_B']
    gate_gce = [c for c in contrasts if c['noise'] != 'clean' and c['comparison'] == 'S_shared_GCE-P_shared_GCE']
    decision = dict(operator_gate=any(c['mean'] >= 1 and c['positive'] >= 2 for c in gate_ce),
                    gce_gate=any(c['mean'] >= 1 and c['positive'] >= 2 for c in gate_gce),
                    clean_harm=[c for c in contrasts if c['noise'] == 'clean' and c['comparison'] == 'S_shared_B-P_ind_A' and c['mean'] < -2],
                    scope='Qualification passed with material clean harm; no universal method claim or strong-method plugin validation')
    atomic_json(output/'FINAL_DESCRIPTIVE_DIAGNOSTICS.json', dict(
        verification_status='ANALYZED; saved artifacts independently checked, no new training reproduction',
        source_sha256=protocol['source_sha256'], config_sha256=protocol['config_sha256'],
        audited=180, verified_summary_groups=len(groups), verified_paired_rows=len(stored_pairs),
        source_checks=source_checks, decision=decision, contrasts=contrasts,
        endpoint_train_predictions=endpoint, posthoc_clean_gini=partition_notes, gini_bins=gini_bins,
        per_epoch_predictions_available=False))
    for name in ['summary.csv', 'paired.csv', 'PREDICTION_AUDIT.json', 'status.json', 'protocol.json']:
        shutil.copy2(root/name, output/name)
    for name, data in [('endpoint_train_predictions.csv', endpoint), ('gini_bins.csv', gini_bins)]:
        with (output/name).open('w', encoding='utf8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    print(json.dumps(dict(audited=180, groups=len(groups), paired=len(stored_pairs), decision=decision)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    main(args.run_root, args.source, args.output)
