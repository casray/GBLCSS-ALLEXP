import csv
import json
import statistics as stats
import torch
import run_kmeans as run


def csv_write(name, rows):
    with (run.ROOT / name).open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def signs(values):
    return dict(mean=stats.mean(values), wins=sum(v > 1e-9 for v in values),
                ties=sum(abs(v) <= 1e-9 for v in values), losses=sum(v < -1e-9 for v in values),
                minimum=min(values), maximum=max(values))


def main():
    run.old.require_gpu()
    run.verify()
    rows, cells, audit = [], [], []
    for dataset in run.DATASETS:
        for rate in (.4, .6):
            data, _ = run.previous.load_data(dataset, rate)
            data = run.previous.gpu_data(data)
            for seed in (0, 1, 2):
                folder = run.folder_for(dataset, rate, seed)
                result = run.read(folder / 'result.json')
                assert result['status'] == 'ok'
                epochs = [json.loads(line) for line in (folder / 'epochs.jsonl').read_text().splitlines()]
                assert len(epochs) == 200
                assert result['best_epoch'] == max(epochs, key=lambda v: v['valid'])['epoch']
                model = run.previous.MLP(data.x.shape[1], 64, data.num_classes, 2, .5).cuda().eval()
                for tag in ['student'] + [f'epoch{e}' for e in run.stable.SNAPSHOTS]:
                    model.load_state_dict(torch.load(folder / f'{tag}.pt', map_location='cuda', weights_only=True))
                    with torch.no_grad():
                        reconstructed = model(data.x).softmax(1).cpu()
                    cached = torch.load(folder / f'{tag}_probabilities.pt', weights_only=True)
                    error = float((reconstructed-cached).abs().max())
                    assert error == 0., (dataset, rate, seed, tag, error)
                    audit.append(dict(dataset=dataset, rate=rate, seed=seed, checkpoint=tag, max_abs_difference=error))
                    if tag == 'student':
                        metrics = run.old.exact_metrics(reconstructed.cuda(), data)
                        assert metrics['test_correct'] == result['test_correct']
                row = dict(dataset=dataset, noise_rate=rate, student_seed=seed, KM=100*result['student_test'])
                for arm in ('R0', 'R1'):
                    ref = run.read(run.stable.folder_for(dataset, rate, seed, arm) / 'result.json')
                    assert result['initial_state_hash'] == ref['initial_state_hash']
                    assert result['initial_rng_hash'] == ref['initial_rng_hash']
                    row[arm] = 100*ref['student_test']
                row.update(KM_minus_R0=row['KM']-row['R0'], GB_minus_R0=row['R1']-row['R0'], GB_minus_KM=row['R1']-row['KM'])
                rows.append(row)
            subset = rows[-3:]
            cell = dict(dataset=dataset, noise_rate=rate)
            for key in ('R0', 'R1', 'KM', 'KM_minus_R0', 'GB_minus_R0', 'GB_minus_KM'):
                values = [r[key] for r in subset]
                cell[key+'_mean'] = stats.mean(values)
                cell[key+'_std'] = stats.stdev(values)
            cells.append(cell)
    comparisons = ('KM_minus_R0', 'GB_minus_R0', 'GB_minus_KM')
    overall = {key: signs([r[key] for r in rows]) for key in comparisons}
    by_seed = {str(seed): {key: signs([r[key] for r in rows if r['student_seed'] == seed]) for key in comparisons} for seed in (0, 1, 2)}
    cell_signs = {key: signs([c[key+'_mean'] for c in cells]) for key in comparisons}
    generic = all(by_seed[str(s)]['KM_minus_R0']['mean'] > 0 for s in (0, 1, 2)) and cell_signs['KM_minus_R0']['wins'] >= 6
    specific = overall['GB_minus_KM']['mean'] >= .20 and all(by_seed[str(s)]['GB_minus_KM']['mean'] > 0 for s in (0, 1, 2)) and cell_signs['GB_minus_KM']['wins'] >= 6
    summary = dict(overall=overall, by_seed=by_seed, cell_mean_signs=cell_signs,
                   decisions=dict(generic_region_support=generic, gb_specific_followup_gate=specific),
                   scope='Fixed teacher/split/noise/representation; student seeds only; no significance claim')
    csv_write('paired_results.csv', rows)
    csv_write('cell_summary.csv', cells)
    run.write(run.ROOT / 'summary.json', summary)
    run.write(run.ROOT / 'output_audit.json', dict(checkpoints=len(audit), passed=True, items=audit))
    run.verify()
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
