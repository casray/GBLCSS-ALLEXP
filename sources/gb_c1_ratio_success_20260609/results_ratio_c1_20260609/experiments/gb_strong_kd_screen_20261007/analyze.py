from __future__ import annotations

import csv
import json
from pathlib import Path
import statistics

import torch

import run_challenge as run
from selection import choose

ROOT = run.ROOT
read, write = run.read, run.write


def save_csv(name, rows):
    with (ROOT / name).open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stats(values):
    return dict(mean_pp=statistics.mean(values), wins=sum(x > 1e-10 for x in values),
                ties=sum(abs(x) <= 1e-10 for x in values), losses=sum(x < -1e-10 for x in values),
                minimum_pp=min(values), maximum_pp=max(values), seed_std=None, student_seeds=[0])


def main():
    run.old.require_gpu()
    run.verify()
    assert read(ROOT / 'execution.json')['status'] == 'ok'
    rows = []
    lookup = {}
    for dataset, rate, family, temperature, region in run.all_cells():
        folder = run.folder_for(dataset, rate, family, temperature, region)
        result = read(folder / 'result.json')
        assert result['status'] == 'ok'
        row = dict(dataset=dataset, noise_rate=rate, family=family, temperature=temperature,
                   region=region, valid_correct=result['valid_correct'], valid_total=result['valid_total'],
                   best_epoch=result['best_epoch'], folder=str(folder))
        rows.append(row)
        lookup[dataset, rate, family, temperature, region] = row
    selections = []
    for dataset in run.DATASETS:
        for rate in (.4, .6):
            no_region = []
            with_region = []
            for family, candidates in run.CANDIDATES.items():
                a = choose([lookup[dataset, rate, family, t, False] for t in candidates])
                matched = lookup[dataset, rate, family, a['temperature'], True]
                b = choose([lookup[dataset, rate, family, t, True] for t in candidates])
                no_region.append(a)
                with_region.append(b)
                selections.append(dict(dataset=dataset, noise_rate=rate, comparison=family,
                                       base=a['folder'], matched_region=matched['folder'], selected_region=b['folder']))
            a, b = choose(no_region), choose(with_region)
            matched = lookup[dataset, rate, a['family'], a['temperature'], True]
            selections.append(dict(dataset=dataset, noise_rate=rate, comparison='overall',
                                   base=a['folder'], matched_region=matched['folder'], selected_region=b['folder']))
    # The selection manifest is persisted before the test labels are loaded.
    write(ROOT / 'validation_selection.json', dict(selections=selections, candidate_order=run.CANDIDATES))
    audit, fixed = [], []
    for dataset in run.DATASETS:
        for rate in (.4, .6):
            data, protocol = run.previous.load_data(dataset, rate)
            data = run.previous.gpu_data(data)
            for row in [r for r in rows if r['dataset'] == dataset and r['noise_rate'] == rate]:
                folder = Path(row['folder'])
                config = read(folder / 'config.json')
                assert config['protocol'] == protocol
                reference = read(run.stable.folder_for(dataset, rate, 0, 'R0')/'config.json')
                assert config['initial_state_hash'] == reference['initial_state_hash']
                assert config['initial_rng_hash'] == reference['initial_rng_hash']
                epochs = [json.loads(line) for line in (folder/'epochs.jsonl').read_text().splitlines()]
                assert [x['epoch'] for x in epochs] == list(range(1, 201))
                assert max(epochs, key=lambda x: x['valid'])['epoch'] == row['best_epoch']
                prob = torch.load(folder/'student_probabilities.pt', weights_only=True).cuda()
                result = run.old.exact_metrics(prob, data)
                assert result['valid_correct'] == row['valid_correct']
                row.update(test_accuracy=100*result['student_test'], test_correct=result['test_correct'],
                           test_total=result['test_total'], valid_accuracy=100*result['student_valid'])
                model = run.previous.MLP(data.x.shape[1], 64, data.num_classes, 2, .5).cuda()
                model.load_state_dict(torch.load(folder/'student.pt', weights_only=True))
                model.eval()
                with torch.no_grad():
                    recovered = model(data.x).softmax(1)
                error = float((recovered-prob).abs().max())
                assert error == 0, (row, error)
                audit.append(dict(folder=str(folder), reload_max_difference=error))
                trajectory = [json.loads(x) for x in (folder/'trajectory.jsonl').read_text().splitlines()]
                row['median_region_to_node_gradient_ratio'] = statistics.median(x['region_to_node_gradient_ratio'] for x in trajectory)
                row['mean_gradient_cosine'] = statistics.mean(x['gradient_cosine'] for x in trajectory)
                for epoch in run.stable.SNAPSHOTS:
                    p = torch.load(folder/f'epoch{epoch}_probabilities.pt', weights_only=True).cuda()
                    met = run.old.exact_metrics(p, data)
                    fixed.append(dict(dataset=dataset, noise_rate=rate, family=row['family'],
                                      temperature=row['temperature'], region=row['region'], epoch=epoch,
                                      test_accuracy=100*met['student_test']))
                del model
            print(f'AUDITED {dataset} u{rate}', flush=True)
    by_folder = {r['folder']: r for r in rows}
    paired = []
    for choice in selections:
        a, m, b = (by_folder[choice[k]] for k in ('base', 'matched_region', 'selected_region'))
        assert a['test_total'] == m['test_total'] == b['test_total']
        paired.append(dict(dataset=choice['dataset'], noise_rate=choice['noise_rate'], comparison=choice['comparison'],
                           base_family=a['family'], base_temperature=a['temperature'],
                           independent_region_family=b['family'], independent_region_temperature=b['temperature'],
                           base_accuracy=a['test_accuracy'], matched_region_accuracy=m['test_accuracy'],
                           independent_region_accuracy=b['test_accuracy'],
                           matched_delta_pp=100*(m['test_correct']-a['test_correct'])/a['test_total'],
                           independent_delta_pp=100*(b['test_correct']-a['test_correct'])/a['test_total']))
    summary = {}
    for family in list(run.CANDIDATES) + ['overall']:
        subset = [x for x in paired if x['comparison'] == family]
        summary[family] = dict(matched=stats([x['matched_delta_pp'] for x in subset]),
                               independent=stats([x['independent_delta_pp'] for x in subset]))
    overall = summary['overall']['matched']
    summary['continue_gate'] = overall['mean_pp'] >= .2 and overall['wins'] >= 7
    summary['scope'] = 'one student seed; fixed split/noise/teacher/GRACE seed0; screening, not significance or SOTA'
    save_csv('all_candidates.csv', rows)
    save_csv('paired_results.csv', paired)
    save_csv('fixed_epoch_results.csv', fixed)
    write(ROOT/'output_audit.json', dict(passed=True, selected_models=len(audit), checks=audit))
    write(ROOT/'summary.json', summary)
    run.verify()
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
