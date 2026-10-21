import csv
import json
import statistics as stats
import numpy as np
import torch
from torch.nn import functional as F
import run_kmeans as run


def write_csv(name, rows):
    with (run.ROOT / name).open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    run.old.require_gpu()
    run.verify()
    sources, fixed, selected, teachers = [], [], [], []
    for dataset in run.DATASETS:
        config = run.read(run.ROOT / 'partitions' / dataset / 'config.json')
        main = run.read(run.ROOT.parent / 'main_protocol_replay_20260922' / 'matrix' / dataset / 'default' / 'seed0.json')
        assert run.Path(main['ball_embedding_path']).resolve() == run.Path(config['embedding_source']).resolve()
        assert main['ball_assignment_hash'] == config['gb_assignment_hash']
        sources.append(dict(dataset=dataset, source_matches_main=True, gb_assignment_matches_main=True,
                            group_count=config['count'], gb_min=min(config['gb_sizes']),
                            km_min=min(config['sizes']), gb_max=max(config['gb_sizes']),
                            km_max=max(config['sizes']), ari=config['ari_with_gb']))
        for rate in (.4, .6):
            data, _ = run.previous.load_data(dataset, rate)
            cached = torch.load(run.old.cache_folder(dataset, rate) / 'regional.pt', weights_only=True)
            teacher = cached['teacher'].cuda()
            regions = dict(GB=run.stable.RegionalKL(teacher, cached['assignments']['R1'].cuda()),
                           KM=run.stable.RegionalKL(teacher, torch.from_numpy(np.load(run.ROOT / 'partitions' / dataset / 'assignment.npy')).cuda()))
            teacher_accuracy = 100*float((teacher.cpu()[data.test_index].argmax(1) == data.y[data.test_index]).double().mean())
            teachers.append(dict(dataset=dataset, rate=rate, teacher_test=teacher_accuracy))
            for seed in (0, 1, 2):
                for arm in ('R0', 'R1', 'KM'):
                    folder = run.folder_for(dataset, rate, seed) if arm == 'KM' else run.stable.folder_for(dataset, rate, seed, arm)
                    for tag in ('student', 'epoch20', 'epoch50', 'epoch100', 'epoch200'):
                        probabilities = torch.load(folder / f'{tag}_probabilities.pt', weights_only=True)
                        assert torch.isfinite(probabilities).all()
                        torch.testing.assert_close(probabilities.sum(1), torch.ones(len(probabilities)), atol=1e-6, rtol=1e-6)
                        accuracy = 100*float((probabilities[data.test_index].argmax(1) == data.y[data.test_index]).double().mean())
                        row = dict(dataset=dataset, rate=rate, seed=seed, arm=arm, checkpoint=tag, test=accuracy,
                                   teacher_gap=teacher_accuracy-accuracy)
                        if tag == 'student':
                            values = probabilities.cuda()
                            row.update(pointwise_kl=float(F.kl_div(values.clamp_min(1e-30).log(), teacher, reduction='batchmean')),
                                       common_gb_region_kl=float(regions['GB'](values)),
                                       common_km_region_kl=float(regions['KM'](values)))
                            selected.append(row)
                        else:
                            fixed.append(row)
    fixed_summary = {}
    for epoch in (20, 50, 100, 200):
        subset = [r for r in fixed if r['checkpoint'] == f'epoch{epoch}']
        paired = []
        for dataset in run.DATASETS:
            for rate in (.4, .6):
                for seed in (0, 1, 2):
                    values = {r['arm']: r['test'] for r in subset if (r['dataset'], r['rate'], r['seed']) == (dataset, rate, seed)}
                    paired.append(dict(KM_minus_R0=values['KM']-values['R0'], GB_minus_KM=values['R1']-values['KM']))
        fixed_summary[str(epoch)] = {k: dict(mean=stats.mean(r[k] for r in paired),
                                       wins=sum(r[k] > 1e-9 for r in paired),
                                       ties=sum(abs(r[k]) <= 1e-9 for r in paired)) for k in paired[0]}
    source_summary = dict(passed=True, sources=sources, fixed_epoch=fixed_summary,
                         frozen_inputs_sha256=run.sha(run.ROOT / 'frozen_inputs.json'),
                         analysis_sha256=run.sha(run.ROOT / 'analyze.py'),
                         diagnostics_sha256=run.sha(run.ROOT / 'diagnostics.py'))
    run.write(run.ROOT / 'diagnostics_summary.json', source_summary)
    write_csv('fixed_epoch_results.csv', fixed)
    write_csv('selected_diagnostics.csv', selected)
    write_csv('teacher_results.csv', teachers)
    run.verify()
    print(json.dumps(source_summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
