from __future__ import annotations

import csv
import json
import statistics as stats

import torch
import run_protocol as run


def accuracy(probabilities, data):
    index = data.test_index.cpu()
    labels = data.y.cpu()[index]
    return int((probabilities[index].argmax(1) == labels).sum()) / len(index) * 100


def summarize(rows):
    delta = [r['delta_pp'] for r in rows]
    return dict(mean_delta_pp=stats.mean(delta), wins=sum(x>0 for x in delta),
                ties=sum(x==0 for x in delta), losses=sum(x<0 for x in delta))


def main():
    run.old.require_gpu()
    run.verify()
    assert run.read(run.ROOT/'execution.json')['status'] == 'ok'
    rows, audit = [], []
    for dataset, rate, seed in run.cells():
        data, meta, _ = run.load_data(dataset, rate, seed)
        folder = run.folder_for(dataset,rate,seed)
        teacher = torch.load(folder/'teacher/probabilities.pt', weights_only=True)
        row = dict(dataset=dataset, rate=rate, seed=seed, teacher=accuracy(teacher,data))
        data = run.previous.gpu_data(data)
        configs = [run.read(folder/f'GB{arm}/config.json') for arm in (0,1)]
        for key in ('initial_state_hash','initial_rng_hash','teacher_hash','assignment_hash'):
            assert configs[0][key] == configs[1][key]
        assert configs[0]['assignment_hash'] == meta['assignment_hash']
        assert configs[0]['teacher_hash'] == run.old.state_hash({'probabilities': teacher})
        for arm in (0,1):
            path = folder/f'GB{arm}'
            result = run.read(path/'result.json')
            epochs = [json.loads(s) for s in (path/'epochs.jsonl').read_text().splitlines()]
            assert len(epochs) == 200
            best = max(epochs,key=lambda r:r['valid_correct'])
            assert best['epoch'] == result['best_epoch']
            assert best['valid_correct'] == result['valid_correct']
            model = run.previous.MLP(data.x.shape[1],64,data.num_classes,2,.5).cuda().eval()
            for stage, state_name, prob_name in (
                    ('selected','student.pt','probabilities.pt'),
                    ('fixed200','epoch200.pt','epoch200_probabilities.pt')):
                model.load_state_dict(torch.load(path/state_name,weights_only=True))
                with torch.no_grad():
                    observed = model(data.x).softmax(1).cpu()
                saved = torch.load(path/prob_name,weights_only=True)
                exact = torch.equal(observed,saved)
                audit.append(dict(dataset=dataset, rate=rate, seed=seed, arm=arm,
                                  stage=stage, exact=exact, max_difference=float((observed-saved).abs().max())))
                assert exact, audit[-1]
                row[f'GB{arm}_{stage}'] = accuracy(saved,data)
            row[f'GB{arm}_epoch'] = result['best_epoch']
            del model
        checkpoint = torch.load(folder/'teacher/teacher.pt',weights_only=True)
        conf = checkpoint['settings']
        Model,_ = run.previous.erase_classes()
        model = Model.GAT.get_model(data.x.shape[1],conf['hidden'],conf['embedding'],
                conf['layers'],conf['heads'],conf['dropout'],'cuda',False,False,False,False).cuda().eval()
        model.load_state_dict(checkpoint['state'])
        with torch.no_grad():
            observed = run.previous.probe_probabilities(model(data.x,data.edge_index),
                      tuple(x.cuda() for x in conf['probe_state']),data.num_classes).cpu()
        audit.append(dict(dataset=dataset,rate=rate,seed=seed,stage='teacher',
                          exact=torch.equal(observed,teacher),
                          class_predictions_identical=torch.equal(observed.argmax(1),teacher.argmax(1)),
                          max_difference=float((observed-teacher).abs().max())))
        assert audit[-1]['max_difference'] <= 1e-6 and audit[-1]['class_predictions_identical'], audit[-1]
        teacher_epochs=[json.loads(s) for s in (folder/'teacher/epochs.jsonl').read_text().splitlines()]
        teacher_result=run.read(folder/'teacher/result.json')
        assert max(teacher_epochs,key=lambda r:r['valid'])['epoch'] == teacher_result['best_epoch']
        saved_correct=int((teacher[data.val_index.cpu()].argmax(1)==data.y[data.val_index].cpu()).sum())
        assert saved_correct == teacher_result['valid_correct']
        row['delta_pp'] = row['GB1_selected']-row['GB0_selected']
        row['fixed200_delta_pp'] = row['GB1_fixed200']-row['GB0_fixed200']
        row['GB0_teacher_gap_pp'] = row['GB0_selected']-row['teacher']
        row['GB1_teacher_gap_pp'] = row['GB1_selected']-row['teacher']
        rows.append(row)
        del model,data
        torch.cuda.empty_cache()
    with (run.ROOT/'paired_results.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    cell_rows=[]
    for dataset in run.DATASETS:
        for rate in (.4,.6):
            pair=[r for r in rows if r['dataset']==dataset and r['rate']==rate]
            cell=dict(dataset=dataset,rate=rate)
            for key in ('teacher','GB0_selected','GB1_selected','delta_pp','fixed200_delta_pp'):
                cell[key+'_mean']=stats.mean(r[key] for r in pair)
                cell[key+'_sample_std']=stats.stdev(r[key] for r in pair)
            cell_rows.append(cell)
    with (run.ROOT/'cell_summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=cell_rows[0].keys())
        writer.writeheader()
        writer.writerows(cell_rows)
    summary=summarize(rows)
    summary['by_seed']={str(seed):summarize([r for r in rows if r['seed']==seed]) for seed in (1,2)}
    summary['positive_cells']=sum(r['delta_pp_mean']>0 for r in cell_rows)
    summary['fixed200_mean_delta_pp']=stats.mean(r['fixed200_delta_pp'] for r in rows)
    summary['gate_passed']=(summary['mean_delta_pp']>=.2 and summary['positive_cells']>=7
                           and all(s['mean_delta_pp']>0 for s in summary['by_seed'].values()))
    summary['teacher_mean']=stats.mean(r['teacher'] for r in rows)
    summary['DIST_mean']=stats.mean(r['GB0_selected'] for r in rows)
    summary['DIST_GB_mean']=stats.mean(r['GB1_selected'] for r in rows)
    summary['crosses_teacher_DIST']=sum(r['GB0_teacher_gap_pp']>=0 for r in rows)
    summary['crosses_teacher_DIST_GB']=sum(r['GB1_teacher_gap_pp']>=0 for r in rows)
    run.write(run.ROOT/'summary.json',summary)
    run.write(run.ROOT/'reload_audit.json',dict(passed=True,checks=audit))
    run.verify()
    print(json.dumps(summary,indent=2),flush=True)
    for row in cell_rows:
        print(row['dataset'],row['rate'],row['delta_pp_mean'],row['delta_pp_sample_std'],flush=True)


if __name__=='__main__':
    main()
