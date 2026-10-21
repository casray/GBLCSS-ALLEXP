from __future__ import annotations

import csv
import json
import statistics as stats

import numpy as np
import torch

import run_oracle as run


def export(name, rows):
    with (run.ROOT/name).open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def scores(probabilities, labels):
    pred = probabilities.argmax(1)
    correct = pred == labels
    recalls = [correct[labels == c].float().mean().item() for c in torch.unique(labels)]
    onehot = torch.nn.functional.one_hot(labels, probabilities.shape[1]).float()
    nll = -probabilities[torch.arange(len(labels)),labels].clamp_min(1e-12).log().mean()
    return dict(accuracy=float(correct.double().mean()*100),
                balanced_accuracy=float(np.mean(recalls)*100), nll=float(nll),
                brier=float((probabilities-onehot).square().sum(1).mean()))


def reload_pair(data, folder):
    teacher_dir=folder/'teacher'
    student_dir=folder/'student'
    teacher=torch.load(teacher_dir/'probabilities.pt',weights_only=True)
    student=torch.load(student_dir/'probabilities.pt',weights_only=True)
    checkpoint=torch.load(teacher_dir/'teacher.pt',weights_only=True)
    conf=checkpoint['settings']
    Model,_=run.previous.erase_classes()
    model=Model.GAT.get_model(data.x.shape[1],conf['hidden'],conf['embedding'],conf['layers'],
                             conf['heads'],conf['dropout'],'cuda',False,False,False,False).cuda().eval()
    model.load_state_dict(checkpoint['state'])
    with torch.no_grad():
        observed=run.previous.probe_probabilities(model(data.x,data.edge_index),
                  tuple(item.cuda() for item in conf['probe_state']),data.num_classes).cpu()
    teacher_difference=float((observed-teacher).abs().max())
    assert teacher_difference <= 1e-6 and torch.equal(observed.argmax(1),teacher.argmax(1))
    student_model=run.previous.MLP(data.x.shape[1],64,data.num_classes,2,.5).cuda().eval()
    student_model.load_state_dict(torch.load(student_dir/'student.pt',weights_only=True))
    with torch.no_grad():
        observed_student=student_model(data.x).softmax(1).cpu()
    student_difference=float((observed_student-student).abs().max())
    assert student_difference <= 1e-6 and torch.equal(observed_student.argmax(1),student.argmax(1))
    teacher_epochs=[json.loads(s) for s in (teacher_dir/'epochs.jsonl').read_text().splitlines()]
    student_epochs=[json.loads(s) for s in (student_dir/'epochs.jsonl').read_text().splitlines()]
    best_t=max(teacher_epochs,key=lambda x:x['valid'])
    best_s=max(student_epochs,key=lambda x:x['valid_correct'])
    result_t=run.read(teacher_dir/'result.json')
    result_s=run.read(student_dir/'result.json')
    assert best_t['epoch'] == result_t['best_epoch']
    assert best_s['epoch'] == result_s['best_epoch']
    assert best_s['valid_correct'] == result_s['valid_correct']
    assert int((teacher[data.val_index.cpu()].argmax(1)==data.y[data.val_index].cpu()).sum()) == result_t['valid_correct']
    student_config=run.read(student_dir/'config.json')
    return teacher,student,dict(teacher_max_difference=teacher_difference,
           teacher_argmax_identical=True,student_max_difference=student_difference,
           student_exact=torch.equal(observed_student,student),validation_selection_checked=True,
           student_initial_state_hash=student_config['initial_state_hash'],
           student_initial_rng_hash=student_config['initial_rng_hash'],
           teacher_epoch=result_t['best_epoch'],student_epoch=result_s['best_epoch'])


def main():
    assert run.read(run.ROOT/'execution.json')['status'] == 'ok'
    run.verify()
    rows,audits=[],[]
    for dataset,rate in run.cells():
        protocol=torch.load(run.source_folder(dataset,rate)/'protocol.pt',weights_only=True)
        test=protocol['test']
        truth=protocol['clean'][test]
        data,_=run.data_for(dataset,rate,'A_noisy')
        data=run.previous.gpu_data(data)
        initial=[]
        teacher_initial=[]
        for arm in run.ARMS:
            folder=run.run_folder(dataset,rate,arm)
            teacher,student,audit=reload_pair(data,folder)
            audit.update(dataset=dataset,rate=rate,arm=arm,folder=str(folder))
            audits.append(audit)
            initial.append((audit['student_initial_state_hash'],audit['student_initial_rng_hash']))
            conf=run.read(folder/'teacher/config.json')
            teacher_initial.append((conf['initial_state_hash'],conf['initial_rng_hash']))
            t,s=teacher[test],student[test]
            correct_t=t.argmax(1)==truth
            correct_s=s.argmax(1)==truth
            _,intervention=run.data_for(dataset,rate,arm)
            row=dict(dataset=dataset,rate=rate,seed=run.SEED,arm=arm,supervised=intervention['supervised'],
                     test_nodes=len(test),teacher_epoch=audit['teacher_epoch'],student_epoch=audit['student_epoch'],
                     teacher_folder=str(folder/'teacher'),student_folder=str(folder/'student'),
                     **{'teacher_'+k:v for k,v in scores(t,truth).items()},
                     **{'student_'+k:v for k,v in scores(s,truth).items()},
                     teacher_correct_student_wrong_pct=float((correct_t & ~correct_s).double().mean()*100),
                     teacher_wrong_student_correct_pct=float((~correct_t & correct_s).double().mean()*100),
                     teacher_student_disagreement_pct=float((t.argmax(1)!=s.argmax(1)).double().mean()*100))
            row['student_minus_teacher_pp']=row['student_accuracy']-row['teacher_accuracy']
            rows.append(row)
        assert len(set(initial))==1,(dataset,rate,initial)
        assert len(set(teacher_initial))==1,(dataset,rate,teacher_initial)
        del data
        torch.cuda.empty_cache()
    export('results.csv',rows)
    pairs=[]
    for dataset,rate in run.cells():
        cell={r['arm']:r for r in rows if r['dataset']==dataset and r['rate']==rate}
        row=dict(dataset=dataset,rate=rate,seed=run.SEED)
        for first,second,key in [('B_all_clean','A_noisy','B_minus_A'),('C_oracle_filter','A_noisy','C_minus_A'),
                                 ('B_all_clean','C_oracle_filter','B_minus_C'),('C_oracle_filter','D_matched_clean','C_minus_D')]:
            for role in ('teacher','student'):
                row[key+'_'+role+'_pp']=cell[first][role+'_accuracy']-cell[second][role+'_accuracy']
        b=cell['B_all_clean']
        row['clean_student_minus_teacher_pp']=b['student_minus_teacher_pp']
        pairs.append(row)
    export('paired_deltas.csv',pairs)
    summaries=[]
    for arm in run.ARMS:
        selected=[r for r in rows if r['arm']==arm]
        summaries.append(dict(arm=arm,logical_cells=len(selected),physical_pairs=3 if arm=='B_all_clean' else 6,
                  teacher_accuracy=stats.mean(r['teacher_accuracy'] for r in selected),
                  student_accuracy=stats.mean(r['student_accuracy'] for r in selected),
                  student_minus_teacher_pp=stats.mean(r['student_minus_teacher_pp'] for r in selected),
                  teacher_correct_student_wrong_pct=stats.mean(r['teacher_correct_student_wrong_pct'] for r in selected),
                  teacher_wrong_student_correct_pct=stats.mean(r['teacher_wrong_student_correct_pct'] for r in selected)))
    export('macro_results.csv',summaries)
    mean={key:stats.mean(r[key] for r in pairs) for key in pairs[0] if key.endswith('_pp')}
    teacher_positive=sum(r['B_minus_A_teacher_pp']>0 for r in pairs)
    student_positive=sum(r['B_minus_A_student_pp']>0 for r in pairs)
    teacher_opportunity=mean['B_minus_A_teacher_pp']>=1. and teacher_positive>=4
    student_opportunity=teacher_opportunity and mean['B_minus_A_student_pp']>=.5 and student_positive>=4
    correction_opportunity=mean['B_minus_C_teacher_pp']>=1. and sum(r['B_minus_C_teacher_pp']>0 for r in pairs)>=4
    clean_student_lag=sum(r['clean_student_minus_teacher_pp']<=-2. for r in pairs)
    transfer_warning=(teacher_opportunity and mean['B_minus_A_student_pp']<.25) or clean_student_lag>=4
    decision=dict(stage='single-protocol-seed oracle screen',mean_paired_deltas=mean,
          teacher_opportunity=teacher_opportunity,transferred_opportunity=student_opportunity,
          recovering_discarded_supervision=correction_opportunity,transfer_warning=transfer_warning,
          teacher_positive_cells=teacher_positive,student_positive_cells=student_positive,
          clean_student_lag_cells=clean_student_lag,statistically_confirmed=False,
          theoretical_ceiling_established=False,new_method_started=False)
    run.write(run.ROOT/'decision.json',decision)
    run.verify()
    run.write(run.ROOT/'audit.json',dict(passed=True,original_sources_unchanged=True,checks=audits,
          validation_selection_checked=True,student_initializations_match=True,
          new_teacher_initializations_match=True,analyze_source_hash=run.sha(run.ROOT/'analyze.py')))
    print(json.dumps(decision,indent=2),flush=True)
    for row in rows:
        print(row['dataset'],row['rate'],row['arm'],round(row['teacher_accuracy'],3),round(row['student_accuracy'],3),flush=True)


if __name__=='__main__':
    torch.set_num_threads(4)
    main()
